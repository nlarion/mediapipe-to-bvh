import cv2
import mediapipe as mp
import numpy as np
import os
import argparse
from tqdm import tqdm
import math
import time
from dataclasses import dataclass

# MediaPipe setup
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

@dataclass
class EmptyLandmark:
    """Simple class to substitute for MediaPipe landmarks when needed"""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    visibility: float = 0.0

class Joint:
    """Class representing a joint in the BVH skeleton"""
    def __init__(self, name, parent=None):
        self.name = name
        self.parent = parent
        self.children = []
        self.offset = np.zeros(3)  # Offset from parent to this joint
        self.channels = []
        self.rotation_order = 'XYZ' # Using XYZ order
        # self.positions = [] # These were not used from original Joint class
        # self.rotations = [] # These were not used from original Joint class
        
    def add_child(self, child):
        self.children.append(child)

# --- Math Helper Functions ---
def normalize_vector(v):
    norm = np.linalg.norm(v)
    if norm < 1e-10: # Avoid division by zero for very small vectors
        return np.zeros_like(v)
    return v / norm

def euler_to_rotation_matrix(euler_angles_deg, order='XYZ'):
    """Convert euler angles (degrees) to rotation matrix (intrinsic rotations).
       For 'XYZ' order, this means R = Rz @ Ry @ Rx.
    """
    euler_angles_rad = np.radians(euler_angles_deg)
    x, y, z = euler_angles_rad
    
    # Rotation matrices for each axis
    Rx = np.array([
        [1, 0, 0],
        [0, math.cos(x), -math.sin(x)],
        [0, math.sin(x), math.cos(x)]
    ])
    Ry = np.array([
        [math.cos(y), 0, math.sin(y)],
        [0, 1, 0],
        [-math.sin(y), 0, math.cos(y)]
    ])
    Rz = np.array([
        [math.cos(z), -math.sin(z), 0],
        [math.sin(z), math.cos(z), 0],
        [0, 0, 1]
    ])
    
    if order == 'XYZ':
        # Apply in order: Rz, then Ry, then Rx for intrinsic XYZ
        # (meaning rotations are about the NEWLY transformed axes)
        # If extrinsic, it would be Rx @ Ry @ Rz.
        # The common interpretation for "Euler angles XYZ" often means R = Rz Ry Rx
        R = Rz @ Ry @ Rx
    elif order == 'ZYX': # Common alternative for intrinsic ZYX: Rx @ Ry @ Rz
        R = Rx @ Ry @ Rz
    else:
        # Default to original script's interpretation if different, but ensure consistency
        # The original preview_bvh's euler_to_rotation_matrix was Rz @ Ry @ Rx for XYZ input
        # This is kept.
        R = Rz @ Ry @ Rx 
    return R

def rotation_matrix_to_euler(R, order='XYZ'):
    """Convert rotation matrix to Euler angles (degrees).
       Inverse of euler_to_rotation_matrix for 'XYZ' (R = Rz Ry Rx).
    """
    sy = math.sqrt(R[0,0]**2 + R[1,0]**2) # sqrt(Rz_c^2 * Ry_c^2 + Rz_s^2 * Ry_c^2) = Ry_c
    singular = sy < 1e-6

    if order == 'XYZ':
        if not singular:
            x_rad = math.atan2(R[2,1], R[2,2]) # atan2(Ry_s*Rx_c, Ry_c*Rx_c)
            y_rad = math.atan2(-R[2,0], sy)    # atan2(-(-Ry_s), Ry_c)
            z_rad = math.atan2(R[1,0], R[0,0]) # atan2(Rz_s*Ry_c, Rz_c*Ry_c)
        else: # Gimbal lock
            # When cos(y) is close to 0 (y is +/- 90 deg)
            # R[2,0] = -sin(y) -> if y = pi/2, R[2,0] = -1. if y = -pi/2, R[2,0] = 1
            # Assume z = 0
            x_rad = math.atan2(-R[1,2], R[1,1]) # (Original from axis_angle_to_euler)
            y_rad = math.atan2(-R[2,0], sy)
            z_rad = 0
    else:
        raise ValueError(f"Rotation order {order} not yet supported for matrix_to_euler accurately.")

    return np.array([x_rad, y_rad, z_rad]) * (180.0 / math.pi)


def get_rotation_axis_angle_from_vectors(v_from, v_to):
    """ Get axis-angle rotation (vector where direction is axis, mag is angle) to align v_from with v_to. """
    v_from_n = normalize_vector(v_from)
    v_to_n = normalize_vector(v_to)
    
    cross_prod = np.cross(v_from_n, v_to_n)
    dot_prod = np.dot(v_from_n, v_to_n)
    
    # Clamp dot_prod to avoid acos domain errors due to precision
    angle_rad = math.acos(np.clip(dot_prod, -1.0, 1.0))
    
    if np.linalg.norm(cross_prod) > 1e-10: # Vectors are not collinear
        axis = normalize_vector(cross_prod)
        return axis * angle_rad
    elif dot_prod > 0.99999: # Vectors are nearly identical (parallel and same direction)
        return np.zeros(3) # No rotation needed
    else: # Vectors are opposite (parallel and opposite direction, 180 deg rotation)
        # Find an arbitrary axis perpendicular to v_from_n
        if abs(v_from_n[0]) < 0.9: # Check if not aligned with X to avoid cross product with itself
            axis = np.cross(v_from_n, np.array([1.0, 0.0, 0.0]))
        else: # Aligned with X, try Y
            axis = np.cross(v_from_n, np.array([0.0, 1.0, 0.0]))
        
        # If still no good axis (e.g., v_from_n was [0,0,1] and we crossed with [0,0,1]),
        # this case should be rare given the dot_prod check.
        if np.linalg.norm(axis) < 1e-10: # Fallback if still collinear after trying standard axes
             # This case means v_from_n was parallel to the chosen cross vector,
             # e.g. v_from_n = [1,0,0] and we crossed with [1,0,0].
             # Try a different fallback perpendicular axis construction.
             # If v_from_n is [1,0,0], use [0,1,0]. If [0,1,0], use [0,0,1]. If [0,0,1], use [1,0,0].
            if abs(v_from_n[0]) > 0.9 : axis = np.array([0.0, 1.0, 0.0])
            elif abs(v_from_n[1]) > 0.9 : axis = np.array([0.0, 0.0, 1.0])
            else : axis = np.array([1.0, 0.0, 0.0])

        return normalize_vector(axis) * angle_rad

def axis_angle_to_euler(axis_angle_rad, order='XYZ'):
    """Convert axis-angle rotation (radians) to Euler angles (degrees).
       Matches the one from the original BVH previewer.
    """
    angle_rad = np.linalg.norm(axis_angle_rad)
    
    if angle_rad < 1e-10:
        return np.zeros(3)
    
    axis = axis_angle_rad / angle_rad
    
    # Convert to rotation matrix (Rodrigues' formula)
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    t = 1.0 - c
    x, y, z = axis
    
    # Rotation matrix elements for XYZ order (R = Rz Ry Rx)
    # This matrix should be the one for which the Euler angle extraction below is valid.
    # R = Rot(axis, angle_rad)
    R_mat = np.array([
        [t*x*x + c,   t*x*y - z*s, t*x*z + y*s],
        [t*x*y + z*s, t*y*y + c,   t*y*z - x*s],
        [t*x*z - y*s, t*y*z + x*s, t*z*z + c]
    ])

    # Euler angle extraction for 'XYZ' intrinsic (Rz Ry Rx)
    # (same as used in preview_bvh's get_joint_positions -> euler_to_rotation_matrix & its implicit inverse)
    if order == 'XYZ':
        sy = math.sqrt(R_mat[0,0]**2 + R_mat[1,0]**2) # R_mat[0,0] = Rz_c*Ry_c, R_mat[1,0] = Rz_s*Ry_c -> sy = Ry_c
        singular = sy < 1e-6
        
        if not singular:
            # x_euler = math.atan2(R_mat[2,1]/sy, R_mat[2,2]/sy) # if sy = Ry_c, then R[2,1]/Ry_c = Rx_s, R[2,2]/Ry_c = Rx_c
            x_euler = math.atan2(R_mat[2,1], R_mat[2,2]) # from original script: atan2(r21, r22)
            y_euler = math.atan2(-R_mat[2,0], sy)        # from original script: atan2(-r20, sy)
            z_euler = math.atan2(R_mat[1,0], R_mat[0,0]) # from original script: atan2(r10, r00)
        else: # Gimbal lock
            x_euler = math.atan2(-R_mat[1,2], R_mat[1,1]) # from original script
            y_euler = math.atan2(-R_mat[2,0], sy)         # from original script
            z_euler = 0
    else:
        raise ValueError(f"Rotation order {order} not supported in axis_angle_to_euler")
    
    return np.array([x_euler, y_euler, z_euler]) * (180.0 / math.pi) # Convert to degrees

def get_rotation_from_vectors(v_from, v_to):
    """Get rotation matrix to align v_from with v_to using Rodrigues' formula."""
    # Get axis-angle representation
    axis_angle_rad = get_rotation_axis_angle_from_vectors(v_from, v_to)
    angle_rad = np.linalg.norm(axis_angle_rad)
    
    if angle_rad < 1e-10:
        return np.identity(3)
    
    axis = axis_angle_rad / angle_rad
    
    # Convert to rotation matrix using Rodrigues' formula
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    t = 1.0 - c
    x, y, z = axis
    
    # Rotation matrix
    R_mat = np.array([
        [t*x*x + c,   t*x*y - z*s, t*x*z + y*s],
        [t*x*y + z*s, t*y*y + c,   t*y*z - x*s],
        [t*x*z - y*s, t*y*z + x*s, t*z*z + c]
    ])
    
    return R_mat

# --- BVH Parsing (from original script, slightly modified for context) ---
def parse_bvh(file_path):
    """Parse a BVH file and extract hierarchy and motion data"""
    print(f"Loading BVH file: {file_path}")
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    if 'MOTION' in content:
        hierarchy_str, motion_str = content.split('MOTION', 1)
    else:
        print("Error: No MOTION section found in BVH file.")
        return None, None, None, None
    
    joints = {}
    joint_order = []
    parent_stack = []
    current_parent_name = None # Store name, not the object yet
    
    active_joint_name_for_config = None # For OFFSET and CHANNELS

    for line in hierarchy_str.split('\n'):
        line = line.strip()
        parts = line.split()
        if not parts:
            continue
            
        keyword = parts[0]

        if keyword == 'ROOT':
            joint_name = parts[1]
            joints[joint_name] = {
                'parent_name': None, 'children_names': [], 'offset': [0,0,0], 'channels': [], 'rot_order': 'XYZ' # default
            }
            active_joint_name_for_config = joint_name
            current_parent_name = joint_name
            joint_order.append(joint_name)
            
        elif keyword == 'JOINT':
            joint_name = parts[1]
            joints[joint_name] = {
                'parent_name': current_parent_name, 'children_names': [], 'offset': [0,0,0], 'channels': [], 'rot_order': 'XYZ'
            }
            if current_parent_name:
                joints[current_parent_name]['children_names'].append(joint_name)
            
            parent_stack.append(current_parent_name)
            current_parent_name = joint_name
            active_joint_name_for_config = joint_name
            joint_order.append(joint_name)
            
        elif keyword == 'End': # End Site
            # Create a dummy name for the end site if needed for structure, or handle appropriately
            end_site_name = f"{current_parent_name}_EndSite"
            joints[end_site_name] = {
                 'parent_name': current_parent_name, 'children_names': [], 'offset': [0,0,0], 'channels': [] , 'rot_order': 'XYZ'
            }
            if current_parent_name:
                joints[current_parent_name]['children_names'].append(end_site_name)
            active_joint_name_for_config = end_site_name # Offset will apply to this
            # No current_parent_name change, End Site doesn't become a parent
            # No joint_order.append for End Site if it's just for offset
            
        elif keyword == 'OFFSET':
            if active_joint_name_for_config:
                offset_vals = [float(x) for x in parts[1:]]
                joints[active_joint_name_for_config]['offset'] = offset_vals
                
        elif keyword == 'CHANNELS':
            if active_joint_name_for_config:
                num_channels = int(parts[1])
                channel_names = parts[2:2+num_channels]
                joints[active_joint_name_for_config]['channels'] = channel_names
                # Infer rotation order (simple check for ZXY, XYZ, etc.)
                rot_channels = [ch for ch in channel_names if 'rotation' in ch.lower()]
                if rot_channels:
                    order = "".join([ch[0].upper() for ch in rot_channels]) # Xrotation -> X
                    joints[active_joint_name_for_config]['rot_order'] = order

        elif keyword == '{':
            pass # Handled by JOINT/ROOT pushing to stack

        elif keyword == '}':
            if parent_stack:
                # If the active joint was an EndSite, its parent is current_parent_name
                # Otherwise, the active joint *was* current_parent_name
                # So, when '}' is encountered, we pop from parent_stack to go up one level
                current_parent_name = parent_stack.pop()
                active_joint_name_for_config = current_parent_name # Next offset/channels belong to this parent
            else:
                # End of root's definition or end of hierarchy
                current_parent_name = None
                active_joint_name_for_config = None
    
    motion_lines = motion_str.strip().split('\n')
    num_frames = None
    frame_time = None
    
    for line in motion_lines[:3]:
        line = line.strip()
        if 'Frames:' in line or 'FRAMES:' in line:
            try: num_frames = int(line.split(':')[-1].strip())
            except ValueError: return None, None, None, None
        elif 'Frame Time:' in line or 'FRAME TIME:' in line:
            try: frame_time = float(line.split(':')[-1].strip())
            except ValueError: return None, None, None, None
    
    if num_frames is None or frame_time is None: return None, None, None, None
    
    data_start_idx = 0
    for i, line in enumerate(motion_lines):
        if 'Frame Time:' in line or 'FRAME TIME:' in line:
            data_start_idx = i + 1; break
    
    motion_data = []
    for i in range(data_start_idx, len(motion_lines)):
        if motion_lines[i].strip():
            try:
                frame_values = [float(x) for x in motion_lines[i].strip().split()]
                motion_data.append(frame_values)
            except ValueError: continue
    
    return joints, joint_order, motion_data, frame_time

# --- BVH Preview (from original script, slightly modified for context) ---
def get_joint_global_positions_from_bvh_data(joints_dict, joint_order, frame_data_row):
    """Calculate global positions of joints from a single frame of BVH motion data."""
    
    joint_global_positions = {}
    joint_global_orientations = {} # Store as rotation matrices

    # Create a flat list of all channel names in order
    ordered_channels = []
    for joint_name in joint_order:
        ordered_channels.extend(joints_dict[joint_name]['channels'])

    data_idx = 0
    
    for joint_name in joint_order:
        joint_info = joints_dict[joint_name]
        parent_name = joint_info['parent_name']

        local_offset = np.array(joint_info['offset'])
        
        # Get rotation and position data for this joint from the frame_data_row
        num_joint_channels = len(joint_info['channels'])
        joint_motion_values = frame_data_row[data_idx : data_idx + num_joint_channels]
        data_idx += num_joint_channels

        position_channels = [val for name, val in zip(joint_info['channels'], joint_motion_values) if 'position' in name.lower()]
        rotation_channels_deg = [val for name, val in zip(joint_info['channels'], joint_motion_values) if 'rotation' in name.lower()]
        
        # Determine rotation order for this joint (e.g., 'XYZ', 'ZYX') from joint_info['rot_order']
        # Default to 'XYZ' if not specified or poorly parsed
        current_joint_rot_order = joint_info.get('rot_order', 'XYZ')
        if len(current_joint_rot_order) != 3: current_joint_rot_order = 'XYZ'


        # Calculate local rotation matrix
        if rotation_channels_deg:
             # Ensure rotation_channels_deg are in the order implied by current_joint_rot_order
             # This requires careful mapping if BVH channels are not e.g. X,Y,Z in order
             # For now, assume they are in the order matching current_joint_rot_order (e.g., Xrot, Yrot, Zrot vals for 'XYZ')
            R_local = euler_to_rotation_matrix(np.array(rotation_channels_deg), order=current_joint_rot_order)
        else:
            R_local = np.identity(3)

        # Calculate global position and orientation
        if parent_name is None: # Root joint
            # Position is directly from channels (if they exist)
            if len(position_channels) == 3:
                global_pos = np.array(position_channels)
            else: # If root has no position channels (e.g. static root)
                global_pos = local_offset # Offset from origin (0,0,0)
            
            global_orientation_matrix = R_local
        else:
            parent_global_pos = joint_global_positions[parent_name]
            parent_global_orientation_matrix = joint_global_orientations[parent_name]
            
            # Transform local offset by parent's global orientation
            global_pos = parent_global_pos + (parent_global_orientation_matrix @ local_offset)
            global_orientation_matrix = parent_global_orientation_matrix @ R_local
            
        joint_global_positions[joint_name] = global_pos
        joint_global_orientations[joint_name] = global_orientation_matrix
        
    return joint_global_positions


def preview_bvh_animation(bvh_file, speed_factor=1.0):
    try:
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation
        # from mpl_toolkits.mplot3d import Axes3D # Not explicitly used but fig.add_subplot needs it implicitly
        
        joints_dict, joint_order, motion_data, frame_time = parse_bvh(bvh_file)
        if None in (joints_dict, joint_order, motion_data, frame_time):
            print("Failed to parse BVH file for preview.")
            return
        
        connections = []
        for joint_name in joint_order:
            joint_info = joints_dict[joint_name]
            if joint_info['parent_name'] and joint_info['parent_name'] in joints_dict : # Ensure parent exists and not end site
                 # Do not draw from parent to an EndSite if EndSites are in joint_order but not meant for drawing bones
                 # This logic depends on how EndSites are handled by parse_bvh
                 is_child_end_site = "_EndSite" in joint_name # Heuristic
                 if not is_child_end_site:
                    connections.append((joint_info['parent_name'], joint_name))

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        all_positions_sample = []
        # Sample a few frames to estimate bounds (using the new global position calculation)
        for frame_idx in range(min(10, len(motion_data))):
            positions = get_joint_global_positions_from_bvh_data(joints_dict, joint_order, motion_data[frame_idx])
            for pos in positions.values():
                all_positions_sample.append(pos)
        
        if not all_positions_sample:
            min_vals, max_vals = np.array([-50,-50,-50]), np.array([50,50,50])
        else:
            all_positions_sample = np.array(all_positions_sample)
            min_vals = np.min(all_positions_sample, axis=0)
            max_vals = np.max(all_positions_sample, axis=0)

        center = (min_vals + max_vals) / 2
        max_range = np.max(max_vals - min_vals) / 2
        max_range = max(max_range, 50) # Ensure a minimum viewing range
        
        ax.set_xlim(center[0] - max_range, center[0] + max_range)
        ax.set_ylim(center[1] - max_range, center[1] + max_range)
        ax.set_zlim(center[2] - max_range, center[2] + max_range)
        ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
        ax.set_title('BVH Animation Preview')
        
        lines = [ax.plot([], [], [], 'b-')[0] for _ in connections]
        points_plot = ax.plot([], [], [], 'ro', ms=4)[0] # Changed for clarity
        frame_text = ax.text2D(0.02, 0.98, '', transform=ax.transAxes)
        
        def update(frame_idx):
            frame_motion_values = motion_data[frame_idx]
            current_positions = get_joint_global_positions_from_bvh_data(joints_dict, joint_order, frame_motion_values)
            
            for i, (p_name, c_name) in enumerate(connections):
                if p_name in current_positions and c_name in current_positions:
                    p_pos, c_pos = current_positions[p_name], current_positions[c_name]
                    lines[i].set_data([p_pos[0], c_pos[0]], [p_pos[1], c_pos[1]])
                    lines[i].set_3d_properties([p_pos[2], c_pos[2]])
                else: # Hide if missing
                    lines[i].set_data([],[]); lines[i].set_3d_properties([])
            
            # Plot all valid joint positions
            plot_xs, plot_ys, plot_zs = [], [], []
            for joint_name_to_plot in joint_order: # Plot all joints in order, not just connected ones
                if "_EndSite" not in joint_name_to_plot and joint_name_to_plot in current_positions : # Don't plot end sites
                    pos = current_positions[joint_name_to_plot]
                    plot_xs.append(pos[0]); plot_ys.append(pos[1]); plot_zs.append(pos[2])
            
            points_plot.set_data(plot_xs, plot_ys)
            points_plot.set_3d_properties(plot_zs)
            frame_text.set_text(f'Frame: {frame_idx}')
            return lines + [points_plot, frame_text]
        
        num_anim_frames = len(motion_data)
        interval_ms = frame_time * 1000 / speed_factor
        anim = FuncAnimation(fig, update, frames=num_anim_frames, interval=interval_ms, blit=True)
        plt.show()
        
    except ImportError: print("Matplotlib needed for BVH preview: pip install matplotlib")
    except Exception as e: print(f"Error previewing BVH: {e}"); import traceback; traceback.print_exc()

# --- Skeleton Creation and Landmark Processing ---
def create_skeleton():
    """Create a skeleton structure (Joint objects) that matches MediaPipe's pose landmarks."""
    hips = Joint("Hips")
    spine = Joint("Spine", hips); hips.add_child(spine)
    chest = Joint("Chest", spine); spine.add_child(chest)
    neck = Joint("Neck", chest); chest.add_child(neck)
    head = Joint("Head", neck); neck.add_child(head)
    
    left_shoulder = Joint("LeftShoulder", chest); chest.add_child(left_shoulder)
    left_arm = Joint("LeftArm", left_shoulder); left_shoulder.add_child(left_arm)
    left_forearm = Joint("LeftForeArm", left_arm); left_arm.add_child(left_forearm)
    left_hand = Joint("LeftHand", left_forearm); left_forearm.add_child(left_hand)
    
    right_shoulder = Joint("RightShoulder", chest); chest.add_child(right_shoulder)
    right_arm = Joint("RightArm", right_shoulder); right_shoulder.add_child(right_arm)
    right_forearm = Joint("RightForeArm", right_arm); right_arm.add_child(right_forearm)
    right_hand = Joint("RightHand", right_forearm); right_forearm.add_child(right_hand)
    
    left_up_leg = Joint("LeftUpLeg", hips); hips.add_child(left_up_leg)
    left_leg = Joint("LeftLeg", left_up_leg); left_up_leg.add_child(left_leg)
    left_foot = Joint("LeftFoot", left_leg); left_leg.add_child(left_foot)
    left_toe = Joint("LeftToeBase", left_foot); left_foot.add_child(left_toe)
    
    right_up_leg = Joint("RightUpLeg", hips); hips.add_child(right_up_leg)
    right_leg = Joint("RightLeg", right_up_leg); right_up_leg.add_child(right_leg)
    right_foot = Joint("RightFoot", right_leg); right_leg.add_child(right_foot)
    right_toe = Joint("RightToeBase", right_foot); right_foot.add_child(right_toe)
    
    return hips

def get_joint_mapping():
    """Map MediaPipe landmarks to BVH skeleton joints."""
    # Using lists of landmarks, average position will be taken.
    return {
        "Hips": [mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.RIGHT_HIP],
        "Spine": [mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.RIGHT_HIP, 
                  mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.RIGHT_SHOULDER], # Midpoint of hips & shoulders
        "Chest": [mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.RIGHT_SHOULDER],
        "Neck": [mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.RIGHT_SHOULDER, # Approx. between shoulders and ears
                 mp_pose.PoseLandmark.LEFT_EAR, mp_pose.PoseLandmark.RIGHT_EAR],
        "Head": [mp_pose.PoseLandmark.NOSE], # Or average of ears/eyes if more stable
        
        "LeftShoulder": [mp_pose.PoseLandmark.LEFT_SHOULDER],
        "LeftArm": [mp_pose.PoseLandmark.LEFT_ELBOW], # LeftArm joint is at the elbow
        "LeftForeArm": [mp_pose.PoseLandmark.LEFT_WRIST], # LeftForeArm joint is at the wrist
        "LeftHand": [mp_pose.PoseLandmark.LEFT_PINKY, mp_pose.PoseLandmark.LEFT_INDEX], # Mid hand
        
        "RightShoulder": [mp_pose.PoseLandmark.RIGHT_SHOULDER],
        "RightArm": [mp_pose.PoseLandmark.RIGHT_ELBOW],
        "RightForeArm": [mp_pose.PoseLandmark.RIGHT_WRIST],
        "RightHand": [mp_pose.PoseLandmark.RIGHT_PINKY, mp_pose.PoseLandmark.RIGHT_INDEX],
        
        "LeftUpLeg": [mp_pose.PoseLandmark.LEFT_HIP], # LeftUpLeg joint is at the hip
        "LeftLeg": [mp_pose.PoseLandmark.LEFT_KNEE], # LeftLeg joint is at the knee
        "LeftFoot": [mp_pose.PoseLandmark.LEFT_ANKLE], # LeftFoot joint is at the ankle
        "LeftToeBase": [mp_pose.PoseLandmark.LEFT_FOOT_INDEX],
        
        "RightUpLeg": [mp_pose.PoseLandmark.RIGHT_HIP],
        "RightLeg": [mp_pose.PoseLandmark.RIGHT_KNEE],
        "RightFoot": [mp_pose.PoseLandmark.RIGHT_ANKLE],
        "RightToeBase": [mp_pose.PoseLandmark.RIGHT_FOOT_INDEX]
    }

def get_landmark_position(landmarks_list, landmark_enum_val):
    """Safely get the position of a landmark by its enum value from the list of landmarks."""
    if landmarks_list and landmark_enum_val.value < len(landmarks_list): # Check index validity
        lm = landmarks_list[landmark_enum_val.value]
        if hasattr(lm, 'x') and hasattr(lm, 'y') and hasattr(lm, 'z') and \
           not (np.isnan(lm.x) or np.isnan(lm.y) or np.isnan(lm.z) or lm.visibility < 0.3): # Added visibility check
            return np.array([lm.x, lm.y, lm.z])
    return None

def get_joint_world_position(joint_name, frame_landmarks, joint_mapping_dict):
    """Get the average world position for a joint from its mapped MediaPipe landmarks for a single frame."""
    if joint_name not in joint_mapping_dict: return None
        
    landmark_indices = joint_mapping_dict[joint_name]
    positions = []
    for lm_enum in landmark_indices:
        pos = get_landmark_position(frame_landmarks, lm_enum)
        if pos is not None:
            positions.append(pos)
    
    if positions: return np.mean(positions, axis=0)
    return None


def build_skeleton_initial_offsets(skeleton_root_joint, ref_frame_landmarks, joint_mapping, scale=100.0):
    """
    Build the skeleton structure (calculates Joint.offset values) based on a reference frame.
    The Joint.offset is the vector from the joint's parent to the joint itself, in the parent's coordinate system.
    """
    all_joints_list = get_all_joints(skeleton_root_joint) # Get flat list for easy lookup
    
    # Calculate world positions for all joints in the reference frame
    ref_world_positions = {}
    for joint_obj in all_joints_list:
        pos = get_joint_world_position(joint_obj.name, ref_frame_landmarks, joint_mapping)
        if pos is not None:
            ref_world_positions[joint_obj.name] = pos * scale # Scale to BVH units
        # else:
            # print(f"Warning: Could not get reference position for {joint_obj.name}")

    # Recursively set offsets
    def set_offsets_recursive(current_joint_obj, parent_world_pos):
        current_world_pos = ref_world_positions.get(current_joint_obj.name)

        if current_world_pos is None: # If a joint's ref pos is missing
            # Try to use a default offset based on parent, or small fixed offset
            print(f"Warning: Missing ref pos for {current_joint_obj.name}. Using default offset.")
            # Default based on common bone directions if parent exists
            # This part needs careful thought for good defaults based on joint type
            if current_joint_obj.parent:
                 if "Leg" in current_joint_obj.name or "Foot" in current_joint_obj.name : current_joint_obj.offset = np.array([0.0, -10.0, 0.0]) # Down
                 elif "Arm" in current_joint_obj.name or "Hand" in current_joint_obj.name : current_joint_obj.offset = np.array([10.0, 0.0, 0.0]) # Assume right side, adjust if left
                 else: current_joint_obj.offset = np.array([0.0, 10.0, 0.0]) # Up
            else: current_joint_obj.offset = np.zeros(3) # Root at origin offset
        else:
            if current_joint_obj.parent is None: # Root joint
                current_joint_obj.offset = current_world_pos # Position relative to global origin (0,0,0)
            else:
                if parent_world_pos is None: # Parent was missing, problematic
                     print(f"Error: Parent {current_joint_obj.parent.name} of {current_joint_obj.name} missing ref_pos. Cannot set offset accurately.")
                     current_joint_obj.offset = np.array([0.0, 1.0, 0.0]) * scale * 0.1 # Small default
                else:
                     current_joint_obj.offset = current_world_pos - parent_world_pos
        
        # Ensure minimum offset length for stability, except for root's initial position.
        if current_joint_obj.parent is not None and np.linalg.norm(current_joint_obj.offset) < 1.0: # 1.0 unit (scaled)
            print(f"Warning: Offset for {current_joint_obj.name} is very small. Using default direction.")
            # Default based on typical bone direction to avoid zero-length bones
            if "Leg" in current_joint_obj.name or "Foot" in current_joint_obj.name: direction = np.array([0.0, -1.0, 0.05]) # Slightly forward for knees/ankles
            elif "Toe" in current_joint_obj.name: direction = np.array([0.0, 0.0, 1.0]) # Forward
            elif "Arm" in current_joint_obj.name or "Hand" in current_joint_obj.name:
                direction = np.array([-1.0 if "Left" in current_joint_obj.name else 1.0, 0.0, 0.0])
            elif "Shoulder" in current_joint_obj.name:
                direction = np.array([-1.0 if "Left" in current_joint_obj.name else 1.0, 0.1, 0.0]) # Slight up
            elif "Head" in current_joint_obj.name or "Neck" in current_joint_obj.name: direction = np.array([0.0, 1.0, 0.0])
            else: direction = np.array([0.0, 0.1, 0.0]) # Default small upward offset
            current_joint_obj.offset = normalize_vector(direction) * (scale * 0.1) # 10 cm default length

        for child_joint_obj in current_joint_obj.children:
            set_offsets_recursive(child_joint_obj, current_world_pos)

    # Root's parent world pos is global origin
    set_offsets_recursive(skeleton_root_joint, np.zeros(3))
    # The root's offset from (0,0,0) will be its initial position in BVH.
    # For BVH, ROOT offset is often 0,0,0, and its position channels define its start.
    # Let's adjust this: root offset for HIERARCHY is 0,0,0. Its position is motion data.
    skeleton_root_joint.offset = np.zeros(3)


def get_all_joints(skeleton_root_joint, include_end_sites=False):
    """Get a flat list of all Joint objects in the skeleton using BFS/DFS."""
    joints_list = []
    queue = [skeleton_root_joint]
    visited_names = set()

    while queue:
        current_joint = queue.pop(0)
        if current_joint.name in visited_names: continue
        visited_names.add(current_joint.name)

        is_end_site = "_EndSite" in current_joint.name # Heuristic from my bvh parser example
        if include_end_sites or not is_end_site:
            joints_list.append(current_joint)
        
        for child in current_joint.children:
            if child.name not in visited_names:
                queue.append(child)
    return joints_list

# --- Core Rotation Logic (New Implementation) ---
def _calculate_frame_rotations_recursive(
    current_joint_obj, parent_world_rotation_matrix,
    frame_landmarks, joint_mapping, skeleton_joint_dict, # For accessing any joint by name
    frame_local_eulers_output): # Dictionary to store results
    """
    Recursively calculates local Euler angles for each joint in the skeleton for a single frame.
    `skeleton_joint_dict` maps joint name to Joint object for easy access to rest offsets.
    """
    current_joint_name = current_joint_obj.name
    current_joint_world_lm_pos = get_joint_world_position(current_joint_name, frame_landmarks, joint_mapping)

    current_local_rotation_matrix = np.identity(3)

    if current_joint_obj.parent is None: # Root Joint (Hips)
        # Calculate world orientation for the Hips joint
        hips_lm_pos = current_joint_world_lm_pos
        # Use chest as a reference for "up" direction from hips, or spine if mapped differently
        chest_lm_pos = get_joint_world_position("Chest", frame_landmarks, joint_mapping) 
        if chest_lm_pos is None: # Fallback if chest is not visible
            chest_lm_pos = get_joint_world_position("Spine", frame_landmarks, joint_mapping)

        left_hip_lm = get_landmark_position(frame_landmarks, mp_pose.PoseLandmark.LEFT_HIP)
        right_hip_lm = get_landmark_position(frame_landmarks, mp_pose.PoseLandmark.RIGHT_HIP)

        if hips_lm_pos is not None and chest_lm_pos is not None and left_hip_lm is not None and right_hip_lm is not None:
            # BVH Y-up, X-right, Z-forward (common convention)
            # MediaPipe world: Y-up, X-right, Z-backward (towards person)
            # Assume direct mapping for now, adjust if BVH Z is forward
            
            # Up vector (Y-axis for Hips)
            up_vec = normalize_vector(chest_lm_pos - hips_lm_pos)
            
            # Side vector (X-axis for Hips)
            hip_lr_vec = normalize_vector(right_hip_lm - left_hip_lm) # Vector from Left to Right hip
            
            # Forward vector (Z-axis for Hips)
            # Ensure Z is pointing away from person if MP Z is backwards
            # If BVH Z is forward, then MP -Z direction is BVH +Z
            # For now, use cross product for orthogonality
            forward_vec = normalize_vector(np.cross(hip_lr_vec, up_vec)) # hip_lr_vec is X-like, up_vec is Y-like, so X cross Y = Z
            
            # Re-orthogonalize X-axis
            side_vec = normalize_vector(np.cross(up_vec, forward_vec)) # Y cross Z = X

            # Construct rotation matrix: columns are the new axes in world space
            # This matrix transforms from Hips local space to world space.
            # If BVH standard is Y-up, X-right, Z-forward for Hips local:
            # Col1 = X_axis_world, Col2 = Y_axis_world, Col3 = Z_axis_world
            current_local_rotation_matrix = np.array([side_vec, up_vec, forward_vec]).T
        else:
            # print(f"Warning: Root {current_joint_name} landmarks missing, using identity rotation.")
            current_local_rotation_matrix = np.identity(3)
        
        # For root, its "local" rotation IS its world orientation
        # The Euler angles are for its CHANNELS (world rotation)
        # Need to ensure rotation_matrix_to_euler uses the same convention as euler_to_rotation_matrix
        frame_local_eulers_output[current_joint_name] = rotation_matrix_to_euler(current_local_rotation_matrix, current_joint_obj.rotation_order)
    
    else: # Non-root joint
        if not current_joint_obj.children: # End effector (e.g. LeftHand, LeftToeBase end here in this structure)
            # For end effectors with no children to orient towards, use a neutral rotation or previous frame's.
            frame_local_eulers_output[current_joint_name] = np.zeros(3) 
            current_local_rotation_matrix = np.identity(3)
        else:
            # Use the first child to define the bone's orientation.
            # Could be improved by selecting a "primary" child or averaging.
            primary_child_obj = current_joint_obj.children[0]
            primary_child_name = primary_child_obj.name
            
            primary_child_lm_pos = get_joint_world_position(primary_child_name, frame_landmarks, joint_mapping)

            if current_joint_world_lm_pos is None or primary_child_lm_pos is None:
                # print(f"Warning: Landmark missing for {current_joint_name} or {primary_child_name}. Using zero local rotation.")
                frame_local_eulers_output[current_joint_name] = np.zeros(3)
                current_local_rotation_matrix = np.identity(3)
            else:
                # Rest pose bone vector: from current_joint to primary_child, in current_joint's local space.
                # This is simply the offset of the primary_child_obj.
                rest_bone_local = normalize_vector(skeleton_joint_dict[primary_child_name].offset)

                # Target bone vector: from current_joint to primary_child, using current frame landmarks, in world space.
                target_bone_world = normalize_vector(primary_child_lm_pos - current_joint_world_lm_pos)

                # We need to find the local rotation for current_joint.
                # This local rotation transforms rest_bone_local (in current_joint's space)
                # so that when combined with parent_world_rotation_matrix, it aligns with target_bone_world.
                # R_parent_world * R_local * rest_bone_local = target_bone_world
                # R_local * rest_bone_local = inv(R_parent_world) * target_bone_world
                
                target_bone_in_parent_space = np.linalg.inv(parent_world_rotation_matrix) @ target_bone_world
                target_bone_in_parent_space_n = normalize_vector(target_bone_in_parent_space)

                if np.linalg.norm(rest_bone_local) < 1e-5 or np.linalg.norm(target_bone_in_parent_space_n) < 1e-5:
                    # print(f"Warning: Near zero vector for {current_joint_name}-{primary_child_name} bone. Using zero local rotation.")
                    current_local_rotation_matrix = np.identity(3)
                else:
                    # This rotation matrix aligns rest_bone_local to target_bone_in_parent_space_n
                    # This IS the local rotation matrix.
                    current_local_rotation_matrix = get_rotation_from_vectors(rest_bone_local, target_bone_in_parent_space_n)
                
                frame_local_eulers_output[current_joint_name] = rotation_matrix_to_euler(current_local_rotation_matrix, current_joint_obj.rotation_order)

    # Accumulate world rotation for children
    current_joint_world_rotation_matrix = parent_world_rotation_matrix @ current_local_rotation_matrix

    for child_obj in current_joint_obj.children:
        _calculate_frame_rotations_recursive(
            child_obj, current_joint_world_rotation_matrix,
            frame_landmarks, joint_mapping, skeleton_joint_dict,
            frame_local_eulers_output)


def process_motion_sapien_style(all_frames_landmarks, skeleton_root, joint_mapping, num_frames_total):
    """
    Process all frames to calculate local joint rotations for animation.
    Returns a list of dictionaries, where each dictionary maps joint_name to its local Euler angles for that frame.
    """
    print(f"Calculating joint rotations for {len(all_frames_landmarks)} frames...")
    
    all_joint_objects_dict = {j.name: j for j in get_all_joints(skeleton_root, include_end_sites=False)} # include_end_sites = True might be needed if they have offsets for bones

    all_frames_local_eulers = []

    for frame_idx in tqdm(range(len(all_frames_landmarks)), desc="Processing frames for rotation"):
        current_frame_landmarks = all_frames_landmarks[frame_idx]
        
        # This dictionary will store {joint_name: [ex, ey, ez]} for the current frame
        current_frame_joint_local_eulers = {} 
        
        # Initial parent rotation is identity matrix (world space)
        initial_parent_world_rotation = np.identity(3)
        
        _calculate_frame_rotations_recursive(
            skeleton_root, initial_parent_world_rotation,
            current_frame_landmarks, joint_mapping, all_joint_objects_dict,
            current_frame_joint_local_eulers)
        
        all_frames_local_eulers.append(current_frame_joint_local_eulers)
        
    return all_frames_local_eulers


# --- BVH Writing ---
def write_bvh_file(skeleton_root, all_frames_local_eulers, frame_time, output_bvh_path, hip_positions_world):
    """Write the BVH file with hierarchy and motion data."""
    print(f"Writing BVH file to {output_bvh_path}...")
    
    # Get joints in the order they should appear in HIERARCHY and MOTION sections (usually depth-first)
    # This order must match how motion data is written.
    bvh_joint_order_list = get_all_joints(skeleton_root, include_end_sites=False) # Exclude pure end sites if they don't have channels

    with open(output_bvh_path, 'w') as f:
        f.write("HIERARCHY\n")
        
        # Write joint hierarchy recursively
        # Root's initial offset in HIERARCHY is (0,0,0). Its actual position is from MOTION.
        write_joint_hierarchy_recursive(f, skeleton_root, 0, is_root=True)
        
        num_motion_frames = len(all_frames_local_eulers)
        f.write("MOTION\n")
        f.write(f"Frames: {num_motion_frames}\n")
        f.write(f"Frame Time: {frame_time:.6f}\n")

        for frame_idx in tqdm(range(num_motion_frames), desc="Writing motion data"):
            frame_euler_data_dict = all_frames_local_eulers[frame_idx]
            motion_line_values = []

            # Iterate through joints in the established BVH order
            for joint_obj in bvh_joint_order_list:
                joint_name = joint_obj.name
                
                # Root joint: Xposition Yposition Zposition then rotations
                if joint_obj.parent is None: # Is Root
                    # Use pre-calculated hip world positions
                    if frame_idx < len(hip_positions_world):
                        pos_data = hip_positions_world[frame_idx]
                        motion_line_values.extend([f"{val:.6f}" for val in pos_data])
                    else: # Should not happen if hip_positions_world is same length
                        motion_line_values.extend(["0.0", "0.0", "0.0"])
                
                # All joints (including root) add their rotation channels
                if joint_name in frame_euler_data_dict:
                    rot_data_deg = frame_euler_data_dict[joint_name]
                    # Ensure rotation data matches joint's channel order (e.g. XYZ)
                    # Assuming frame_euler_data_dict stores them as [Rx, Ry, Rz] for 'XYZ'
                    motion_line_values.extend([f"{val:.6f}" for val in rot_data_deg])
                else:
                    # If a joint's rotation is missing (should not happen if all joints processed)
                    # Provide default (0,0,0) for the number of rotation channels
                    num_rot_channels = sum(1 for ch in joint_obj.channels if "rotation" in ch.lower())
                    motion_line_values.extend(["0.0"] * num_rot_channels)
            
            f.write(" ".join(motion_line_values) + "\n")
            
    print(f"BVH file created: {output_bvh_path}")


def write_joint_hierarchy_recursive(f, joint_obj, indent_level, is_root=False):
    indent = "  " * indent_level
    joint_name = joint_obj.name

    if is_root:
        f.write(f"{indent}ROOT {joint_name}\n")
    else:
        f.write(f"{indent}JOINT {joint_name}\n")
    
    f.write(f"{indent}{{\n")
    
    # Offset from parent to this joint. For ROOT, this is (0,0,0) by convention in HIERARCHY.
    # Its actual world position is handled by Xposition, Yposition, Zposition channels in MOTION.
    current_offset = joint_obj.offset
    if is_root:
        current_offset = np.zeros(3) 

    f.write(f"{indent}  OFFSET {current_offset[0]:.6f} {current_offset[1]:.6f} {current_offset[2]:.6f}\n")
    
    # Define channels based on whether it's root or regular joint
    # Rotation order is on joint_obj.rotation_order (e.g., 'XYZ')
    rot_order = joint_obj.rotation_order
    if len(rot_order) != 3 : rot_order = 'XYZ' # Default

    channel_str_parts = []
    if is_root:
        channel_str_parts.extend(["Xposition", "Yposition", "Zposition"])
    
    # Add rotation channels in the specified order
    channel_str_parts.extend([f"{axis.upper()}rotation" for axis in rot_order])
    joint_obj.channels = channel_str_parts # Store for writer

    f.write(f"{indent}  CHANNELS {len(channel_str_parts)} {' '.join(channel_str_parts)}\n")
    
    if not joint_obj.children: # Leaf node in our defined skeleton (e.g. Head, LeftHand, LeftToeBase)
        # Write End Site for BVH compliance, representing the end of this bone segment
        f.write(f"{indent}  End Site\n")
        f.write(f"{indent}  {{\n")
        # End site offset is typically small, representing bone tip, relative to this joint
        # e.g., for Head, a small offset upwards from the Head joint.
        # These can be fine-tuned for better visual representation if using a BVH player
        # that draws End Sites as bone tips.
        end_site_offset = np.array([0.0, 0.0, 0.0]) # Default, can customize
        if "Hand" in joint_name: end_site_offset = np.array([5.0 if "Right" in joint_name else -5.0, 0.0, 0.0]) 
        elif "Toe" in joint_name: end_site_offset = np.array([0.0, 0.0, 5.0])
        elif "Head" == joint_name: end_site_offset = np.array([0.0, 5.0, 0.0])

        f.write(f"{indent}    OFFSET {end_site_offset[0]:.6f} {end_site_offset[1]:.6f} {end_site_offset[2]:.6f}\n")
        f.write(f"{indent}  }}\n")
    else:
        for child_obj in joint_obj.children:
            write_joint_hierarchy_recursive(f, child_obj, indent_level + 1, is_root=False)
    
    f.write(f"{indent}}}\n")


def calculate_hip_world_positions(all_frames_landmarks, joint_mapping, scale=100.0):
    """Calculates the world position of the Hips joint for each frame."""
    num_frames = len(all_frames_landmarks)
    hip_positions = np.zeros((num_frames, 3))
    
    initial_hip_pos_world = None

    for i, frame_landmarks in enumerate(all_frames_landmarks):
        # Get Hips landmark position (average of left/right hip)
        current_hip_center_lm = get_joint_world_position("Hips", frame_landmarks, joint_mapping)
        
        if current_hip_center_lm is not None:
            current_hip_pos_world = current_hip_center_lm * scale
            if i == 0: # First frame sets the reference origin
                initial_hip_pos_world = current_hip_pos_world.copy()
            
            # Position relative to the initial hip position (making first frame Hips effectively at (0,0,0) in its own data stream)
            # The BVH root's OFFSET is (0,0,0), its Xpos,Ypos,Zpos are these relative values.
            if initial_hip_pos_world is not None:
                 hip_positions[i] = current_hip_pos_world - initial_hip_pos_world
            else: # Should not happen after first frame with valid hip
                 hip_positions[i] = current_hip_pos_world # Absolute if initial failed (bad)
        else:
            # If hips not found, use previous frame's position or (0,0,0)
            if i > 0: hip_positions[i] = hip_positions[i-1]
            else: hip_positions[i] = np.zeros(3)
            print(f"Warning: Hips not detected in frame {i}. Using previous/zero position.")
            if i==0 : initial_hip_pos_world = np.zeros(3) # Set initial if first frame is bad

    return hip_positions

# --- Video Processing and Main ---
def process_video(video_path, output_bvh, confidence_threshold=0.5, sample_rate=1, preview=False):
    print(f"Opening video file: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}"); return
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Video: {width}x{height}, {fps:.2f} FPS, {frame_count} frames. Sampling every {sample_rate} frames.")
    
    actual_fps_for_bvh = fps / sample_rate
    bvh_frame_time = 1.0 / actual_fps_for_bvh
    
    if preview:
        cv2.namedWindow('MediaPipe Pose Preview', cv2.WINDOW_NORMAL)
        # cv2.resizeWindow('MediaPipe Pose Preview', width // 2, height // 2) # Optional resize

    print("Initializing MediaPipe Pose...")
    with mp_pose.Pose(
        static_image_mode=False, model_complexity=1,
        smooth_landmarks=True, enable_segmentation=False, smooth_segmentation=False,
        min_detection_confidence=confidence_threshold, min_tracking_confidence=confidence_threshold
    ) as pose_detector:
        
        collected_landmarks_for_bvh = []
        processed_frame_count = 0
        
        print(f"Processing video frames (sampling 1 every {sample_rate} frames)...")
        pbar = tqdm(total=frame_count, desc="Extracting landmarks")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            pbar.update(1)
            
            if processed_frame_count % sample_rate == 0:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_rgb.flags.writeable = False
                results = pose_detector.process(frame_rgb)
                frame_rgb.flags.writeable = True # For drawing if previewing

                if results.pose_world_landmarks:
                    collected_landmarks_for_bvh.append(results.pose_world_landmarks.landmark)
                else:
                    # Add empty landmarks if no pose detected to maintain frame count
                    collected_landmarks_for_bvh.append([EmptyLandmark()] * 33) 
                    # print(f"Warning: No pose detected in source frame {processed_frame_count}.")
            
            if preview: # Show preview for all frames, not just sampled ones, for smoother viz
                frame_to_draw_on = frame.copy() # Draw on BGR frame
                if results and results.pose_landmarks: # Use screen landmarks for preview
                     mp_drawing.draw_landmarks(
                        frame_to_draw_on, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(0,255,0), thickness=1, circle_radius=1),
                        mp_drawing.DrawingSpec(color=(0,0,255), thickness=1, circle_radius=1))
                cv2.putText(frame_to_draw_on, f"Frame: {processed_frame_count}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0),2)
                cv2.imshow('MediaPipe Pose Preview', frame_to_draw_on)
                if cv2.waitKey(1) & 0xFF == ord('q'): print("Preview stopped."); break
            
            processed_frame_count += 1
        
        pbar.close()
        cap.release()
        if preview: cv2.destroyAllWindows()

        if not collected_landmarks_for_bvh:
            print("Error: No pose landmarks collected from the video.")
            return
        
        print(f"Landmark extraction complete. {len(collected_landmarks_for_bvh)} frames for BVH.")

        joint_mapping_dict = get_joint_mapping()
        skeleton_hierarchy_root = create_skeleton()
        
        # Use the first frame with good visibility for initial offsets.
        # Could be improved by finding "best" T-pose frame.
        ref_frame_idx_for_offsets = 0
        for idx, lms in enumerate(collected_landmarks_for_bvh):
            if any(lm.visibility > 0.7 for lm in lms if not isinstance(lm, EmptyLandmark)): # Check if not all empty
                ref_frame_idx_for_offsets = idx
                break
        print(f"Using frame {ref_frame_idx_for_offsets} for initial skeleton bone offsets.")
        build_skeleton_initial_offsets(skeleton_hierarchy_root, collected_landmarks_for_bvh[ref_frame_idx_for_offsets], joint_mapping_dict)
        
        # Calculate world positions of Hips for all frames (for root motion)
        hip_world_positions = calculate_hip_world_positions(collected_landmarks_for_bvh, joint_mapping_dict)

        # Process all frames to get local Euler angle rotations for each joint
        all_frames_rotation_data = process_motion_sapien_style(
            collected_landmarks_for_bvh, skeleton_hierarchy_root, joint_mapping_dict, len(collected_landmarks_for_bvh)
        )
        
        write_bvh_file(skeleton_hierarchy_root, all_frames_rotation_data, bvh_frame_time, output_bvh, hip_world_positions)

def main():
    parser = argparse.ArgumentParser(description="Convert video to BVH using MediaPipe")
    parser.add_argument("--video", required=True, help="Path to input video file")
    parser.add_argument("--output", required=True, help="Path to output BVH file")
    parser.add_argument("--confidence", type=float, default=0.5, help="Min detection confidence for MediaPipe Pose")
    parser.add_argument("--sample-rate", type=int, default=2, help="Process every Nth video frame for BVH (e.g., 2 means 30fps video -> 15fps BVH)")
    parser.add_argument("--preview", action="store_true", help="Show live MediaPipe pose detection preview")
    parser.add_argument("--preview-bvh", action="store_true", help="Preview the generated BVH animation using Matplotlib")
    parser.add_argument("--speed", type=float, default=1.0, help="Speed factor for BVH preview (default: 1.0)")
    
    args = parser.parse_args()
    
    print("Starting MediaPipe to BVH conversion...")
    start_time = time.time()
    
    process_video(args.video, args.output, args.confidence, args.sample_rate, args.preview)
    
    end_time = time.time()
    print(f"Conversion process finished in {end_time - start_time:.2f} seconds.")
    
    if args.preview_bvh:
        if os.path.exists(args.output):
            print("Launching BVH animation preview...")
            preview_bvh_animation(args.output, args.speed)
        else:
            print(f"Cannot preview BVH: Output file {args.output} not found.")

if __name__ == "__main__":
    main()