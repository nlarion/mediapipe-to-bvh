import numpy as np
from scipy.spatial.transform import Rotation as R
import mediapipe as mp # For PoseLandmark enum, actual processing is placeholder
import json # For potentially loading T-pose data

# --- Configuration Constants ---
OUTPUT_BVH_FILE = "motion_output.bvh"
FRAME_TIME = 1/30  # Assuming 30 FPS
SCALE_FACTOR = 100.0  # Convert meters (MediaPipe) to centimeters (BVH)
EULER_ORDER = 'zxy' # Euler angle order for BVH rotations

# --- MediaPipe Pose Landmark Indices (for reference) ---
# These are based on mp.solutions.pose.PoseLandmark
# NOSE = 0
# LEFT_SHOULDER = 11
# RIGHT_SHOULDER = 12
# LEFT_ELBOW = 13
# RIGHT_ELBOW = 14
# LEFT_WRIST = 15
# RIGHT_WRIST = 16
# LEFT_HIP = 23
# RIGHT_HIP = 24
# LEFT_KNEE = 25
# RIGHT_KNEE = 26
# LEFT_ANKLE = 27
# RIGHT_ANKLE = 28
# LEFT_HEEL = 29
# RIGHT_HEEL = 30
# LEFT_FOOT_INDEX = 31
# RIGHT_FOOT_INDEX = 32

# --- BVH Skeleton Definition ---
# This defines the hierarchy and which MediaPipe landmarks map to which BVH joints.
# You might need to customize this for your specific character or needs.
BVH_SKELETON_STRUCTURE = {
    "Hips": { # Root
        "mp_landmark": "CALCULATED_HIPS", # Midpoint of LEFT_HIP and RIGHT_HIP
        "channels": ["Xposition", "Yposition", "Zposition", "Zrotation", "Xrotation", "Yrotation"]
    },
    "Spine": {
        "mp_landmark": "CALCULATED_SPINE", # Midpoint of Hips and Chest
        "children": ["Chest"],
        "channels": ["Zrotation", "Xrotation", "Yrotation"]
    },
    "Chest": {
        "mp_landmark": "CALCULATED_CHEST", # Midpoint of LEFT_SHOULDER and RIGHT_SHOULDER
        "channels": ["Zrotation", "Xrotation", "Yrotation"]
    },
    "Neck": {
        "mp_landmark": "CALCULATED_NECK", # Approx. from Chest towards Nose
        "children": ["Head"],
        "channels": ["Zrotation", "Xrotation", "Yrotation"]
    },
    "Head": {
        "mp_landmark": mp.solutions.pose.PoseLandmark.NOSE,
        "children": ["Head_End"], # End Site
        "channels": ["Zrotation", "Xrotation", "Yrotation"]
    },
    "Head_End": {"mp_landmark": None }, # End Site for Head

    "LeftShoulder": {
        "mp_landmark": mp.solutions.pose.PoseLandmark.LEFT_SHOULDER,
        "children": ["LeftArm"],
        "channels": ["Zrotation", "Xrotation", "Yrotation"]
    },
    "LeftArm": { # Represents Upper Arm
        "mp_landmark": mp.solutions.pose.PoseLandmark.LEFT_ELBOW,
        "children": ["LeftForeArm"],
        "channels": ["Zrotation", "Xrotation", "Yrotation"]
    },
    "LeftForeArm": {
        "mp_landmark": mp.solutions.pose.PoseLandmark.LEFT_WRIST,
        "children": ["LeftHand"],
        "channels": ["Zrotation", "Xrotation", "Yrotation"]
    },
    "LeftHand": { # Simplified hand
        "mp_landmark": mp.solutions.pose.PoseLandmark.LEFT_WRIST, # Placeholder, could use specific hand landmarks
        "children": ["LeftHand_End"],
        "channels": ["Zrotation", "Xrotation", "Yrotation"]
    },
    "LeftHand_End": {"mp_landmark": None},

    "RightShoulder": {
        "mp_landmark": mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER,
        "channels": ["Zrotation", "Xrotation", "Yrotation"]
    },
    "RightArm": {
        "mp_landmark": mp.solutions.pose.PoseLandmark.RIGHT_ELBOW,
    
        "channels": ["Zrotation", "Xrotation", "Yrotation"]
    },
    "RightForeArm": {
        "mp_landmark": mp.solutions.pose.PoseLandmark.RIGHT_WRIST,
    
        "channels": ["Zrotation", "Xrotation", "Yrotation"]
    },
    "RightHand": {
        "mp_landmark": mp.solutions.pose.PoseLandmark.RIGHT_WRIST, # Placeholder
    
        "channels": ["Zrotation", "Xrotation", "Yrotation"]
    },
    "RightHand_End": {"mp_landmark": None},

    "LeftUpLeg": {
        "mp_landmark": mp.solutions.pose.PoseLandmark.LEFT_KNEE,
        "children": ["LeftLeg"],
        "channels": ["Zrotation", "Xrotation", "Yrotation"]
    },
    "LeftLeg": {
        "mp_landmark": mp.solutions.pose.PoseLandmark.LEFT_ANKLE,
        "children": ["LeftFoot"],
        "channels": ["Zrotation", "Xrotation", "Yrotation"]
    },
    "LeftFoot": {
        "mp_landmark": mp.solutions.pose.PoseLandmark.LEFT_FOOT_INDEX,
        "children": ["LeftFoot_End"],
        "channels": ["Zrotation", "Xrotation", "Yrotation"]
    },
    "LeftFoot_End": {"mp_landmark": None},

    "RightUpLeg": {
        "mp_landmark": mp.solutions.pose.PoseLandmark.RIGHT_KNEE,
    
        "channels": ["Zrotation", "Xrotation", "Yrotation"]
    },
    "RightLeg": {
        "mp_landmark": mp.solutions.pose.PoseLandmark.RIGHT_ANKLE,
    
        "channels": ["Zrotation", "Xrotation", "Yrotation"]
    },
    "RightFoot": {
        "mp_landmark": mp.solutions.pose.PoseLandmark.RIGHT_FOOT_INDEX,
    
        "channels": ["Zrotation", "Xrotation", "Yrotation"]
    },
    "RightFoot_End": {"mp_landmark": None},
}

# --- Helper Functions ---

def get_landmark_coords(landmarks_list, landmark_index):
    """Safely retrieves x, y, z coordinates from MediaPipe landmarks list."""
    if landmarks_list and 0 <= landmark_index < len(landmarks_list):
        lm = landmarks_list[landmark_index]
        return np.array([lm.x, lm.y, lm.z])
    return np.array([0.0, 0.0, 0.0]) # Default if landmark is not found

def transform_mp_to_bvh_coords(mp_coords_m):
    """
    Transforms MediaPipe world coordinates (meters, Y-up relative to image, Z towards camera)
    to BVH coordinates (scaled, Y-up world, Z-forward world).
    """
    # MediaPipe: X right, Y up (often screen up), Z towards camera (smaller z is closer)
    # BVH: X right, Y up (world up), Z forward
    # Transformation:
    # BVH_X = MP_X
    # BVH_Y = MP_Y
    # BVH_Z = -MP_Z (to flip Z from towards camera to forward)
    # Scale from meters to desired BVH units (e.g., cm)
    return np.array([mp_coords_m, mp_coords_m[1], -mp_coords_m[2]]) * SCALE_FACTOR

def calculate_bvh_joint_world_pos(landmarks_list, joint_name, skeleton_def):
    """Calculates the world position of a BVH joint from MediaPipe landmarks."""
    spec = skeleton_def.get(joint_name)
    if not spec:
        return np.array([0.0, 0.0, 0.0])

    mp_idx = spec["mp_landmark"]

    if isinstance(mp_idx, str) and mp_idx.startswith("CALCULATED_"):
        if mp_idx == "CALCULATED_HIPS":
            lm_left_hip = get_landmark_coords(landmarks_list, mp.solutions.pose.PoseLandmark.LEFT_HIP)
            lm_right_hip = get_landmark_coords(landmarks_list, mp.solutions.pose.PoseLandmark.RIGHT_HIP)
            raw_pos = (lm_left_hip + lm_right_hip) / 2.0
        elif mp_idx == "CALCULATED_CHEST":
            lm_left_shoulder = get_landmark_coords(landmarks_list, mp.solutions.pose.PoseLandmark.LEFT_SHOULDER)
            lm_right_shoulder = get_landmark_coords(landmarks_list, mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER)
            raw_pos = (lm_left_shoulder + lm_right_shoulder) / 2.0
        elif mp_idx == "CALCULATED_SPINE": # Approx. midpoint of Hips and Chest
            lm_left_hip = get_landmark_coords(landmarks_list, mp.solutions.pose.PoseLandmark.LEFT_HIP)
            lm_right_hip = get_landmark_coords(landmarks_list, mp.solutions.pose.PoseLandmark.RIGHT_HIP)
            hips_pos = (lm_left_hip + lm_right_hip) / 2.0
            lm_left_shoulder = get_landmark_coords(landmarks_list, mp.solutions.pose.PoseLandmark.LEFT_SHOULDER)
            lm_right_shoulder = get_landmark_coords(landmarks_list, mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER)
            chest_pos = (lm_left_shoulder + lm_right_shoulder) / 2.0
            raw_pos = (hips_pos + chest_pos) / 2.0
        elif mp_idx == "CALCULATED_NECK": # Approx. from Chest towards Nose
            lm_left_shoulder = get_landmark_coords(landmarks_list, mp.solutions.pose.PoseLandmark.LEFT_SHOULDER)
            lm_right_shoulder = get_landmark_coords(landmarks_list, mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER)
            chest_pos = (lm_left_shoulder + lm_right_shoulder) / 2.0
            nose_pos = get_landmark_coords(landmarks_list, mp.solutions.pose.PoseLandmark.NOSE)
            # Simple midpoint, or could be a point along the chest-nose vector
            raw_pos = chest_pos + 0.5 * (nose_pos - chest_pos) # Midpoint
        else:
            raw_pos = np.array([0.0, 0.0, 0.0]) # Unknown calculated joint
    elif isinstance(mp_idx, mp.solutions.pose.PoseLandmark):
        raw_pos = get_landmark_coords(landmarks_list, mp_idx.value)
    else: # End sites or unmapped
        raw_pos = np.array([0.0, 0.0, 0.0])

    return transform_mp_to_bvh_coords(raw_pos)


def get_tpose_landmarks_from_source(source_path=None):
    """
    Placeholder function to load MediaPipe pose_world_landmarks from a T-pose.
    IMPLEMENT THIS YOURSELF: Load landmarks from a video frame where the subject is in T-pose,
    or from a saved JSON/file containing these landmarks.
    The landmarks should be a list of MediaPipe Landmark objects (or dicts with x, y, z).
    """
    print("WARNING: Using dummy T-pose data. Implement `get_tpose_landmarks_from_source` for accurate results.")
    # Example: Create dummy T-pose landmarks (replace with actual data)
    # This is a very rough approximation and WILL NOT result in a good T-pose.
    # A real T-pose would have arms straight out, legs straight down.
    tpose_landmarks = [ {'x':0.0, 'y':0.0, 'z':0.0} ] * 33 # Initialize all to origin

    # Approximate some key T-pose landmarks (in MediaPipe's world coordinate system - meters)
    # Hips centered at origin
    tpose_landmarks = {'x': -0.1, 'y': -0.8, 'z': 0.0}
    tpose_landmarks = {'x': 0.1, 'y': -0.8, 'z': 0.0}

    # Shoulders
    tpose_landmarks = {'x': -0.25, 'y': -0.4, 'z': 0.0}
    tpose_landmarks = {'x': 0.25, 'y': -0.4, 'z': 0.0}

    # Elbows (arms straight out for T-pose)
    tpose_landmarks = {'x': -0.6, 'y': -0.4, 'z': 0.0}
    tpose_landmarks = {'x': 0.6, 'y': -0.4, 'z': 0.0}

    # Wrists
    tpose_landmarks = {'x': -0.9, 'y': -0.4, 'z': 0.0}
    tpose_landmarks = {'x': 0.9, 'y': -0.4, 'z': 0.0}

    # Knees
    tpose_landmarks = {'x': -0.1, 'y': -1.2, 'z': 0.0}
    tpose_landmarks = {'x': 0.1, 'y': -1.2, 'z': 0.0}

    # Ankles
    tpose_landmarks = {'x': -0.1, 'y': -1.6, 'z': 0.0}
    tpose_landmarks = {'x': 0.1, 'y': -1.6, 'z': 0.0}

    # Feet
    tpose_landmarks = {'x': -0.1, 'y': -1.6, 'z': 0.1} # Slightly forward
    tpose_landmarks = {'x': 0.1, 'y': -1.6, 'z': 0.1}

    # Nose (Head)
    tpose_landmarks = {'x': 0.0, 'y': -0.1, 'z': 0.0}

    # Convert dicts to Landmark-like objects for compatibility with get_landmark_coords
    class DummyLandmark:
        def __init__(self, d):
            self.x = d['x']
            self.y = d['y']
            self.z = d['z']
    
    return


def calculate_bvh_offsets(tpose_landmarks_mp, skeleton_def):
    """Calculates BVH OFFSETs from T-pose MediaPipe landmarks."""
    tpose_bvh_joint_world_pos = {}
    for joint_name in skeleton_def.keys():
        tpose_bvh_joint_world_pos[joint_name] = calculate_bvh_joint_world_pos(tpose_landmarks_mp, joint_name, skeleton_def)

    bvh_offsets = {}
    for joint_name, spec in skeleton_def.items():
        if joint_name == "Hips": # Root
            bvh_offsets[joint_name] = np.array([0.0, 0.0, 0.0])
        else:
            # Find parent
            parent_name = None
            for p_name, p_spec in skeleton_def.items():
                if joint_name in p_spec["children"]:
                    parent_name = p_name
                    break
            if parent_name:
                offset = tpose_bvh_joint_world_pos[joint_name] - tpose_bvh_joint_world_pos[parent_name]
                bvh_offsets[joint_name] = offset
            else: # Should not happen for non-root joints if hierarchy is correct
                bvh_offsets[joint_name] = np.array([0.0, 0.0, 0.0])
    return bvh_offsets

def generate_hierarchy_string(skeleton_def, offsets, initial_joint="Hips", indent_level=0):
    """Recursively generates the BVH HIERARCHY string."""
    spec = skeleton_def[initial_joint]
    indent = "\t" * indent_level
    hierarchy_str = ""

    if initial_joint == "Hips": # Root
        hierarchy_str += f"{indent}ROOT {initial_joint}\n"
    elif "_End" in initial_joint : # End Site
        hierarchy_str += f"{indent}End Site\n"
    else:
        hierarchy_str += f"{indent}JOINT {initial_joint}\n"

    hierarchy_str += f"{indent}{{\n"
    indent_inner = "\t" * (indent_level + 1)

    offset_vals = offsets.get(initial_joint, np.array([0.0, 0.0, 0.0]))
    # For End Sites, the offset is from its parent, but its "own" offset is typically small or zero if it's just a marker.
    # The BVH standard uses the End Site's offset to define the length of the parent bone.
    # If an End Site has a specific offset in our definition, use it. Otherwise, a small default.
    if "_End" in initial_joint and initial_joint not in offsets:
         # A small offset for the end effector bone if not explicitly defined by a landmark.
         # This value determines the length of the parent bone segment.
         # Example: if Head_End is an End Site for Head, its offset defines Head's length.
         # We use the calculated offset for the "parent" of the end site if available.
         # The offset for an End Site is relative to its parent joint.
         # Here, we assume the offset for "Head_End" was calculated as if it were a child of "Head".
         # If "Head_End" itself has no landmark, its offset from "Head" in T-pose needs to be defined.
         # For simplicity, let's use a small default if not in offsets.
         # A better approach would be to define these end-effector offsets in the T-pose.
        offset_vals = np.array([0.0, 5.0, 0.0]) # Default small length along Y for terminal bones

    hierarchy_str += f"{indent_inner}OFFSET {offset_vals:.6f} {offset_vals[1]:.6f} {offset_vals[2]:.6f}\n"

    if spec["channels"]:
        hierarchy_str += f"{indent_inner}CHANNELS {len(spec['channels'])} {' '.join(spec['channels'])}\n"

    for child_joint in spec["children"]:
        hierarchy_str += generate_hierarchy_string(skeleton_def, offsets, child_joint, indent_level + 1)

    hierarchy_str += f"{indent}}}\n"
    return hierarchy_str


def calculate_root_motion_and_orientation(current_landmarks_mp, skeleton_def):
    """Calculates root (Hips) translation and orientation for the current frame."""
    # Root position (Hips)
    lm_left_hip = get_landmark_coords(current_landmarks_mp, mp.solutions.pose.PoseLandmark.LEFT_HIP)
    lm_right_hip = get_landmark_coords(current_landmarks_mp, mp.solutions.pose.PoseLandmark.RIGHT_HIP)
    root_pos_mp = (lm_left_hip + lm_right_hip) / 2.0
    root_pos_bvh = transform_mp_to_bvh_coords(root_pos_mp)

    # Root orientation
    # Use hips and shoulders to define pelvis orientation
    # These are already in MediaPipe's world coords (meters)
    p_left_hip_mp = lm_left_hip
    p_right_hip_mp = lm_right_hip
    p_left_shoulder_mp = get_landmark_coords(current_landmarks_mp, mp.solutions.pose.PoseLandmark.LEFT_SHOULDER)
    p_right_shoulder_mp = get_landmark_coords(current_landmarks_mp, mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER)

    # Transform to BVH coordinate space before calculating axes
    p_left_hip_bvh = transform_mp_to_bvh_coords(p_left_hip_mp)
    p_right_hip_bvh = transform_mp_to_bvh_coords(p_right_hip_mp)
    p_left_shoulder_bvh = transform_mp_to_bvh_coords(p_left_shoulder_mp)
    p_right_shoulder_bvh = transform_mp_to_bvh_coords(p_right_shoulder_mp)
    
    root_center_bvh = (p_left_hip_bvh + p_right_hip_bvh) / 2.0 # This is root_pos_bvh

    # Pelvis X-axis (right): from left hip to right hip
    pelvis_X_axis = p_right_hip_bvh - p_left_hip_bvh
    if np.linalg.norm(pelvis_X_axis) < 1e-6: pelvis_X_axis = np.array([1.0, 0.0, 0.0]) # Avoid zero vector
    else: pelvis_X_axis = pelvis_X_axis / np.linalg.norm(pelvis_X_axis)

    # Approximate Spine vector (Y-axis direction for pelvis): from hip center to shoulder center
    shoulder_center_bvh = (p_left_shoulder_bvh + p_right_shoulder_bvh) / 2.0
    spine_vec_approx = shoulder_center_bvh - root_center_bvh
    if np.linalg.norm(spine_vec_approx) < 1e-6: spine_vec_approx = np.array([0.0, 1.0, 0.0]) # Avoid zero vector
    else: spine_vec_approx = spine_vec_approx / np.linalg.norm(spine_vec_approx)
    
    # Pelvis Z-axis (forward): cross product of X and approx Y (spine_vec)
    # Ensure X and Spine are not collinear
    if np.abs(np.dot(pelvis_X_axis, spine_vec_approx)) > 0.99: # Highly collinear
        # If X and Spine are collinear, pick an arbitrary Z perpendicular to X
        # This can happen if shoulders are directly above hips and hips are aligned with X
        # A common fallback is to assume Z is world Z if X is world X, or world X if X is world Y etc.
        # For simplicity, if X is roughly , Z is . If X is , Z is 
        if np.abs(pelvis_X_axis) > 0.9: # X is mainly along world X
            pelvis_Z_axis = np.cross(pelvis_X_axis, np.array([0.0, 1.0, 0.0])) # Cross with world Y
        else: # X is not mainly along world X (e.g. could be Y if person is lying down)
             pelvis_Z_axis = np.cross(pelvis_X_axis, np.array([0.0, 0.0, 1.0])) # Cross with world Z
    else:
        pelvis_Z_axis = np.cross(pelvis_X_axis, spine_vec_approx)

    if np.linalg.norm(pelvis_Z_axis) < 1e-6: pelvis_Z_axis = np.array([0.0, 0.0, 1.0]) # Avoid zero vector
    else: pelvis_Z_axis = pelvis_Z_axis / np.linalg.norm(pelvis_Z_axis)

    # Pelvis Y-axis (up): cross product of Z and X (to ensure orthogonality)
    pelvis_Y_axis = np.cross(pelvis_Z_axis, pelvis_X_axis)
    if np.linalg.norm(pelvis_Y_axis) < 1e-6: pelvis_Y_axis = np.array([0.0, 1.0, 0.0]) # Avoid zero vector
    else: pelvis_Y_axis = pelvis_Y_axis / np.linalg.norm(pelvis_Y_axis)

    # Create rotation matrix (columns are the new X, Y, Z axes in the world frame)
    # Scipy's R.from_matrix expects rows to be basis vectors of original frame in new frame,
    # OR columns to be basis vectors of new frame in original frame. We have the latter.
    root_rotation_matrix = np.array([pelvis_X_axis, pelvis_Y_axis, pelvis_Z_axis]).T
    
    root_orientation_obj = R.from_matrix(root_rotation_matrix)
    root_euler_angles = root_orientation_obj.as_euler(EULER_ORDER, degrees=True)
    
    return root_pos_bvh, root_euler_angles, root_orientation_obj


def calculate_local_rotation(current_joint_bvh_pos, parent_bvh_pos,
                             parent_world_rotation_obj, tpose_offset_vector):
    """Calculates local joint rotation relative to its parent."""
    if np.linalg.norm(tpose_offset_vector) < 1e-6: # Zero offset, implies no rotation or coincident with parent
        return np.array([0.0, 0.0, 0.0])

    # Current bone vector in world space
    current_bone_vec_world = current_joint_bvh_pos - parent_bvh_pos
    if np.linalg.norm(current_bone_vec_world) < 1e-6: # current bone has zero length
        return np.array([0.0, 0.0, 0.0])

    # The tpose_offset_vector is the bone's direction in the parent's T-pose local space.
    # We need to find the rotation that aligns this T-pose vector (in parent's current orientation)
    # with the current bone vector (in parent's current orientation).

    # Vector representing the bone in T-pose, but rotated by parent's current world orientation.
    # This is where the bone *would* point if it had T-pose rotation relative to current parent orientation.
    tpose_bone_in_parent_current_orientation_world = parent_world_rotation_obj.apply(tpose_offset_vector)

    # We want to find the rotation R_local such that:
    # current_bone_vec_world = R_local * tpose_bone_in_parent_current_orientation_world
    # This is equivalent to aligning tpose_bone_in_parent_current_orientation_world (vector_a)
    # with current_bone_vec_world (vector_b)
    # R_local = align_vectors([current_bone_vec_world], [tpose_bone_in_parent_current_orientation_world])
    
    # Normalize vectors for align_vectors, as it's about direction
    vec_a_norm = tpose_bone_in_parent_current_orientation_world / np.linalg.norm(tpose_bone_in_parent_current_orientation_world)
    vec_b_norm = current_bone_vec_world / np.linalg.norm(current_bone_vec_world)

    # align_vectors finds rotation R such that R @ vec_a_norm ~= vec_b_norm
    # Here, 'a' is the vector in the "from" orientation (T-pose relative to parent)
    # and 'b' is the vector in the "to" orientation (current pose relative to parent)
    # The documentation for align_vectors(a,b) finds rotation C such that C @ b ~= a.
    # So we need align_vectors(b, a) to get C such that C @ a ~= b.
    # Let target_vectors = [current_bone_vec_world_norm]
    # Let source_vectors = [tpose_bone_in_parent_current_orientation_world_norm]
    # local_rotation_obj, _ = R.align_vectors(target_vectors, source_vectors)
    
    # Simpler: transform current_bone_vec_world into parent's local space
    current_bone_vec_parent_local = parent_world_rotation_obj.inv().apply(current_bone_vec_world)

    # Now align tpose_offset_vector (which is already in parent's T-pose local space)
    # with current_bone_vec_parent_local (current bone in parent's current local space)
    
    source_vec_norm = tpose_offset_vector / np.linalg.norm(tpose_offset_vector)
    target_vec_norm = current_bone_vec_parent_local / np.linalg.norm(current_bone_vec_parent_local)
    
    # Find rotation R such that R @ source_vec_norm ~= target_vec_norm
    local_rotation_obj, _ = R.align_vectors([target_vec_norm], [source_vec_norm])

    local_euler_angles = local_rotation_obj.as_euler(EULER_ORDER, degrees=True)
    return local_euler_angles, local_rotation_obj


def get_ordered_joint_names(skeleton_def, root_name="Hips"):
    """Performs a BFS to get joint names in a processable order."""
    ordered_names = []
    queue = [root_name]
    visited = {root_name}
    
    while queue:
        current_joint = queue.pop(0)
        ordered_names.append(current_joint)
        if current_joint in skeleton_def and "children" in skeleton_def[current_joint]:
            for child in skeleton_def[current_joint]["children"]:
                if child not in visited and "_End" not in child : # Don't add End Sites to this processing list
                    queue.append(child)
                    visited.add(child)
    return ordered_names

# --- Main BVH Generation Function ---
def mediapipe_pose_to_bvh(all_frames_landmarks_mp, output_filepath):
    """
    Converts a sequence of MediaPipe pose_world_landmarks to a BVH file.
    all_frames_landmarks_mp: A list of lists. Each inner list contains MediaPipe Landmark
                             objects (or dicts with x,y,z) for one frame.
    """
    if not all_frames_landmarks_mp:
        print("Error: No landmark data provided.")
        return

    # 1. Get T-Pose landmarks and calculate BVH offsets
    #    IMPORTANT: User must provide accurate T-pose data here.
    #    Using the first frame as T-pose is a fallback and likely inaccurate.
    tpose_landmarks_mp = get_tpose_landmarks_from_source() # Or pass all_frames_landmarks_mp as a rough guess
    if not tpose_landmarks_mp:
         tpose_landmarks_mp = all_frames_landmarks_mp # Fallback: use first frame for T-pose (NOT RECOMMENDED)
         print("WARNING: Using first frame's landmarks for T-pose. This is likely inaccurate.")

    bvh_offsets = calculate_bvh_offsets(tpose_landmarks_mp, BVH_SKELETON_STRUCTURE)

    # 2. Generate HIERARCHY string
    hierarchy_string = "HIERARCHY\n"
    hierarchy_string += generate_hierarchy_string(BVH_SKELETON_STRUCTURE, bvh_offsets)

    # 3. Process MOTION data for each frame
    motion_data_strings = []
    num_frames = len(all_frames_landmarks_mp)
    
    # Get a processing order for joints (ensures parent is processed before child)
    joint_processing_order = get_ordered_joint_names(BVH_SKELETON_STRUCTURE)
    
    # Store parent info for quick lookup
    parent_map = {}
    for p_name, p_spec in BVH_SKELETON_STRUCTURE.items():
        for c_name in p_spec["children"]:
            parent_map[c_name] = p_name

    for frame_idx, current_landmarks_mp_frame in enumerate(all_frames_landmarks_mp):
        if not current_landmarks_mp_frame:
            print(f"Warning: Empty landmarks for frame {frame_idx}. Skipping.")
            # Add a line of zeros or repeat last frame's data if necessary
            # For simplicity, we'll just add zeros if this happens, assuming fixed channel count
            # This needs a robust way to get channel count if we don't process Hips first
            if motion_data_strings: # if not the first frame
                motion_data_strings.append(motion_data_strings[-1]) # repeat last frame
            else: # if first frame has no data, this is problematic
                 # Count total channels
                total_channels = 0
                for joint_name in joint_processing_order:
                    if "_End" not in joint_name:
                        total_channels += len(BVH_SKELETON_STRUCTURE[joint_name]["channels"])
                motion_data_strings.append(" ".join(["0.000"] * total_channels))
            continue

        frame_motion_values = []
        
        # Store world rotations and positions for hierarchical calculation
        # Keyed by BVH joint name
        current_frame_bvh_world_positions = {} 
        current_frame_world_rotations_obj = {} # Stores Scipy Rotation objects

        # Calculate current world positions for all BVH joints
        for joint_name in BVH_SKELETON_STRUCTURE.keys():
            if "_End" not in joint_name: # Don't need world pos for end sites directly for rotation calc
                 current_frame_bvh_world_positions[joint_name] = calculate_bvh_joint_world_pos(
                    current_landmarks_mp_frame, joint_name, BVH_SKELETON_STRUCTURE
                )

        # Process joints in hierarchical order
        for joint_name in joint_processing_order:
            spec = BVH_SKELETON_STRUCTURE[joint_name]

            if joint_name == "Hips": # Root joint
                root_pos_bvh, root_euler_angles, root_orientation_obj = \
                    calculate_root_motion_and_orientation(current_landmarks_mp_frame, BVH_SKELETON_STRUCTURE)
                
                current_frame_world_rotations_obj[joint_name] = root_orientation_obj
                # current_frame_bvh_world_positions[joint_name] is already calculated

                frame_motion_values.extend([f"{v:.3f}" for v in root_pos_bvh])
                frame_motion_values.extend([f"{v:.3f}" for v in root_euler_angles])
            
            else: # Non-root joints
                parent_name = parent_map.get(joint_name)
                if not parent_name or parent_name not in current_frame_bvh_world_positions or parent_name not in current_frame_world_rotations_obj:
                    print(f"Error: Parent {parent_name} not processed for joint {joint_name}. Skipping.")
                    # Add placeholder rotations (zeros)
                    frame_motion_values.extend(["0.000"] * len(spec["channels"]))
                    current_frame_world_rotations_obj[joint_name] = R.identity() # Identity rotation
                    continue

                parent_bvh_pos = current_frame_bvh_world_positions[parent_name]
                parent_world_rotation_obj = current_frame_world_rotations_obj[parent_name]
                
                current_joint_bvh_pos = current_frame_bvh_world_positions[joint_name]
                tpose_offset_vec = bvh_offsets[joint_name]

                local_euler_angles, local_rotation_obj = calculate_local_rotation(
                    current_joint_bvh_pos, parent_bvh_pos,
                    parent_world_rotation_obj, tpose_offset_vec
                )
                
                frame_motion_values.extend([f"{v:.3f}" for v in local_euler_angles])
                
                # Update current joint's world rotation for its children
                current_frame_world_rotations_obj[joint_name] = parent_world_rotation_obj * local_rotation_obj
        
        motion_data_strings.append(" ".join(frame_motion_values))

    # 4. Write BVH file
    bvh_content = hierarchy_string
    bvh_content += "MOTION\n"
    bvh_content += f"Frames: {num_frames}\n"
    bvh_content += f"Frame Time: {FRAME_TIME:.8f}\n"
    bvh_content += "\n".join(motion_data_strings)

    with open(output_filepath, "w") as f:
        f.write(bvh_content)
    print(f"BVH file saved to {output_filepath}")


# --- Example Usage ---
if __name__ == "__main__":
    # --- This is where you would load your MediaPipe pose_world_landmarks ---
    # For demonstration, let's create some dummy landmark data for a few frames.
    # Each frame should be a list of 33 MediaPipe Landmark-like objects/dicts.
    
    print("Generating dummy MediaPipe landmark data for demonstration...")
    num_dummy_frames = 10
    dummy_all_frames_landmarks = []

    # Create a base pose (e.g., from the T-pose function)
    base_landmarks_data = get_tpose_landmarks_from_source() 

    for i in range(num_dummy_frames):
        frame_landmarks = []
        # Simulate some movement by slightly altering base T-pose landmarks
        for lm_idx, base_lm_data in enumerate(base_landmarks_data):
            # Make a copy to modify
            new_lm_data = {'x': base_lm_data.x, 'y': base_lm_data.y, 'z': base_lm_data.z}
            
            # Example: make the wrists move slightly
            if lm_idx == mp.solutions.pose.PoseLandmark.LEFT_WRIST.value:
                new_lm_data['y'] += 0.05 * np.sin(i * np.pi / 5) # Simple sinusoidal movement
                new_lm_data['x'] += 0.02 * np.cos(i * np.pi / 5)
            if lm_idx == mp.solutions.pose.PoseLandmark.RIGHT_WRIST.value:
                new_lm_data['y'] -= 0.05 * np.sin(i * np.pi / 5)
                new_lm_data['x'] -= 0.02 * np.cos(i * np.pi / 5)
            
            # Convert dict to a simple object with x, y, z attributes
            class FrameLandmark:
                def __init__(self, d):
                    self.x = d['x']; self.y = d['y']; self.z = d['z']
            frame_landmarks.append(FrameLandmark(new_lm_data))
        dummy_all_frames_landmarks.append(frame_landmarks)
    
    print(f"Generated {len(dummy_all_frames_landmarks)} frames of dummy data.")

    # --- Call the conversion function ---
    mediapipe_pose_to_bvh(dummy_all_frames_landmarks, OUTPUT_BVH_FILE)

    # --- How to use with your actual MediaPipe data: ---
    # 1. Initialize MediaPipe Pose Landmarker
    #    pose_landmarker = mp.solutions.pose.PoseLandmarker(
    #        static_image_mode=False, # or True if processing single images
    #        model_complexity=1,      # 0, 1, or 2
    #        enable_segmentation=False,
    #        min_detection_confidence=0.5,
    #        min_tracking_confidence=0.5)
    #
    # 2. Loop through your video frames:
    #    all_my_frames_landmarks_mp =
    #    cap = cv2.VideoCapture("your_video.mp4")
    #    while cap.isOpened():
    #        ret, frame = cap.read()
    #        if not ret: break
    #        
    #        # Convert frame to RGB for MediaPipe
    #        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #        results = pose_landmarker.process(image_rgb)
    #        
    #        if results.pose_world_landmarks:
    #            all_my_frames_landmarks_mp.append(results.pose_world_landmarks.landmark)
    #        else:
    #            all_my_frames_landmarks_mp.append() # Handle frames with no detection
    #    cap.release()
    #    pose_landmarker.close()
    #
    # 3. Call the conversion:
    #    mediapipe_pose_to_bvh(all_my_frames_landmarks_mp, "my_real_motion.bvh")
    #
    # 4. CRITICAL: Implement `get_tpose_landmarks_from_source()` to load
    #    landmarks from an actual T-pose of your subject for accurate BVH offsets.