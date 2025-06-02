import cv2
import mediapipe as mp
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.signal import savgol_filter
import bvhio # For writing BVH files: pip install bvhio
import collections
import argparse

# --- Skeleton Definition ---
# This defines the BVH hierarchy, MediaPipe landmark mapping, and channels.
# 'children': list of child joint names
# 'mp_landmark': MediaPipe landmark enum(s) for this joint
# 'channels': BVH channels for this joint
# Other keys like 't_pose_xyz', 'offset', 't_pose_orientation_matrix',
# 'current_xyz', 'current_orientation_matrix', 'local_rotation_object',
# 'motion_data' will be populated during processing.

MP_LANDMARKS = mp.solutions.pose.PoseLandmark

SKELETON_DEF = collections.OrderedDict({
    'Hips': {
        'children': ['Spine', 'LeftUpLeg', 'RightUpLeg'],
        'mp_landmark': [MP_LANDMARKS.LEFT_HIP.value, MP_LANDMARKS.RIGHT_HIP.value], # Averaged
        'channels': ['Xposition', 'Yposition', 'Zposition', 'Zrotation', 'Yrotation', 'Xrotation']
    },
    'Spine': {
        'children': ['Chest'],
        'mp_landmark': [MP_LANDMARKS.LEFT_HIP.value, MP_LANDMARKS.RIGHT_HIP.value, MP_LANDMARKS.LEFT_SHOULDER.value, MP_LANDMARKS.RIGHT_SHOULDER.value], # Midpoint
        'channels': ['Zrotation', 'Yrotation', 'Xrotation']
    },
    'Chest': {
        'children': ['Neck', 'LeftShoulder', 'RightShoulder'],
        'mp_landmark': [MP_LANDMARKS.LEFT_SHOULDER.value, MP_LANDMARKS.RIGHT_SHOULDER.value], # Averaged
        'channels': ['Zrotation', 'Yrotation', 'Xrotation']
    },
    'Neck': {
        'children': ['Head'],
        'mp_landmark': MP_LANDMARKS.NOSE.value, # Approximate, or use midpoint of shoulders to nose
        'channels': ['Zrotation', 'Yrotation', 'Xrotation']
    },
    'Head': {
        'children': ['HeadEnd'],
        'mp_landmark': MP_LANDMARKS.NOSE.value, # Using NOSE as a reference point for Head joint
        'channels': ['Zrotation', 'Yrotation', 'Xrotation']
    },
    'HeadEnd': {
        'children': [],
        'mp_landmark': MP_LANDMARKS.NOSE.value,  # Same as Head for end site
        'channels': [],  # No channels for end site
        'is_end_site': True,
        'end_site_offset': [0.0, 0.15, 0.0] # Example offset for head's end site
    },
    'LeftShoulder': {
        'children': ['LeftArm'],
        'mp_landmark': MP_LANDMARKS.LEFT_SHOULDER.value,
        'channels': ['Zrotation', 'Yrotation', 'Xrotation']
    },
    'LeftArm': { # Represents the Upper Arm segment
        'children': ['LeftForeArm'],
        'mp_landmark': MP_LANDMARKS.LEFT_ELBOW.value,
        'channels': ['Zrotation', 'Yrotation', 'Xrotation']
    },
    'LeftForeArm': { # Represents the Forearm segment
        'children': ['LeftHand'],
        'mp_landmark': MP_LANDMARKS.LEFT_WRIST.value,
        'channels': ['Zrotation', 'Yrotation', 'Xrotation']
    },
    'LeftHand': {
        'children': ['LeftHandEnd'],
        'mp_landmark': MP_LANDMARKS.LEFT_INDEX.value, # Or average of LEFT_PINKY, LEFT_INDEX, LEFT_THUMB
        'channels': ['Zrotation', 'Yrotation', 'Xrotation']
    },
    'LeftHandEnd': {
        'children': [],
        'mp_landmark': MP_LANDMARKS.LEFT_INDEX.value,
        'channels': [],  # No channels for end site
        'is_end_site': True,
        'end_site_offset': [0.0, 0.1, 0.0] # Example offset for hand's end site
    },
    'RightShoulder': {
        'children': ['RightArm'],
        'mp_landmark': MP_LANDMARKS.RIGHT_SHOULDER.value,
        'channels': ['Zrotation', 'Yrotation', 'Xrotation']
    },
    'RightArm': {
        'children': ['RightForeArm'],
        'mp_landmark': MP_LANDMARKS.RIGHT_ELBOW.value,
        'channels': ['Zrotation', 'Yrotation', 'Xrotation']
    },
    'RightForeArm': {
        'children': ['RightHand'],
        'mp_landmark': MP_LANDMARKS.RIGHT_WRIST.value,
        'channels': ['Zrotation', 'Yrotation', 'Xrotation']
    },
    'RightHand': {
        'children': ['RightHandEnd'],
        'mp_landmark': MP_LANDMARKS.RIGHT_INDEX.value,
        'channels': ['Zrotation', 'Yrotation', 'Xrotation']
    },
    'RightHandEnd': {
        'children': [],
        'mp_landmark': MP_LANDMARKS.RIGHT_INDEX.value,
        'channels': [],  # No channels for end site
        'is_end_site': True,
        'end_site_offset': [0.0, 0.1, 0.0]
    },
    'LeftUpLeg': {
        'children': ['LeftLeg'],
        'mp_landmark': MP_LANDMARKS.LEFT_HIP.value,
        'channels': ['Zrotation', 'Yrotation', 'Xrotation']
    },
    'LeftLeg': { # Represents Lower Leg
        'children': ['LeftFoot'],
        'mp_landmark': MP_LANDMARKS.LEFT_KNEE.value,
        'channels': ['Zrotation', 'Yrotation', 'Xrotation']
    },
    'LeftFoot': {
        'children': ['LeftToeBase'],
        'mp_landmark': MP_LANDMARKS.LEFT_ANKLE.value,
        'channels': ['Zrotation', 'Yrotation', 'Xrotation']
    },
    'LeftToeBase': {
        'children': [],
        'mp_landmark': MP_LANDMARKS.LEFT_FOOT_INDEX.value,
        'channels': [], # No channels for End Site
        'is_end_site': True,
        'end_site_offset': [0.0, -0.1, 0.0] # Example offset for toe base
    },
    'RightUpLeg': {
        'children': ['RightLeg'],
        'mp_landmark': MP_LANDMARKS.RIGHT_HIP.value,
        'channels': ['Zrotation', 'Yrotation', 'Xrotation']
    },
    'RightLeg': {
        'children': ['RightFoot'],
        'mp_landmark': MP_LANDMARKS.RIGHT_KNEE.value,
        'channels': ['Zrotation', 'Yrotation', 'Xrotation']
    },
    'RightFoot': {
        'children': ['RightToeBase'],
        'mp_landmark': MP_LANDMARKS.RIGHT_ANKLE.value,
        'channels': ['Zrotation', 'Yrotation', 'Xrotation']
    },
    'RightToeBase': {
        'children': [],
        'mp_landmark': MP_LANDMARKS.RIGHT_FOOT_INDEX.value,
        'channels': [],
        'is_end_site': True,
        'end_site_offset': [0.0, -0.1, 0.0]
    }
})

# Populate parent names and initialize motion data lists
for joint_name, data in SKELETON_DEF.items():
    data['motion_data'] = [] # To store (translation_xyz, euler_angles_xyz) for root, or euler_angles_xyz for others
    for child_name in data.get('children', []):
        if child_name in SKELETON_DEF:
            SKELETON_DEF[child_name]['parent_name'] = joint_name
        else:
            print(f"Warning: Child joint '{child_name}' defined in '{joint_name}' not found in SKELETON_DEF.")


# --- Helper Functions ---

def get_landmark_coords(landmarks_list, mp_landmark_indices, visibility_threshold=0.5):
    """
    Extracts and averages specified MediaPipe landmark coordinates.
    Converts to BVH coordinate system (Y-up, X-right, Z-forward).
    MediaPipe world_landmarks: origin at hip midpoint. X to subject's right,
                               Y up (approx), Z towards camera (smaller Z is closer).
    """
    coords = []
    if not isinstance(mp_landmark_indices, list):
        mp_landmark_indices = [mp_landmark_indices]

    # Handle both list and object with landmark attribute
    if isinstance(landmarks_list, list):
        landmarks = landmarks_list
    elif hasattr(landmarks_list, 'landmark'):
        landmarks = landmarks_list.landmark
    else:
        print(f"Error: Invalid landmarks_list type: {type(landmarks_list)}")
        if hasattr(landmarks_list, '__dict__'):
            print(f"Available attributes: {list(landmarks_list.__dict__.keys())}")
        return None

    for lm_index in mp_landmark_indices:
        if lm_index < len(landmarks):
            lm = landmarks[lm_index]
            if lm.visibility >= visibility_threshold: # Check visibility
                 # MediaPipe X, Y, Z are in meters.
                 # BVH: Y-up, X-right, Z-forward (from camera perspective if camera is at -Z)
                 # MediaPipe: X-right, Y-up (approx), Z-towards camera (smaller is closer)
                 # To map to BVH (Y-up, X-right, Z-forward):
                 # BVH_X = MP_X
                 # BVH_Y = MP_Y
                 # BVH_Z = -MP_Z (this makes landmarks further from camera have positive Z)
                coords.append(np.array([lm.x, lm.y, -lm.z]))
            else:
                # print(f"Warning: Landmark {MP_LANDMARKS(lm_index).name} has low visibility ({lm.visibility:.2f}).")
                pass # Optionally return None or a placeholder if a key landmark is not visible
        else:
            print(f"Warning: Landmark index {lm_index} out of bounds.")
            return None # Critical error

    if not coords:
        # print(f"Warning: No valid landmarks found for indices {mp_landmark_indices} with visibility > {visibility_threshold}.")
        return None
    return np.mean(coords, axis=0)


def get_bone_orientation_matrix(p_joint_coord, c_joint_coord, parent_world_orientation_matrix=None):
    """
    Calculates the local coordinate system (orientation matrix) for a bone.
    The bone's Y-axis points from parent (proximal) to child (distal).
    Uses parent's orientation to help define a stable X (side) and Z (front) axis.
    Returns a 3x3 orientation matrix (columns are X, Y, Z axes).
    """
    if p_joint_coord is None or c_joint_coord is None or np.allclose(p_joint_coord, c_joint_coord):
        # print(f"Warning: Invalid joint coords for orientation: P={p_joint_coord}, C={c_joint_coord}")
        return np.identity(3) # Default to identity if bone length is zero or coords are invalid

    y_axis = c_joint_coord - p_joint_coord
    if np.linalg.norm(y_axis) < 1e-6: # Check for zero length vector
        return np.identity(3)
    y_axis = y_axis / np.linalg.norm(y_axis)

    # Determine X and Z axes
    if parent_world_orientation_matrix is not None:
        # Try to use parent's Z-axis (forward) to determine current bone's X-axis (side)
        parent_z_axis_world = parent_world_orientation_matrix[:, 2]
        x_axis = np.cross(y_axis, parent_z_axis_world)
        if np.linalg.norm(x_axis) < 1e-6: # y_axis is collinear with parent_z_axis_world
            # Fallback: use parent's X-axis (side) to determine current bone's Z-axis (forward)
            parent_x_axis_world = parent_world_orientation_matrix[:, 0]
            z_axis_temp = np.cross(parent_x_axis_world, y_axis) # Temp Z
            if np.linalg.norm(z_axis_temp) < 1e-6: # Still collinear, fallback to global
                x_axis = np.cross(y_axis, np.array([0,0,1.0])) # Global Z as reference for X
                if np.linalg.norm(x_axis) < 1e-6:
                    x_axis = np.cross(y_axis, np.array([0,1.0,0])) # Global Y if Z fails
            else:
                x_axis = np.cross(y_axis, z_axis_temp / np.linalg.norm(z_axis_temp))

    else: # For root or if no parent orientation provided, use global axes
        # Assume default "front" is global Z, "side" is global X
        x_axis = np.cross(y_axis, np.array([0,0,1.0])) # Global Z as reference for X
        if np.linalg.norm(x_axis) < 1e-6: # y_axis is collinear with global Z
            x_axis = np.cross(y_axis, np.array([0,1.0,0])) # Global Y if Z fails

    if np.linalg.norm(x_axis) < 1e-6: # If x_axis is still zero (e.g. y_axis aligns with all fallbacks)
        # This can happen if y_axis is perfectly aligned with world X or Y.
        # Create an arbitrary perpendicular vector.
        if not np.allclose(y_axis, [1, 0, 0]):
            x_axis = np.cross(y_axis, np.array([1, 0, 0]))
        else: # y_axis is [1, 0, 0]
            x_axis = np.cross(y_axis, np.array([0, 1, 0]))

    x_axis = x_axis / np.linalg.norm(x_axis)
    z_axis = np.cross(x_axis, y_axis)
    z_axis = z_axis / np.linalg.norm(z_axis) # Should be unit if x and y are unit and perp.

    return np.column_stack((x_axis, y_axis, z_axis))


def get_hierarchical_joint_order(skeleton_definition, root_name):
    """Performs a BFS traversal to get joint order for processing."""
    ordered_joints = []
    queue = collections.deque([root_name])
    visited = {root_name}
    while queue:
        current_joint_name = queue.popleft()
        ordered_joints.append(current_joint_name)
        for child_name in skeleton_definition.get(current_joint_name, {}).get('children', []):
            if child_name in skeleton_definition and child_name not in visited:
                queue.append(child_name)
                visited.add(child_name)
    return ordered_joints

JOINT_ORDER = get_hierarchical_joint_order(SKELETON_DEF, 'Hips')


def get_t_pose_landmarks_from_video(video_path, pose_landmarker):
    """
    Processes the first frame of a video to get T-Pose landmarks.
    Assumes the first frame IS the T-Pose.
    Modify this to load from a specific image if needed.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return None

    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("Error: Could not read the first frame for T-Pose.")
        return None

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    
    try:
        detection_result = pose_landmarker.detect(mp_image) # Use detect for single image
        if detection_result and detection_result.pose_world_landmarks:
            # Check if pose_world_landmarks is a list and not empty
            if isinstance(detection_result.pose_world_landmarks, list) and detection_result.pose_world_landmarks:
                # Return the first detected pose's landmarks (which should have .landmark attribute)
                return detection_result.pose_world_landmarks[0]
            else: # If it's not a list (older MediaPipe versions might return directly)
                return detection_result.pose_world_landmarks 
        else:
            print("Error: No pose landmarks detected in the T-Pose frame.")
            return None
    except Exception as e:
        print(f"Error during T-Pose landmark detection: {e}")
        import traceback
        traceback.print_exc()
        return None


def initialize_skeleton_for_bvh(skeleton_def, t_pose_world_landmarks):
    """
    Calculates T-Pose XYZ coordinates, offsets, and T-Pose orientation matrices.
    """
    if t_pose_world_landmarks is None:
        print("Error: T-Pose landmarks are None. Cannot initialize skeleton.")
        return False

    # 1. Store T-Pose landmark coordinates
    for joint_name, data in skeleton_def.items():
        if 'mp_landmark' in data:
            coords = get_landmark_coords(t_pose_world_landmarks, data['mp_landmark'])
            if coords is None:
                print(f"Critical Error: Could not get T-Pose coords for {joint_name}. Check landmark definitions and T-Pose image.")
                return False
            data['t_pose_xyz'] = coords
        elif not data.get('is_end_site', False):
             print(f"Warning: Joint {joint_name} has no 'mp_landmark' and is not an end site.")


    # 2. Calculate BVH OFFSETS
    for joint_name in JOINT_ORDER: # Process in hierarchical order
        data = skeleton_def[joint_name]
        if data.get('is_end_site', False):
            data['offset'] = np.array(data.get('end_site_offset', [0.0, 0.0, 0.0]))
            continue

        if 'parent_name' in data:
            parent_data = skeleton_def[data['parent_name']]
            if 't_pose_xyz' in data and 't_pose_xyz' in parent_data:
                data['offset'] = data['t_pose_xyz'] - parent_data['t_pose_xyz']
            else:
                print(f"Error calculating offset for {joint_name}: T-pose XYZ missing for self or parent.")
                return False
        else: # Root joint
            data['offset'] = np.array([0.0, 0.0, 0.0])

    # 3. Calculate T-Pose Orientation Matrices
    # This defines the "zero rotation" orientation for each bone.
    for joint_name in JOINT_ORDER:
        data = skeleton_def[joint_name]
        if data.get('is_end_site', False):
            data['t_pose_orientation_matrix'] = np.identity(3) # End sites don't have orientation in this context
            continue

        parent_name = data.get('parent_name')
        parent_t_pose_orientation = np.identity(3) # For root or if parent is not found
        if parent_name and parent_name in skeleton_def:
            parent_t_pose_orientation = skeleton_def[parent_name].get('t_pose_orientation_matrix', np.identity(3))
        
        # For non-end-site joints, find the primary child to define bone direction
        # If no children, it's effectively an end effector for orientation purposes,
        # but BVH still needs a JOINT definition if it has channels.
        # We'll use its own t_pose_xyz and parent's t_pose_xyz to define its orientation.
        p_coord = data['t_pose_xyz'] # Proximal end of the "bone" is the joint itself
        
        # Find a distal point to define the bone's primary axis in T-Pose
        # This could be its first child, or if it's a leaf JOINT (not End Site), use a nominal direction
        distal_coord_for_orientation = None
        if data.get('children'):
            first_child_name = data['children'][0]
            if first_child_name in skeleton_def and 't_pose_xyz' in skeleton_def[first_child_name]:
                distal_coord_for_orientation = skeleton_def[first_child_name]['t_pose_xyz']
        
        if distal_coord_for_orientation is None:
            # If it's a JOINT but has no children defined for orientation (e.g. Hand, Foot before End Site)
            # or if it's the root, we need a different strategy for its "default" direction.
            if joint_name == 'Hips': # Root
                # Define Hips T-pose orientation: Y up (Hips to Spine), X right (RightHip to LeftHip), Z forward
                spine_tpose = skeleton_def['Spine']['t_pose_xyz']
                left_hip_tpose = get_landmark_coords(t_pose_world_landmarks, MP_LANDMARKS.LEFT_HIP.value)
                right_hip_tpose = get_landmark_coords(t_pose_world_landmarks, MP_LANDMARKS.RIGHT_HIP.value)

                if spine_tpose is None or left_hip_tpose is None or right_hip_tpose is None:
                    print("Error: Missing critical landmarks for Hips T-pose orientation.")
                    return False

                root_y_axis = spine_tpose - data['t_pose_xyz']
                if np.linalg.norm(root_y_axis) < 1e-6: root_y_axis = np.array([0, 1, 0]) # Default up
                else: root_y_axis /= np.linalg.norm(root_y_axis)
                
                # X-axis from right hip to left hip (points to character's left in MP coords)
                # To get character's right (BVH X+), use left_hip_tpose - right_hip_tpose
                root_x_axis_temp = left_hip_tpose - right_hip_tpose
                if np.linalg.norm(root_x_axis_temp) < 1e-6: root_x_axis_temp = np.array([1, 0, 0]) # Default right

                # Make sure X is orthogonal to Y
                root_z_axis_temp = np.cross(root_x_axis_temp, root_y_axis) # Temp Z (forward-ish)
                if np.linalg.norm(root_z_axis_temp) < 1e-6: root_z_axis_temp = np.array([0, 0, 1])

                root_x_axis = np.cross(root_y_axis, root_z_axis_temp) # Final X
                if np.linalg.norm(root_x_axis) < 1e-6: root_x_axis = np.array([1, 0, 0])
                else: root_x_axis /= np.linalg.norm(root_x_axis)

                root_z_axis = np.cross(root_x_axis, root_y_axis) # Final Z
                if np.linalg.norm(root_z_axis) < 1e-6: root_z_axis = np.array([0, 0, 1])
                else: root_z_axis /= np.linalg.norm(root_z_axis)

                data['t_pose_orientation_matrix'] = np.column_stack((root_x_axis, root_y_axis, root_z_axis))
                continue # Skip to next joint
            else:
                # For other leaf JOINTs without children for orientation, assume they align with parent's orientation initially
                # or use a nominal offset if available (e.g. for hands, feet pointing slightly forward)
                # This is a simplification; ideally, their T-pose orientation is also well-defined.
                # For now, let's assume its primary axis in T-pose aligns with parent's primary axis.
                # This means its t_pose_orientation_matrix would be identity relative to parent's t_pose_orientation
                # So, its world t_pose_orientation_matrix is same as parent's.
                # This might not be ideal for all cases.
                # A better way: use its own offset vector to define its primary Y axis in T-pose.
                # distal_coord_for_orientation = p_coord + data['offset'] # Use its own offset vector
                # A simple default: align with parent's orientation
                data['t_pose_orientation_matrix'] = parent_t_pose_orientation
                # print(f"Warning: Leaf JOINT {joint_name} using parent's T-pose orientation. Consider defining its T-pose direction.")
                continue


        if distal_coord_for_orientation is None: # Should not happen if logic above is correct for root/leaves
             print(f"Error: Could not determine distal point for T-pose orientation of {joint_name}")
             return False

        data['t_pose_orientation_matrix'] = get_bone_orientation_matrix(
            p_coord, distal_coord_for_orientation, parent_t_pose_orientation
        )
    return True

# --- Main Processing ---
def main(video_path, bvh_output_path, bvh_fps=30, pose_model_complexity=2, 
         min_detection_confidence=0.5, min_tracking_confidence=0.5,
         smoothing_window_length=5, smoothing_polyorder=2):
    # Initialize MediaPipe Pose
    mp_pose = mp.solutions.pose
    pose_options = mp.tasks.vision.PoseLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path='model/pose_landmarker_heavy.task' if pose_model_complexity == 2 else ('pose_landmarker_full.task' if pose_model_complexity == 1 else 'pose_landmarker_lite.task')),
        running_mode=mp.tasks.vision.RunningMode.VIDEO, # Changed to VIDEO for detect_for_video
        num_poses=1,
        min_pose_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
        output_segmentation_masks=False
    )
    # Create landmarker for T-Pose (using IMAGE mode temporarily for single frame)
    tpose_options_dict = pose_options.__dict__.copy()
    tpose_options_dict['running_mode'] = mp.tasks.vision.RunningMode.IMAGE
    tpose_landmarker_options = mp.tasks.vision.PoseLandmarkerOptions(**tpose_options_dict)
    
    try:
        with mp.tasks.vision.PoseLandmarker.create_from_options(tpose_landmarker_options) as tpose_landmarker, \
             mp.tasks.vision.PoseLandmarker.create_from_options(pose_options) as landmarker:

            # 1. T-Pose Calibration
            print("Attempting to get T-Pose landmarks from the first frame of the video...")
            t_pose_world_landmarks = get_t_pose_landmarks_from_video(video_path, tpose_landmarker)
            if not t_pose_world_landmarks:
                print("Failed to get T-Pose landmarks. Exiting.")
                return

            if not initialize_skeleton_for_bvh(SKELETON_DEF, t_pose_world_landmarks):
                print("Failed to initialize skeleton from T-Pose. Exiting.")
                return
            print("T-Pose calibration successful. Offsets and T-Pose orientations calculated.")
            
            # Store initial Hips position from T-Pose to make motion relative
            initial_hips_world_pos = SKELETON_DEF['Hips']['t_pose_xyz'].copy()


            # 2. Process Video and Extract All Landmarks
            print(f"Processing video: {video_path}")
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"Error: Could not open video {video_path}")
                return

            all_frames_world_landmarks = []
            frame_idx = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                
                timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
                detection_result = landmarker.detect_for_video(mp_image, timestamp_ms)

                if detection_result and detection_result.pose_world_landmarks:
                    if isinstance(detection_result.pose_world_landmarks, list) and detection_result.pose_world_landmarks:
                        # Store the first detected pose (which should have .landmark attribute)
                        all_frames_world_landmarks.append(detection_result.pose_world_landmarks[0])
                    else:
                        all_frames_world_landmarks.append(detection_result.pose_world_landmarks)
                else:
                    all_frames_world_landmarks.append(None) # Keep track of frames with no detection
                
                frame_idx += 1
                if frame_idx % 100 == 0:
                    print(f"Processed {frame_idx} frames...")

            cap.release()
            print(f"Finished processing video. Total frames with landmark data: {len(all_frames_world_landmarks)}")

            if not any(all_frames_world_landmarks):
                print("No landmarks detected in any frame. Exiting.")
                return

            # 3. Temporal Smoothing of Landmark Data
            num_frames = len(all_frames_world_landmarks)
            num_mp_landmarks = 33 # MediaPipe Pose outputs 33 landmarks
            
            landmark_data_xyz_raw = np.zeros((num_frames, num_mp_landmarks, 3))
            landmark_visibility = np.zeros((num_frames, num_mp_landmarks))
            valid_frame_indices = []


            for i, frame_landmarks_mp in enumerate(all_frames_world_landmarks):
                if frame_landmarks_mp:
                    valid_frame_indices.append(i)
                    # Handle both list and object with landmark attribute
                    if isinstance(frame_landmarks_mp, list):
                        landmarks = frame_landmarks_mp
                    elif hasattr(frame_landmarks_mp, 'landmark'):
                        landmarks = frame_landmarks_mp.landmark
                    else:
                        print(f"Warning: Frame {i} has invalid landmark structure")
                        landmark_data_xyz_raw[i, :, :] = np.nan
                        landmark_visibility[i,:] = 0
                        continue
                    
                    for j in range(num_mp_landmarks):
                        if j < len(landmarks):
                            lm = landmarks[j]
                            landmark_data_xyz_raw[i, j, :] = [lm.x, lm.y, -lm.z] # Coordinate transform
                            landmark_visibility[i,j] = lm.visibility
                        else: # Should not happen if MediaPipe always returns 33
                            landmark_data_xyz_raw[i, j, :] = np.nan # Mark as missing
                            landmark_visibility[i,j] = 0

                else: # Frame with no detection
                    landmark_data_xyz_raw[i, :, :] = np.nan # Mark all landmarks as missing
                    landmark_visibility[i,:] = 0


            smoothed_landmark_data_xyz = np.copy(landmark_data_xyz_raw)
            if num_frames > smoothing_window_length:
                print("Applying Savitzky-Golay filter for smoothing...")
                for landmark_idx in range(num_mp_landmarks):
                    for axis_idx in range(3): # x, y, z
                        # Only filter valid (non-NaN) segments
                        raw_series = landmark_data_xyz_raw[:, landmark_idx, axis_idx]
                        valid_mask = ~np.isnan(raw_series)
                        
                        # Find contiguous blocks of valid data to apply filter
                        # This is a simplified approach; more robust NaN interpolation might be needed
                        # For now, filter only if a sufficient block of data is present
                        if np.sum(valid_mask) > smoothing_window_length:
                            # Create a temporary array with only valid points for filtering
                            series_to_filter = raw_series[valid_mask]
                            if len(series_to_filter) > smoothing_window_length :
                                smoothed_series = savgol_filter(
                                    series_to_filter,
                                    smoothing_window_length,
                                    smoothing_polyorder
                                )
                                # Place smoothed data back
                                smoothed_landmark_data_xyz[valid_mask, landmark_idx, axis_idx] = smoothed_series
                            # else: too short to filter, keep raw
                        # else: not enough data or too many NaNs, keep raw
            else:
                print("Not enough frames for smoothing, using raw landmark data.")

            # Create a list of smoothed MediaPipe Landmark objects for get_landmark_coords
            smoothed_mp_landmark_frames = []
            for i in range(num_frames):
                if i not in valid_frame_indices: # if original frame had no detection
                    smoothed_mp_landmark_frames.append(None)
                    continue

                # Reconstruct a LandmarkList-like object
                frame_lms_data = []
                for j in range(num_mp_landmarks):
                    # Create a simple object that mimics MediaPipe's Landmark structure
                    lm_obj = type('Landmark', (object,), {
                        'x': smoothed_landmark_data_xyz[i,j,0],
                        'y': smoothed_landmark_data_xyz[i,j,1],
                        'z': -smoothed_landmark_data_xyz[i,j,2], # Convert back to MP Z-convention for get_landmark_coords
                        'visibility': landmark_visibility[i,j] # Use original visibility
                    })()
                    frame_lms_data.append(lm_obj)
                
                # Create a simple object that mimics MediaPipe's PoseLandmarkerResult structure
                smoothed_frame_result = type('PoseWorldLandmarks', (object,), {'landmark': frame_lms_data})()
                smoothed_mp_landmark_frames.append(smoothed_frame_result)


            # 4. Calculate Rotations for Each Frame
            print("Calculating joint rotations for BVH...")
            for frame_idx_loop in range(num_frames):
                current_smoothed_mp_landmarks = smoothed_mp_landmark_frames[frame_idx_loop]

                if current_smoothed_mp_landmarks is None:
                    # Handle frames with no detection: repeat last valid pose or use default
                    # For simplicity, we'll mark rotations as None and handle in BVH writing
                    for joint_name in JOINT_ORDER:
                        SKELETON_DEF[joint_name]['motion_data'].append(None)
                    continue

                # Update current_xyz for all joints based on smoothed landmarks
                all_landmarks_valid_for_frame = True
                for joint_name, data in SKELETON_DEF.items():
                    if 'mp_landmark' in data: # Skip end sites for this
                        coords = get_landmark_coords(current_smoothed_mp_landmarks, data['mp_landmark'])
                        if coords is None:
                            # print(f"Frame {frame_idx_loop}: Missing/low-vis landmarks for {joint_name}. Using T-pose as fallback.")
                            # Fallback to T-pose if landmarks are missing for a joint in a frame
                            data['current_xyz'] = data['t_pose_xyz']
                            all_landmarks_valid_for_frame = False # Mark frame as potentially problematic
                        else:
                            data['current_xyz'] = coords
                
                # if not all_landmarks_valid_for_frame:
                    # print(f"Warning: Frame {frame_idx_loop} has missing landmarks for some joints.")

                # Calculate current world orientation matrix for each joint
                for joint_name in JOINT_ORDER:
                    data = SKELETON_DEF[joint_name]
                    if data.get('is_end_site', False):
                        data['current_orientation_matrix'] = np.identity(3)
                        continue

                    parent_name = data.get('parent_name')
                    parent_current_world_orientation = np.identity(3)
                    p_coord_current = data['current_xyz'] # Proximal is the joint itself

                    if parent_name:
                        parent_current_world_orientation = SKELETON_DEF[parent_name]['current_orientation_matrix']
                        # p_coord_current = SKELETON_DEF[parent_name]['current_xyz'] # This was wrong, p_coord is the joint itself
                    
                    # Determine distal coordinate for orientation calculation
                    distal_coord_current = None
                    if data.get('children'):
                        first_child_name = data['children'][0]
                        if first_child_name in SKELETON_DEF and 'current_xyz' in SKELETON_DEF[first_child_name]:
                             distal_coord_current = SKELETON_DEF[first_child_name]['current_xyz']
                    
                    if distal_coord_current is None:
                        if joint_name == 'Hips': # Root
                            current_hips_avg = data['current_xyz']
                            current_spine_avg = SKELETON_DEF['Spine']['current_xyz']
                            current_left_hip = get_landmark_coords(current_smoothed_mp_landmarks, MP_LANDMARKS.LEFT_HIP.value)
                            current_right_hip = get_landmark_coords(current_smoothed_mp_landmarks, MP_LANDMARKS.RIGHT_HIP.value)

                            if current_spine_avg is None or current_left_hip is None or current_right_hip is None:
                                # print(f"Frame {frame_idx_loop}: Missing critical landmarks for Hips current orientation. Using T-pose.")
                                data['current_orientation_matrix'] = data['t_pose_orientation_matrix']
                                continue
                            
                            cur_root_y = (current_spine_avg - current_hips_avg)
                            if np.linalg.norm(cur_root_y) < 1e-6: cur_root_y = data['t_pose_orientation_matrix'][:,1] # from T-pose
                            else: cur_root_y /= np.linalg.norm(cur_root_y)
                            
                            cur_root_x_temp = current_left_hip - current_right_hip
                            if np.linalg.norm(cur_root_x_temp) < 1e-6: cur_root_x_temp = data['t_pose_orientation_matrix'][:,0] # from T-pose

                            cur_root_z_temp = np.cross(cur_root_x_temp, cur_root_y)
                            if np.linalg.norm(cur_root_z_temp) < 1e-6: cur_root_z_temp = data['t_pose_orientation_matrix'][:,2]
                            
                            cur_root_x = np.cross(cur_root_y, cur_root_z_temp)
                            if np.linalg.norm(cur_root_x) < 1e-6: cur_root_x = data['t_pose_orientation_matrix'][:,0]
                            else: cur_root_x /= np.linalg.norm(cur_root_x)
                            
                            cur_root_z = np.cross(cur_root_x, cur_root_y)
                            if np.linalg.norm(cur_root_z) < 1e-6: cur_root_z = data['t_pose_orientation_matrix'][:,2]
                            else: cur_root_z /= np.linalg.norm(cur_root_z)
                            
                            data['current_orientation_matrix'] = np.column_stack((cur_root_x, cur_root_y, cur_root_z))
                            continue
                        else: # Leaf JOINT
                            # distal_coord_current = p_coord_current + data['offset'] # Use its own offset vector from T-pose
                            # A simple default: align with parent's current orientation
                            data['current_orientation_matrix'] = parent_current_world_orientation
                            # print(f"Frame {frame_idx_loop}: Leaf JOINT {joint_name} using parent's current orientation.")
                            continue
                    
                    if distal_coord_current is None: # Should not happen
                        data['current_orientation_matrix'] = data['t_pose_orientation_matrix'] # Fallback
                        continue

                    data['current_orientation_matrix'] = get_bone_orientation_matrix(
                        p_coord_current, distal_coord_current, parent_current_world_orientation
                    )

                # Calculate local rotations for BVH
                frame_motion_data = {}
                for joint_name in JOINT_ORDER:
                    data = SKELETON_DEF[joint_name]
                    if data.get('is_end_site', False):
                        continue # End sites don't have rotations in BVH motion

                    m_bone_current_world = data['current_orientation_matrix']
                    m_bone_tpose_world = data['t_pose_orientation_matrix']
                    
                    r_bone_current_world = R.from_matrix(m_bone_current_world)
                    r_bone_tpose_world = R.from_matrix(m_bone_tpose_world)

                    # Rotation of this bone from its T-pose to its current world orientation
                    rot_bone_from_tpose_to_current = r_bone_current_world * r_bone_tpose_world.inv()

                    if 'parent_name' in data:
                        parent_data = SKELETON_DEF[data['parent_name']]
                        m_parent_current_world = parent_data['current_orientation_matrix']
                        m_parent_tpose_world = parent_data['t_pose_orientation_matrix']
                        
                        r_parent_current_world = R.from_matrix(m_parent_current_world)
                        r_parent_tpose_world = R.from_matrix(m_parent_tpose_world)

                        # Rotation of parent from its T-pose to its current world orientation
                        rot_parent_from_tpose_to_current = r_parent_current_world * r_parent_tpose_world.inv()
                        
                        # Local rotation for BVH: R_local = inv(Rot_parent_from_Tpose) * Rot_bone_from_Tpose
                        r_local_bvh = rot_parent_from_tpose_to_current.inv() * rot_bone_from_tpose_to_current
                    else: # Root joint
                        r_local_bvh = rot_bone_from_tpose_to_current # Root's "local" is its rotation from T-pose

                    # Convert to Euler angles (ZXY order is common for BVH)
                    # CHANNELS Zrotation Yrotation Xrotation means intrinsic z, then y, then x.
                    # scipy's as_euler('zyx') means intrinsic z, then y, then x.
                    # For BVH Zrotation Xrotation Yrotation, use 'zxy' in scipy
                    euler_angles_deg = r_local_bvh.as_euler('zxy', degrees=True)
                    
                    if joint_name == 'Hips':
                        # Root translation relative to initial T-pose position
                        translation = data['current_xyz'] - initial_hips_world_pos
                        # BVH typically expects translation in cm or same units as offsets.
                        # MediaPipe is in meters. If your BVH viewer expects cm, multiply by 100.
                        # For now, keep in meters.
                        frame_motion_data[joint_name] = (translation, euler_angles_deg)
                    else:
                        frame_motion_data[joint_name] = euler_angles_deg
                
                # Append calculated motion for this frame
                for joint_name_in_order in JOINT_ORDER:
                    if not SKELETON_DEF[joint_name_in_order].get('is_end_site', False):
                         SKELETON_DEF[joint_name_in_order]['motion_data'].append(frame_motion_data.get(joint_name_in_order))


            # 5. Write BVH File using bvhio
            print(f"Writing BVH file to: {bvh_output_path}")
            
            # Helper function to recursively build the hierarchy
            def build_joint_hierarchy(joint_name, parent_bvh_joint=None):
                data = SKELETON_DEF[joint_name]
                
                # Skip pure end sites - bvhio will create them automatically
                if data.get('is_end_site', False) and not data.get('channels'):
                    return None
                
                # Create the joint
                bvh_joint = bvhio.Joint(name=joint_name)
                bvh_joint.Offset = data.get('offset', [0.0, 0.0, 0.0])
                bvh_joint.channels = data.get('channels', [])
                
                # Attach to parent if not root
                if parent_bvh_joint is not None:
                    parent_bvh_joint.attach(bvh_joint)
                
                # Store in map for later reference
                bvh_joints_map[joint_name] = bvh_joint
                
                # Recursively create and attach children
                for child_name in data.get('children', []):
                    build_joint_hierarchy(child_name, bvh_joint)
                
                return bvh_joint
            
            # Build hierarchy starting from root
            bvh_joints_map = {}
            root_bvh_joint = build_joint_hierarchy('Hips')
            
            if not root_bvh_joint:
                print("Error: Root BVH joint not created.")
                return

            # Set rest pose (bvhio uses current pose as rest pose if writeRestPose is called)
            # Our offsets already define the T-Pose structure.
            # The rotations in motion data are relative to this T-Pose.
            # So, the "rest pose" in bvhio terms should have zero rotations.
            root_bvh_joint.setEuler((0,0,0), order='ZXY') # Set root to zero rotation for rest
            for joint_bvh_obj in root_bvh_joint.filter('*'): # All joints
                joint_bvh_obj.setEuler((0,0,0), order='ZXY')
            root_bvh_joint.writeRestPose(recursive=True)


            # Add motion data
            num_motion_frames = len(SKELETON_DEF['Hips']['motion_data']) # Assuming Hips has data for all frames
            
            for frame_i in range(num_motion_frames):
                pose_data_for_frame = []
                valid_frame = True
                for joint_name in JOINT_ORDER: # Ensure BVH channel order
                    data = SKELETON_DEF[joint_name]
                    if data.get('is_end_site', False) or not data.get('channels'):
                        continue # Skip end sites or joints with no channels in motion

                    motion_entry = data['motion_data'][frame_i]
                    if motion_entry is None:
                        # Use last valid frame's rotation or T-pose (0,0,0)
                        # For simplicity, use T-pose (zeros) for missing data
                        # print(f"Warning: Frame {frame_i}, Joint {joint_name} missing motion. Using T-pose (zeros).")
                        if joint_name == 'Hips':
                            pose_data_for_frame.extend([0.0, 0.0, 0.0]) # Translation
                            pose_data_for_frame.extend([0.0, 0.0, 0.0]) # Rotation
                        else:
                            pose_data_for_frame.extend([0.0, 0.0, 0.0]) # Rotation
                        valid_frame = False # Mark frame as having missing data
                        continue

                    if joint_name == 'Hips':
                        translation, euler_angles = motion_entry
                        # Order for Hips: Xpos, Ypos, Zpos, Zrot, Yrot, Xrot
                        # Our Euler angles are Z, Y, X from as_euler('zxy')
                        pose_data_for_frame.extend(translation)
                        pose_data_for_frame.extend(euler_angles) # Z, Y, X
                    else:
                        euler_angles = motion_entry
                        # Order for other joints: Zrot, Yrot, Xrot
                        pose_data_for_frame.extend(euler_angles) # Z, Y, X
                
                if valid_frame: # Only write pose if all data for this frame was valid
                    # bvhio expects rotations to be set on the joint objects, then writePose
                    for joint_name_bvhio in JOINT_ORDER:
                        bvh_joint_obj = bvh_joints_map.get(joint_name_bvhio)
                        if bvh_joint_obj and SKELETON_DEF[joint_name_bvhio]['motion_data'][frame_i] is not None:
                            if not SKELETON_DEF[joint_name_bvhio].get('is_end_site', False) and SKELETON_DEF[joint_name_bvhio].get('channels'):
                                motion_val = SKELETON_DEF[joint_name_bvhio]['motion_data'][frame_i]
                                if joint_name_bvhio == 'Hips':
                                    trans, euls = motion_val
                                    bvh_joint_obj.Position = list(trans) # bvhio expects list or tuple
                                    bvh_joint_obj.setEuler(list(euls), order='ZXY') # Z, Y, X
                                else:
                                    euls = motion_val
                                    bvh_joint_obj.setEuler(list(euls), order='ZXY') # Z, Y, X
                    
                    root_bvh_joint.writePose(frame_i, recursive=True)
                # else: skip writing this frame if any joint had missing data (already filled with zeros)
                # A better strategy for missing frames would be interpolation or holding last pose.

            bvhio.writeHierarchy(bvh_output_path, root_bvh_joint, frameTime=1.0/bvh_fps, precision=6)
            print("BVH file written successfully.")

    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert video to BVH using MediaPipe Pose.")
    parser.add_argument("--video", type=str, required=True, 
                        help="Path to the input video file.")
    parser.add_argument("--output", type=str, default="output_animation.bvh",
                        help="Path to save the output BVH file. (default: output_animation.bvh)")
    parser.add_argument("--fps", type=int, default=30,
                        help="Desired FPS for the BVH file. (default: 30)")
    parser.add_argument("--model-complexity", type=int, default=2, choices=[0, 1, 2],
                        help="MediaPipe model complexity: 0=lite, 1=full, 2=heavy. (default: 2)")
    parser.add_argument("--min-detection-confidence", type=float, default=0.5,
                        help="Minimum confidence for pose detection. (default: 0.5)")
    parser.add_argument("--min-tracking-confidence", type=float, default=0.5,
                        help="Minimum confidence for pose tracking. (default: 0.5)")
    parser.add_argument("--smoothing-window", type=int, default=5,
                        help="Window length for Savitzky-Golay smoothing filter (must be odd). (default: 5)")
    parser.add_argument("--smoothing-poly", type=int, default=2,
                        help="Polynomial order for Savitzky-Golay smoothing filter. (default: 2)")

    args = parser.parse_args()
    
    # Validate smoothing window is odd
    if args.smoothing_window % 2 == 0:
        parser.error("--smoothing-window must be an odd number")
    
    main(args.video, args.output, args.fps, args.model_complexity,
         args.min_detection_confidence, args.min_tracking_confidence,
         args.smoothing_window, args.smoothing_poly)