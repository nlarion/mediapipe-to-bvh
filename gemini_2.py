Pythonimport cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
# No direct import of landmark_pb2 needed if using PoseLandmark enums
import numpy as np
from scipy.spatial.transform import Rotation as R
import bvhio # For writing BVH files [4, 3]
import argparse
import os

# Use MediaPipe's PoseLandmark enum directly
# POSE_LANDMARK_NAMES = mp.solutions.pose.PoseLandmark # Not strictly needed as a separate var if using mp.solutions.pose.PoseLandmark directly

# --- BVH Skeleton Definition ---
BVH_SKELETON_DEF = + lm_dict) / 2,
     'channels': ['Xposition', 'Yposition', 'Zposition', 'Zrotation', 'Yrotation', 'Xrotation'], 'order': 'ZYX'},

    {'name': 'Spine', 'parent': 'Hips', 
     'lm_source': lambda lm_dict: ((lm_dict + lm_dict) / 2 + 
                                   (lm_dict + lm_dict) / 2) / 2,
     'channels': ['Zrotation', 'Yrotation', 'Xrotation'], 'order': 'ZYX'},
    {'name': 'Chest', 'parent': 'Spine', 
     'lm_source': lambda lm_dict: (lm_dict + lm_dict) / 2,
     'channels': ['Zrotation', 'Yrotation', 'Xrotation'], 'order': 'ZYX'},
    {'name': 'Neck', 'parent': 'Chest', 
     'lm_source': lambda lm_dict: ((lm_dict + lm_dict) / 2 + lm_dict) / 2,
     'channels': ['Zrotation', 'Yrotation', 'Xrotation'], 'order': 'ZYX'},
    {'name': 'Head', 'parent': 'Neck', 
     'lm_source': mp.solutions.pose.PoseLandmark.NOSE,
     'channels': ['Zrotation', 'Yrotation', 'Xrotation'], 'order': 'ZYX'},
    {'name': 'HeadEnd', 'parent': 'Head', 
     'lm_source': mp.solutions.pose.PoseLandmark.NOSE, 'offset_dir': np.array([0, 0.1, 0]), # Offset from NOSE to define head top
     'channels':, 'order': 'ZYX'}, # End Site

    # Left Arm
    {'name': 'LeftShoulder', 'parent': 'Chest', 
     'lm_source': mp.solutions.pose.PoseLandmark.LEFT_SHOULDER,
     'channels': ['Zrotation', 'Yrotation', 'Xrotation'], 'order': 'ZYX'},
    {'name': 'LeftElbow', 'parent': 'LeftShoulder', 
     'lm_source': mp.solutions.pose.PoseLandmark.LEFT_ELBOW,
     'channels': ['Zrotation', 'Yrotation', 'Xrotation'], 'order': 'ZYX'},
    {'name': 'LeftWrist', 'parent': 'LeftElbow', 
     'lm_source': mp.solutions.pose.PoseLandmark.LEFT_WRIST,
     'channels': ['Zrotation', 'Yrotation', 'Xrotation'], 'order': 'ZYX'},
    {'name': 'LeftHandEnd', 'parent': 'LeftWrist', 
     'lm_source': mp.solutions.pose.PoseLandmark.LEFT_INDEX, 'offset_dir': np.array([0,0,-0.05]), # Offset from wrist/index
     'channels':, 'order': 'ZYX'},

    # Right Arm
    {'name': 'RightShoulder', 'parent': 'Chest', 
     'lm_source': mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER,
     'channels': ['Zrotation', 'Yrotation', 'Xrotation'], 'order': 'ZYX'},
    {'name': 'RightElbow', 'parent': 'RightShoulder', 
     'lm_source': mp.solutions.pose.PoseLandmark.RIGHT_ELBOW,
     'channels': ['Zrotation', 'Yrotation', 'Xrotation'], 'order': 'ZYX'},
    {'name': 'RightWrist', 'parent': 'RightElbow', 
     'lm_source': mp.solutions.pose.PoseLandmark.RIGHT_WRIST,
     'channels': ['Zrotation', 'Yrotation', 'Xrotation'], 'order': 'ZYX'},
    {'name': 'RightHandEnd', 'parent': 'RightWrist', 
     'lm_source': mp.solutions.pose.PoseLandmark.RIGHT_INDEX, 'offset_dir': np.array([0,0,-0.05]),
     'channels':, 'order': 'ZYX'},

    # Left Leg
    {'name': 'LeftUpLeg', 'parent': 'Hips', 
     'lm_source': mp.solutions.pose.PoseLandmark.LEFT_HIP,
     'channels': ['Zrotation', 'Yrotation', 'Xrotation'], 'order': 'ZYX'},
    {'name': 'LeftLeg', 'parent': 'LeftUpLeg', 
     'lm_source': mp.solutions.pose.PoseLandmark.LEFT_KNEE,
     'channels': ['Zrotation', 'Yrotation', 'Xrotation'], 'order': 'ZYX'},
    {'name': 'LeftFoot', 'parent': 'LeftLeg', 
     'lm_source': mp.solutions.pose.PoseLandmark.LEFT_ANKLE,
     'channels': ['Zrotation', 'Yrotation', 'Xrotation'], 'order': 'ZYX'},
    {'name': 'LeftToeEnd', 'parent': 'LeftFoot', 
     'lm_source': mp.solutions.pose.PoseLandmark.LEFT_FOOT_INDEX, 'offset_dir': np.array([0,0,0.05]),
     'channels':, 'order': 'ZYX'},

    # Right Leg
    {'name': 'RightUpLeg', 'parent': 'Hips', 
     'lm_source': mp.solutions.pose.PoseLandmark.RIGHT_HIP,
     'channels': ['Zrotation', 'Yrotation', 'Xrotation'], 'order': 'ZYX'},
    {'name': 'RightLeg', 'parent': 'RightUpLeg', 
     'lm_source': mp.solutions.pose.PoseLandmark.RIGHT_KNEE,
     'channels': ['Zrotation', 'Yrotation', 'Xrotation'], 'order': 'ZYX'},
    {'name': 'RightFoot', 'parent': 'RightLeg', 
     'lm_source': mp.solutions.pose.PoseLandmark.RIGHT_ANKLE,
     'channels': ['Zrotation', 'Yrotation', 'Xrotation'], 'order': 'ZYX'},
    {'name': 'RightToeEnd', 'parent': 'RightFoot', 
     'lm_source': mp.solutions.pose.PoseLandmark.RIGHT_FOOT_INDEX, 'offset_dir': np.array([0,0,0.05]),
     'channels':, 'order': 'ZYX'}
]

# --- Helper Functions ---
def get_joint_world_pos(landmarks_mp_dict, joint_def_entry):
    """Calculates the 3D world position of a BVH joint from MediaPipe landmarks."""
    source = joint_def_entry['lm_source']
    if callable(source):
        return source(landmarks_mp_dict)
    elif isinstance(source, mp.solutions.pose.PoseLandmark):
        return landmarks_mp_dict[source]
    elif isinstance(source, list): # Average multiple landmarks
        pts = np.array([landmarks_mp_dict[lm_idx] for lm_idx in source])
        return np.mean(pts, axis=0)
    elif 'offset_dir' in joint_def_entry: # For End Sites, offset from parent's landmark
        # lm_source for EndSite is the landmark from which offset_dir is applied
        return landmarks_mp_dict[source] + joint_def_entry['offset_dir']
    raise ValueError(f"Invalid lm_source for joint: {joint_def_entry['name']}")


def convert_mediapipe_landmarks_to_dict(pose_world_landmarks_list_of_lists):
    """
    Converts MediaPipe PoseWorldLandmarks (from results.pose_world_landmarks, which is a list of lists)
    to a dictionary of NumPy arrays, applying Z-axis inversion.
    Assumes a single pose detection (results.pose_world_landmarks).
    """
    landmarks_dict = {}
    if pose_world_landmarks_list_of_lists and pose_world_landmarks_list_of_lists:
        for i, landmark_proto in enumerate(pose_world_landmarks_list_of_lists):
            # MediaPipe: Y up, X right, Z towards viewer (closer to camera = smaller Z)
            # Common 3D/BVH: Y up, X right, Z forward (away from viewer)
            # So, negate MediaPipe's Z.
            landmarks_dict[mp.solutions.pose.PoseLandmark(i)] = np.array([landmark_proto.x, landmark_proto.y, -landmark_proto.z])
    return landmarks_dict

def calculate_bvh_offsets_and_tpose_positions(tpose_landmarks_dict, skeleton_definition_list):
    """Calculates BVH offsets and stores T-pose world positions for each joint."""
    bvh_offsets = {}
    tpose_world_positions = {}

    for joint_def_entry in skeleton_definition_list:
        joint_name = joint_def_entry['name']
        parent_name = joint_def_entry['parent']

        current_joint_tpose_pos = get_joint_world_pos(tpose_landmarks_dict, joint_def_entry)
        tpose_world_positions[joint_name] = current_joint_tpose_pos

        if parent_name is None: # Root joint
            bvh_offsets[joint_name] = np.array([0.0, 0.0, 0.0]) # Root offset is often 0,0,0 as its position is absolute
        else:
            parent_tpose_pos = tpose_world_positions[parent_name]
            bvh_offsets[joint_name] = current_joint_tpose_pos - parent_tpose_pos
            
    return bvh_offsets, tpose_world_positions

def get_rotation_from_vectors(vec1, vec2, fallback_axis=None):
    """Computes the rotation matrix that aligns normalized vec1 to normalized vec2."""
    vec1_norm = np.linalg.norm(vec1)
    vec2_norm = np.linalg.norm(vec2)

    if vec1_norm < 1e-6 or vec2_norm < 1e-6: # Check for zero vectors
        return R.identity()

    v1 = vec1 / vec1_norm
    v2 = vec2 / vec2_norm
    
    if np.allclose(v1, v2):
        return R.identity()
    if np.allclose(v1, -v2): # 180 degree rotation
        if fallback_axis is None or np.linalg.norm(fallback_axis) < 1e-6:
            # Try to find an orthogonal vector to v1
            if not np.allclose(v1, ) and not np.allclose(v1, [-1,0,0]):
                 axis = np.cross(v1, )
            else: # v1 is along X, use Y
                 axis = np.cross(v1, )
        else:
            axis = fallback_axis
        
        if np.linalg.norm(axis) < 1e-6: # Still couldn't find a good axis
            return R.identity() # Should not happen if v1 is not zero
            
        axis = axis / np.linalg.norm(axis)
        return R.from_rotvec(np.pi * axis)

    # Using scipy's align_vectors: finds rotation C such that a is aligned with C @ b
    # We want to rotate v1 to v2, so v2 = C @ v1.  align_vectors(a,b) -> C where a ~ C@b
    # So, a=v2, b=v1
    rotation, _ = R.align_vectors([v2], [v1]) 
    return rotation


# --- Main Processing ---
def main(video_path, output_bvh_path, tpose_frame_num, mediapipe_model_path):
    if not os.path.exists(mediapipe_model_path):
        print(f"MediaPipe model file not found at {mediapipe_model_path}")
        print("Please download it from https://developers.google.com/mediapipe/solutions/vision/pose_landmarker/index#models")
        return

    # Initialize MediaPipe PoseLandmarker
    base_options = python.BaseOptions(model_asset_path=mediapipe_model_path)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1, 
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_segmentation_masks=False)
    landmarker = vision.PoseLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        print("Warning: Could not get FPS from video. Defaulting to 30 FPS.")
        fps = 30.0
    frame_time = 1.0 / fps
    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Processing video: {video_path} ({fps:.2f} FPS, {total_video_frames} frames)")

    all_frame_motion_data = # List to store dicts of joint_name: (tx,ty,tz,rz,ry,rx) or (rz,ry,rx)
    tpose_landmarks_dict = None
    tpose_world_positions = None
    bvh_offsets = None
    tpose_hips_pos = None
    tpose_hips_orientation_vectors = None 

    processed_frame_count = 0

    print(f"Attempting to capture T-pose from frame {tpose_frame_num}...")
    current_frame_idx = 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0) # Ensure we start from the beginning

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if current_frame_idx == tpose_frame_num:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
            if timestamp_ms == 0 and current_frame_idx > 0 : # OpenCV might return 0 for first few frames if not perfectly synced
                timestamp_ms = int(current_frame_idx * frame_time * 1000)

            results = landmarker.detect_for_video(mp_image, timestamp_ms)

            if results and results.pose_world_landmarks:
                tpose_landmarks_dict = convert_mediapipe_landmarks_to_dict(results.pose_world_landmarks)
                if not tpose_landmarks_dict: # Check if conversion was successful
                    print(f"Warning: No landmarks converted from T-pose frame {tpose_frame_num}.")
                else:
                    bvh_offsets, tpose_world_positions = calculate_bvh_offsets_and_tpose_positions(tpose_landmarks_dict, BVH_SKELETON_DEF)
                    tpose_hips_pos = tpose_world_positions['Hips']

                    hips_tpose_pos = tpose_world_positions['Hips']
                    chest_tpose_pos = tpose_world_positions.get('Chest', hips_tpose_pos + np.array([0, 0.5, 0])) # Fallback for chest
                    left_hip_tpose_pos = tpose_world_positions.get('LeftUpLeg', hips_tpose_pos - np.array([0.1, 0, 0]))
                    right_hip_tpose_pos = tpose_world_positions.get('RightUpLeg', hips_tpose_pos + np.array([0.1, 0, 0]))

                    t_forward = chest_tpose_pos - hips_tpose_pos
                    t_inter_hips = right_hip_tpose_pos - left_hip_tpose_pos
                    
                    if np.linalg.norm(t_forward) < 1e-6: t_forward = np.array() # Default forward Z
                    if np.linalg.norm(t_inter_hips) < 1e-6: t_inter_hips = np.array() # Default right X

                    t_forward_norm = t_forward / np.linalg.norm(t_forward)
                    
                    # Create orthonormal basis for T-pose Hips
                    # Assuming Y is up, Z is forward, X is right for the Hips orientation
                    temp_up = np.array()
                    t_actual_right = np.cross(temp_up, t_forward_norm)
                    if np.linalg.norm(t_actual_right) < 1e-6: # If t_forward_norm is aligned with temp_up
                        t_actual_right = np.cross(t_forward_norm, np.array()) # Use X as temp_right
                        if np.linalg.norm(t_actual_right) < 1e-6: # If t_forward_norm is also aligned with X (should not happen)
                            t_actual_right = np.array() # Fallback
                    t_actual_right_norm = t_actual_right / np.linalg.norm(t_actual_right)
                    t_actual_up_norm = np.cross(t_forward_norm, t_actual_right_norm) # Should be normalized due to cross product properties

                    tpose_hips_orientation_vectors = (t_forward_norm, t_actual_up_norm, t_actual_right_norm) # (forward, up, right)
                    print(f"T-pose captured successfully from frame {tpose_frame_num}.")
                    break 
        current_frame_idx += 1
    
    if tpose_landmarks_dict is None or bvh_offsets is None:
        print(f"Error: Could not capture T-pose from frame {tpose_frame_num} or no valid pose detected/landmarks converted.")
        cap.release()
        landmarker.close()
        return

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0) # Reset video capture for the main processing pass
    print("Processing video for motion data...")
    processed_frame_count = 0

    hips_joint_def = next(j_def for j_def in BVH_SKELETON_DEF if j_def['name'] == 'Hips')


    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        current_pos_msec = cap.get(cv2.CAP_PROP_POS_MSEC)
        timestamp_ms = int(current_pos_msec if current_pos_msec > 0 else processed_frame_count * frame_time * 1000)
        
        results = landmarker.detect_for_video(mp_image, timestamp_ms)

        current_frame_motion = {}
        if results and results.pose_world_landmarks:
            current_landmarks_dict = convert_mediapipe_landmarks_to_dict(results.pose_world_landmarks)
            if not current_landmarks_dict: # Skip if no landmarks converted
                if all_frame_motion_data: all_frame_motion_data.append(all_frame_motion_data[-1])
                processed_frame_count +=1
                continue

            current_world_positions = {}
            for joint_def_entry in BVH_SKELETON_DEF:
                if not joint_def_entry['channels']: 
                    continue
                try:
                    current_world_positions[joint_def_entry['name']] = get_joint_world_pos(current_landmarks_dict, joint_def_entry)
                except KeyError as e: # A landmark needed for a joint might be missing if not detected well
                    print(f"Warning: Landmark {e} not found for joint {joint_def_entry['name']} in frame {processed_frame_count}. Using T-pose position.")
                    current_world_positions[joint_def_entry['name']] = tpose_world_positions[joint_def_entry['name']]


            current_hips_pos = current_world_positions['Hips']
            root_translation = current_hips_pos - tpose_hips_pos
            
            curr_hips_pos = current_world_positions['Hips']
            curr_chest_pos = current_world_positions.get('Chest', curr_hips_pos + np.array([0,0.5,0]))
            
            c_forward = curr_chest_pos - curr_hips_pos
            
            root_euler_angles = [0.0, 0.0, 0.0] # Default
            root_world_rotation = R.identity()

            if np.linalg.norm(c_forward) < 1e-6:
                if all_frame_motion_data and 'Hips' in all_frame_motion_data[-1]:
                    root_euler_angles = list(all_frame_motion_data[-1]['Hips'][3:]) 
                root_world_rotation = R.from_euler(hips_joint_def['order'], root_euler_angles, degrees=True)
            else:
                c_forward_norm = c_forward / np.linalg.norm(c_forward)
                
                temp_up = np.array()
                c_actual_right = np.cross(temp_up, c_forward_norm)
                if np.linalg.norm(c_actual_right) < 1e-6:
                    c_actual_right = np.cross(c_forward_norm, np.array())
                    if np.linalg.norm(c_actual_right) < 1e-6: c_actual_right = np.array() # Fallback
                c_actual_right_norm = c_actual_right / np.linalg.norm(c_actual_right)
                c_actual_up_norm = np.cross(c_forward_norm, c_actual_right_norm)

                # Target vectors (current frame Hips orientation: forward, up, right)
                target_basis_vectors = np.array([c_forward_norm, c_actual_up_norm, c_actual_right_norm])
                # Source vectors (T-pose Hips orientation: forward, up, right)
                source_basis_vectors = np.array([tpose_hips_orientation_vectors, tpose_hips_orientation_vectors[1], tpose_hips_orientation_vectors[2]])
                
                # Find rotation from T-pose Hips orientation to current Hips orientation
                # R.align_vectors(A, B) finds C such that A is aligned with C @ B
                # We want C such that target_basis = C @ source_basis
                root_world_rotation, _ = R.align_vectors(target_basis_vectors, source_basis_vectors)
                root_euler_angles = root_world_rotation.as_euler(hips_joint_def['order'], degrees=True)

            current_frame_motion['Hips'] = (*root_translation, *root_euler_angles)
            
            parent_world_rotations = {'Hips': root_world_rotation}

            for joint_def_entry in BVH_SKELETON_DEF:
                joint_name = joint_def_entry['name']
                parent_name = joint_def_entry['parent']

                if parent_name is None or not joint_def_entry['channels']: 
                    continue

                parent_tpose_pos = tpose_world_positions[parent_name]
                child_tpose_pos = tpose_world_positions[joint_name]
                tpose_bone_vector = child_tpose_pos - parent_tpose_pos

                parent_current_pos = current_world_positions[parent_name]
                child_current_pos = current_world_positions[joint_name]
                current_bone_vector = child_current_pos - parent_current_pos
                
                local_euler_angles = [0.0, 0.0, 0.0] # Default
                local_rotation = R.identity()

                if np.linalg.norm(tpose_bone_vector) < 1e-6 or np.linalg.norm(current_bone_vector) < 1e-6:
                    if all_frame_motion_data and joint_name in all_frame_motion_data[-1]:
                        local_euler_angles = list(all_frame_motion_data[-1][joint_name])
                    local_rotation = R.from_euler(joint_def_entry['order'], local_euler_angles, degrees=True)
                else:
                    # Rotation that aligns T-pose bone vector to current bone vector in world space
                    # This is the change in the bone's world orientation from its T-pose state
                    bone_orientation_change_world = get_rotation_from_vectors(tpose_bone_vector, current_bone_vector, fallback_axis=np.array())
                    
                    parent_abs_world_rotation = parent_world_rotations[parent_name]
                    
                    # local_rotation = ParentWorldAbsolute.inv() * ChildWorldAbsolute
                    # ChildWorldAbsolute = ParentWorldAbsolute * LocalRotation
                    # We have change of bone orientation: bone_orientation_change_world
                    # This is R_current_bone_world * (R_tpose_bone_world)^-1
                    # This part of the logic can be complex. The previous script's logic:
                    # local_rotation = parent_abs_world_rotation.inv() * bone_orientation_change_world
                    # This implies bone_orientation_change_world was treated as an absolute orientation, which is not quite right.
                    # A more standard approach for local rotation:
                    # 1. Get parent's T-pose world rotation (R_p_wt) and current world rotation (R_p_wc)
                    # 2. Get child's T-pose world rotation (R_c_wt) and current world rotation (R_c_wc)
                    # 3. Local T-pose: L_t = R_p_wt.inv() * R_c_wt
                    # 4. Local Current: L_c = R_p_wc.inv() * R_c_wc
                    # This requires defining full orientations, not just bone vectors.
                    # Sticking to the simplified approach for now:
                    local_rotation = parent_abs_world_rotation.inv() * bone_orientation_change_world
                    local_euler_angles = local_rotation.as_euler(joint_def_entry['order'], degrees=True)

                current_frame_motion[joint_name] = local_euler_angles
                # This is key: child's absolute world rotation
                parent_world_rotations[joint_name] = parent_world_rotations[parent_name] * local_rotation 
            
            all_frame_motion_data.append(current_frame_motion)
        else:
            if all_frame_motion_data:
                all_frame_motion_data.append(all_frame_motion_data[-1])
            else: 
                zero_motion = {}
                for joint_def_entry in BVH_SKELETON_DEF:
                    if joint_def_entry['channels']:
                        zero_motion[joint_def_entry['name']] = [0.0] * len(joint_def_entry['channels'])
                all_frame_motion_data.append(zero_motion)
        
        processed_frame_count +=1
        if processed_frame_count % 100 == 0 or processed_frame_count == total_video_frames:
            print(f"Processed {processed_frame_count}/{total_video_frames} frames...")

    cap.release()
    landmarker.close()
    print(f"Finished processing video. Total motion frames: {len(all_frame_motion_data)}")

    if not all_frame_motion_data:
        print("No motion data was generated. Exiting.")
        return

    bvh_file = bvhio.Bvh()
    joint_to_bvh_joint_map = {}

    def add_joint_to_bvh_recursive(skeleton_def_list, current_parent_bvh_joint=None, parent_name_filter=None):
        for joint_def_entry in skeleton_def_list:
            joint_name = joint_def_entry['name']
            parent_name = joint_def_entry['parent']

            # Filter to only process direct children of current_parent_bvh_joint
            if parent_name!= parent_name_filter:
                continue
            
            offset = bvh_offsets[joint_name]
            channels = joint_def_entry['channels']
            
            keyframes_data_for_joint =
            for frame_motion_dict in all_frame_motion_data:
                if joint_name in frame_motion_dict:
                    keyframes_data_for_joint.append(tuple(frame_motion_dict[joint_name]))
                elif channels: 
                     keyframes_data_for_joint.append(tuple([0.0] * len(channels)))

            if not channels: # End Site
                if current_parent_bvh_joint:
                    current_parent_bvh_joint.add_end_site(list(offset))
                continue

            bvh_joint_node = bvhio.BvhJoint(name=joint_name, offset=list(offset), channels=channels)
            if keyframes_data_for_joint: # Ensure there's data to assign
                 bvh_joint_node.keyframes = keyframes_data_for_joint
            else: # If somehow no keyframes were generated for a channelled joint
                 bvh_joint_node.keyframes = [(0.0,) * len(channels)] * len(all_frame_motion_data)


            if current_parent_bvh_joint is None: # This is the ROOT
                bvh_file.root = bvh_joint_node
            else:
                current_parent_bvh_joint.add_child(bvh_joint_node)
            
            joint_to_bvh_joint_map[joint_name] = bvh_joint_node
            
            add_joint_to_bvh_recursive(skeleton_def_list, bvh_joint_node, joint_name)

    # Start building the hierarchy from the root
    root_joint_def = next(j for j in BVH_SKELETON_DEF if j['parent'] is None)
    add_joint_to_bvh_recursive(BVH_SKELETON_DEF, None, None) # Initial call for root
    
    bvh_file.frames = len(all_frame_motion_data)
    bvh_file.frame_time = frame_time

    print(f"Writing BVH file to: {output_bvh_path}")
    bvhio.writeBvh(output_bvh_path, bvh_file, percision=6) # Using 'percision' as per bvhio examples [3]
    print("BVH file written successfully.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Convert video to BVH using MediaPipe Pose.")
    parser.add_argument("video_path", type=str, help="Path to the input video file.")
    parser.add_argument("output_bvh_path", type=str, help="Path to save the output BVH file.")
    parser.add_argument("--tpose_frame", type=int, default=0, help="Frame number to use for T-pose calibration (0-indexed).")
    parser.add_argument("--model_path", type=str, default="pose_landmarker_full.task",
                        help="Path to the MediaPipe Pose Landmarker model file (.task).")
    
    args = parser.parse_args()
    main(args.video_path, args.output_bvh_path, args.tpose_frame, args.model_path)

