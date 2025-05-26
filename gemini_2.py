Pythonimport cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
from scipy.spatial.transform import Rotation as R
import bvhio # For writing BVH files [3, 4, 5, 6]
import argparse
import os

# --- BVH Skeleton Definition ---
# This defines the hierarchy, the MediaPipe landmarks used for each joint's position,
# and the channels for the BVH file.
# 'lm_source' can be a single landmark, a list of landmarks to average, or a lambda for custom logic.
# The order of joints matters for BVH hierarchy construction and motion data ordering.
# Rotation order for Euler angles is ZYX.
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
     'lm_source': lambda lm_dict: ((lm_dict + lm_dict) / 2 + lm_dict) / 2, # Approx between shoulders and nose
     'channels': ['Zrotation', 'Yrotation', 'Xrotation'], 'order': 'ZYX'},
    {'name': 'Head', 'parent': 'Neck',
     'lm_source': mp.solutions.pose.PoseLandmark.NOSE,
     'channels': ['Zrotation', 'Yrotation', 'Xrotation'], 'order': 'ZYX'},
    {'name': 'HeadEnd', 'parent': 'Head',
     'lm_source': mp.solutions.pose.PoseLandmark.NOSE, 'offset_dir': np.array([0, 0.1, 0]), # Offset from NOSE
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
     'lm_source': mp.solutions.pose.PoseLandmark.LEFT_INDEX, 'offset_dir': np.array([0,0,-0.05]),
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
        # Ensure the landmark key exists, otherwise return a default or raise error
        if source not in landmarks_mp_dict:
            # Fallback or error handling for missing critical landmarks
            # print(f"Warning: Landmark {source.name} not found in current frame for joint {joint_def_entry['name']}. Using zero vector.")
            return np.array([0.0, 0.0, 0.0]) # Or handle more gracefully
        return landmarks_mp_dict[source]
    elif isinstance(source, list): # Average multiple landmarks
        pts =
        for lm_idx in source:
            if lm_idx not in landmarks_mp_dict:
                # print(f"Warning: Landmark {lm_idx.name} not found for averaging in joint {joint_def_entry['name']}. Skipping.")
                continue
            pts.append(landmarks_mp_dict[lm_idx])
        if not pts: return np.array([0.0, 0.0, 0.0]) # All landmarks for averaging were missing
        return np.mean(np.array(pts), axis=0)
    elif 'offset_dir' in joint_def_entry: # For End Sites, offset from its source landmark
        if source not in landmarks_mp_dict:
            # print(f"Warning: Landmark {source.name} for EndSite {joint_def_entry['name']} not found. Using zero vector for base.")
            base_pos = np.array([0.0, 0.0, 0.0])
        else:
            base_pos = landmarks_mp_dict[source]
        return base_pos + joint_def_entry['offset_dir']
    raise ValueError(f"Invalid lm_source for joint: {joint_def_entry['name']}")


def convert_mediapipe_landmarks_to_dict(pose_world_landmarks_from_result):
    """
    Converts MediaPipe PoseWorldLandmarks (from results.pose_world_landmarks)
    to a dictionary of {PoseLandmark_enum: NumPy_array}, applying Z-axis inversion.
    Assumes a single pose detection (results.pose_world_landmarks).
    """
    landmarks_dict = {}
    if pose_world_landmarks_from_result and pose_world_landmarks_from_result:
        for i, landmark_proto in enumerate(pose_world_landmarks_from_result): # Access landmarks for the first (and assumed only) pose
            # MediaPipe: Y up, X right, Z towards viewer (closer to camera = smaller Z) [1, 7, 8]
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
            bvh_offsets[joint_name] = np.array([0.0, 0.0, 0.0])
        else:
            parent_tpose_pos = tpose_world_positions[parent_name]
            bvh_offsets[joint_name] = current_joint_tpose_pos - parent_tpose_pos

    return bvh_offsets, tpose_world_positions

def get_rotation_from_vectors(vec1, vec2, fallback_axis=None):
    """Computes the rotation matrix that aligns normalized vec1 to normalized vec2."""
    vec1_norm_val = np.linalg.norm(vec1)
    vec2_norm_val = np.linalg.norm(vec2)

    if vec1_norm_val < 1e-6 or vec2_norm_val < 1e-6:
        return R.identity()

    v1_normalized = vec1 / vec1_norm_val
    v2_normalized = vec2 / vec2_norm_val

    if np.allclose(v1_normalized, v2_normalized):
        return R.identity()
    if np.allclose(v1_normalized, -v2_normalized):
        axis_to_use = fallback_axis
        if axis_to_use is None or np.linalg.norm(axis_to_use) < 1e-6:
            if not np.allclose(v1_normalized, ) and not np.allclose(v1_normalized, [0, -1, 0]):
                 axis_to_use = np.cross(v1_normalized, )
            else:
                 axis_to_use = np.cross(v1_normalized, )
        if np.linalg.norm(axis_to_use) < 1e-6: return R.identity() # Should be rare
        axis_to_use = axis_to_use / np.linalg.norm(axis_to_use)
        return R.from_rotvec(np.pi * axis_to_use)
    # Scipy's align_vectors: finds rotation C such that a is aligned with C @ b [9]
    # We want to rotate v1 to v2, so v2 = C @ v1. align_vectors(a,b) -> C where a ~ C@b
    # So, a=v2_normalized, b=v1_normalized
    rotation, _ = R.align_vectors([v2_normalized], [v1_normalized])
    return rotation


# --- Main Processing ---
def main(video_path, output_bvh_path, tpose_frame_num, mediapipe_model_path):
    if not os.path.exists(mediapipe_model_path):
        print(f"MediaPipe model file not found at {mediapipe_model_path}")
        print("Please download it from https://developers.google.com/mediapipe/solutions/vision/pose_landmarker/index#models")
        return

    base_options = python.BaseOptions(model_asset_path=mediapipe_model_path)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_segmentation_masks=False) # [1, 2]
    landmarker = vision.PoseLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0: fps = 30.0 # Default FPS
    frame_time = 1.0 / fps
    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Processing video: {video_path} ({fps:.2f} FPS, {total_video_frames} frames)")

    all_frame_motion_data = # CORRECTED: Initialize list
    tpose_landmarks_dict = None
    tpose_world_positions = None
    bvh_offsets = None
    tpose_hips_pos = None
    tpose_hips_orientation_vectors = None

    print(f"Attempting to capture T-pose from frame {tpose_frame_num}...")
    current_frame_idx = 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        if current_frame_idx == tpose_frame_num:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC) if cap.get(cv2.CAP_PROP_POS_MSEC) > 0 else current_frame_idx * frame_time * 1000)
            results = landmarker.detect_for_video(mp_image, timestamp_ms) # [1]

            if results and results.pose_world_landmarks:
                tpose_landmarks_dict = convert_mediapipe_landmarks_to_dict(results.pose_world_landmarks)
                if not tpose_landmarks_dict:
                    print(f"Warning: No landmarks converted from T-pose frame {tpose_frame_num}.")
                else:
                    bvh_offsets, tpose_world_positions = calculate_bvh_offsets_and_tpose_positions(tpose_landmarks_dict, BVH_SKELETON_DEF)
                    tpose_hips_pos = tpose_world_positions['Hips']

                    hips_tpose_pos_calc = tpose_world_positions['Hips']
                    chest_tpose_pos_calc = tpose_world_positions.get('Chest', hips_tpose_pos_calc + np.array([0, 0.5, 0]))
                    
                    t_forward = chest_tpose_pos_calc - hips_tpose_pos_calc
                    if np.linalg.norm(t_forward) < 1e-6: t_forward = np.array() # Default forward Z
                    t_forward_norm = t_forward / np.linalg.norm(t_forward)
                    
                    temp_up_vector = np.array() # Assuming Y is generally up
                    t_right = np.cross(temp_up_vector, t_forward_norm)
                    if np.linalg.norm(t_right) < 1e-6: t_right = np.cross(t_forward_norm, np.array()) # If forward is Y-up, use X for right
                    if np.linalg.norm(t_right) < 1e-6: t_right = np.array() # Absolute fallback
                        
                    t_right_norm = t_right / np.linalg.norm(t_right)
                    t_up_norm = np.cross(t_forward_norm, t_right_norm) # Recalculate Up to be orthogonal

                    tpose_hips_orientation_vectors = (t_forward_norm, t_up_norm, t_right_norm) # (forward, up, right)
                    print(f"T-pose captured successfully from frame {tpose_frame_num}.")
                    break
        current_frame_idx += 1

    if tpose_landmarks_dict is None or bvh_offsets is None:
        print(f"Error: Could not capture T-pose from frame {tpose_frame_num}. Exiting.")
        cap.release()
        landmarker.close()
        return

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    print("Processing video for motion data...")
    processed_frame_count = 0
    hips_joint_def_entry = next(j_def for j_def in BVH_SKELETON_DEF if j_def['name'] == 'Hips')

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        current_pos_msec = cap.get(cv2.CAP_PROP_POS_MSEC)
        timestamp_ms = int(current_pos_msec if current_pos_msec > 0 else processed_frame_count * frame_time * 1000)
        results = landmarker.detect_for_video(mp_image, timestamp_ms)

        current_frame_motion = {}
        if results and results.pose_world_landmarks:
            current_landmarks_dict = convert_mediapipe_landmarks_to_dict(results.pose_world_landmarks)
            if not current_landmarks_dict:
                if all_frame_motion_data: all_frame_motion_data.append(all_frame_motion_data[-1])
                processed_frame_count +=1
                continue

            current_world_positions = {}
            valid_pose = True
            for joint_def_entry in BVH_SKELETON_DEF:
                if not joint_def_entry['channels']: continue
                try:
                    current_world_positions[joint_def_entry['name']] = get_joint_world_pos(current_landmarks_dict, joint_def_entry)
                except KeyError: # Should be handled by get_joint_world_pos now
                    print(f"Critical landmark missing for {joint_def_entry['name']} in frame {processed_frame_count}. Using T-pose.")
                    current_world_positions[joint_def_entry['name']] = tpose_world_positions[joint_def_entry['name']]


            current_hips_pos = current_world_positions['Hips']
            root_translation = current_hips_pos - tpose_hips_pos

            curr_hips_pos_calc = current_world_positions['Hips']
            curr_chest_pos_calc = current_world_positions.get('Chest', curr_hips_pos_calc + np.array([0,0.5,0]))
            
            c_forward = curr_chest_pos_calc - curr_hips_pos_calc
            root_euler_angles = [0.0, 0.0, 0.0]
            root_world_rotation = R.identity()

            if np.linalg.norm(c_forward) < 1e-6:
                if all_frame_motion_data and 'Hips' in all_frame_motion_data[-1]:
                    root_euler_angles = list(all_frame_motion_data[-1]['Hips'][3:])
                root_world_rotation = R.from_euler(hips_joint_def_entry['order'], root_euler_angles, degrees=True) # [10, 11, 12, 13]
            else:
                c_forward_norm = c_forward / np.linalg.norm(c_forward)
                temp_up_vector = np.array()
                c_right = np.cross(temp_up_vector, c_forward_norm)
                if np.linalg.norm(c_right) < 1e-6: c_right = np.cross(c_forward_norm, np.array())
                if np.linalg.norm(c_right) < 1e-6: c_right = np.array()

                c_right_norm = c_right / np.linalg.norm(c_right)
                c_up_norm = np.cross(c_forward_norm, c_right_norm)

                target_basis_vectors = np.array([c_forward_norm, c_up_norm, c_right_norm]).T # Columns: Fwd, Up, Right
                source_basis_vectors = np.array([tpose_hips_orientation_vectors, tpose_hips_orientation_vectors[1], tpose_hips_orientation_vectors[2]]).T # Columns: Fwd, Up, Right
                
                # Create rotation matrices from these basis vectors
                # R_target @ R_source.T gives rotation from source to target
                # Scipy align_vectors is more direct: C such that target_basis ~ C @ source_basis
                # We want C such that current_orientation_matrix = C @ tpose_orientation_matrix
                # So, C = current_orientation_matrix @ tpose_orientation_matrix.inv()
                # Or, using align_vectors to find rotation from T-pose basis to current basis:
                # align_vectors(target_vectors, source_vectors) -> C such that target ~ C @ source
                rot_matrix_target = target_basis_vectors
                rot_matrix_source = source_basis_vectors
                
                # We want rotation FROM source TO target
                root_world_rotation = R.from_matrix(rot_matrix_target @ rot_matrix_source.T)
                root_euler_angles = root_world_rotation.as_euler(hips_joint_def_entry['order'], degrees=True)


            current_frame_motion['Hips'] = (*root_translation, *root_euler_angles)
            parent_world_rotations = {'Hips': root_world_rotation}

            for joint_def_entry in BVH_SKELETON_DEF:
                joint_name = joint_def_entry['name']
                parent_name = joint_def_entry['parent']
                if parent_name is None or not joint_def_entry['channels']: continue

                parent_tpose_pos = tpose_world_positions[parent_name]
                child_tpose_pos = tpose_world_positions[joint_name]
                tpose_bone_vector = child_tpose_pos - parent_tpose_pos

                parent_current_pos = current_world_positions[parent_name]
                child_current_pos = current_world_positions[joint_name]
                current_bone_vector = child_current_pos - parent_current_pos

                local_euler_angles = [0.0, 0.0, 0.0]
                local_rotation = R.identity()

                if np.linalg.norm(tpose_bone_vector) < 1e-6 or np.linalg.norm(current_bone_vector) < 1e-6:
                    if all_frame_motion_data and joint_name in all_frame_motion_data[-1]:
                        local_euler_angles = list(all_frame_motion_data[-1][joint_name])
                    local_rotation = R.from_euler(joint_def_entry['order'], local_euler_angles, degrees=True)
                else:
                    bone_orientation_change_world = get_rotation_from_vectors(tpose_bone_vector, current_bone_vector, fallback_axis=np.array())
                    parent_abs_world_rotation = parent_world_rotations[parent_name]
                    local_rotation = parent_abs_world_rotation.inv() * bone_orientation_change_world # [14, 15, 16]
                    local_euler_angles = local_rotation.as_euler(joint_def_entry['order'], degrees=True)

                current_frame_motion[joint_name] = local_euler_angles
                parent_world_rotations[joint_name] = parent_world_rotations[parent_name] * local_rotation

            all_frame_motion_data.append(current_frame_motion)
        else: # No pose detected
            if all_frame_motion_data: all_frame_motion_data.append(all_frame_motion_data[-1]) # Duplicate last frame
            else: # First frame and no pose, add a zero motion frame
                zero_motion = {}
                for j_def in BVH_SKELETON_DEF:
                    if j_def['channels']: zero_motion[j_def['name']] = [0.0] * len(j_def['channels'])
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
            if parent_name!= parent_name_filter: continue

            offset = bvh_offsets[joint_name]
            channels = joint_def_entry['channels']
            keyframes_data_for_joint = # CORRECTED: Initialize list

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
            if keyframes_data_for_joint:
                 bvh_joint_node.keyframes = keyframes_data_for_joint
            else:
                 bvh_joint_node.keyframes = [(0.0,) * len(channels)] * len(all_frame_motion_data)

            if current_parent_bvh_joint is None:
                bvh_file.root = bvh_joint_node
            else:
                current_parent_bvh_joint.add_child(bvh_joint_node)

            joint_to_bvh_joint_map[joint_name] = bvh_joint_node
            add_joint_to_bvh_recursive(skeleton_def_list, bvh_joint_node, joint_name)

    add_joint_to_bvh_recursive(BVH_SKELETON_DEF, None, None)

    bvh_file.frames = len(all_frame_motion_data)
    bvh_file.frame_time = frame_time

    print(f"Writing BVH file to: {output_bvh_path}")
    bvhio.writeBvh(output_bvh_path, bvh_file, percision=6) # [3, 4]
    print("BVH file written successfully.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Convert video to BVH using MediaPipe Pose.")
    parser.add_argument("video_path", type=str, help="Path to the input video file.")
    parser.add_argument("output_bvh_path", type=str, help="Path to save the output BVH file.")
    parser.add_argument("--tpose_frame", type=int, default=0, help="Frame number for T-pose (0-indexed).")
    parser.add_argument("--model_path", type=str, default="pose_landmarker_full.task",
                        help="Path to MediaPipe Pose Landmarker model (.task).")

    args = parser.parse_args()
    main(args.video_path, args.output_bvh_path, args.tpose_frame, args.model_path)
