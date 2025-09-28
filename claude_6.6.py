#!/usr/bin/env python3
"""
Video to BVH Pose Converter
Converts video of human motion to BVH format using ViTPose and mathematical transformations
"""

import cv2
import numpy as np
import os
import argparse
from typing import List, Dict, Tuple
import json
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from tqdm import tqdm

# Import required libraries
try:
    from easy_ViTPose import VitInference
except ImportError:
    print("Please install easy_ViTPose: pip install easy-ViTPose")
    exit(1)

try:
    import bvhio
except ImportError:
    print("Please install bvhio: pip install bvhio")
    exit(1)


class PoseLifter:
    """Simple 3D pose lifting using geometric constraints"""
    
    def __init__(self):
        # Define bone connections based on COCO-17 format
        self.connections = [
            (0, 1), (0, 2), (1, 3), (2, 4),  # Head
            (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Arms
            (5, 11), (6, 12), (11, 12),  # Torso
            (11, 13), (13, 15), (12, 14), (14, 16)  # Legs
        ]
        
        # Average bone lengths (in meters) for normalization
        self.bone_lengths = {
            (0, 1): 0.15, (0, 2): 0.15,  # Neck to eyes
            (1, 3): 0.15, (2, 4): 0.15,  # Eyes to ears
            (5, 6): 0.25,  # Shoulders
            (5, 7): 0.25, (6, 8): 0.25,  # Upper arms
            (7, 9): 0.25, (8, 10): 0.25,  # Lower arms
            (5, 11): 0.30, (6, 12): 0.30,  # Torso sides
            (11, 12): 0.25,  # Hip width
            (11, 13): 0.40, (12, 14): 0.40,  # Upper legs
            (13, 15): 0.40, (14, 16): 0.40  # Lower legs
        }
    
    def lift_2d_to_3d(self, keypoints_2d: np.ndarray, confidence: np.ndarray) -> np.ndarray:
        """
        Lift 2D keypoints to 3D using geometric constraints
        
        Args:
            keypoints_2d: 2D keypoints of shape (17, 2)
            confidence: Confidence scores of shape (17,)
            
        Returns:
            3D keypoints of shape (17, 3)
        """
        # Initialize 3D points
        keypoints_3d = np.zeros((keypoints_2d.shape[0], 3))
        keypoints_3d[:, :2] = keypoints_2d
        
        # Simple depth estimation based on perspective and bone constraints
        # This is a simplified version - for better results use VideoPose3D or similar
        
        # Set hip center as origin
        hip_center = (keypoints_2d[11] + keypoints_2d[12]) / 2
        
        # Estimate depth based on vertical position and typical proportions
        for i in range(keypoints_2d.shape[0]):
            # Basic depth from perspective (higher points are further)
            y_offset = keypoints_2d[i, 1] - hip_center[1]
            keypoints_3d[i, 2] = y_offset * 0.001  # Scale factor
            
        # Normalize by hip center
        keypoints_3d[:, :2] -= hip_center
        
        # Apply bone length constraints
        for (i, j), target_length in self.bone_lengths.items():
            if confidence[i] > 0.3 and confidence[j] > 0.3:
                current_vec = keypoints_3d[j] - keypoints_3d[i]
                current_length = np.linalg.norm(current_vec)
                if current_length > 0:
                    # Adjust depth to match target bone length
                    scale = target_length / current_length
                    keypoints_3d[j] = keypoints_3d[i] + current_vec * scale
        
        return keypoints_3d


class Pose2BVHConverter:
    """Convert 3D pose sequences to BVH format"""
    
    def __init__(self):
        # Define skeleton hierarchy for BVH
        self.joint_names = [
            'Hips',
            'RightHip', 'RightKnee', 'RightAnkle',
            'LeftHip', 'LeftKnee', 'LeftAnkle',
            'Spine', 'Spine1', 'Neck', 'Head',
            'LeftShoulder', 'LeftElbow', 'LeftWrist',
            'RightShoulder', 'RightElbow', 'RightWrist'
        ]
        
        # Mapping from COCO keypoints to BVH joints
        self.coco_to_bvh = {
            'Hips': (11, 12),  # Average of hips
            'LeftHip': 11,
            'LeftKnee': 13,
            'LeftAnkle': 15,
            'RightHip': 12,
            'RightKnee': 14,
            'RightAnkle': 16,
            'Spine': (5, 6, 11, 12),  # Average of shoulders and hips
            'Spine1': (5, 6),  # Average of shoulders
            'Neck': (5, 6, 0),  # Shoulders to nose
            'Head': 0,  # Nose
            'LeftShoulder': 5,
            'LeftElbow': 7,
            'LeftWrist': 9,
            'RightShoulder': 6,
            'RightElbow': 8,
            'RightWrist': 10
        }
        
        # Define parent-child relationships
        self.hierarchy = {
            'Hips': None,
            'LeftHip': 'Hips',
            'LeftKnee': 'LeftHip',
            'LeftAnkle': 'LeftKnee',
            'RightHip': 'Hips',
            'RightKnee': 'RightHip',
            'RightAnkle': 'RightKnee',
            'Spine': 'Hips',
            'Spine1': 'Spine',
            'Neck': 'Spine1',
            'Head': 'Neck',
            'LeftShoulder': 'Spine1',
            'LeftElbow': 'LeftShoulder',
            'LeftWrist': 'LeftElbow',
            'RightShoulder': 'Spine1',
            'RightElbow': 'RightShoulder',
            'RightWrist': 'RightElbow'
        }
        
        # Initialize reference pose (T-pose)
        self.reference_vectors = {}
        self._initialize_reference_pose()
    
    def _initialize_reference_pose(self):
        """Initialize reference vectors for T-pose"""
        # Define reference directions for each bone in T-pose
        self.reference_vectors = {
            'LeftHip': np.array([0, -1, 0]),
            'LeftKnee': np.array([0, -1, 0]),
            'LeftAnkle': np.array([0, -1, 0]),
            'RightHip': np.array([0, -1, 0]),
            'RightKnee': np.array([0, -1, 0]),
            'RightAnkle': np.array([0, -1, 0]),
            'Spine': np.array([0, 1, 0]),
            'Spine1': np.array([0, 1, 0]),
            'Neck': np.array([0, 1, 0]),
            'Head': np.array([0, 1, 0]),
            'LeftShoulder': np.array([-1, 0, 0]),
            'LeftElbow': np.array([-1, 0, 0]),
            'LeftWrist': np.array([-1, 0, 0]),
            'RightShoulder': np.array([1, 0, 0]),
            'RightElbow': np.array([1, 0, 0]),
            'RightWrist': np.array([1, 0, 0])
        }
    
    def _get_joint_position(self, joint_name: str, keypoints_3d: np.ndarray) -> np.ndarray:
        """Get 3D position for a BVH joint from COCO keypoints"""
        mapping = self.coco_to_bvh[joint_name]
        
        if isinstance(mapping, tuple):
            # Average multiple keypoints
            positions = [keypoints_3d[idx] for idx in mapping]
            return np.mean(positions, axis=0)
        else:
            return keypoints_3d[mapping]
    
    def _calculate_rotation(self, parent_pos: np.ndarray, joint_pos: np.ndarray, 
                          child_pos: np.ndarray, ref_vector: np.ndarray) -> Tuple[float, float, float]:
        """
        Calculate ZXY Euler angles for joint rotation
        Following the mathematical formulas from the PDF
        """
        # Current bone vector
        v_current = child_pos - joint_pos
        v_current_norm = v_current / (np.linalg.norm(v_current) + 1e-8)
        
        # Calculate rotation from reference to current
        axis = np.cross(ref_vector, v_current_norm)
        axis_norm = np.linalg.norm(axis)
        
        if axis_norm < 1e-8:
            # Vectors are parallel
            if np.dot(ref_vector, v_current_norm) > 0:
                return 0, 0, 0  # No rotation needed
            else:
                return 180, 0, 0  # 180 degree rotation
        
        axis = axis / axis_norm
        angle = np.arccos(np.clip(np.dot(ref_vector, v_current_norm), -1, 1))
        
        # Convert axis-angle to rotation matrix
        rot = R.from_rotvec(axis * angle)
        rot_matrix = rot.as_matrix()
        
        # Extract ZXY Euler angles (as per PDF formula)
        z_rot = np.arctan2(-rot_matrix[0, 1], rot_matrix[1, 1])
        x_rot = np.arcsin(np.clip(rot_matrix[2, 1], -1, 1))
        y_rot = np.arctan2(-rot_matrix[2, 0], rot_matrix[2, 2])
        
        # Convert to degrees
        return np.degrees(z_rot), np.degrees(x_rot), np.degrees(y_rot)
    
    def convert_sequence_to_bvh(self, pose_sequence: List[np.ndarray], fps: float = 30.0) -> bvhio.BvhJoint:
        """
        Convert a sequence of 3D poses to BVH format
        
        Args:
            pose_sequence: List of 3D keypoints arrays, each of shape (17, 3)
            fps: Frames per second
            
        Returns:
            BVH root joint
        """
        # Create skeleton hierarchy
        joints = {}
        
        # Create root joint
        root = bvhio.BvhJoint('Hips', parent=None)
        root.offset = [0, 0, 0]
        joints['Hips'] = root
        
        # Create other joints
        for joint_name in self.joint_names[1:]:
            parent_name = self.hierarchy[joint_name]
            parent_joint = joints[parent_name]
            
            joint = bvhio.BvhJoint(joint_name, parent=parent_joint)
            
            # Calculate offset from first frame
            first_frame = pose_sequence[0]
            parent_pos = self._get_joint_position(parent_name, first_frame)
            joint_pos = self._get_joint_position(joint_name, first_frame)
            offset = joint_pos - parent_pos
            
            joint.offset = offset.tolist()
            parent_joint.add_child(joint)
            joints[joint_name] = joint
        
        # Process motion data
        motion_data = []
        
        for frame_idx, keypoints_3d in enumerate(pose_sequence):
            frame_data = []
            
            # Root position (Hips)
            hip_pos = self._get_joint_position('Hips', keypoints_3d)
            frame_data.extend(hip_pos.tolist())  # X, Y, Z position
            
            # Process rotations for each joint
            for joint_name in self.joint_names:
                if joint_name == 'Hips':
                    # Add root rotation (initially zero)
                    frame_data.extend([0, 0, 0])  # Z, X, Y rotation
                else:
                    parent_name = self.hierarchy[joint_name]
                    
                    # Get positions
                    parent_pos = self._get_joint_position(parent_name, keypoints_3d)
                    joint_pos = self._get_joint_position(joint_name, keypoints_3d)
                    
                    # Find child for bone vector (use joint position if no child)
                    child_joints = [j for j, p in self.hierarchy.items() if p == joint_name]
                    if child_joints:
                        child_pos = self._get_joint_position(child_joints[0], keypoints_3d)
                    else:
                        # For end joints, extend in the same direction
                        child_pos = joint_pos + (joint_pos - parent_pos)
                    
                    # Calculate rotation
                    ref_vector = self.reference_vectors.get(joint_name, np.array([0, 1, 0]))
                    z_rot, x_rot, y_rot = self._calculate_rotation(
                        parent_pos, joint_pos, child_pos, ref_vector
                    )
                    
                    frame_data.extend([z_rot, x_rot, y_rot])
            
            motion_data.append(frame_data)
        
        # Set animation data
        root.frame_time = 1.0 / fps
        root.frames = motion_data
        
        return root


def process_video(video_path: str, output_path: str, visualize: bool = False):
    """
    Process video to extract poses and convert to BVH
    
    Args:
        video_path: Path to input video
        output_path: Path to output BVH file
        visualize: Whether to show visualization
    """
    # Initialize pose detector
    print("Initializing ViTPose...")
    model = VitInference()
    
    # Initialize pose lifter and converter
    lifter = PoseLifter()
    converter = Pose2BVHConverter()
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Processing video: {video_path}")
    print(f"FPS: {fps}, Total frames: {total_frames}")
    
    # Process frames
    pose_sequence_3d = []
    frame_count = 0
    first_frame_debug = True
    
    with tqdm(total=total_frames, desc="Extracting poses") as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Detect 2D pose
            results = model.inference(frame)
            
            # Skip if no results
            if results is None:
                frame_count += 1
                pbar.update(1)
                continue
            
            # Debug output for first frame
            if first_frame_debug:
                print(f"\nDebug - ViTPose output type: {type(results)}")
                if isinstance(results, dict):
                    print(f"Debug - Dictionary keys: {results.keys()}")
                    for key, value in results.items():
                        if isinstance(value, np.ndarray):
                            print(f"Debug - {key} shape: {value.shape}")
                elif isinstance(results, (list, tuple)):
                    print(f"Debug - List/tuple length: {len(results)}")
                    if len(results) > 0:
                        print(f"Debug - First element type: {type(results[0])}")
                        if isinstance(results[0], np.ndarray):
                            print(f"Debug - First element shape: {results[0].shape}")
                elif isinstance(results, np.ndarray):
                    print(f"Debug - Array shape: {results.shape}")
                first_frame_debug = False
            
            # Handle different possible output formats from easy_ViTPose
            keypoints = None
            
            if isinstance(results, dict):
                # Check for common keys
                if 'keypoints' in results:
                    keypoints = results['keypoints']
                elif 'poses' in results:
                    keypoints = results['poses']
                else:
                    # Try to find the keypoints in the dict
                    for key, value in results.items():
                        if isinstance(value, np.ndarray) and len(value.shape) >= 2:
                            keypoints = value
                            break
            elif isinstance(results, (list, tuple)):
                # Results might be a list of detections
                if len(results) > 0:
                    keypoints = results
                else:
                    frame_count += 1
                    pbar.update(1)
                    continue
            else:
                keypoints = results
            
            if keypoints is None:
                frame_count += 1
                pbar.update(1)
                continue
            
            # Convert to numpy array if needed
            if isinstance(keypoints, (list, tuple)):
                keypoints = np.array(keypoints)
            
            # Handle empty keypoints
            if isinstance(keypoints, np.ndarray) and keypoints.size == 0:
                frame_count += 1
                pbar.update(1)
                continue
            
            # Handle different shapes
            person_keypoints = None
            if len(keypoints.shape) == 2:
                # Single person detected - shape (17, 3)
                if keypoints.shape[0] >= 17:
                    person_keypoints = keypoints[:17]
            elif len(keypoints.shape) == 3 and keypoints.shape[0] > 0:
                # Multiple people detected - shape (N, 17, 3)
                if keypoints.shape[1] >= 17:
                    person_keypoints = keypoints[0, :17]
            
            if person_keypoints is None or person_keypoints.shape[0] < 17:
                frame_count += 1
                pbar.update(1)
                continue
                
            # Extract 2D points and confidence
            points_2d = person_keypoints[:, :2]
            confidence = person_keypoints[:, 2] if person_keypoints.shape[1] >= 3 else np.ones(person_keypoints.shape[0])
                
            # Lift to 3D
            points_3d = lifter.lift_2d_to_3d(points_2d, confidence)
            pose_sequence_3d.append(points_3d)
            
            # Visualize if requested
            if visualize and frame_count % 10 == 0:
                vis_frame = frame.copy()
                # Draw keypoints manually since we're not sure about the draw method format
                
                # Define COCO skeleton connections
                connections = [
                    (0, 1), (0, 2), (1, 3), (2, 4),  # Face
                    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Arms
                    (5, 11), (6, 12), (11, 12),  # Torso
                    (11, 13), (13, 15), (12, 14), (14, 16)  # Legs
                ]
                
                # Draw skeleton
                for i, j in connections:
                    if i < len(points_2d) and j < len(points_2d):
                        if confidence[i] > 0.3 and confidence[j] > 0.3:
                            pt1 = (int(points_2d[i, 0]), int(points_2d[i, 1]))
                            pt2 = (int(points_2d[j, 0]), int(points_2d[j, 1]))
                            cv2.line(vis_frame, pt1, pt2, (0, 255, 255), 2)
                
                # Draw keypoints
                for i, (x, y) in enumerate(points_2d):
                    if confidence[i] > 0.3:
                        cv2.circle(vis_frame, (int(x), int(y)), 4, (0, 255, 0), -1)
                        cv2.circle(vis_frame, (int(x), int(y)), 6, (0, 0, 255), 2)
                
                cv2.imshow('Pose Detection', vis_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            frame_count += 1
            pbar.update(1)
    
    cap.release()
    cv2.destroyAllWindows()
    
    print(f"Extracted {len(pose_sequence_3d)} poses")
    
    if len(pose_sequence_3d) == 0:
        print("No poses detected in video!")
        return
    
    # Smooth the sequence
    print("Smoothing pose sequence...")
    pose_sequence_3d = smooth_pose_sequence(pose_sequence_3d)
    
    # Convert to BVH
    print("Converting to BVH format...")
    bvh_root = converter.convert_sequence_to_bvh(pose_sequence_3d, fps)
    
    # Save BVH file
    print(f"Saving BVH to: {output_path}")
    with open(output_path, 'w') as f:
        bvhio.write_bvh(f, bvh_root)
    
    print("Done!")


def smooth_pose_sequence(pose_sequence: List[np.ndarray], window_size: int = 5) -> List[np.ndarray]:
    """
    Apply temporal smoothing to pose sequence
    
    Args:
        pose_sequence: List of 3D keypoints arrays
        window_size: Size of smoothing window
        
    Returns:
        Smoothed pose sequence
    """
    if len(pose_sequence) < window_size:
        return pose_sequence
    
    smoothed = []
    half_window = window_size // 2
    
    for i in range(len(pose_sequence)):
        start = max(0, i - half_window)
        end = min(len(pose_sequence), i + half_window + 1)
        
        # Average poses in window
        window_poses = pose_sequence[start:end]
        avg_pose = np.mean(window_poses, axis=0)
        smoothed.append(avg_pose)
    
    return smoothed


def main():
    parser = argparse.ArgumentParser(description='Convert video to BVH using pose estimation')
    parser.add_argument('input', help='Input video file path')
    parser.add_argument('-o', '--output', help='Output BVH file path', default=None)
    parser.add_argument('-v', '--visualize', action='store_true', help='Show visualization')
    
    args = parser.parse_args()
    
    # Set output path if not provided
    if args.output is None:
        base_name = os.path.splitext(os.path.basename(args.input))[0]
        args.output = f"{base_name}.bvh"
    
    # Process video
    process_video(args.input, args.output, args.visualize)


if __name__ == "__main__":
    main()