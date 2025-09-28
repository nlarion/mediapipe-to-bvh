#!/usr/bin/env python3
"""
Video to BVH Converter using MediaPipe and bvhio
Converts MP4 video of a person to BVH motion capture format
"""

import cv2
import numpy as np
import mediapipe as mp
from bvhio import BvhJoint, BvhContainer
import argparse
import os
from typing import List, Dict, Tuple
import math

# MediaPipe pose landmark indices
class PoseLandmark:
    NOSE = 0
    LEFT_EYE_INNER = 1
    LEFT_EYE = 2
    LEFT_EYE_OUTER = 3
    RIGHT_EYE_INNER = 4
    RIGHT_EYE = 5
    RIGHT_EYE_OUTER = 6
    LEFT_EAR = 7
    RIGHT_EAR = 8
    MOUTH_LEFT = 9
    MOUTH_RIGHT = 10
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_ELBOW = 13
    RIGHT_ELBOW = 14
    LEFT_WRIST = 15
    RIGHT_WRIST = 16
    LEFT_PINKY = 17
    RIGHT_PINKY = 18
    LEFT_INDEX = 19
    RIGHT_INDEX = 20
    LEFT_THUMB = 21
    RIGHT_THUMB = 22
    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_KNEE = 25
    RIGHT_KNEE = 26
    LEFT_ANKLE = 27
    RIGHT_ANKLE = 28
    LEFT_HEEL = 29
    RIGHT_HEEL = 30
    LEFT_FOOT_INDEX = 31
    RIGHT_FOOT_INDEX = 32

class MediaPipeToBVH:
    def __init__(self):
        # Initialize MediaPipe
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,  # Use highest accuracy model
            enable_segmentation=False,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Define skeleton mapping from MediaPipe to BVH joints
        self.skeleton_mapping = {
            'Hips': (PoseLandmark.LEFT_HIP, PoseLandmark.RIGHT_HIP),  # Average of hips
            'Spine': None,  # Will be estimated
            'Spine1': None,  # Will be estimated
            'Spine2': None,  # Will be estimated
            'Neck': (PoseLandmark.LEFT_SHOULDER, PoseLandmark.RIGHT_SHOULDER),  # Average of shoulders
            'Head': PoseLandmark.NOSE,
            'LeftShoulder': PoseLandmark.LEFT_SHOULDER,
            'LeftArm': PoseLandmark.LEFT_ELBOW,
            'LeftForeArm': PoseLandmark.LEFT_WRIST,
            'LeftHand': PoseLandmark.LEFT_INDEX,
            'RightShoulder': PoseLandmark.RIGHT_SHOULDER,
            'RightArm': PoseLandmark.RIGHT_ELBOW,
            'RightForeArm': PoseLandmark.RIGHT_WRIST,
            'RightHand': PoseLandmark.RIGHT_INDEX,
            'LeftUpLeg': PoseLandmark.LEFT_HIP,
            'LeftLeg': PoseLandmark.LEFT_KNEE,
            'LeftFoot': PoseLandmark.LEFT_ANKLE,
            'LeftToeBase': PoseLandmark.LEFT_FOOT_INDEX,
            'RightUpLeg': PoseLandmark.RIGHT_HIP,
            'RightLeg': PoseLandmark.RIGHT_KNEE,
            'RightFoot': PoseLandmark.RIGHT_ANKLE,
            'RightToeBase': PoseLandmark.RIGHT_FOOT_INDEX
        }
        
    def create_bvh_skeleton(self) -> BvhJoint:
        """Create BVH skeleton hierarchy"""
        # Root joint
        root = BvhJoint('Hips', parent=None)
        root.offset = np.array([0.0, 95.0, 0.0])  # Approximate hip height in cm
        root.channels = ['Xposition', 'Yposition', 'Zposition', 'Zrotation', 'Xrotation', 'Yrotation']
        
        # Spine chain
        spine = BvhJoint('Spine', parent=root)
        spine.offset = np.array([0.0, 10.0, 0.0])
        spine.channels = ['Zrotation', 'Xrotation', 'Yrotation']
        
        spine1 = BvhJoint('Spine1', parent=spine)
        spine1.offset = np.array([0.0, 10.0, 0.0])
        spine1.channels = ['Zrotation', 'Xrotation', 'Yrotation']
        
        spine2 = BvhJoint('Spine2', parent=spine1)
        spine2.offset = np.array([0.0, 10.0, 0.0])
        spine2.channels = ['Zrotation', 'Xrotation', 'Yrotation']
        
        neck = BvhJoint('Neck', parent=spine2)
        neck.offset = np.array([0.0, 15.0, 0.0])
        neck.channels = ['Zrotation', 'Xrotation', 'Yrotation']
        
        head = BvhJoint('Head', parent=neck)
        head.offset = np.array([0.0, 10.0, 0.0])
        head.channels = ['Zrotation', 'Xrotation', 'Yrotation']
        
        # Left arm chain
        left_shoulder = BvhJoint('LeftShoulder', parent=spine2)
        left_shoulder.offset = np.array([7.0, 5.0, 0.0])
        left_shoulder.channels = ['Zrotation', 'Xrotation', 'Yrotation']
        
        left_arm = BvhJoint('LeftArm', parent=left_shoulder)
        left_arm.offset = np.array([10.0, 0.0, 0.0])
        left_arm.channels = ['Zrotation', 'Xrotation', 'Yrotation']
        
        left_forearm = BvhJoint('LeftForeArm', parent=left_arm)
        left_forearm.offset = np.array([25.0, 0.0, 0.0])
        left_forearm.channels = ['Zrotation', 'Xrotation', 'Yrotation']
        
        left_hand = BvhJoint('LeftHand', parent=left_forearm)
        left_hand.offset = np.array([25.0, 0.0, 0.0])
        left_hand.channels = ['Zrotation', 'Xrotation', 'Yrotation']
        
        # Right arm chain
        right_shoulder = BvhJoint('RightShoulder', parent=spine2)
        right_shoulder.offset = np.array([-7.0, 5.0, 0.0])
        right_shoulder.channels = ['Zrotation', 'Xrotation', 'Yrotation']
        
        right_arm = BvhJoint('RightArm', parent=right_shoulder)
        right_arm.offset = np.array([-10.0, 0.0, 0.0])
        right_arm.channels = ['Zrotation', 'Xrotation', 'Yrotation']
        
        right_forearm = BvhJoint('RightForeArm', parent=right_arm)
        right_forearm.offset = np.array([-25.0, 0.0, 0.0])
        right_forearm.channels = ['Zrotation', 'Xrotation', 'Yrotation']
        
        right_hand = BvhJoint('RightHand', parent=right_forearm)
        right_hand.offset = np.array([-25.0, 0.0, 0.0])
        right_hand.channels = ['Zrotation', 'Xrotation', 'Yrotation']
        
        # Left leg chain
        left_upleg = BvhJoint('LeftUpLeg', parent=root)
        left_upleg.offset = np.array([7.0, -5.0, 0.0])
        left_upleg.channels = ['Zrotation', 'Xrotation', 'Yrotation']
        
        left_leg = BvhJoint('LeftLeg', parent=left_upleg)
        left_leg.offset = np.array([0.0, -40.0, 0.0])
        left_leg.channels = ['Zrotation', 'Xrotation', 'Yrotation']
        
        left_foot = BvhJoint('LeftFoot', parent=left_leg)
        left_foot.offset = np.array([0.0, -40.0, 0.0])
        left_foot.channels = ['Zrotation', 'Xrotation', 'Yrotation']
        
        left_toe = BvhJoint('LeftToeBase', parent=left_foot)
        left_toe.offset = np.array([0.0, -5.0, 10.0])
        left_toe.channels = ['Zrotation', 'Xrotation', 'Yrotation']
        
        # Right leg chain
        right_upleg = BvhJoint('RightUpLeg', parent=root)
        right_upleg.offset = np.array([-7.0, -5.0, 0.0])
        right_upleg.channels = ['Zrotation', 'Xrotation', 'Yrotation']
        
        right_leg = BvhJoint('RightLeg', parent=right_upleg)
        right_leg.offset = np.array([0.0, -40.0, 0.0])
        right_leg.channels = ['Zrotation', 'Xrotation', 'Yrotation']
        
        right_foot = BvhJoint('RightFoot', parent=right_leg)
        right_foot.offset = np.array([0.0, -40.0, 0.0])
        right_foot.channels = ['Zrotation', 'Xrotation', 'Yrotation']
        
        right_toe = BvhJoint('RightToeBase', parent=right_foot)
        right_toe.offset = np.array([0.0, -5.0, 10.0])
        right_toe.channels = ['Zrotation', 'Xrotation', 'Yrotation']
        
        return root
    
    def extract_3d_positions(self, landmarks) -> Dict[str, np.ndarray]:
        """Extract 3D positions from MediaPipe landmarks"""
        positions = {}
        
        # Get hip center (root position)
        left_hip = np.array([landmarks[PoseLandmark.LEFT_HIP].x,
                            landmarks[PoseLandmark.LEFT_HIP].y,
                            landmarks[PoseLandmark.LEFT_HIP].z])
        right_hip = np.array([landmarks[PoseLandmark.RIGHT_HIP].x,
                             landmarks[PoseLandmark.RIGHT_HIP].y,
                             landmarks[PoseLandmark.RIGHT_HIP].z])
        positions['Hips'] = (left_hip + right_hip) / 2.0
        
        # Get shoulder center for spine estimation
        left_shoulder = np.array([landmarks[PoseLandmark.LEFT_SHOULDER].x,
                                 landmarks[PoseLandmark.LEFT_SHOULDER].y,
                                 landmarks[PoseLandmark.LEFT_SHOULDER].z])
        right_shoulder = np.array([landmarks[PoseLandmark.RIGHT_SHOULDER].x,
                                  landmarks[PoseLandmark.RIGHT_SHOULDER].y,
                                  landmarks[PoseLandmark.RIGHT_SHOULDER].z])
        shoulder_center = (left_shoulder + right_shoulder) / 2.0
        
        # Estimate spine positions by interpolating between hips and shoulders
        positions['Spine'] = positions['Hips'] + (shoulder_center - positions['Hips']) * 0.25
        positions['Spine1'] = positions['Hips'] + (shoulder_center - positions['Hips']) * 0.5
        positions['Spine2'] = positions['Hips'] + (shoulder_center - positions['Hips']) * 0.75
        positions['Neck'] = shoulder_center
        
        # Get other joint positions
        for joint_name, landmark_idx in self.skeleton_mapping.items():
            if joint_name not in ['Hips', 'Spine', 'Spine1', 'Spine2', 'Neck']:
                if isinstance(landmark_idx, int):
                    positions[joint_name] = np.array([
                        landmarks[landmark_idx].x,
                        landmarks[landmark_idx].y,
                        landmarks[landmark_idx].z
                    ])
        
        return positions
    
    def calculate_rotation_matrix(self, parent_pos: np.ndarray, child_pos: np.ndarray, 
                                 offset: np.ndarray) -> np.ndarray:
        """Calculate rotation matrix from parent to child joint"""
        # Calculate the vector from parent to child
        bone_vector = child_pos - parent_pos
        bone_length = np.linalg.norm(bone_vector)
        
        if bone_length < 1e-6:
            return np.eye(3)
        
        bone_direction = bone_vector / bone_length
        
        # Calculate the offset direction
        offset_length = np.linalg.norm(offset)
        if offset_length < 1e-6:
            return np.eye(3)
        
        offset_direction = offset / offset_length
        
        # Calculate rotation axis and angle
        rotation_axis = np.cross(offset_direction, bone_direction)
        axis_length = np.linalg.norm(rotation_axis)
        
        if axis_length < 1e-6:
            # Vectors are parallel
            if np.dot(offset_direction, bone_direction) > 0:
                return np.eye(3)
            else:
                # 180 degree rotation
                # Find an orthogonal vector
                if abs(offset_direction[0]) < 0.9:
                    rotation_axis = np.cross(offset_direction, np.array([1, 0, 0]))
                else:
                    rotation_axis = np.cross(offset_direction, np.array([0, 1, 0]))
                rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
                angle = np.pi
        else:
            rotation_axis = rotation_axis / axis_length
            angle = np.arccos(np.clip(np.dot(offset_direction, bone_direction), -1.0, 1.0))
        
        # Create rotation matrix using Rodrigues' formula
        K = np.array([[0, -rotation_axis[2], rotation_axis[1]],
                     [rotation_axis[2], 0, -rotation_axis[0]],
                     [-rotation_axis[1], rotation_axis[0], 0]])
        
        R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * np.dot(K, K)
        
        return R
    
    def rotation_matrix_to_euler(self, R: np.ndarray, order: str = 'ZXY') -> Tuple[float, float, float]:
        """Convert rotation matrix to Euler angles"""
        # Ensure the matrix is orthogonal
        U, _, Vt = np.linalg.svd(R)
        R = np.dot(U, Vt)
        
        if order == 'ZXY':
            sy = np.sqrt(R[0, 0]**2 + R[1, 0]**2)
            singular = sy < 1e-6
            
            if not singular:
                x = np.arctan2(R[2, 1], R[2, 2])
                y = np.arctan2(-R[2, 0], sy)
                z = np.arctan2(R[1, 0], R[0, 0])
            else:
                x = np.arctan2(-R[1, 2], R[1, 1])
                y = np.arctan2(-R[2, 0], sy)
                z = 0
            
            return np.degrees(z), np.degrees(x), np.degrees(y)
        
        # Default XYZ order
        sy = np.sqrt(R[0, 0]**2 + R[0, 1]**2)
        singular = sy < 1e-6
        
        if not singular:
            x = np.arctan2(R[2, 1], R[2, 2])
            y = np.arctan2(-R[2, 0], sy)
            z = np.arctan2(R[1, 0], R[0, 0])
        else:
            x = np.arctan2(-R[1, 2], R[1, 1])
            y = np.arctan2(-R[2, 0], sy)
            z = 0
        
        return np.degrees(x), np.degrees(y), np.degrees(z)
    
    def process_frame(self, frame: np.ndarray, skeleton: BvhJoint) -> Dict[str, List[float]]:
        """Process a single frame and return joint rotations"""
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process with MediaPipe
        results = self.pose.process(rgb_frame)
        
        if not results.pose_world_landmarks:
            return None
        
        # Extract 3D positions
        positions = self.extract_3d_positions(results.pose_world_landmarks.landmark)
        
        # Calculate rotations for each joint
        frame_data = {}
        
        # Process each joint in the skeleton
        def process_joint(joint: BvhJoint, parent_rotation: np.ndarray = np.eye(3)):
            if joint.name == 'Hips':
                # Root position (convert to cm and adjust coordinate system)
                root_pos = positions['Hips'] * 100.0  # Convert to cm
                frame_data['Hips'] = [
                    root_pos[0],  # X position
                    root_pos[1] * -1 + 95.0,  # Y position (invert and offset)
                    root_pos[2] * -1,  # Z position (invert for BVH coordinate system)
                    0.0, 0.0, 0.0  # Root rotation (simplified)
                ]
            else:
                # Calculate rotation from parent to current joint
                parent_name = joint.parent.name
                if parent_name in positions and joint.name in positions:
                    parent_pos = positions[parent_name]
                    child_pos = positions[joint.name]
                    
                    # Calculate local rotation
                    R_local = self.calculate_rotation_matrix(parent_pos, child_pos, joint.offset)
                    
                    # Convert to Euler angles
                    euler = self.rotation_matrix_to_euler(R_local, 'ZXY')
                    frame_data[joint.name] = list(euler)
                else:
                    frame_data[joint.name] = [0.0, 0.0, 0.0]
            
            # Process children
            for child in joint.children:
                if not isinstance(child, BvhJoint):
                    continue
                process_joint(child)
        
        process_joint(skeleton)
        
        return frame_data
    
    def smooth_motion_data(self, motion_data: List[Dict[str, List[float]]], 
                          window_size: int = 5) -> List[Dict[str, List[float]]]:
        """Apply Savitzky-Golay filter to smooth motion data"""
        if len(motion_data) < window_size:
            return motion_data
        
        from scipy.signal import savgol_filter
        
        # Convert to numpy arrays for easier processing
        joint_names = list(motion_data[0].keys())
        smoothed_data = []
        
        for joint_name in joint_names:
            joint_trajectory = []
            for frame in motion_data:
                joint_trajectory.append(frame[joint_name])
            
            joint_trajectory = np.array(joint_trajectory)
            
            # Apply smoothing to each channel
            smoothed_trajectory = np.zeros_like(joint_trajectory)
            for channel in range(joint_trajectory.shape[1]):
                smoothed_trajectory[:, channel] = savgol_filter(
                    joint_trajectory[:, channel], 
                    window_size, 
                    min(3, window_size - 1)  # polynomial order
                )
            
            # Store smoothed data
            for i, frame in enumerate(motion_data):
                if i >= len(smoothed_data):
                    smoothed_data.append({})
                smoothed_data[i][joint_name] = list(smoothed_trajectory[i])
        
        return smoothed_data
    
    def convert_video_to_bvh(self, video_path: str, output_path: str):
        """Main conversion function"""
        # Open video
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Processing video: {video_path}")
        print(f"FPS: {fps}, Total frames: {frame_count}")
        
        # Create BVH skeleton
        skeleton = self.create_bvh_skeleton()
        
        # Process frames
        motion_data = []
        frame_idx = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Preprocess frame
            # Resize for consistent processing (as recommended in the document)
            target_size = (640, 480)
            frame = cv2.resize(frame, target_size)
            
            # Process frame
            frame_data = self.process_frame(frame, skeleton)
            
            if frame_data:
                motion_data.append(frame_data)
            
            frame_idx += 1
            if frame_idx % 30 == 0:
                print(f"Processed {frame_idx}/{frame_count} frames")
        
        cap.release()
        
        # Smooth motion data (as recommended in the document)
        print("Applying motion smoothing...")
        motion_data = self.smooth_motion_data(motion_data, window_size=7)
        
        # Create BVH container
        print("Creating BVH file...")
        container = BvhContainer()
        container.skeleton = skeleton
        container.frame_time = 1.0 / fps
        
        # Convert motion data to BVH format
        for frame_data in motion_data:
            frame_values = []
            
            # Traverse skeleton in order and collect values
            def collect_values(joint: BvhJoint):
                if joint.name in frame_data:
                    frame_values.extend(frame_data[joint.name])
                
                for child in joint.children:
                    if isinstance(child, BvhJoint):
                        collect_values(child)
            
            collect_values(skeleton)
            container.add_frame(frame_values)
        
        # Save BVH file
        with open(output_path, 'w') as f:
            container.save(f)
        
        print(f"BVH file saved to: {output_path}")
        
    def cleanup(self):
        """Clean up resources"""
        self.pose.close()

def main():
    parser = argparse.ArgumentParser(description='Convert video to BVH using MediaPipe')
    parser.add_argument('--video', help='Input MP4 video file')
    parser.add_argument('--output', help='Output BVH file')
    parser.add_argument('--preview', action='store_true', help='Show pose detection preview')
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.exists(args.video):
        print(f"Error: Input file '{args.video}' not found")
        return
    
    # Create converter
    converter = MediaPipeToBVH()
    
    try:
        # Convert video to BVH
        converter.convert_video_to_bvh(args.video, args.output)
    finally:
        converter.cleanup()

if __name__ == "__main__":
    main()