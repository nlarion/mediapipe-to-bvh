#!/usr/bin/env python3
"""
MediaPipe to BVH Converter

This script processes a video file using MediaPipe to extract pose landmarks,
then converts the landmarks to BVH format for animation.

Usage:
    python mediapipe_to_bvh.py input_video.mp4 output.bvh [--fps FPS]

Arguments:
    input_video.mp4   Path to the input video file
    output.bvh        Path to save the output BVH file
    --fps FPS         Frame rate for BVH output (default: 30)
"""

import argparse
import cv2
import numpy as np
import mediapipe as mp
import time
import math
import os
from datetime import datetime


class MediaPipeToBVH:
    def __init__(self, fps=30):
        self.mp_holistic = mp.solutions.holistic
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.fps = fps
        self.frames = []
        self.first_frame_joints = None
        
        # Mapping between MediaPipe pose landmarks and BVH joints
        self.landmark_to_joint = {
            # Main body
            'Hips': 'center_hip',
            'Spine': 'spine0',
            'Spine1': 'spine1',
            'Spine2': 'spine2',
            'Neck': 'neck',
            'Head': 'head',
            # Left arm
            'LeftShoulder': 'shoulder_left',
            'LeftArm': 'left_shoulder',
            'LeftForeArm': 'left_elbow',
            'LeftHand': 'left_wrist',
            # Right arm
            'RightShoulder': 'shoulder_right',
            'RightArm': 'right_shoulder',
            'RightForeArm': 'right_elbow',
            'RightHand': 'right_wrist',
            # Left leg
            'LeftUpLeg': 'left_hip',
            'LeftLeg': 'left_knee',
            'LeftFoot': 'left_ankle',
            'LeftToeBase': 'left_toebase',
            # Right leg
            'RightUpLeg': 'right_hip',
            'RightLeg': 'right_knee',
            'RightFoot': 'right_ankle',
            'RightToeBase': 'right_toebase',
        }
        
        # Define MediaPipe landmark indices
        self.mp_indices = {
            'nose': 0,
            'left_eye_inner': 1,
            'left_eye': 2,
            'left_eye_outer': 3,
            'right_eye_inner': 4,
            'right_eye': 5,
            'right_eye_outer': 6,
            'left_ear': 7,
            'right_ear': 8,
            'mouth_left': 9,
            'mouth_right': 10,
            'left_shoulder': 11,
            'right_shoulder': 12,
            'left_elbow': 13,
            'right_elbow': 14,
            'left_wrist': 15,
            'right_wrist': 16,
            'left_pinky': 17,
            'right_pinky': 18,
            'left_index': 19,
            'right_index': 20,
            'left_thumb': 21,
            'right_thumb': 22,
            'left_hip': 23,
            'right_hip': 24,
            'left_knee': 25,
            'right_knee': 26,
            'left_ankle': 27,
            'right_ankle': 28,
            'left_heel': 29,
            'right_heel': 30,
            'left_foot_index': 31,
            'right_foot_index': 32
        }
        
        # Define joint hierarchy
        self.joint_hierarchy = [
            ('Hips', None),
            ('Spine', 'Hips'),
            ('Spine1', 'Spine'),
            ('Spine2', 'Spine1'),
            ('Neck', 'Spine2'),
            ('Head', 'Neck'),
            ('LeftShoulder', 'Spine2'),
            ('LeftArm', 'LeftShoulder'),
            ('LeftForeArm', 'LeftArm'),
            ('LeftHand', 'LeftForeArm'),
            ('RightShoulder', 'Spine2'),
            ('RightArm', 'RightShoulder'),
            ('RightForeArm', 'RightArm'),
            ('RightHand', 'RightForeArm'),
            ('LeftUpLeg', 'Hips'),
            ('LeftLeg', 'LeftUpLeg'),
            ('LeftFoot', 'LeftLeg'),
            ('LeftToeBase', 'LeftFoot'),
            ('RightUpLeg', 'Hips'),
            ('RightLeg', 'RightUpLeg'),
            ('RightFoot', 'RightLeg'),
            ('RightToeBase', 'RightFoot')
        ]

    def process_video(self, video_path):
        """Process a video file and extract pose landmarks"""
        print(f"Processing video: {video_path}")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
            
        frame_count = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        print(f"Video FPS: {fps}")
        print(f"Total frames: {total_frames}")
        print(f"Output BVH FPS: {self.fps}")
        
        # Calculate frame sampling rate to match desired BVH fps
        sample_rate = max(1, round(fps / self.fps))
        print(f"Sampling every {sample_rate} frames")
        
        with self.mp_holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=2,
            smooth_landmarks=True) as holistic:
            
            start_time = time.time()
            frame_idx = 0
            
            while cap.isOpened():
                success, image = cap.read()
                if not success:
                    break
                
                # Only process every sample_rate frames
                if frame_idx % sample_rate != 0:
                    frame_idx += 1
                    continue
                
                # To improve performance, optionally mark the image as not writeable
                image.flags.writeable = False
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = holistic.process(image)

                # Skip frames where no pose was detected
                if not results.pose_landmarks:
                    frame_idx += 1
                    continue
                
                # Process landmarks
                joint_data = self.process_landmarks(results.pose_landmarks.landmark)
                
                # Store the first frame joint data separately
                if self.first_frame_joints is None:
                    self.first_frame_joints = joint_data
                
                # Store frame data
                self.frames.append(joint_data)
                
                # Progress update
                frame_count += 1
                if frame_count % 10 == 0:
                    elapsed = time.time() - start_time
                    fps_processed = frame_count / elapsed if elapsed > 0 else 0
                    percent_done = (frame_idx / total_frames) * 100
                    print(f"\rProcessed {frame_count} frames ({percent_done:.1f}%, {fps_processed:.1f} fps)", end="")
                
                frame_idx += 1
            
            cap.release()
            print(f"\nProcessed {frame_count} frames in {time.time() - start_time:.2f} seconds")
            print(f"Captured {len(self.frames)} frames for BVH")
            
            if len(self.frames) == 0:
                raise ValueError("No usable frames detected in the video")

    def process_landmarks(self, landmarks):
        """Process the landmarks from MediaPipe into our joint structure"""
        joint_data = []
        
        # First create a dictionary of landmark positions
        landmark_dict = {}
        for name, idx in self.mp_indices.items():
            landmark = landmarks[idx]
            # Convert from MediaPipe coordinate system to 3D space
            # MediaPipe: (0,0) is top-left of image, y-down, x-right, z-toward camera
            # BVH space: Usually y-up, x-right, z-forward or backward
            pos = [
                landmark.x,
                -landmark.y,  # Flip Y to make Y-up
                landmark.z
            ]
            landmark_dict[name] = pos
        
        # Calculate additional landmarks (spine segments, etc.)
        # Center hip
        left_hip = np.array(landmark_dict['left_hip'])
        right_hip = np.array(landmark_dict['right_hip'])
        center_hip = (left_hip + right_hip) / 2
        landmark_dict['center_hip'] = center_hip.tolist()
        
        # Center shoulder
        left_shoulder = np.array(landmark_dict['left_shoulder'])
        right_shoulder = np.array(landmark_dict['right_shoulder'])
        center_shoulder = (left_shoulder + right_shoulder) / 2
        landmark_dict['center_shoulder'] = center_shoulder.tolist()
        
        # Center ear (for head orientation)
        left_ear = np.array(landmark_dict['left_ear'])
        right_ear = np.array(landmark_dict['right_ear'])
        center_ear = (left_ear + right_ear) / 2
        landmark_dict['center_ear'] = center_ear.tolist()
        
        # Spine segments
        spine_vec = center_shoulder - center_hip
        spine_length = np.linalg.norm(spine_vec)
        spine_dir = spine_vec / spine_length if spine_length > 0 else np.array([0, 1, 0])
        
        landmark_dict['spine0'] = (center_hip + spine_dir * (spine_length / 9.0) * 3).tolist()
        landmark_dict['spine1'] = (center_hip + spine_dir * (spine_length / 9.0) * 5).tolist()
        landmark_dict['spine2'] = (center_hip + spine_dir * (spine_length / 9.0) * 7).tolist()
        
        # Neck
        landmark_dict['neck'] = (center_shoulder + spine_dir * (spine_length / 9.0)).tolist()
        
        # Head
        head_dir = center_ear - landmark_dict['neck']
        landmark_dict['head'] = (np.array(landmark_dict['neck']) + head_dir * 0.5).tolist()
        
        # Shoulder joints (between spine and arm)
        shoulder_vec = right_shoulder - left_shoulder
        landmark_dict['shoulder_left'] = (left_shoulder + shoulder_vec * (1/3)).tolist()
        landmark_dict['shoulder_right'] = (left_shoulder + shoulder_vec * (2/3)).tolist()
        
        # Toe bases
        left_foot_dir = np.array(landmark_dict['left_foot_index']) - np.array(landmark_dict['left_heel'])
        landmark_dict['left_toebase'] = (np.array(landmark_dict['left_heel']) + left_foot_dir * 0.6).tolist()
        
        right_foot_dir = np.array(landmark_dict['right_foot_index']) - np.array(landmark_dict['right_heel'])
        landmark_dict['right_toebase'] = (np.array(landmark_dict['right_heel']) + right_foot_dir * 0.6).tolist()
        
        # Calculate joint positions and rotations
        for joint_name, parent_name in self.joint_hierarchy:
            mp_name = self.landmark_to_joint.get(joint_name)
            
            if mp_name is None or mp_name not in landmark_dict:
                continue
                
            position = landmark_dict[mp_name]
            
            # Calculate joint rotation (simplified for this example)
            # In a real application, you'd compute proper rotations between joints
            # Using techniques like look-at matrices or forward kinematics
            rotation = self.calculate_rotation(joint_name, landmark_dict)
            
            # Scale positions for better visualization
            scaled_position = [p * 100 for p in position]  # Scale up for better visibility
            
            # For the root joint (Hips), we use absolute position
            # For other joints, we'll use relative positions in the final BVH
            joint_info = {
                'name': joint_name,
                'position': scaled_position,
                'rotation': rotation,
                'level': self.get_joint_level(joint_name)
            }
            
            joint_data.append(joint_info)
            
        return joint_data

    def calculate_rotation(self, joint_name, landmark_dict):
        """
        Calculate rotation for a joint based on the MediaPipe landmarks.
        Uses the skeletal hierarchy to calculate proper joint rotations.
        """
        # Special case for root joint (Hips)
        if joint_name == 'Hips':
            return self._calculate_hips_rotation(landmark_dict)
        
        # Find parent joint name
        parent_name = None
        for j, p in self.joint_hierarchy:
            if j == joint_name:
                parent_name = p
                break
                
        if parent_name is None or parent_name not in self.landmark_to_joint:
            return [0.0, 0.0, 0.0]  # Unknown parent
            
        # Get the MediaPipe names for current joint and parent
        mp_name = self.landmark_to_joint[joint_name]
        parent_mp_name = self.landmark_to_joint[parent_name]
        
        # Get coordinates for current joint and parent
        if mp_name not in landmark_dict or parent_mp_name not in landmark_dict:
            return [0.0, 0.0, 0.0]
            
        # Current joint position
        joint_pos = np.array(landmark_dict[mp_name])
        
        # Parent joint position
        parent_pos = np.array(landmark_dict[parent_mp_name])
        
        # Handle specific joints with custom logic
        if joint_name == 'Head':
            return self._calculate_head_rotation(joint_name, parent_name, landmark_dict)
        elif 'Arm' in joint_name or 'ForeArm' in joint_name:
            return self._calculate_arm_rotation(joint_name, parent_name, landmark_dict)
        elif 'Leg' in joint_name or 'Foot' in joint_name:
            return self._calculate_leg_rotation(joint_name, parent_name, landmark_dict)
        
        # Find child joints to help with orientation
        child_joints = []
        for j, p in self.joint_hierarchy:
            if p == joint_name:
                child_joints.append(j)
                
        # If there's a child joint, use it to create a plane and calculate rotation
        if child_joints and self.landmark_to_joint[child_joints[0]] in landmark_dict:
            child_mp_name = self.landmark_to_joint[child_joints[0]]
            child_pos = np.array(landmark_dict[child_mp_name])
            
            # Calculate joint direction vector (parent to current)
            joint_dir = joint_pos - parent_pos
            joint_length = np.linalg.norm(joint_dir)
            joint_dir = joint_dir / joint_length if joint_length > 0 else np.array([0, 1, 0])
            
            # Calculate child direction vector (current to child)
            child_dir = child_pos - joint_pos
            child_length = np.linalg.norm(child_dir)
            child_dir = child_dir / child_length if child_length > 0 else np.array([0, 1, 0])
            
            # Create a local coordinate system at the joint
            y_axis = joint_dir  # Primary axis along bone (matching BVH Y-up convention)
            
            # For forward vector (z_axis), use child direction or a perpendicular direction
            temp_z = child_dir - np.dot(child_dir, y_axis) * y_axis  # Project child onto plane perpendicular to y_axis
            temp_z_length = np.linalg.norm(temp_z)
            
            if temp_z_length > 0.001:  # If the projection is significant
                z_axis = temp_z / temp_z_length
            else:
                # Find a perpendicular axis
                ref_dir = np.array([0, 0, 1])  # Use forward as reference
                if np.allclose(y_axis, ref_dir) or np.allclose(y_axis, -ref_dir):
                    ref_dir = np.array([1, 0, 0])  # Use another reference if needed
                
                z_axis = np.cross(y_axis, ref_dir)
                z_axis = z_axis / np.linalg.norm(z_axis) if np.linalg.norm(z_axis) > 0 else np.array([0, 0, 1])
            
            x_axis = np.cross(y_axis, z_axis)
            x_axis = x_axis / np.linalg.norm(x_axis) if np.linalg.norm(x_axis) > 0 else np.array([1, 0, 0])
            
            # Ensure we have an orthogonal basis
            z_axis = np.cross(x_axis, y_axis)
            
            # Now we have a local coordinate system (x_axis, y_axis, z_axis)
            # Construct rotation matrix (column-major order)
            rotation_matrix = np.column_stack((x_axis, y_axis, z_axis))
            
            # Check for valid rotation matrix
            det = np.linalg.det(rotation_matrix)
            if abs(det - 1.0) > 0.1:  # Not a proper rotation matrix
                # Fix: make sure it's orthogonal
                u, _, vh = np.linalg.svd(rotation_matrix, full_matrices=True)
                rotation_matrix = u @ vh
            
            # Extract Euler angles (ZYX order, to match BVH convention)
            # This is the math for extracting Euler angles from rotation matrix in ZYX order
            if abs(rotation_matrix[0, 2]) >= 0.99999:  # Gimbal lock case
                # Handle gimbal lock
                z_angle = 0  # Can't determine individual z and x rotations in gimbal lock
                y_angle = -math.pi/2 if rotation_matrix[0, 2] > 0 else math.pi/2
                x_angle = math.atan2(rotation_matrix[1, 0], rotation_matrix[2, 0])
            else:
                y_angle = math.asin(-rotation_matrix[0, 2])
                x_angle = math.atan2(rotation_matrix[1, 2], rotation_matrix[2, 2])
                z_angle = math.atan2(rotation_matrix[0, 1], rotation_matrix[0, 0])
            
            # Convert from radians to degrees
            x_deg = math.degrees(x_angle)
            y_deg = math.degrees(y_angle)
            z_deg = math.degrees(z_angle)
            
            return [x_deg, y_deg, z_deg]
        else:
            # For joints without children, calculate a simplified rotation
            # based on the parent-to-joint direction vector
            joint_dir = joint_pos - parent_pos
            joint_dir = joint_dir / np.linalg.norm(joint_dir) if np.linalg.norm(joint_dir) > 0 else np.array([0, 1, 0])
            
            # Create a coordinate system
            y_axis = joint_dir  # Primary axis along bone
            
            # Find a perpendicular axis
            ref_dir = np.array([0, 0, 1])  # Use forward as reference
            if np.allclose(y_axis, ref_dir) or np.allclose(y_axis, -ref_dir):
                ref_dir = np.array([1, 0, 0])  # Use another reference if needed
            
            z_axis = np.cross(y_axis, ref_dir)
            z_axis = z_axis / np.linalg.norm(z_axis) if np.linalg.norm(z_axis) > 0 else np.array([0, 0, 1])
            
            x_axis = np.cross(y_axis, z_axis)
            x_axis = x_axis / np.linalg.norm(x_axis) if np.linalg.norm(x_axis) > 0 else np.array([1, 0, 0])
            
            # Ensure we have an orthogonal basis
            z_axis = np.cross(x_axis, y_axis)
            
            # Construct rotation matrix
            rotation_matrix = np.column_stack((x_axis, y_axis, z_axis))
            
            # Check for valid rotation matrix
            det = np.linalg.det(rotation_matrix)
            if abs(det - 1.0) > 0.1:  # Not a proper rotation matrix
                # Fix: make sure it's orthogonal
                u, _, vh = np.linalg.svd(rotation_matrix, full_matrices=True)
                rotation_matrix = u @ vh
            
            # Extract Euler angles (ZYX order)
            if abs(rotation_matrix[0, 2]) >= 0.99999:  # Gimbal lock case
                z_angle = 0
                y_angle = -math.pi/2 if rotation_matrix[0, 2] > 0 else math.pi/2
                x_angle = math.atan2(rotation_matrix[1, 0], rotation_matrix[2, 0])
            else:
                y_angle = math.asin(-rotation_matrix[0, 2])
                x_angle = math.atan2(rotation_matrix[1, 2], rotation_matrix[2, 2])
                z_angle = math.atan2(rotation_matrix[0, 1], rotation_matrix[0, 0])
            
            # Convert from radians to degrees
            x_deg = math.degrees(x_angle)
            y_deg = math.degrees(y_angle)
            z_deg = math.degrees(z_angle)
            
            return [x_deg, y_deg, z_deg]
    
    def _calculate_hips_rotation(self, landmark_dict):
        """Calculate rotation specifically for the hips (root) joint"""
        # Get reference positions
        left_hip = np.array(landmark_dict['left_hip'])
        right_hip = np.array(landmark_dict['right_hip'])
        left_shoulder = np.array(landmark_dict['left_shoulder'])
        right_shoulder = np.array(landmark_dict['right_shoulder'])
        
        # Calculate hip direction (right to left hip)
        hip_dir = left_hip - right_hip
        hip_dir = hip_dir / np.linalg.norm(hip_dir) if np.linalg.norm(hip_dir) > 0 else np.array([1, 0, 0])
        
        # Calculate approximate up direction
        center_hip = (left_hip + right_hip) / 2
        center_shoulder = (left_shoulder + right_shoulder) / 2
        up_dir = center_shoulder - center_hip
        up_dir = up_dir / np.linalg.norm(up_dir) if np.linalg.norm(up_dir) > 0 else np.array([0, 1, 0])
        
        # Create coordinate system (X forward, Y up, Z right - standard BVH convention)
        z_axis = hip_dir  # Right to left corresponds to Z axis in BVH
        y_axis = up_dir   # Up corresponds to Y axis in BVH
        
        # Ensure z_axis is orthogonal to y_axis
        z_axis = z_axis - np.dot(z_axis, y_axis) * y_axis
        z_axis = z_axis / np.linalg.norm(z_axis) if np.linalg.norm(z_axis) > 0 else np.array([0, 0, 1])
        
        # Compute x_axis as cross product (pointing forward)
        x_axis = np.cross(y_axis, z_axis)
        x_axis = x_axis / np.linalg.norm(x_axis) if np.linalg.norm(x_axis) > 0 else np.array([1, 0, 0])
        
        # Ensure perfect orthogonality
        z_axis = np.cross(x_axis, y_axis)
        
        # Construct rotation matrix
        rotation_matrix = np.column_stack((x_axis, y_axis, z_axis))
        
        # Ensure it's a valid rotation matrix
        det = np.linalg.det(rotation_matrix)
        if abs(det - 1.0) > 0.1:
            u, _, vh = np.linalg.svd(rotation_matrix, full_matrices=True)
            rotation_matrix = u @ vh
        
        # Extract Euler angles (ZYX order)
        if abs(rotation_matrix[0, 2]) >= 0.99999:  # Gimbal lock case
            z_angle = 0
            y_angle = -math.pi/2 if rotation_matrix[0, 2] > 0 else math.pi/2
            x_angle = math.atan2(rotation_matrix[1, 0], rotation_matrix[2, 0])
        else:
            y_angle = math.asin(-rotation_matrix[0, 2])
            x_angle = math.atan2(rotation_matrix[1, 2], rotation_matrix[2, 2])
            z_angle = math.atan2(rotation_matrix[0, 1], rotation_matrix[0, 0])
        
        # Convert from radians to degrees
        x_deg = math.degrees(x_angle)
        y_deg = math.degrees(y_angle)
        z_deg = math.degrees(z_angle)
        
        return [x_deg, y_deg, z_deg]
    
    def _calculate_head_rotation(self, joint_name, parent_name, landmark_dict):
        """Calculate rotation specifically for the head joint"""
        # Get reference positions
        neck_pos = np.array(landmark_dict[self.landmark_to_joint[parent_name]])
        head_pos = np.array(landmark_dict[self.landmark_to_joint[joint_name]])
        nose = np.array(landmark_dict['nose'])
        left_ear = np.array(landmark_dict['left_ear'])
        right_ear = np.array(landmark_dict['right_ear'])
        
        # Calculate head direction (neck to head)
        head_dir = head_pos - neck_pos
        head_dir = head_dir / np.linalg.norm(head_dir) if np.linalg.norm(head_dir) > 0 else np.array([0, 1, 0])
        
        # Calculate approximate forward direction using nose
        forward_dir = nose - head_pos
        forward_dir = forward_dir / np.linalg.norm(forward_dir) if np.linalg.norm(forward_dir) > 0 else np.array([0, 0, 1])
        
        # Calculate ear-to-ear direction
        ear_dir = left_ear - right_ear
        ear_dir = ear_dir / np.linalg.norm(ear_dir) if np.linalg.norm(ear_dir) > 0 else np.array([1, 0, 0])
        
        # Create coordinate system
        y_axis = head_dir   # Up corresponds to Y axis in BVH
        
        # Make sure forward_dir is orthogonal to y_axis
        forward_dir = forward_dir - np.dot(forward_dir, y_axis) * y_axis
        forward_length = np.linalg.norm(forward_dir)
        
        if forward_length > 0.001:
            x_axis = forward_dir / forward_length
        else:
            # If forward direction is too close to head direction, use ear direction
            x_axis = np.cross(ear_dir, y_axis)
            x_axis = x_axis / np.linalg.norm(x_axis) if np.linalg.norm(x_axis) > 0 else np.array([1, 0, 0])
        
        # Compute z_axis
        z_axis = np.cross(x_axis, y_axis)
        z_axis = z_axis / np.linalg.norm(z_axis) if np.linalg.norm(z_axis) > 0 else np.array([0, 0, 1])
        
        # Ensure orthogonality
        x_axis = np.cross(y_axis, z_axis)
        
        # Construct rotation matrix
        rotation_matrix = np.column_stack((x_axis, y_axis, z_axis))
        
        # Ensure valid rotation matrix
        det = np.linalg.det(rotation_matrix)
        if abs(det - 1.0) > 0.1:
            u, _, vh = np.linalg.svd(rotation_matrix, full_matrices=True)
            rotation_matrix = u @ vh
        
        # Extract Euler angles (ZYX order)
        if abs(rotation_matrix[0, 2]) >= 0.99999:  # Gimbal lock case
            z_angle = 0
            y_angle = -math.pi/2 if rotation_matrix[0, 2] > 0 else math.pi/2
            x_angle = math.atan2(rotation_matrix[1, 0], rotation_matrix[2, 0])
        else:
            y_angle = math.asin(-rotation_matrix[0, 2])
            x_angle = math.atan2(rotation_matrix[1, 2], rotation_matrix[2, 2])
            z_angle = math.atan2(rotation_matrix[0, 1], rotation_matrix[0, 0])
        
        # Convert from radians to degrees
        x_deg = math.degrees(x_angle)
        y_deg = math.degrees(y_angle)
        z_deg = math.degrees(z_angle)
        
        return [x_deg, y_deg, z_deg]
    
    def _calculate_arm_rotation(self, joint_name, parent_name, landmark_dict):
        """Calculate rotation specifically for arm joints"""
        # Get relevant joint positions
        joint_pos = np.array(landmark_dict[self.landmark_to_joint[joint_name]])
        parent_pos = np.array(landmark_dict[self.landmark_to_joint[parent_name]])
        
        # Find child joint (if any)
        child_name = None
        for j, p in self.joint_hierarchy:
            if p == joint_name:
                child_name = j
                break
        
        # Calculate joint direction vector (parent to current)
        joint_dir = joint_pos - parent_pos
        joint_dir = joint_dir / np.linalg.norm(joint_dir) if np.linalg.norm(joint_dir) > 0 else np.array([0, 1, 0])
        
        # If we have a child joint, use it to determine twist
        if child_name and self.landmark_to_joint[child_name] in landmark_dict:
            child_pos = np.array(landmark_dict[self.landmark_to_joint[child_name]])
            child_dir = child_pos - joint_pos
            child_dir = child_dir / np.linalg.norm(child_dir) if np.linalg.norm(child_dir) > 0 else np.array([0, 1, 0])
            
            # For arm joints, twist is particularly important
            # We'll use the plane formed by parent-joint-child to help determine twist
            
            # Primary axis is along the bone
            y_axis = joint_dir
            
            # Use child direction to inform the plane
            # Project child direction onto plane perpendicular to joint direction
            temp_z = child_dir - np.dot(child_dir, y_axis) * y_axis
            temp_z_length = np.linalg.norm(temp_z)
            
            if temp_z_length > 0.001:
                z_axis = temp_z / temp_z_length
            else:
                # Default to a standard orientation
                ref_dir = np.array([0, 0, 1])  # Use forward direction
                if np.allclose(y_axis, ref_dir) or np.allclose(y_axis, -ref_dir):
                    ref_dir = np.array([1, 0, 0])
                    
                z_axis = np.cross(y_axis, ref_dir)
                z_axis = z_axis / np.linalg.norm(z_axis) if np.linalg.norm(z_axis) > 0 else np.array([0, 0, 1])
            
            # Complete the coordinate system
            x_axis = np.cross(y_axis, z_axis)
            x_axis = x_axis / np.linalg.norm(x_axis) if np.linalg.norm(x_axis) > 0 else np.array([1, 0, 0])
            
            # Ensure orthogonality
            z_axis = np.cross(x_axis, y_axis)
        else:
            # Without a child joint, use a default orientation
            y_axis = joint_dir
            
            # Find perpendicular axes
            ref_dir = np.array([0, 0, 1])  # Use forward direction
            if np.allclose(y_axis, ref_dir) or np.allclose(y_axis, -ref_dir):
                ref_dir = np.array([1, 0, 0])
                
            z_axis = np.cross(y_axis, ref_dir)
            z_axis = z_axis / np.linalg.norm(z_axis) if np.linalg.norm(z_axis) > 0 else np.array([0, 0, 1])
            
            x_axis = np.cross(y_axis, z_axis)
            x_axis = x_axis / np.linalg.norm(x_axis) if np.linalg.norm(x_axis) > 0 else np.array([1, 0, 0])
            
            # Ensure orthogonality
            z_axis = np.cross(x_axis, y_axis)
        
        # Construct rotation matrix
        rotation_matrix = np.column_stack((x_axis, y_axis, z_axis))
        
        # Ensure valid rotation matrix
        det = np.linalg.det(rotation_matrix)
        if abs(det - 1.0) > 0.1:
            u, _, vh = np.linalg.svd(rotation_matrix, full_matrices=True)
            rotation_matrix = u @ vh
        
        # Extract Euler angles (ZYX order)
        if abs(rotation_matrix[0, 2]) >= 0.99999:  # Gimbal lock case
            z_angle = 0
            y_angle = -math.pi/2 if rotation_matrix[0, 2] > 0 else math.pi/2
            x_angle = math.atan2(rotation_matrix[1, 0], rotation_matrix[2, 0])
        else:
            y_angle = math.asin(-rotation_matrix[0, 2])
            x_angle = math.atan2(rotation_matrix[1, 2], rotation_matrix[2, 2])
            z_angle = math.atan2(rotation_matrix[0, 1], rotation_matrix[0, 0])
        
        # Convert from radians to degrees
        x_deg = math.degrees(x_angle)
        y_deg = math.degrees(y_angle)
        z_deg = math.degrees(z_angle)
        
        return [x_deg, y_deg, z_deg]
    
    def _calculate_leg_rotation(self, joint_name, parent_name, landmark_dict):
        """Calculate rotation specifically for leg joints"""
        # Similar approach to arms, but handling the different orientations of leg joints
        
        # Get relevant joint positions
        joint_pos = np.array(landmark_dict[self.landmark_to_joint[joint_name]])
        parent_pos = np.array(landmark_dict[self.landmark_to_joint[parent_name]])
        
        # Find child joint (if any)
        child_name = None
        for j, p in self.joint_hierarchy:
            if p == joint_name:
                child_name = j
                break
        
        # Calculate joint direction vector (parent to current)
        joint_dir = joint_pos - parent_pos
        joint_dir = joint_dir / np.linalg.norm(joint_dir) if np.linalg.norm(joint_dir) > 0 else np.array([0, -1, 0])
        
        # If we have a child joint, use it to determine twist
        if child_name and self.landmark_to_joint[child_name] in landmark_dict:
            child_pos = np.array(landmark_dict[self.landmark_to_joint[child_name]])
            child_dir = child_pos - joint_pos
            child_dir = child_dir / np.linalg.norm(child_dir) if np.linalg.norm(child_dir) > 0 else np.array([0, -1, 0])
            
            # Primary axis is along the bone
            y_axis = joint_dir
            
            # Use child direction to inform twist
            temp_z = child_dir - np.dot(child_dir, y_axis) * y_axis
            temp_z_length = np.linalg.norm(temp_z)
            
            if temp_z_length > 0.001:
                z_axis = temp_z / temp_z_length
            else:
                # Default to a standard orientation for legs
                # For legs, use left-right direction to help determine twist
                left_hip = np.array(landmark_dict['left_hip'])
                right_hip = np.array(landmark_dict['right_hip'])
                hip_dir = left_hip - right_hip
                hip_dir = hip_dir / np.linalg.norm(hip_dir) if np.linalg.norm(hip_dir) > 0 else np.array([1, 0, 0])
                
                # Project hip direction onto plane perpendicular to joint direction
                temp_z = hip_dir - np.dot(hip_dir, y_axis) * y_axis
                temp_z_length = np.linalg.norm(temp_z)
                
                if temp_z_length > 0.001:
                    z_axis = temp_z / temp_z_length
                else:
                    # Final fallback
                    ref_dir = np.array([1, 0, 0])  # Use left-right direction
                    if np.allclose(y_axis, ref_dir) or np.allclose(y_axis, -ref_dir):
                        ref_dir = np.array([0, 0, 1])
                        
                    z_axis = np.cross(y_axis, ref_dir)
                    z_axis = z_axis / np.linalg.norm(z_axis) if np.linalg.norm(z_axis) > 0 else np.array([0, 0, 1])
            
            # Complete the coordinate system
            x_axis = np.cross(y_axis, z_axis)
            x_axis = x_axis / np.linalg.norm(x_axis) if np.linalg.norm(x_axis) > 0 else np.array([1, 0, 0])
            
            # Ensure orthogonality
            z_axis = np.cross(x_axis, y_axis)
        else:
            # Without a child joint, use a default orientation
            y_axis = joint_dir
            
            # Default to a standard orientation for legs
            left_hip = np.array(landmark_dict['left_hip'])
            right_hip = np.array(landmark_dict['right_hip'])
            hip_dir = left_hip - right_hip
            hip_dir = hip_dir / np.linalg.norm(hip_dir) if np.linalg.norm(hip_dir) > 0 else np.array([1, 0, 0])
            
            # Project hip direction onto plane perpendicular to joint direction
            temp_z = hip_dir - np.dot(hip_dir, y_axis) * y_axis
            temp_z_length = np.linalg.norm(temp_z)
            
            if temp_z_length > 0.001:
                z_axis = temp_z / temp_z_length
            else:
                # Final fallback
                ref_dir = np.array([1, 0, 0])  # Use left-right direction
                if np.allclose(y_axis, ref_dir) or np.allclose(y_axis, -ref_dir):
                    ref_dir = np.array([0, 0, 1])
                    
                z_axis = np.cross(y_axis, ref_dir)
                z_axis = z_axis / np.linalg.norm(z_axis) if np.linalg.norm(z_axis) > 0 else np.array([0, 0, 1])
            
            # Complete the coordinate system
            x_axis = np.cross(y_axis, z_axis)
            x_axis = x_axis / np.linalg.norm(x_axis) if np.linalg.norm(x_axis) > 0 else np.array([1, 0, 0])
            
            # Ensure orthogonality
            z_axis = np.cross(x_axis, y_axis)
        
        # Construct rotation matrix
        rotation_matrix = np.column_stack((x_axis, y_axis, z_axis))
        
        # Ensure valid rotation matrix
        det = np.linalg.det(rotation_matrix)
        if abs(det - 1.0) > 0.1:
            u, _, vh = np.linalg.svd(rotation_matrix, full_matrices=True)
            rotation_matrix = u @ vh
        
        # Extract Euler angles (ZYX order)
        if abs(rotation_matrix[0, 2]) >= 0.99999:  # Gimbal lock case
            z_angle = 0
            y_angle = -math.pi/2 if rotation_matrix[0, 2] > 0 else math.pi/2
            x_angle = math.atan2(rotation_matrix[1, 0], rotation_matrix[2, 0])
        else:
            y_angle = math.asin(-rotation_matrix[0, 2])
            x_angle = math.atan2(rotation_matrix[1, 2], rotation_matrix[2, 2])
            z_angle = math.atan2(rotation_matrix[0, 1], rotation_matrix[0, 0])
        
        # Convert from radians to degrees
        x_deg = math.degrees(x_angle)
        y_deg = math.degrees(y_angle)
        z_deg = math.degrees(z_angle)
        
        return [x_deg, y_deg, z_deg]

    def get_joint_level(self, joint_name):
        """Determine the hierarchical level of a joint"""
        level = 0
        current = joint_name
        
        while True:
            parent = None
            for j, p in self.joint_hierarchy:
                if j == current:
                    parent = p
                    break
            
            if parent is None:
                break
                
            level += 1
            current = parent
            
        return level

    def generate_bvh(self):
        """Generate BVH file content from the processed frames"""
        if not self.frames or not self.first_frame_joints:
            raise ValueError("No motion data available")
            
        bvh_content = "HIERARCHY\n"
        
        # Process hierarchy
        root_joint = None
        for joint in self.first_frame_joints:
            if joint['level'] == 0:  # This is the root
                root_joint = joint
                break
                
        if not root_joint:
            raise ValueError("No root joint found in hierarchy")
            
        # Process the hierarchy starting from the root
        bvh_content += self._process_joint_hierarchy(root_joint, self.first_frame_joints)
        
        # Process motion
        bvh_content += "MOTION\n"
        bvh_content += f"Frames: {len(self.frames)}\n"
        bvh_content += f"Frame Time: {1.0/self.fps}\n"
        
        # Process each frame
        for frame in self.frames:
            frame_data = []
            
            # First, add root position and rotation
            root = None
            for joint in frame:
                if joint['name'] == 'Hips':  # Root joint
                    root = joint
                    break
                    
            if root:
                # Root position
                frame_data.extend(root['position'])
                # Root rotation
                frame_data.extend(root['rotation'])
                
                # Add rotations for all other joints in hierarchy order
                for joint in frame:
                    if joint['name'] != 'Hips':  # Skip root (already added)
                        frame_data.extend(joint['rotation'])
            
            # Write the frame data
            bvh_content += " ".join(map(str, frame_data)) + "\n"
            
        return bvh_content

    def _process_joint_hierarchy(self, joint, all_joints, indent=0):
        """Recursively process the joint hierarchy for BVH format"""
        indent_str = "  " * indent
        name = joint['name']
        
        # Start with root
        if joint['level'] == 0:
            bvh_content = f"{indent_str}ROOT {name}\n"
        else:
            bvh_content = f"{indent_str}JOINT {name}\n"
            
        bvh_content += f"{indent_str}{{\n"
        indent_str = "  " * (indent + 1)
        
        # Add offset
        position = joint['position']
        if joint['level'] > 0:
            # For non-root joints, we need to calculate relative position to parent
            # This is simplified - a real implementation would compute actual offsets
            parent_name = None
            for j_name, p_name in self.joint_hierarchy:
                if j_name == name:
                    parent_name = p_name
                    break
                    
            if parent_name:
                parent = None
                for j in all_joints:
                    if j['name'] == parent_name:
                        parent = j
                        break
                        
                if parent:
                    # Calculate relative position (simplified)
                    # In a real implementation, this would be a proper offset in the joint's parent space
                    position = [
                        position[0] - parent['position'][0],
                        position[1] - parent['position'][1],
                        position[2] - parent['position'][2]
                    ]
        
        bvh_content += f"{indent_str}OFFSET {position[0]} {position[1]} {position[2]}\n"
        
        # Add channels
        if joint['level'] == 0:  # Root
            bvh_content += f"{indent_str}CHANNELS 6 Xposition Yposition Zposition Xrotation Yrotation Zrotation\n"
        else:  # Other joints
            bvh_content += f"{indent_str}CHANNELS 3 Xrotation Yrotation Zrotation\n"
            
        # Process children
        children = []
        for j_name, p_name in self.joint_hierarchy:
            if p_name == name:
                for j in all_joints:
                    if j['name'] == j_name:
                        children.append(j)
                        break
        
        for child in children:
            bvh_content += self._process_joint_hierarchy(child, all_joints, indent + 1)
            
        # If no children, add End Site
        if not children:
            bvh_content += f"{indent_str}End Site\n"
            bvh_content += f"{indent_str}{{\n"
            bvh_content += f"{indent_str}  OFFSET 0 0 0\n"
            bvh_content += f"{indent_str}}}\n"
            
        bvh_content += f"{'  ' * indent}}}\n"
        return bvh_content

    def save_bvh(self, output_path):
        """Save the BVH data to a file"""
        bvh_content = self.generate_bvh()
        
        with open(output_path, 'w') as f:
            f.write(bvh_content)
            
        print(f"BVH file saved to: {output_path}")
        print(f"Total frames: {len(self.frames)}")
        print(f"Duration: {len(self.frames)/self.fps:.2f} seconds")


def main():
    """Main function to run the converter"""
    parser = argparse.ArgumentParser(description='Convert video to BVH using MediaPipe')
    parser.add_argument('input', help='Input video file path')
    parser.add_argument('output', help='Output BVH file path')
    parser.add_argument('--fps', type=float, default=30.0, help='Output BVH frame rate (default: 30)')
    
    args = parser.parse_args()
    
    try:
        converter = MediaPipeToBVH(fps=args.fps)
        converter.process_video(args.input)
        converter.save_bvh(args.output)
        print("Conversion completed successfully!")
    except Exception as e:
        print(f"Error: {e}")
        return 1
        
    return 0


if __name__ == "__main__":
    main()