#!/usr/bin/env python3
"""
MediaPipe to BVH Converter
Converts video input to BVH motion capture files using MediaPipe pose estimation
"""

import cv2
import mediapipe as mp
import numpy as np
import argparse
from datetime import datetime
import os

class MediaPipeToBVH:
    def __init__(self):
        # Initialize MediaPipe
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Define BVH joint hierarchy based on MediaPipe landmarks
        self.joint_hierarchy = {
            'Hips': {
                'parent': None,
                'offset': [0, 0, 0],
                'channels': ['Xposition', 'Yposition', 'Zposition', 'Zrotation', 'Xrotation', 'Yrotation'],
                'landmark_indices': [23, 24]  # Average of left and right hip
            },
            'Spine': {
                'parent': 'Hips',
                'offset': [0, 10, 0],
                'channels': ['Zrotation', 'Xrotation', 'Yrotation'],
                'landmark_indices': [23, 24, 11, 12]  # Hips to shoulders
            },
            'Spine1': {
                'parent': 'Spine',
                'offset': [0, 10, 0],
                'channels': ['Zrotation', 'Xrotation', 'Yrotation'],
                'landmark_indices': [11, 12]  # Shoulders
            },
            'Neck': {
                'parent': 'Spine1',
                'offset': [0, 10, 0],
                'channels': ['Zrotation', 'Xrotation', 'Yrotation'],
                'landmark_indices': [11, 12, 0]  # Shoulders to nose
            },
            'Head': {
                'parent': 'Neck',
                'offset': [0, 10, 0],
                'channels': ['Zrotation', 'Xrotation', 'Yrotation'],
                'landmark_indices': [0]  # Nose
            },
            'LeftShoulder': {
                'parent': 'Spine1',
                'offset': [-5, 0, 0],
                'channels': ['Zrotation', 'Xrotation', 'Yrotation'],
                'landmark_indices': [11]
            },
            'LeftArm': {
                'parent': 'LeftShoulder',
                'offset': [-10, 0, 0],
                'channels': ['Zrotation', 'Xrotation', 'Yrotation'],
                'landmark_indices': [11, 13]  # Shoulder to elbow
            },
            'LeftForeArm': {
                'parent': 'LeftArm',
                'offset': [-10, 0, 0],
                'channels': ['Zrotation', 'Xrotation', 'Yrotation'],
                'landmark_indices': [13, 15]  # Elbow to wrist
            },
            'LeftHand': {
                'parent': 'LeftForeArm',
                'offset': [-5, 0, 0],
                'channels': ['Zrotation', 'Xrotation', 'Yrotation'],
                'landmark_indices': [15, 17, 19, 21]  # Wrist and fingers
            },
            'RightShoulder': {
                'parent': 'Spine1',
                'offset': [5, 0, 0],
                'channels': ['Zrotation', 'Xrotation', 'Yrotation'],
                'landmark_indices': [12]
            },
            'RightArm': {
                'parent': 'RightShoulder',
                'offset': [10, 0, 0],
                'channels': ['Zrotation', 'Xrotation', 'Yrotation'],
                'landmark_indices': [12, 14]
            },
            'RightForeArm': {
                'parent': 'RightArm',
                'offset': [10, 0, 0],
                'channels': ['Zrotation', 'Xrotation', 'Yrotation'],
                'landmark_indices': [14, 16]
            },
            'RightHand': {
                'parent': 'RightForeArm',
                'offset': [5, 0, 0],
                'channels': ['Zrotation', 'Xrotation', 'Yrotation'],
                'landmark_indices': [16, 18, 20, 22]
            },
            'LeftUpLeg': {
                'parent': 'Hips',
                'offset': [-5, -5, 0],
                'channels': ['Zrotation', 'Xrotation', 'Yrotation'],
                'landmark_indices': [23, 25]
            },
            'LeftLeg': {
                'parent': 'LeftUpLeg',
                'offset': [0, -20, 0],
                'channels': ['Zrotation', 'Xrotation', 'Yrotation'],
                'landmark_indices': [25, 27]
            },
            'LeftFoot': {
                'parent': 'LeftLeg',
                'offset': [0, -20, 0],
                'channels': ['Zrotation', 'Xrotation', 'Yrotation'],
                'landmark_indices': [27, 29, 31]
            },
            'RightUpLeg': {
                'parent': 'Hips',
                'offset': [5, -5, 0],
                'channels': ['Zrotation', 'Xrotation', 'Yrotation'],
                'landmark_indices': [24, 26]
            },
            'RightLeg': {
                'parent': 'RightUpLeg',
                'offset': [0, -20, 0],
                'channels': ['Zrotation', 'Xrotation', 'Yrotation'],
                'landmark_indices': [26, 28]
            },
            'RightFoot': {
                'parent': 'RightLeg',
                'offset': [0, -20, 0],
                'channels': ['Zrotation', 'Xrotation', 'Yrotation'],
                'landmark_indices': [28, 30, 32]
            }
        }
        
        self.frames = []
        self.frame_time = 1.0 / 30.0  # Default 30 FPS
        
    def landmarks_to_world_coords(self, landmarks, image_width, image_height):
        """Convert normalized landmarks to world coordinates"""
        world_coords = []
        for landmark in landmarks.landmark:
            # Convert normalized coordinates to pixel coordinates
            x = landmark.x * image_width - image_width / 2
            y = image_height / 2 - landmark.y * image_height
            z = landmark.z * image_width  # Z is relative to hip point
            world_coords.append([x, y, z])
        return np.array(world_coords)
    
    def calculate_bone_rotation(self, parent_pos, child_pos, up_vector=[0, 1, 0]):
        """Calculate rotation angles for a bone from parent to child position"""
        # Calculate bone vector
        bone_vector = child_pos - parent_pos
        bone_length = np.linalg.norm(bone_vector)
        
        if bone_length < 0.001:
            return [0, 0, 0]
        
        bone_vector = bone_vector / bone_length
        
        # Calculate rotation angles (simplified version)
        # Y rotation (yaw)
        y_rot = np.arctan2(bone_vector[0], bone_vector[2])
        
        # X rotation (pitch)
        x_rot = np.arcsin(-bone_vector[1])
        
        # Z rotation (roll) - simplified
        z_rot = 0
        
        # Convert to degrees
        return [
            np.degrees(x_rot),
            np.degrees(y_rot),
            np.degrees(z_rot)
        ]
    
    def get_joint_position(self, joint_name, landmarks):
        """Get the position of a joint based on its landmark indices"""
        indices = self.joint_hierarchy[joint_name]['landmark_indices']
        
        if len(indices) == 1:
            return landmarks[indices[0]]
        else:
            # Average multiple landmarks
            positions = [landmarks[idx] for idx in indices]
            return np.mean(positions, axis=0)
    
    def process_frame(self, landmarks):
        """Process a single frame of pose landmarks"""
        frame_data = {}
        
        # Process each joint
        for joint_name, joint_info in self.joint_hierarchy.items():
            joint_pos = self.get_joint_position(joint_name, landmarks)
            
            if joint_name == 'Hips':
                # Root joint has position and rotation
                frame_data[joint_name] = {
                    'position': joint_pos.tolist(),
                    'rotation': [0, 0, 0]  # Simplified - could calculate from body orientation
                }
            else:
                # Calculate rotation based on parent-child relationship
                parent_name = joint_info['parent']
                if parent_name:
                    parent_pos = self.get_joint_position(parent_name, landmarks)
                    rotation = self.calculate_bone_rotation(parent_pos, joint_pos)
                    frame_data[joint_name] = {
                        'rotation': rotation
                    }
        
        return frame_data
    
    def write_bvh_header(self, file):
        """Write BVH header with joint hierarchy"""
        file.write("HIERARCHY\n")
        self._write_joint(file, 'Hips', 0)
        file.write(f"MOTION\n")
        file.write(f"Frames: {len(self.frames)}\n")
        file.write(f"Frame Time: {self.frame_time}\n")
    
    def _write_joint(self, file, joint_name, indent_level):
        """Recursively write joint hierarchy"""
        indent = "  " * indent_level
        joint_info = self.joint_hierarchy[joint_name]
        
        # Write joint declaration
        if joint_info['parent'] is None:
            file.write(f"{indent}ROOT {joint_name}\n")
        else:
            file.write(f"{indent}JOINT {joint_name}\n")
        
        file.write(f"{indent}{{\n")
        
        # Write offset
        offset = joint_info['offset']
        file.write(f"{indent}  OFFSET {offset[0]:.6f} {offset[1]:.6f} {offset[2]:.6f}\n")
        
        # Write channels
        channels = joint_info['channels']
        file.write(f"{indent}  CHANNELS {len(channels)} {' '.join(channels)}\n")
        
        # Write children
        children = [j for j, info in self.joint_hierarchy.items() if info['parent'] == joint_name]
        for child in children:
            self._write_joint(file, child, indent_level + 1)
        
        # Add end site for leaf joints
        if not children:
            file.write(f"{indent}  End Site\n")
            file.write(f"{indent}  {{\n")
            file.write(f"{indent}    OFFSET 0.0 -5.0 0.0\n")
            file.write(f"{indent}  }}\n")
        
        file.write(f"{indent}}}\n")
    
    def write_bvh_motion(self, file):
        """Write motion data for all frames"""
        for frame_data in self.frames:
            motion_values = []
            
            # Write values in hierarchy order
            for joint_name, joint_info in self.joint_hierarchy.items():
                if joint_name in frame_data:
                    data = frame_data[joint_name]
                    
                    # Add values based on channels
                    for channel in joint_info['channels']:
                        if 'position' in channel:
                            axis = channel[0].lower()
                            idx = {'x': 0, 'y': 1, 'z': 2}[axis]
                            motion_values.append(data['position'][idx])
                        elif 'rotation' in channel:
                            axis = channel[0].lower()
                            idx = {'x': 0, 'y': 1, 'z': 2}[axis]
                            rotation = data.get('rotation', [0, 0, 0])
                            motion_values.append(rotation[idx])
            
            # Write frame data
            file.write(" ".join([f"{v:.6f}" for v in motion_values]) + "\n")
    
    def process_video(self, video_path, output_path):
        """Process video file and generate BVH"""
        cap = cv2.VideoCapture(video_path)
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        self.frame_time = 1.0 / fps
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Processing video: {video_path}")
        print(f"FPS: {fps}, Total frames: {frame_count}")
        
        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process with MediaPipe
            results = self.pose.process(rgb_frame)
            
            if results.pose_landmarks:
                # Convert landmarks to world coordinates
                height, width, _ = frame.shape
                world_coords = self.landmarks_to_world_coords(results.pose_landmarks, width, height)
                
                # Process frame
                frame_data = self.process_frame(world_coords)
                self.frames.append(frame_data)
            
            frame_idx += 1
            if frame_idx % 30 == 0:
                print(f"Processed {frame_idx}/{frame_count} frames")
        
        cap.release()
        
        # Write BVH file
        print(f"Writing BVH file: {output_path}")
        with open(output_path, 'w') as f:
            self.write_bvh_header(f)
            self.write_bvh_motion(f)
        
        print(f"Successfully created BVH file with {len(self.frames)} frames")

def main():
    parser = argparse.ArgumentParser(description='Convert video to BVH using MediaPipe')
    parser.add_argument('--video', help='Path to input video file')
    parser.add_argument('-o', '--output', help='Output BVH file path', default=None)
    
    args = parser.parse_args()
    
    # Generate output filename if not provided
    if args.output is None:
        base_name = os.path.splitext(os.path.basename(args.video))[0]
        args.output = f"{base_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.bvh"
    
    # Create converter and process video
    converter = MediaPipeToBVH()
    converter.process_video(args.video, args.output)

if __name__ == "__main__":
    main()