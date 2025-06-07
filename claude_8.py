#!/usr/bin/env python3
"""
MediaPipe Video to BVH Converter
Converts video files to BVH motion capture format using MediaPipe pose estimation.
Compatible with Blender and other 3D animation software.
"""

import cv2
import mediapipe as mp
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
import argparse
import os
from typing import Dict, List, Tuple, Optional
import json
from dataclasses import dataclass
from datetime import datetime


@dataclass
class BVHJoint:
    """Represents a joint in the BVH hierarchy"""
    name: str
    offset: List[float]
    channels: List[str]
    children: List[str]


class MediaPipeToBVHConverter:
    """Main converter class for MediaPipe to BVH conversion"""
    
    def __init__(self, scale_factor: float = 100.0):
        """
        Initialize the converter.
        
        Args:
            scale_factor: Scale factor for converting normalized coordinates to cm
        """
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.scale_factor = scale_factor
        
        # Initialize MediaPipe Pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Define the joint mapping from MediaPipe landmarks to BVH joints
        self.joint_mapping = {
            'Hips': {'indices': [23, 24], 'type': 'average'},
            'Spine': {'indices': [23, 24, 11, 12], 'type': 'interpolate', 'factor': 0.33},
            'Spine1': {'indices': [23, 24, 11, 12], 'type': 'interpolate', 'factor': 0.66},
            'Neck': {'indices': [11, 12], 'type': 'average'},
            'Head': {'indices': [0], 'type': 'direct'},
            
            'LeftShoulder': {'indices': [11], 'type': 'direct'},
            'LeftArm': {'indices': [13], 'type': 'direct'},
            'LeftForeArm': {'indices': [15], 'type': 'direct'},
            'LeftHand': {'indices': [19], 'type': 'direct'},
            
            'RightShoulder': {'indices': [12], 'type': 'direct'},
            'RightArm': {'indices': [14], 'type': 'direct'},
            'RightForeArm': {'indices': [16], 'type': 'direct'},
            'RightHand': {'indices': [20], 'type': 'direct'},
            
            'LeftUpLeg': {'indices': [23], 'type': 'direct'},
            'LeftLeg': {'indices': [25], 'type': 'direct'},
            'LeftFoot': {'indices': [27], 'type': 'direct'},
            'LeftToeBase': {'indices': [31], 'type': 'direct'},
            
            'RightUpLeg': {'indices': [24], 'type': 'direct'},
            'RightLeg': {'indices': [26], 'type': 'direct'},
            'RightFoot': {'indices': [28], 'type': 'direct'},
            'RightToeBase': {'indices': [32], 'type': 'direct'}
        }
        
        # Define BVH skeleton hierarchy
        self.skeleton_hierarchy = {
            'Hips': BVHJoint(
                name='Hips',
                offset=[0.0, 0.0, 0.0],
                channels=['Xposition', 'Yposition', 'Zposition', 
                         'Zrotation', 'Xrotation', 'Yrotation'],
                children=['Spine', 'LeftUpLeg', 'RightUpLeg']
            ),
            'Spine': BVHJoint(
                name='Spine',
                offset=[0.0, 10.0, 0.0],
                channels=['Zrotation', 'Xrotation', 'Yrotation'],
                children=['Spine1']
            ),
            'Spine1': BVHJoint(
                name='Spine1',
                offset=[0.0, 10.0, 0.0],
                channels=['Zrotation', 'Xrotation', 'Yrotation'],
                children=['Neck']
            ),
            'Neck': BVHJoint(
                name='Neck',
                offset=[0.0, 10.0, 0.0],
                channels=['Zrotation', 'Xrotation', 'Yrotation'],
                children=['Head', 'LeftShoulder', 'RightShoulder']
            ),
            'Head': BVHJoint(
                name='Head',
                offset=[0.0, 10.0, 0.0],
                channels=['Zrotation', 'Xrotation', 'Yrotation'],
                children=[]
            ),
            'LeftShoulder': BVHJoint(
                name='LeftShoulder',
                offset=[-5.0, 5.0, 0.0],
                channels=['Zrotation', 'Xrotation', 'Yrotation'],
                children=['LeftArm']
            ),
            'LeftArm': BVHJoint(
                name='LeftArm',
                offset=[-15.0, 0.0, 0.0],
                channels=['Zrotation', 'Xrotation', 'Yrotation'],
                children=['LeftForeArm']
            ),
            'LeftForeArm': BVHJoint(
                name='LeftForeArm',
                offset=[-12.0, 0.0, 0.0],
                channels=['Zrotation', 'Xrotation', 'Yrotation'],
                children=['LeftHand']
            ),
            'LeftHand': BVHJoint(
                name='LeftHand',
                offset=[-8.0, 0.0, 0.0],
                channels=['Zrotation', 'Xrotation', 'Yrotation'],
                children=[]
            ),
            'RightShoulder': BVHJoint(
                name='RightShoulder',
                offset=[5.0, 5.0, 0.0],
                channels=['Zrotation', 'Xrotation', 'Yrotation'],
                children=['RightArm']
            ),
            'RightArm': BVHJoint(
                name='RightArm',
                offset=[15.0, 0.0, 0.0],
                channels=['Zrotation', 'Xrotation', 'Yrotation'],
                children=['RightForeArm']
            ),
            'RightForeArm': BVHJoint(
                name='RightForeArm',
                offset=[12.0, 0.0, 0.0],
                channels=['Zrotation', 'Xrotation', 'Yrotation'],
                children=['RightHand']
            ),
            'RightHand': BVHJoint(
                name='RightHand',
                offset=[8.0, 0.0, 0.0],
                channels=['Zrotation', 'Xrotation', 'Yrotation'],
                children=[]
            ),
            'LeftUpLeg': BVHJoint(
                name='LeftUpLeg',
                offset=[-10.0, 0.0, 0.0],
                channels=['Zrotation', 'Xrotation', 'Yrotation'],
                children=['LeftLeg']
            ),
            'LeftLeg': BVHJoint(
                name='LeftLeg',
                offset=[0.0, -40.0, 0.0],
                channels=['Zrotation', 'Xrotation', 'Yrotation'],
                children=['LeftFoot']
            ),
            'LeftFoot': BVHJoint(
                name='LeftFoot',
                offset=[0.0, -40.0, 0.0],
                channels=['Zrotation', 'Xrotation', 'Yrotation'],
                children=['LeftToeBase']
            ),
            'LeftToeBase': BVHJoint(
                name='LeftToeBase',
                offset=[0.0, -5.0, 10.0],
                channels=['Zrotation', 'Xrotation', 'Yrotation'],
                children=[]
            ),
            'RightUpLeg': BVHJoint(
                name='RightUpLeg',
                offset=[10.0, 0.0, 0.0],
                channels=['Zrotation', 'Xrotation', 'Yrotation'],
                children=['RightLeg']
            ),
            'RightLeg': BVHJoint(
                name='RightLeg',
                offset=[0.0, -40.0, 0.0],
                channels=['Zrotation', 'Xrotation', 'Yrotation'],
                children=['RightFoot']
            ),
            'RightFoot': BVHJoint(
                name='RightFoot',
                offset=[0.0, -40.0, 0.0],
                channels=['Zrotation', 'Xrotation', 'Yrotation'],
                children=['RightToeBase']
            ),
            'RightToeBase': BVHJoint(
                name='RightToeBase',
                offset=[0.0, -5.0, 10.0],
                channels=['Zrotation', 'Xrotation', 'Yrotation'],
                children=[]
            )
        }
        
        # Store frames data
        self.frames_data = []
        self.joint_positions_over_time = []
        
    def process_video(self, video_path: str, output_path: str, 
                     show_preview: bool = False, 
                     max_frames: Optional[int] = None) -> bool:
        """
        Process video file and convert to BVH.
        
        Args:
            video_path: Path to input video file
            output_path: Path for output BVH file
            show_preview: Whether to show preview window
            max_frames: Maximum number of frames to process (None for all)
            
        Returns:
            bool: Success status
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Cannot open video file {video_path}")
            return False
            
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Processing video: {video_path}")
        print(f"FPS: {fps}, Total frames: {total_frames}")
        
        frame_count = 0
        processed_frames = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            if max_frames and frame_count >= max_frames:
                break
                
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process with MediaPipe
            results = self.pose.process(rgb_frame)
            
            if results.pose_world_landmarks:
                # Extract and convert landmarks
                joint_positions = self._process_landmarks(results.pose_world_landmarks.landmark)
                self.joint_positions_over_time.append(joint_positions)
                
                # Calculate rotations and add frame
                self._add_frame(joint_positions)
                processed_frames += 1
                
                # Draw pose on frame if preview is enabled
                if show_preview:
                    self.mp_drawing.draw_landmarks(
                        frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
                    cv2.imshow('MediaPipe Pose', frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
            
            frame_count += 1
            if frame_count % 30 == 0:
                print(f"Processed {frame_count}/{total_frames} frames...")
        
        cap.release()
        cv2.destroyAllWindows()
        
        print(f"Processed {processed_frames} frames with valid poses")
        
        # Apply smoothing if we have enough frames
        if processed_frames > 5:
            self._smooth_animation()
        
        # Write BVH file
        success = self._write_bvh(output_path, fps)
        
        if success:
            print(f"BVH file successfully written to: {output_path}")
        
        return success
    
    def _process_landmarks(self, landmarks) -> Dict[str, np.ndarray]:
        """Convert MediaPipe landmarks to joint positions."""
        joint_positions = {}
        
        for joint_name, mapping in self.joint_mapping.items():
            indices = mapping['indices']
            
            if mapping['type'] == 'direct':
                # Direct mapping from single landmark
                landmark = landmarks[indices[0]]
                joint_positions[joint_name] = np.array([
                    landmark.x * self.scale_factor,
                    landmark.y * self.scale_factor,
                    -landmark.z * self.scale_factor  # Flip Z axis
                ])
                
            elif mapping['type'] == 'average':
                # Average multiple landmarks
                positions = []
                for idx in indices:
                    landmark = landmarks[idx]
                    positions.append([
                        landmark.x * self.scale_factor,
                        landmark.y * self.scale_factor,
                        -landmark.z * self.scale_factor
                    ])
                joint_positions[joint_name] = np.mean(positions, axis=0)
                
            elif mapping['type'] == 'interpolate':
                # Interpolate between landmarks
                if len(indices) == 4:  # Hip to shoulder interpolation
                    hip_left = landmarks[indices[0]]
                    hip_right = landmarks[indices[1]]
                    shoulder_left = landmarks[indices[2]]
                    shoulder_right = landmarks[indices[3]]
                    
                    hip_center = np.array([
                        (hip_left.x + hip_right.x) / 2,
                        (hip_left.y + hip_right.y) / 2,
                        -(hip_left.z + hip_right.z) / 2
                    ]) * self.scale_factor
                    
                    shoulder_center = np.array([
                        (shoulder_left.x + shoulder_right.x) / 2,
                        (shoulder_left.y + shoulder_right.y) / 2,
                        -(shoulder_left.z + shoulder_right.z) / 2
                    ]) * self.scale_factor
                    
                    factor = mapping.get('factor', 0.5)
                    joint_positions[joint_name] = hip_center + (shoulder_center - hip_center) * factor
        
        return joint_positions
    
    def _calculate_rotation_matrix(self, parent_pos: np.ndarray, 
                                 child_pos: np.ndarray, 
                                 reference_vector: np.ndarray = np.array([0, 1, 0])) -> np.ndarray:
        """Calculate rotation matrix from parent to child joint."""
        bone_vector = child_pos - parent_pos
        bone_length = np.linalg.norm(bone_vector)
        
        if bone_length < 1e-6:
            return np.eye(3)
        
        bone_vector = bone_vector / bone_length
        
        # Calculate rotation from reference vector to bone vector
        v = np.cross(reference_vector, bone_vector)
        s = np.linalg.norm(v)
        c = np.dot(reference_vector, bone_vector)
        
        if s < 1e-10:  # Vectors are parallel
            if c > 0:
                return np.eye(3)
            else:
                # 180 degree rotation around perpendicular axis
                perp = np.array([1, 0, 0]) if abs(reference_vector[0]) < 0.9 else np.array([0, 1, 0])
                perp = perp - np.dot(perp, reference_vector) * reference_vector
                perp = perp / np.linalg.norm(perp)
                return 2 * np.outer(perp, perp) - np.eye(3)
        
        vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        rotation_matrix = np.eye(3) + vx + np.dot(vx, vx) * ((1 - c) / (s ** 2))
        
        return rotation_matrix
    
    def _calculate_joint_rotations(self, joint_positions: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Calculate joint rotations from positions."""
        rotations = {}
        
        # Define parent-child relationships
        parent_child_pairs = [
            ('Hips', 'Spine'),
            ('Spine', 'Spine1'),
            ('Spine1', 'Neck'),
            ('Neck', 'Head'),
            ('Neck', 'LeftShoulder'),
            ('LeftShoulder', 'LeftArm'),
            ('LeftArm', 'LeftForeArm'),
            ('LeftForeArm', 'LeftHand'),
            ('Neck', 'RightShoulder'),
            ('RightShoulder', 'RightArm'),
            ('RightArm', 'RightForeArm'),
            ('RightForeArm', 'RightHand'),
            ('Hips', 'LeftUpLeg'),
            ('LeftUpLeg', 'LeftLeg'),
            ('LeftLeg', 'LeftFoot'),
            ('LeftFoot', 'LeftToeBase'),
            ('Hips', 'RightUpLeg'),
            ('RightUpLeg', 'RightLeg'),
            ('RightLeg', 'RightFoot'),
            ('RightFoot', 'RightToeBase')
        ]
        
        # Calculate rotation for each joint
        for parent, child in parent_child_pairs:
            if parent in joint_positions and child in joint_positions:
                # Get positions
                parent_pos = joint_positions[parent]
                child_pos = joint_positions[child]
                
                # Calculate rotation matrix
                rotation_matrix = self._calculate_rotation_matrix(parent_pos, child_pos)
                
                # Convert to Euler angles (ZXY order for BVH)
                rotation = R.from_matrix(rotation_matrix)
                euler_angles = rotation.as_euler('zxy', degrees=True)
                
                rotations[child] = euler_angles
        
        # Special case for root (Hips) - calculate from hip-to-spine orientation
        if 'Hips' in joint_positions and 'Spine' in joint_positions:
            up_vector = joint_positions['Spine'] - joint_positions['Hips']
            up_vector = up_vector / np.linalg.norm(up_vector)
            
            # Default forward direction
            forward = np.array([0, 0, 1])
            right = np.cross(up_vector, forward)
            right = right / np.linalg.norm(right)
            forward = np.cross(right, up_vector)
            
            rotation_matrix = np.column_stack([right, up_vector, forward])
            rotation = R.from_matrix(rotation_matrix)
            euler_angles = rotation.as_euler('zxy', degrees=True)
            
            rotations['Hips'] = euler_angles
        
        return rotations
    
    def _add_frame(self, joint_positions: Dict[str, np.ndarray]):
        """Add a frame of animation data."""
        rotations = self._calculate_joint_rotations(joint_positions)
        
        frame_data = []
        
        # Add root position (Hips)
        if 'Hips' in joint_positions:
            hip_pos = joint_positions['Hips']
            frame_data.extend([hip_pos[0], hip_pos[1], hip_pos[2]])
        else:
            frame_data.extend([0.0, 0.0, 0.0])
        
        # Add rotations in hierarchy order
        joint_order = self._get_joint_order()
        
        for joint_name in joint_order:
            if joint_name in rotations:
                rotation = rotations[joint_name]
                frame_data.extend(rotation)
            else:
                frame_data.extend([0.0, 0.0, 0.0])
        
        self.frames_data.append(frame_data)
    
    def _get_joint_order(self) -> List[str]:
        """Get joints in hierarchy order for frame data."""
        order = []
        
        def traverse(joint_name: str):
            order.append(joint_name)
            joint = self.skeleton_hierarchy.get(joint_name)
            if joint:
                for child in joint.children:
                    traverse(child)
        
        traverse('Hips')
        return order
    
    def _smooth_animation(self, window_size: int = 5):
        """Apply smoothing to reduce jitter."""
        if len(self.frames_data) < window_size:
            return
        
        frames_array = np.array(self.frames_data)
        smoothed = np.zeros_like(frames_array)
        
        # Apply Savitzky-Golay filter to each channel
        for channel in range(frames_array.shape[1]):
            try:
                smoothed[:, channel] = savgol_filter(
                    frames_array[:, channel],
                    window_size,
                    3,  # polynomial order
                    mode='nearest'
                )
            except:
                # If smoothing fails, keep original data
                smoothed[:, channel] = frames_array[:, channel]
        
        self.frames_data = smoothed.tolist()
    
    def _write_bvh(self, filename: str, fps: float = 30.0) -> bool:
        """Write BVH file."""
        try:
            with open(filename, 'w') as f:
                # Write HIERARCHY section
                f.write("HIERARCHY\n")
                self._write_joint(f, 'Hips', 0)
                
                # Write MOTION section
                f.write("MOTION\n")
                f.write(f"Frames: {len(self.frames_data)}\n")
                f.write(f"Frame Time: {1.0/fps:.6f}\n")
                
                # Write frame data
                for frame in self.frames_data:
                    frame_str = " ".join(f"{val:.6f}" for val in frame)
                    f.write(frame_str + "\n")
            
            return True
        except Exception as e:
            print(f"Error writing BVH file: {e}")
            return False
    
    def _write_joint(self, f, joint_name: str, depth: int):
        """Recursively write joint hierarchy."""
        indent = "  " * depth
        joint = self.skeleton_hierarchy.get(joint_name)
        
        if not joint:
            return
        
        # Write joint declaration
        if depth == 0:
            f.write(f"{indent}ROOT {joint_name}\n")
        else:
            f.write(f"{indent}JOINT {joint_name}\n")
        
        f.write(f"{indent}{{\n")
        
        # Write offset
        offset = joint.offset
        f.write(f"{indent}  OFFSET {offset[0]:.6f} {offset[1]:.6f} {offset[2]:.6f}\n")
        
        # Write channels
        channels = joint.channels
        f.write(f"{indent}  CHANNELS {len(channels)} ")
        f.write(" ".join(channels) + "\n")
        
        # Write children
        if joint.children:
            for child_name in joint.children:
                self._write_joint(f, child_name, depth + 1)
        else:
            # Write end site for leaf joints
            f.write(f"{indent}  End Site\n")
            f.write(f"{indent}  {{\n")
            f.write(f"{indent}    OFFSET 0.000000 -10.000000 0.000000\n")
            f.write(f"{indent}  }}\n")
        
        f.write(f"{indent}}}\n")
    
    def cleanup(self):
        """Clean up resources."""
        self.pose.close()


def main():
    parser = argparse.ArgumentParser(
        description='Convert video to BVH motion capture file using MediaPipe'
    )
    parser.add_argument('input', help='Input video file path')
    parser.add_argument('-o', '--output', help='Output BVH file path')
    parser.add_argument('--preview', action='store_true', 
                       help='Show preview window during processing')
    parser.add_argument('--max-frames', type=int, default=None,
                       help='Maximum number of frames to process')
    parser.add_argument('--scale', type=float, default=100.0,
                       help='Scale factor for coordinates (default: 100.0)')
    
    args = parser.parse_args()
    
    # Generate output filename if not provided
    if not args.output:
        base_name = os.path.splitext(os.path.basename(args.input))[0]
        args.output = f"{base_name}_motion.bvh"
    
    # Create converter and process video
    converter = MediaPipeToBVHConverter(scale_factor=args.scale)
    
    try:
        success = converter.process_video(
            args.input, 
            args.output, 
            show_preview=args.preview,
            max_frames=args.max_frames
        )
        
        if success:
            print(f"\nConversion successful!")
            print(f"BVH file saved to: {args.output}")
            print(f"\nTo use in Blender:")
            print(f"1. File > Import > Motion Capture (.bvh)")
            print(f"2. Select the BVH file: {args.output}")
            print(f"3. Adjust scale if needed (try 0.01 for meter-based scenes)")
        else:
            print("\nConversion failed!")
            
    finally:
        converter.cleanup()


if __name__ == "__main__":
    main()