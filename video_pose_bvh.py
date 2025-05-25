import cv2
import mediapipe as mp
import numpy as np
import argparse
from typing import List, Tuple, Dict, Optional
import math
import os
from scipy.spatial.transform import Rotation as R

class MediaPipeToBVH:
    def __init__(self, visualize=False):
        # Initialize MediaPipe
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        self.visualize = visualize
        
        # MediaPipe landmark indices
        self.MP_LANDMARKS = {
            'NOSE': 0,
            'LEFT_EYE_INNER': 1,
            'LEFT_EYE': 2,
            'LEFT_EYE_OUTER': 3,
            'RIGHT_EYE_INNER': 4,
            'RIGHT_EYE': 5,
            'RIGHT_EYE_OUTER': 6,
            'LEFT_EAR': 7,
            'RIGHT_EAR': 8,
            'MOUTH_LEFT': 9,
            'MOUTH_RIGHT': 10,
            'LEFT_SHOULDER': 11,
            'RIGHT_SHOULDER': 12,
            'LEFT_ELBOW': 13,
            'RIGHT_ELBOW': 14,
            'LEFT_WRIST': 15,
            'RIGHT_WRIST': 16,
            'LEFT_PINKY': 17,
            'RIGHT_PINKY': 18,
            'LEFT_INDEX': 19,
            'RIGHT_INDEX': 20,
            'LEFT_THUMB': 21,
            'RIGHT_THUMB': 22,
            'LEFT_HIP': 23,
            'RIGHT_HIP': 24,
            'LEFT_KNEE': 25,
            'RIGHT_KNEE': 26,
            'LEFT_ANKLE': 27,
            'RIGHT_ANKLE': 28,
            'LEFT_HEEL': 29,
            'RIGHT_HEEL': 30,
            'LEFT_FOOT_INDEX': 31,
            'RIGHT_FOOT_INDEX': 32
        }
        
        # Define BVH skeleton hierarchy with MediaPipe mappings
        self.skeleton_hierarchy = {
            'Hips': {
                'parent': None,
                'children': ['Spine', 'LeftUpLeg', 'RightUpLeg'],
                'mp_joint': None,  # Will be calculated as midpoint
                'mp_parent': None,
                'offset': [0, 0, 0],
                'channels': ['Xposition', 'Yposition', 'Zposition', 'Zrotation', 'Xrotation', 'Yrotation']
            },
            'Spine': {
                'parent': 'Hips',
                'children': ['Spine1'],
                'mp_joint': None,  # Midpoint between shoulders
                'mp_parent': None,  # Midpoint between hips
                'offset': [0, 15, 0],
                'channels': ['Zrotation', 'Xrotation', 'Yrotation']
            },
            'Spine1': {
                'parent': 'Spine',
                'children': ['Neck', 'LeftShoulder', 'RightShoulder'],
                'mp_joint': None,
                'mp_parent': None,
                'offset': [0, 15, 0],
                'channels': ['Zrotation', 'Xrotation', 'Yrotation']
            },
            'Neck': {
                'parent': 'Spine1',
                'children': ['Head'],
                'mp_joint': None,  # Midpoint between shoulders
                'mp_parent': None,
                'offset': [0, 10, 0],
                'channels': ['Zrotation', 'Xrotation', 'Yrotation']
            },
            'Head': {
                'parent': 'Neck',
                'children': [],
                'mp_joint': 'NOSE',
                'mp_parent': None,
                'offset': [0, 10, 0],
                'channels': ['Zrotation', 'Xrotation', 'Yrotation']
            },
            'LeftShoulder': {
                'parent': 'Spine1',
                'children': ['LeftArm'],
                'mp_joint': 'LEFT_SHOULDER',
                'mp_parent': None,
                'offset': [-5, 5, 0],
                'channels': ['Zrotation', 'Xrotation', 'Yrotation']
            },
            'LeftArm': {
                'parent': 'LeftShoulder',
                'children': ['LeftForeArm'],
                'mp_joint': 'LEFT_ELBOW',
                'mp_parent': 'LEFT_SHOULDER',
                'offset': [-5, -15, 0],  # Arms down by default
                'channels': ['Zrotation', 'Xrotation', 'Yrotation']
            },
            'LeftForeArm': {
                'parent': 'LeftArm',
                'children': ['LeftHand'],
                'mp_joint': 'LEFT_WRIST',
                'mp_parent': 'LEFT_ELBOW',
                'offset': [-3, -15, 0],  # Arms down by default
                'channels': ['Zrotation', 'Xrotation', 'Yrotation']
            },
            'LeftHand': {
                'parent': 'LeftForeArm',
                'children': [],
                'mp_joint': 'LEFT_INDEX',
                'mp_parent': 'LEFT_WRIST',
                'offset': [-2, -7, 0],  # Arms down by default
                'channels': ['Zrotation', 'Xrotation', 'Yrotation']
            },
            'RightShoulder': {
                'parent': 'Spine1',
                'children': ['RightArm'],
                'mp_joint': 'RIGHT_SHOULDER',
                'mp_parent': None,
                'offset': [5, 5, 0],
                'channels': ['Zrotation', 'Xrotation', 'Yrotation']
            },
            'RightArm': {
                'parent': 'RightShoulder',
                'children': ['RightForeArm'],
                'mp_joint': 'RIGHT_ELBOW',
                'mp_parent': 'RIGHT_SHOULDER',
                'offset': [5, -15, 0],  # Arms down by default
                'channels': ['Zrotation', 'Xrotation', 'Yrotation']
            },
            'RightForeArm': {
                'parent': 'RightArm',
                'children': ['RightHand'],
                'mp_joint': 'RIGHT_WRIST',
                'mp_parent': 'RIGHT_ELBOW',
                'offset': [3, -15, 0],  # Arms down by default
                'channels': ['Zrotation', 'Xrotation', 'Yrotation']
            },
            'RightHand': {
                'parent': 'RightForeArm',
                'children': [],
                'mp_joint': 'RIGHT_INDEX',
                'mp_parent': 'RIGHT_WRIST',
                'offset': [2, -7, 0],  # Arms down by default
                'channels': ['Zrotation', 'Xrotation', 'Yrotation']
            },
            'LeftUpLeg': {
                'parent': 'Hips',
                'children': ['LeftLeg'],
                'mp_joint': 'LEFT_HIP',
                'mp_parent': None,
                'offset': [-5, 0, 0],
                'channels': ['Zrotation', 'Xrotation', 'Yrotation']
            },
            'LeftLeg': {
                'parent': 'LeftUpLeg',
                'children': ['LeftFoot'],
                'mp_joint': 'LEFT_KNEE',
                'mp_parent': 'LEFT_HIP',
                'offset': [0, -20, 0],
                'channels': ['Zrotation', 'Xrotation', 'Yrotation']
            },
            'LeftFoot': {
                'parent': 'LeftLeg',
                'children': [],
                'mp_joint': 'LEFT_ANKLE',
                'mp_parent': 'LEFT_KNEE',
                'offset': [0, -20, 0],
                'channels': ['Zrotation', 'Xrotation', 'Yrotation']
            },
            'RightUpLeg': {
                'parent': 'Hips',
                'children': ['RightLeg'],
                'mp_joint': 'RIGHT_HIP',
                'mp_parent': None,
                'offset': [5, 0, 0],
                'channels': ['Zrotation', 'Xrotation', 'Yrotation']
            },
            'RightLeg': {
                'parent': 'RightUpLeg',
                'children': ['RightFoot'],
                'mp_joint': 'RIGHT_KNEE',
                'mp_parent': 'RIGHT_HIP',
                'offset': [0, -20, 0],
                'channels': ['Zrotation', 'Xrotation', 'Yrotation']
            },
            'RightFoot': {
                'parent': 'RightLeg',
                'children': [],
                'mp_joint': 'RIGHT_ANKLE',
                'mp_parent': 'RIGHT_KNEE',
                'offset': [0, -20, 0],
                'channels': ['Zrotation', 'Xrotation', 'Yrotation']
            }
        }
        
        self.frames_data = []
        self.rest_pose = None
        
    def process_video(self, video_path: str, output_path: str):
        """Process video and generate BVH file"""
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = 0
        
        # Get video properties for visualization
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Create video writer if visualization is enabled
        if self.visualize:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out_video_path = output_path.replace('.bvh', '_pose.mp4')
            out = cv2.VideoWriter(out_video_path, fourcc, fps, (width, height))
        
        print(f"Processing video: {video_path}")
        print(f"FPS: {fps}")
        print(f"Resolution: {width}x{height}")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process the frame
            results = self.pose.process(rgb_frame)
            
            if results.pose_world_landmarks:
                # Extract frame data
                frame_data = self.extract_frame_data(results.pose_world_landmarks)
                self.frames_data.append(frame_data)
                frame_count += 1
                
                # Visualize if enabled
                if self.visualize:
                    # Draw pose landmarks
                    self.mp_drawing.draw_landmarks(
                        frame,
                        results.pose_landmarks,
                        self.mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
                    )
                    
                    # Add frame info
                    cv2.putText(frame, f'Frame: {frame_count}', (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    
                    # Write frame
                    out.write(frame)
                
                if frame_count % 30 == 0:
                    print(f"Processed {frame_count} frames...")
            else:
                print(f"Warning: No pose detected in frame {frame_count}")
        
        cap.release()
        if self.visualize:
            out.release()
            print(f"Visualization saved to: {out_video_path}")
        
        print(f"Total frames processed: {frame_count}")
        print(f"Frames with pose data: {len(self.frames_data)}")
        
        # Generate BVH file
        if self.frames_data:
            self.write_bvh(output_path, fps)
            print(f"BVH file saved to: {output_path}")
        else:
            print("No pose data detected in video!")
    
    def get_joint_position(self, landmarks, joint_name: str) -> np.ndarray:
        """Get 3D position of a joint from MediaPipe landmarks"""
        if joint_name in self.MP_LANDMARKS:
            idx = self.MP_LANDMARKS[joint_name]
            landmark = landmarks.landmark[idx]
            # Convert to cm and adjust coordinate system
            # MediaPipe: X-right, Y-down, Z-forward
            # BVH: X-right, Y-up, Z-forward
            return np.array([landmark.x * 100, -landmark.y * 100, landmark.z * 100])
        return None
    
    def extract_frame_data(self, landmarks) -> Dict:
        """Extract pose data for a single frame with proper rotation calculations"""
        frame_data = {}
        
        # Get all joint positions
        joint_positions = {}
        for name, idx in self.MP_LANDMARKS.items():
            landmark = landmarks.landmark[idx]
            # Convert coordinates: MediaPipe Y-down to BVH Y-up
            joint_positions[name] = np.array([
                landmark.x * 100,
                -landmark.y * 100,
                landmark.z * 100
            ])
        
        # Calculate special joint positions
        joint_positions['HIPS'] = (joint_positions['LEFT_HIP'] + joint_positions['RIGHT_HIP']) / 2
        joint_positions['SPINE'] = (joint_positions['LEFT_SHOULDER'] + joint_positions['RIGHT_SHOULDER']) / 2
        joint_positions['NECK'] = joint_positions['SPINE'] + np.array([0, 5, 0])
        
        # Store rest pose on first frame
        if self.rest_pose is None:
            self.rest_pose = joint_positions.copy()
            # Calculate initial bone orientations for better offset estimation
            self._calculate_initial_offsets(joint_positions)
            
            # For first frame, all rotations should be zero since it's the rest pose
            for joint_name in self.skeleton_hierarchy:
                if joint_name == 'Hips':
                    frame_data[joint_name] = {
                        'position': joint_positions['HIPS'],
                        'rotation': np.array([0, 0, 0])
                    }
                else:
                    frame_data[joint_name] = {
                        'rotation': np.array([0, 0, 0])
                    }
            return frame_data
        
        # Calculate rotations for each joint
        for joint_name, joint_info in self.skeleton_hierarchy.items():
            if joint_name == 'Hips':
                # Root position
                frame_data[joint_name] = {
                    'position': joint_positions['HIPS'],
                    'rotation': self.calculate_root_rotation(joint_positions)
                }
            else:
                # Calculate joint rotation
                rotation = self.calculate_joint_rotation_improved(
                    joint_name, 
                    joint_info, 
                    joint_positions
                )
                frame_data[joint_name] = {
                    'rotation': rotation
                }
        
        return frame_data
    
    def calculate_root_rotation(self, joint_positions: Dict) -> np.ndarray:
        """Calculate root (hip) rotation"""
        # Calculate hip orientation based on hip-to-hip vector
        left_hip = joint_positions['LEFT_HIP']
        right_hip = joint_positions['RIGHT_HIP']
        
        # Hip right vector (from left to right hip)
        hip_right = right_hip - left_hip
        hip_right = hip_right / np.linalg.norm(hip_right)
        
        # Hip forward vector (perpendicular to hip-shoulder plane)
        left_shoulder = joint_positions['LEFT_SHOULDER']
        right_shoulder = joint_positions['RIGHT_SHOULDER']
        
        shoulder_center = (left_shoulder + right_shoulder) / 2
        hip_center = (left_hip + right_hip) / 2
        
        hip_up = shoulder_center - hip_center
        hip_up = hip_up / np.linalg.norm(hip_up)
        
        # Calculate forward vector
        hip_forward = np.cross(hip_right, hip_up)
        hip_forward = hip_forward / np.linalg.norm(hip_forward)
        
        # Recalculate up vector to ensure orthogonality
        hip_up = np.cross(hip_forward, hip_right)
        
        # Create rotation matrix
        rotation_matrix = np.array([
            hip_right,
            hip_up,
            hip_forward
        ]).T
        
        # Convert to Euler angles (ZYX order for BVH)
        euler_angles = self.rotation_matrix_to_euler_zyx(rotation_matrix)
        
        return euler_angles
    
    def _calculate_initial_offsets(self, joint_positions: Dict):
        """Calculate actual offsets based on first frame pose"""
        for joint_name, joint_info in self.skeleton_hierarchy.items():
            if joint_info['parent'] is None:
                continue
                
            # Get joint and parent positions
            joint_pos = self._get_joint_position(joint_name, joint_positions)
            parent_pos = self._get_joint_position(joint_info['parent'], joint_positions)
            
            if joint_pos is not None and parent_pos is not None:
                # Calculate actual offset from first frame
                offset = joint_pos - parent_pos
                # Only update if offset is significant
                if np.linalg.norm(offset) > 0.1:
                    joint_info['offset'] = offset.tolist()
    
    def _get_joint_position(self, joint_name: str, joint_positions: Dict) -> Optional[np.ndarray]:
        """Helper to get joint position with proper mapping"""
        joint_info = self.skeleton_hierarchy.get(joint_name, {})
        
        # Direct mapping
        if joint_info.get('mp_joint') and joint_info['mp_joint'] in joint_positions:
            return joint_positions[joint_info['mp_joint']]
        
        # Special cases
        if joint_name == 'Hips':
            return joint_positions.get('HIPS')
        elif joint_name in ['Spine', 'Spine1']:
            return joint_positions.get('SPINE')
        elif joint_name == 'Neck':
            return joint_positions.get('NECK')
        
        return None
    
    def calculate_joint_rotation_improved(self, joint_name: str, joint_info: Dict, 
                                        joint_positions: Dict) -> np.ndarray:
        """Improved rotation calculation that handles arbitrary poses"""
        parent_name = joint_info['parent']
        
        if parent_name is None:
            return np.array([0, 0, 0])
        
        # Get current positions
        current_pos = self._get_joint_position(joint_name, joint_positions)
        parent_pos = self._get_joint_position(parent_name, joint_positions)
        
        # Get rest pose positions
        rest_current = self._get_joint_position(joint_name, self.rest_pose)
        rest_parent = self._get_joint_position(parent_name, self.rest_pose)
        
        if any(p is None for p in [current_pos, parent_pos, rest_current, rest_parent]):
            return np.array([0, 0, 0])
        
        # Calculate bone vectors
        current_bone = current_pos - parent_pos
        rest_bone = rest_current - rest_parent
        
        # Normalize
        current_length = np.linalg.norm(current_bone)
        rest_length = np.linalg.norm(rest_bone)
        
        if current_length < 0.001 or rest_length < 0.001:
            return np.array([0, 0, 0])
        
        current_bone = current_bone / current_length
        rest_bone = rest_bone / rest_length
        
        # For limbs, we need to consider the plane of rotation
        if joint_name in ['LeftForeArm', 'RightForeArm', 'LeftLeg', 'RightLeg']:
            # Get grandparent for plane calculation
            grandparent_name = self.skeleton_hierarchy[parent_name]['parent']
            if grandparent_name:
                grandparent_pos = self._get_joint_position(grandparent_name, joint_positions)
                rest_grandparent = self._get_joint_position(grandparent_name, self.rest_pose)
                
                if grandparent_pos is not None and rest_grandparent is not None:
                    # Calculate rotation that preserves the limb plane
                    return self.calculate_limb_rotation(
                        grandparent_pos, parent_pos, current_pos,
                        rest_grandparent, rest_parent, rest_current
                    )
        
        # For other joints, use standard rotation
        rotation_matrix = self.calculate_rotation_matrix(rest_bone, current_bone)
        euler_angles = self.rotation_matrix_to_euler_zyx(rotation_matrix)
        
        return euler_angles
    
    def calculate_limb_rotation(self, grandparent: np.ndarray, parent: np.ndarray, 
                               current: np.ndarray, rest_grandparent: np.ndarray,
                               rest_parent: np.ndarray, rest_current: np.ndarray) -> np.ndarray:
        """Calculate rotation for limb joints (elbows, knees) preserving natural constraints"""
        # Calculate the rotation of the parent bone first
        parent_bone = parent - grandparent
        rest_parent_bone = rest_parent - rest_grandparent
        
        parent_bone = parent_bone / np.linalg.norm(parent_bone)
        rest_parent_bone = rest_parent_bone / np.linalg.norm(rest_parent_bone)
        
        parent_rotation = self.calculate_rotation_matrix(rest_parent_bone, parent_bone)
        
        # Apply parent rotation to rest child bone
        rest_child_bone = rest_current - rest_parent
        rest_child_bone = rest_child_bone / np.linalg.norm(rest_child_bone)
        expected_child = np.dot(parent_rotation, rest_child_bone)
        
        # Current child bone
        current_child_bone = current - parent
        current_child_bone = current_child_bone / np.linalg.norm(current_child_bone)
        
        # Calculate the additional rotation needed
        local_rotation = self.calculate_rotation_matrix(expected_child, current_child_bone)
        
        # Convert to Euler angles
        euler_angles = self.rotation_matrix_to_euler_zyx(local_rotation)
        
        # Apply constraints for natural limb movement
        # Elbows and knees primarily bend in one direction
        if 'Arm' in rest_parent_bone.__str__() or 'Leg' in rest_parent_bone.__str__():
            # Limit the rotation to realistic ranges
            euler_angles[1] = np.clip(euler_angles[1], -150, 10)  # X rotation (main bend)
            euler_angles[0] = np.clip(euler_angles[0], -45, 45)   # Z rotation (twist)
            euler_angles[2] = np.clip(euler_angles[2], -45, 45)   # Y rotation (side)
        
        return euler_angles
        """Calculate rotation for a specific joint using proper 3D math"""
        parent_name = joint_info['parent']
        
        if parent_name is None:
            return np.array([0, 0, 0])
        
        # Get current joint position
        if joint_info['mp_joint']:
            if joint_info['mp_joint'] in joint_positions:
                current_pos = joint_positions[joint_info['mp_joint']]
            else:
                current_pos = joint_positions.get(joint_info['mp_joint'].replace('_', ''), None)
        else:
            # Handle special cases
            if joint_name in ['Spine', 'Spine1', 'Neck']:
                current_pos = joint_positions.get(joint_name.upper(), 
                                                joint_positions.get('SPINE', None))
            else:
                current_pos = joint_positions.get('HIPS')
        
        # Get parent position
        parent_info = self.skeleton_hierarchy[parent_name]
        if parent_info['mp_joint']:
            if parent_info['mp_joint'] in joint_positions:
                parent_pos = joint_positions[parent_info['mp_joint']]
            else:
                parent_pos = joint_positions.get(parent_info['mp_joint'].replace('_', ''), None)
        else:
            # Handle special cases
            if parent_name == 'Hips':
                parent_pos = joint_positions['HIPS']
            elif parent_name in ['Spine', 'Spine1']:
                parent_pos = joint_positions.get(parent_name.upper(), 
                                               joint_positions.get('SPINE', None))
            else:
                parent_pos = joint_positions.get('HIPS')
        
        if current_pos is None or parent_pos is None:
            return np.array([0, 0, 0])
        
        # Calculate bone vector
        bone_vector = current_pos - parent_pos
        bone_length = np.linalg.norm(bone_vector)
        
        if bone_length < 0.001:
            return np.array([0, 0, 0])
        
        bone_vector = bone_vector / bone_length
        
        # Get reference vector (rest pose direction)
        rest_offset = np.array(joint_info['offset'])
        rest_length = np.linalg.norm(rest_offset)
        if rest_length > 0:
            rest_direction = rest_offset / rest_length
        else:
            rest_direction = np.array([0, -1, 0])
        
        # Calculate rotation from rest pose to current pose
        rotation_matrix = self.calculate_rotation_matrix(rest_direction, bone_vector)
        
        # Convert to Euler angles (ZYX order for BVH)
        euler_angles = self.rotation_matrix_to_euler_zyx(rotation_matrix)
        
        return euler_angles
    
    def calculate_rotation_matrix(self, vec_from: np.ndarray, vec_to: np.ndarray) -> np.ndarray:
        """Calculate rotation matrix that rotates vec_from to vec_to"""
        # Normalize vectors
        vec_from = vec_from / np.linalg.norm(vec_from)
        vec_to = vec_to / np.linalg.norm(vec_to)
        
        # Calculate rotation axis
        rotation_axis = np.cross(vec_from, vec_to)
        rotation_axis_length = np.linalg.norm(rotation_axis)
        
        # Check if vectors are parallel
        if rotation_axis_length < 0.001:
            # Vectors are parallel
            if np.dot(vec_from, vec_to) > 0:
                # Same direction
                return np.eye(3)
            else:
                # Opposite direction - rotate 180 degrees around any perpendicular axis
                # Find a perpendicular axis
                if abs(vec_from[0]) < 0.9:
                    perp_axis = np.array([1, 0, 0])
                else:
                    perp_axis = np.array([0, 1, 0])
                
                rotation_axis = np.cross(vec_from, perp_axis)
                rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
                
                # 180 degree rotation matrix
                return 2 * np.outer(rotation_axis, rotation_axis) - np.eye(3)
        
        rotation_axis = rotation_axis / rotation_axis_length
        
        # Calculate rotation angle
        cos_angle = np.clip(np.dot(vec_from, vec_to), -1.0, 1.0)
        angle = np.arccos(cos_angle)
        
        # Rodrigues' rotation formula
        K = np.array([
            [0, -rotation_axis[2], rotation_axis[1]],
            [rotation_axis[2], 0, -rotation_axis[0]],
            [-rotation_axis[1], rotation_axis[0], 0]
        ])
        
        rotation_matrix = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * np.dot(K, K)
        
        return rotation_matrix
    
    def rotation_matrix_to_euler_zyx(self, R: np.ndarray) -> np.ndarray:
        """Convert rotation matrix to Euler angles in ZYX order (BVH standard)"""
        # Check for gimbal lock
        sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
        
        singular = sy < 1e-6
        
        if not singular:
            x = math.atan2(R[2, 1], R[2, 2])
            y = math.atan2(-R[2, 0], sy)
            z = math.atan2(R[1, 0], R[0, 0])
        else:
            x = math.atan2(-R[1, 2], R[1, 1])
            y = math.atan2(-R[2, 0], sy)
            z = 0
        
        # Convert to degrees
        return np.array([
            math.degrees(z),
            math.degrees(x),
            math.degrees(y)
        ])
    
    def write_bvh(self, output_path: str, fps: float):
        """Write BVH file with proper formatting"""
        with open(output_path, 'w') as f:
            # Write hierarchy
            f.write("HIERARCHY\n")
            self._write_joint_hierarchy(f, 'Hips', 0)
            
            # Write motion data
            f.write("\nMOTION\n")
            f.write(f"Frames: {len(self.frames_data)}\n")
            f.write(f"Frame Time: {1.0/fps:.6f}\n")
            
            # Write frame data
            for frame_idx, frame_data in enumerate(self.frames_data):
                frame_values = []
                
                # Process each joint in hierarchy order
                self._collect_frame_values('Hips', frame_data, frame_values)
                
                # Write frame line
                f.write(' '.join([f"{v:.6f}" for v in frame_values]) + '\n')
    
    def _write_joint_hierarchy(self, f, joint_name: str, indent: int):
        """Recursively write joint hierarchy"""
        indent_str = '  ' * indent
        joint_info = self.skeleton_hierarchy[joint_name]
        
        # Write joint declaration
        if joint_info['parent'] is None:
            f.write(f"{indent_str}ROOT {joint_name}\n")
        else:
            f.write(f"{indent_str}JOINT {joint_name}\n")
        
        f.write(f"{indent_str}{{\n")
        
        # Write offset
        offset = joint_info['offset']
        f.write(f"{indent_str}  OFFSET {offset[0]:.6f} {offset[1]:.6f} {offset[2]:.6f}\n")
        
        # Write channels
        channels = joint_info['channels']
        f.write(f"{indent_str}  CHANNELS {len(channels)} {' '.join(channels)}\n")
        
        # Write children
        children = joint_info['children']
        if children:
            for child in children:
                self._write_joint_hierarchy(f, child, indent + 1)
        else:
            # End effector
            f.write(f"{indent_str}  End Site\n")
            f.write(f"{indent_str}  {{\n")
            f.write(f"{indent_str}    OFFSET 0.0 -5.0 0.0\n")
            f.write(f"{indent_str}  }}\n")
        
        f.write(f"{indent_str}}}\n")
    
    def _collect_frame_values(self, joint_name: str, frame_data: Dict, values: List):
        """Recursively collect frame values in hierarchy order"""
        if joint_name in frame_data:
            data = frame_data[joint_name]
            
            # Add position for root joint
            if 'position' in data:
                pos = data['position']
                values.extend([pos[0], pos[1], pos[2]])
            
            # Add rotation
            if 'rotation' in data:
                rot = data['rotation']
                values.extend([rot[0], rot[1], rot[2]])
        else:
            # Default values if joint data is missing
            joint_info = self.skeleton_hierarchy[joint_name]
            if 'position' in joint_info.get('channels', []):
                values.extend([0.0, 0.0, 0.0])
            values.extend([0.0, 0.0, 0.0])  # rotation
        
        # Process children
        joint_info = self.skeleton_hierarchy[joint_name]
        for child in joint_info['children']:
            self._collect_frame_values(child, frame_data, values)


def main():
    parser = argparse.ArgumentParser(description='Convert video to BVH using MediaPipe')
    parser.add_argument('--video', help='Path to input video file')
    parser.add_argument('--output', help='Path to output BVH file')
    parser.add_argument('--visualize', '-v', action='store_true',
                       help='Generate visualization video with pose overlay')
    
    args = parser.parse_args()
    
    # Validate input
    if not os.path.exists(args.video):
        print(f"Error: Input video '{args.video}' not found!")
        return
    
    # Create output directory if needed
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Create converter and process video
    converter = MediaPipeToBVH(visualize=args.visualize)
    converter.process_video(args.video, args.output)

if __name__ == "__main__":
    main()

"""
IMPROVED MEDIAPIPE TO BVH CONVERTER

KEY IMPROVEMENTS:
1. Proper 3D rotation calculations using rotation matrices
2. Correct coordinate system conversion (MediaPipe Y-down to BVH Y-up)
3. Hierarchical skeleton structure matching BVH standards
4. Rotation matrix to Euler angle conversion (ZYX order for BVH)
5. Better joint mapping between MediaPipe and BVH skeleton
6. First frame calibration - uses actual pose instead of T-pose
7. Natural arm positioning (arms down by default, not T-pose)
8. Improved limb rotation handling with constraints

USAGE:
python video_to_bvh.py input_video.mp4 output_motion.bvh --visualize

REQUIREMENTS:
pip install opencv-python mediapipe numpy scipy

TECHNICAL DETAILS:
- Uses Rodrigues' rotation formula for calculating rotation matrices
- Implements proper parent-child relationships in skeleton hierarchy
- Handles gimbal lock in Euler angle conversion
- Scales positions to centimeters for standard BVH units
- First frame is used as rest pose with zero rotations
- Subsequent frames calculate rotations relative to first frame

NOTES:
- The rotation calculations now properly account for 3D orientations
- Joint offsets are calculated from actual first frame pose
- Arms start in natural down position instead of T-pose
- The output BVH should work better in 3D animation software
- Some manual cleanup may still be needed for production use

FUTURE IMPROVEMENTS:
- Add temporal smoothing to reduce jitter
- Implement bone length constraints
- Add support for hand and face tracking
- Improve foot contact detection
- Add IK constraints for more natural motion
"""