import cv2
import numpy as np
import torch
import os
from pathlib import Path
import logging
from typing import List, Dict, Tuple, Optional
import argparse
from scipy.spatial.transform import Rotation as R
from easy_ViTPose import VitInference

class MathematicallyCorrectMP4ToBVH:
    """
    Video to BVH converter using proper mathematical formulas for joint rotations
    Based on: pos_j = pos_P(j) + R_P(j) · offset_j
    Where BVH stores: R_P(j)^(-1) × R_j
    """
    
    def __init__(self, vitpose_model_path: str, yolo_model_path: str, 
                 model_size: str = 'b', device: str = 'cuda'):
        self.device = device
        self.model_size = model_size
        self.setup_logging()
        
        # Initialize ViTPose model
        self.pose_model = VitInference(
            vitpose_model_path,
            yolo_model_path,
            model_name=model_size,
            yolo_size=320,
            is_video=True,
            device=device
        )
        
        # Setup skeleton
        self.setup_skeleton()
        
        # Reference T-pose for offset calculation
        self.reference_pose = self.create_reference_pose()
        
    def setup_logging(self):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def setup_skeleton(self):
        """Define skeleton hierarchy and properties"""
        # COCO to BVH joint mapping
        self.coco_to_bvh = {
            'nose': 0, 'left_eye': 1, 'right_eye': 2, 
            'left_ear': 3, 'right_ear': 4,
            'left_shoulder': 5, 'right_shoulder': 6,
            'left_elbow': 7, 'right_elbow': 8,
            'left_wrist': 9, 'right_wrist': 10,
            'left_hip': 11, 'right_hip': 12,
            'left_knee': 13, 'right_knee': 14,
            'left_ankle': 15, 'right_ankle': 16
        }
        
        # BVH skeleton hierarchy
        self.skeleton_hierarchy = {
            'Hips': {
                'parent': None,
                'children': ['Spine'],
                'channels': ['Xposition', 'Yposition', 'Zposition', 
                            'Zrotation', 'Xrotation', 'Yrotation']
            },
            'Spine': {
                'parent': 'Hips',
                'children': ['Spine1'],
                'channels': ['Zrotation', 'Xrotation', 'Yrotation']
            },
            'Spine1': {
                'parent': 'Spine',
                'children': ['Spine2'],
                'channels': ['Zrotation', 'Xrotation', 'Yrotation']
            },
            'Spine2': {
                'parent': 'Spine1',
                'children': ['Neck', 'LeftShoulder', 'RightShoulder'],
                'channels': ['Zrotation', 'Xrotation', 'Yrotation']
            },
            'Neck': {
                'parent': 'Spine2',
                'children': ['Head'],
                'channels': ['Zrotation', 'Xrotation', 'Yrotation']
            },
            'Head': {
                'parent': 'Neck',
                'children': [],
                'channels': ['Zrotation', 'Xrotation', 'Yrotation']
            },
            # Left arm
            'LeftShoulder': {
                'parent': 'Spine2',
                'children': ['LeftArm'],
                'channels': ['Zrotation', 'Xrotation', 'Yrotation']
            },
            'LeftArm': {
                'parent': 'LeftShoulder',
                'children': ['LeftForeArm'],
                'channels': ['Zrotation', 'Xrotation', 'Yrotation']
            },
            'LeftForeArm': {
                'parent': 'LeftArm',
                'children': ['LeftHand'],
                'channels': ['Zrotation', 'Xrotation', 'Yrotation']
            },
            'LeftHand': {
                'parent': 'LeftForeArm',
                'children': [],
                'channels': ['Zrotation', 'Xrotation', 'Yrotation']
            },
            # Right arm
            'RightShoulder': {
                'parent': 'Spine2',
                'children': ['RightArm'],
                'channels': ['Zrotation', 'Xrotation', 'Yrotation']
            },
            'RightArm': {
                'parent': 'RightShoulder',
                'children': ['RightForeArm'],
                'channels': ['Zrotation', 'Xrotation', 'Yrotation']
            },
            'RightForeArm': {
                'parent': 'RightArm',
                'children': ['RightHand'],
                'channels': ['Zrotation', 'Xrotation', 'Yrotation']
            },
            'RightHand': {
                'parent': 'RightForeArm',
                'children': [],
                'channels': ['Zrotation', 'Xrotation', 'Yrotation']
            },
            # Legs handled separately due to Hips having multiple children
            'LeftUpLeg': {
                'parent': 'Hips',
                'children': ['LeftLeg'],
                'channels': ['Zrotation', 'Xrotation', 'Yrotation']
            },
            'LeftLeg': {
                'parent': 'LeftUpLeg',
                'children': ['LeftFoot'],
                'channels': ['Zrotation', 'Xrotation', 'Yrotation']
            },
            'LeftFoot': {
                'parent': 'LeftLeg',
                'children': [],
                'channels': ['Zrotation', 'Xrotation', 'Yrotation']
            },
            'RightUpLeg': {
                'parent': 'Hips',
                'children': ['RightLeg'],
                'channels': ['Zrotation', 'Xrotation', 'Yrotation']
            },
            'RightLeg': {
                'parent': 'RightUpLeg',
                'children': ['RightFoot'],
                'channels': ['Zrotation', 'Xrotation', 'Yrotation']
            },
            'RightFoot': {
                'parent': 'RightLeg',
                'children': [],
                'channels': ['Zrotation', 'Xrotation', 'Yrotation']
            }
        }
        
        # Joint order for BVH output
        self.joint_order = [
            'Hips', 'Spine', 'Spine1', 'Spine2', 'Neck', 'Head',
            'LeftShoulder', 'LeftArm', 'LeftForeArm', 'LeftHand',
            'RightShoulder', 'RightArm', 'RightForeArm', 'RightHand',
            'LeftUpLeg', 'LeftLeg', 'LeftFoot',
            'RightUpLeg', 'RightLeg', 'RightFoot'
        ]
        
        # Update Hips children to include legs
        self.skeleton_hierarchy['Hips']['children'] = ['Spine', 'LeftUpLeg', 'RightUpLeg']
    
    def create_reference_pose(self):
        """Create T-pose reference for offset calculation"""
        ref_pose = {}
        
        # Standard T-pose positions (relative to hip at origin)
        ref_pose['Hips'] = np.array([0.0, 0.0, 0.0])
        ref_pose['Spine'] = np.array([0.0, 10.0, 0.0])
        ref_pose['Spine1'] = np.array([0.0, 20.0, 0.0])
        ref_pose['Spine2'] = np.array([0.0, 30.0, 0.0])
        ref_pose['Neck'] = np.array([0.0, 40.0, 0.0])
        ref_pose['Head'] = np.array([0.0, 50.0, 0.0])
        
        # Arms (T-pose)
        ref_pose['LeftShoulder'] = np.array([-5.0, 35.0, 0.0])
        ref_pose['LeftArm'] = np.array([-15.0, 35.0, 0.0])
        ref_pose['LeftForeArm'] = np.array([-25.0, 35.0, 0.0])
        ref_pose['LeftHand'] = np.array([-35.0, 35.0, 0.0])
        
        ref_pose['RightShoulder'] = np.array([5.0, 35.0, 0.0])
        ref_pose['RightArm'] = np.array([15.0, 35.0, 0.0])
        ref_pose['RightForeArm'] = np.array([25.0, 35.0, 0.0])
        ref_pose['RightHand'] = np.array([35.0, 35.0, 0.0])
        
        # Legs
        ref_pose['LeftUpLeg'] = np.array([0.0, 0.0, 0.0])
        ref_pose['LeftLeg'] = np.array([0.0, 0.0, 0.0])
        ref_pose['LeftFoot'] = np.array([0.0, 0.0, 0.0])
        
        ref_pose['RightUpLeg'] = np.array([5.0, -5.0, 0.0])
        ref_pose['RightLeg'] = np.array([5.0, -25.0, 0.0])
        ref_pose['RightFoot'] = np.array([5.0, -45.0, 0.0])
        
        return ref_pose
    
    def calculate_bone_offsets(self):
        """Calculate bone offsets from reference pose"""
        offsets = {}
        
        for joint_name in self.joint_order:
            joint_info = self.skeleton_hierarchy[joint_name]
            
            if joint_info['parent'] is None:
                # Root joint
                offsets[joint_name] = np.array([0.0, 0.0, 0.0])
            else:
                # Child joint - offset from parent
                parent_name = joint_info['parent']
                offset = self.reference_pose[joint_name] - self.reference_pose[parent_name]
                offsets[joint_name] = offset
        
        return offsets
    
    def extract_frames_from_video(self, video_path: str, 
                                 max_frames: Optional[int] = None) -> Tuple[List[np.ndarray], float]:
        """Extract frames from video"""
        frames = []
        cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
        
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        self.video_dims = (width, height)
        frame_count = 0
        
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(rgb_frame)
                frame_count += 1
                
                if max_frames and frame_count >= max_frames:
                    break
        finally:
            cap.release()
        
        self.logger.info(f"Extracted {len(frames)} frames at {fps} FPS")
        return frames, fps
    
    def estimate_poses_from_frames(self, frames: List[np.ndarray]) -> List[Dict]:
        """Extract 2D poses from frames"""
        pose_data = []
        
        for i, frame in enumerate(frames):
            try:
                results = self.pose_model.inference(frame)
                
                frame_poses = {}
                if results:
                    for person_id, keypoints in results.items():
                        # Convert to [x, y, conf] format
                        processed_kp = []
                        for kp in keypoints:
                            y, x, conf = kp
                            processed_kp.append([x, y, conf])
                        
                        frame_poses[person_id] = {
                            'keypoints': np.array(processed_kp),
                            'confidence': np.mean([kp[2] for kp in processed_kp])
                        }
                
                pose_data.append(frame_poses)
                
            except Exception as e:
                self.logger.error(f"Error in frame {i}: {str(e)}")
                pose_data.append({})
        
        self.pose_model.reset()
        return pose_data
    
    def lift_2d_to_3d(self, keypoints_2d: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Lift 2D keypoints to 3D positions
        This is a simplified version - for better results use VideoPose3D
        """
        # Normalize 2D keypoints
        hip_center_2d = (keypoints_2d[11][:2] + keypoints_2d[12][:2]) / 2
        normalized_kp = keypoints_2d.copy()
        
        # Center on hips
        for i in range(len(keypoints_2d)):
            normalized_kp[i][:2] -= hip_center_2d
        
        # Scale normalization
        torso_length = np.linalg.norm(
            (keypoints_2d[5][:2] + keypoints_2d[6][:2]) / 2 - hip_center_2d
        )
        if torso_length > 0:
            scale = 30.0 / torso_length  # Reference torso length
        else:
            scale = 1.0
        
        # Convert to 3D with depth estimation
        positions_3d = {}
        
        # Hip center (root)
        positions_3d['Hips'] = np.array([0.0, 0.0, 0.0])
        
        # Spine chain
        shoulder_center = (normalized_kp[5][:2] + normalized_kp[6][:2]) / 2 * scale
        positions_3d['Spine'] = np.array([shoulder_center[0] * 0.3, 10.0, 0.0])
        positions_3d['Spine1'] = np.array([shoulder_center[0] * 0.6, 20.0, 0.0])
        positions_3d['Spine2'] = np.array([shoulder_center[0], 30.0, 0.0])
        
        # Head/Neck
        if keypoints_2d[0][2] > 0.3:  # Nose visible
            head_2d = normalized_kp[0][:2] * scale
            positions_3d['Neck'] = np.array([head_2d[0] * 0.8, 40.0, 0.0])
            positions_3d['Head'] = np.array([head_2d[0], 50.0, 0.0])
        else:
            positions_3d['Neck'] = np.array([shoulder_center[0], 40.0, 0.0])
            positions_3d['Head'] = np.array([shoulder_center[0], 50.0, 0.0])
        
        # Arms
        # Left arm
        positions_3d['LeftShoulder'] = np.array([
            normalized_kp[5][0] * scale, 35.0, normalized_kp[5][1] * scale * 0.1
        ])
        positions_3d['LeftArm'] = np.array([
            normalized_kp[7][0] * scale, 35.0 + normalized_kp[7][1] * scale * 0.3,
            normalized_kp[7][1] * scale * 0.2
        ])
        positions_3d['LeftForeArm'] = np.array([
            normalized_kp[9][0] * scale, 35.0 + normalized_kp[9][1] * scale * 0.5,
            normalized_kp[9][1] * scale * 0.3
        ])
        positions_3d['LeftHand'] = positions_3d['LeftForeArm'] + np.array([-10.0, -5.0, 0.0])
        
        # Right arm (mirrored)
        positions_3d['RightShoulder'] = np.array([
            normalized_kp[6][0] * scale, 35.0, normalized_kp[6][1] * scale * 0.1
        ])
        positions_3d['RightArm'] = np.array([
            normalized_kp[8][0] * scale, 35.0 + normalized_kp[8][1] * scale * 0.3,
            normalized_kp[8][1] * scale * 0.2
        ])
        positions_3d['RightForeArm'] = np.array([
            normalized_kp[10][0] * scale, 35.0 + normalized_kp[10][1] * scale * 0.5,
            normalized_kp[10][1] * scale * 0.3
        ])
        positions_3d['RightHand'] = positions_3d['RightForeArm'] + np.array([10.0, -5.0, 0.0])
        
        # Legs
        # Left leg
        positions_3d['LeftUpLeg'] = np.array([
            normalized_kp[11][0] * scale, -5.0, normalized_kp[11][1] * scale * 0.1
        ])
        positions_3d['LeftLeg'] = np.array([
            normalized_kp[13][0] * scale, 
            -5.0 + normalized_kp[13][1] * scale * 0.8,
            normalized_kp[13][1] * scale * 0.2
        ])
        positions_3d['LeftFoot'] = np.array([
            normalized_kp[15][0] * scale,
            -5.0 + normalized_kp[15][1] * scale,
            normalized_kp[15][1] * scale * 0.1
        ])
        
        # Right leg
        positions_3d['RightUpLeg'] = np.array([
            normalized_kp[12][0] * scale, -5.0, normalized_kp[12][1] * scale * 0.1
        ])
        positions_3d['RightLeg'] = np.array([
            normalized_kp[14][0] * scale,
            -5.0 + normalized_kp[14][1] * scale * 0.8,
            normalized_kp[14][1] * scale * 0.2
        ])
        positions_3d['RightFoot'] = np.array([
            normalized_kp[16][0] * scale,
            -5.0 + normalized_kp[16][1] * scale,
            normalized_kp[16][1] * scale * 0.1
        ])
        
        return positions_3d
    
    def calculate_rotation_matrix(self, v_from: np.ndarray, v_to: np.ndarray) -> np.ndarray:
        """
        Calculate rotation matrix that rotates v_from to v_to
        Using Rodrigues' rotation formula
        """
        v_from = v_from / np.linalg.norm(v_from)
        v_to = v_to / np.linalg.norm(v_to)
        
        # Check if vectors are parallel
        cross = np.cross(v_from, v_to)
        dot = np.dot(v_from, v_to)
        
        if np.allclose(cross, 0):
            if dot > 0.99999:
                # Vectors are identical
                return np.eye(3)
            else:
                # Vectors are opposite
                # Find an orthogonal vector
                orthogonal = np.array([1, 0, 0]) if abs(v_from[0]) < 0.9 else np.array([0, 1, 0])
                orthogonal = orthogonal - np.dot(orthogonal, v_from) * v_from
                orthogonal = orthogonal / np.linalg.norm(orthogonal)
                return 2 * np.outer(orthogonal, orthogonal) - np.eye(3)
        
        # General case
        axis = cross / np.linalg.norm(cross)
        angle = np.arccos(np.clip(dot, -1.0, 1.0))
        
        # Rodrigues formula
        K = np.array([[0, -axis[2], axis[1]],
                      [axis[2], 0, -axis[0]],
                      [-axis[1], axis[0], 0]])
        
        R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * np.dot(K, K)
        return R
    
    def rotation_matrix_to_euler_zxy(self, R: np.ndarray) -> Tuple[float, float, float]:
        """
        Extract Euler angles from rotation matrix in ZXY order
        Following BVH standard
        """
        # Clamp values to avoid numerical issues
        R = np.clip(R, -1.0, 1.0)
        
        # Extract angles using ZXY order
        sin_x = R[2, 1]
        
        # Check for gimbal lock
        if abs(sin_x) >= 0.99999:
            # Gimbal lock case
            z_rot = 0  # Arbitrary
            x_rot = np.pi/2 if sin_x > 0 else -np.pi/2
            y_rot = np.arctan2(-R[0, 2], R[0, 0])
        else:
            z_rot = np.arctan2(-R[0, 1], R[1, 1])
            x_rot = np.arcsin(sin_x)
            y_rot = np.arctan2(-R[2, 0], R[2, 2])
        
        # Convert to degrees
        return (np.degrees(z_rot), np.degrees(x_rot), np.degrees(y_rot))
    
    def calculate_joint_rotations(self, positions_3d: Dict[str, np.ndarray]) -> Dict[str, List[float]]:
        """
        Calculate joint rotations using the mathematical formula:
        pos_j = pos_P(j) + R_P(j) · offset_j
        BVH stores: R_P(j)^(-1) × R_j
        """
        rotations = {}
        offsets = self.calculate_bone_offsets()
        
        # Process joints in hierarchical order
        for joint_name in self.joint_order:
            joint_info = self.skeleton_hierarchy[joint_name]
            
            if joint_info['parent'] is None:
                # Root joint - include position
                pos = positions_3d[joint_name]
                rotations[joint_name] = [
                    pos[0], pos[1] + 90.0, pos[2],  # Position (Y offset for ground)
                    0.0, 0.0, 0.0  # Rotation (identity for root)
                ]
            else:
                # Child joint - calculate relative rotation
                parent_name = joint_info['parent']
                
                # Get the first child to calculate bone direction
                children = joint_info['children']
                if children:
                    # Has children - calculate rotation to align with child
                    child_name = children[0]
                    
                    # Reference bone vector (from offsets)
                    ref_offset = offsets[child_name]
                    ref_vector = ref_offset / np.linalg.norm(ref_offset) if np.linalg.norm(ref_offset) > 0.001 else np.array([0, 1, 0])
                    
                    # Current bone vector
                    current_vector = positions_3d[child_name] - positions_3d[joint_name]
                    current_vector = current_vector / np.linalg.norm(current_vector) if np.linalg.norm(current_vector) > 0.001 else np.array([0, 1, 0])
                    
                    # Calculate rotation matrix
                    R_local = self.calculate_rotation_matrix(ref_vector, current_vector)
                    
                    # Extract Euler angles in ZXY order
                    z_rot, x_rot, y_rot = self.rotation_matrix_to_euler_zxy(R_local)
                    
                    rotations[joint_name] = [z_rot, x_rot, y_rot]
                else:
                    # End effector - no rotation
                    rotations[joint_name] = [0.0, 0.0, 0.0]
        
        return rotations
    
    def smooth_motion_data(self, motion_frames: List[Dict], window_size: int = 5) -> List[Dict]:
        """Apply temporal smoothing with quaternion interpolation"""
        if len(motion_frames) < window_size:
            return motion_frames
        
        smoothed_frames = []
        
        for i in range(len(motion_frames)):
            smoothed_frame = {}
            
            start_idx = max(0, i - window_size // 2)
            end_idx = min(len(motion_frames), i + window_size // 2 + 1)
            
            for joint_name in self.joint_order:
                joint_data = []
                
                for j in range(start_idx, end_idx):
                    if joint_name in motion_frames[j]:
                        joint_data.append(motion_frames[j][joint_name])
                
                if joint_data:
                    # For root joint with position
                    if len(joint_data[0]) == 6:
                        # Average positions
                        positions = np.array([d[:3] for d in joint_data])
                        avg_pos = np.mean(positions, axis=0)
                        
                        # Average rotations using quaternions
                        euler_angles = np.array([d[3:] for d in joint_data])
                        quats = []
                        for angles in euler_angles:
                            r = R.from_euler('zxy', angles, degrees=True)
                            quats.append(r.as_quat())
                        
                        # Average quaternions
                        avg_quat = np.mean(quats, axis=0)
                        avg_quat = avg_quat / np.linalg.norm(avg_quat)
                        
                        # Convert back to Euler
                        r_avg = R.from_quat(avg_quat)
                        z, x, y = r_avg.as_euler('zxy', degrees=True)
                        
                        smoothed_frame[joint_name] = [
                            avg_pos[0], avg_pos[1], avg_pos[2],
                            z, x, y
                        ]
                    else:
                        # Regular joint - average rotations
                        euler_angles = np.array(joint_data)
                        quats = []
                        for angles in euler_angles:
                            r = R.from_euler('zxy', angles, degrees=True)
                            quats.append(r.as_quat())
                        
                        avg_quat = np.mean(quats, axis=0)
                        avg_quat = avg_quat / np.linalg.norm(avg_quat)
                        
                        r_avg = R.from_quat(avg_quat)
                        z, x, y = r_avg.as_euler('zxy', degrees=True)
                        
                        smoothed_frame[joint_name] = [z, x, y]
                else:
                    # Default values
                    if joint_name == 'Hips':
                        smoothed_frame[joint_name] = [0.0, 90.0, 0.0, 0.0, 0.0, 0.0]
                    else:
                        smoothed_frame[joint_name] = [0.0, 0.0, 0.0]
            
            smoothed_frames.append(smoothed_frame)
        
        return smoothed_frames
    
    def write_bvh_file(self, motion_data: List[Dict], output_path: str, fps: float = 30.0):
        """Write BVH file with proper formatting"""
        offsets = self.calculate_bone_offsets()
        
        with open(output_path, 'w') as f:
            # Write HIERARCHY section
            f.write("HIERARCHY\n")
            self._write_joint_hierarchy(f, 'Hips', 0, offsets)
            
            # Write MOTION section
            f.write("MOTION\n")
            f.write(f"Frames: {len(motion_data)}\n")
            f.write(f"Frame Time: {1.0/fps:.6f}\n")
            
            # Write motion data
            for frame_data in motion_data:
                frame_values = []
                
                for joint_name in self.joint_order:
                    if joint_name in frame_data:
                        values = frame_data[joint_name]
                        frame_values.extend(values)
                    else:
                        # Default values
                        if joint_name == 'Hips':
                            frame_values.extend([0.0, 90.0, 0.0, 0.0, 0.0, 0.0])
                        else:
                            frame_values.extend([0.0, 0.0, 0.0])
                
                f.write(" ".join(f"{val:.6f}" for val in frame_values) + "\n")
        
        self.logger.info(f"BVH file written: {output_path}")
    
    def _write_joint_hierarchy(self, f, joint_name: str, indent_level: int, offsets: Dict):
        """Write joint hierarchy recursively"""
        indent = "  " * indent_level
        joint_info = self.skeleton_hierarchy[joint_name]
        
        # Write joint declaration
        if joint_info['parent'] is None:
            f.write(f"{indent}ROOT {joint_name}\n")
        else:
            f.write(f"{indent}JOINT {joint_name}\n")
        
        f.write(f"{indent}{{\n")
        
        # Write offset
        offset = offsets[joint_name]
        f.write(f"{indent}  OFFSET {offset[0]:.6f} {offset[1]:.6f} {offset[2]:.6f}\n")
        
        # Write channels
        channels = joint_info['channels']
        f.write(f"{indent}  CHANNELS {len(channels)} {' '.join(channels)}\n")
        
        # Write children
        for child_name in joint_info['children']:
            self._write_joint_hierarchy(f, child_name, indent_level + 1, offsets)
        
        # End site for leaf joints
        if not joint_info['children']:
            f.write(f"{indent}  End Site\n")
            f.write(f"{indent}  {{\n")
            f.write(f"{indent}    OFFSET 0.0 -10.0 0.0\n")
            f.write(f"{indent}  }}\n")
        
        f.write(f"{indent}}}\n")
    
    def convert_video_to_bvh(self, video_path: str, output_path: str, 
                           max_frames: Optional[int] = None) -> bool:
        """Complete pipeline with mathematical corrections"""
        try:
            self.logger.info(f"Starting conversion: {video_path} -> {output_path}")
            
            # Extract frames
            frames, fps = self.extract_frames_from_video(video_path, max_frames)
            if not frames:
                self.logger.error("No frames extracted")
                return False
            
            # Estimate 2D poses
            pose_data_2d = self.estimate_poses_from_frames(frames)
            
            # Convert to 3D and calculate rotations
            motion_data = []
            
            for i, frame_poses in enumerate(pose_data_2d):
                if frame_poses:
                    # Use first detected person
                    person_id = list(frame_poses.keys())[0]
                    keypoints_2d = frame_poses[person_id]['keypoints']
                    
                    # Lift 2D to 3D positions
                    positions_3d = self.lift_2d_to_3d(keypoints_2d)
                    
                    # Calculate joint rotations using mathematical formula
                    joint_rotations = self.calculate_joint_rotations(positions_3d)
                    
                    motion_data.append(joint_rotations)
                else:
                    # Default T-pose
                    default_rotations = {}
                    for joint in self.joint_order:
                        if joint == 'Hips':
                            default_rotations[joint] = [0.0, 90.0, 0.0, 0.0, 0.0, 0.0]
                        else:
                            default_rotations[joint] = [0.0, 0.0, 0.0]
                    motion_data.append(default_rotations)
            
            # Apply temporal smoothing
            smoothed_motion = self.smooth_motion_data(motion_data, window_size=7)
            
            # Write BVH file
            self.write_bvh_file(smoothed_motion, output_path, fps)
            
            self.logger.info("Conversion completed successfully!")
            return True
            
        except Exception as e:
            self.logger.error(f"Conversion failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

def main():
    """Example usage"""
    # Model paths
    parser = argparse.ArgumentParser(description='Convert video to BVH using MediaPipe Pose')
    parser.add_argument('--video', required=True, help='Input video file')
    parser.add_argument('--output', required=True, help='Output BVH file')

    vitpose_model = "model/vitpose-s-coco_25.pth"
    yolo_model = "model/yolov8l.pt"
    args = parser.parse_args()
    
    # Initialize converter
    converter = MathematicallyCorrectMP4ToBVH(
        vitpose_model_path=vitpose_model,
        yolo_model_path=yolo_model,
        model_size='s',
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # Convert video
    success = converter.convert_video_to_bvh(
        video_path=args.video,
        output_path=args.output,
        max_frames=300
    )
    
    if success:
        print("✅ Conversion successful!")
        print("\nImport settings for Blender:")
        print("1. File > Import > Motion Capture (.bvh)")
        print("2. Import settings:")
        print("   - Forward: -Z Forward")
        print("   - Up: Y Up")
        print("   - Scale: 0.01")
        print("   - Start Frame: 1")
        print("\nThe animation should now properly match your video!")
    else:
        print("❌ Conversion failed.")

if __name__ == "__main__":
    main()