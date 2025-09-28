#!/usr/bin/env python3
"""
Improved BVH Accuracy Testing System

This version includes additional metrics to capture visual quality:
- Visual naturalness score
- Temporal drift measurement
- Over-smoothing detection
- Ground contact realism
- Better symmetry interpretation
"""

import cv2
import mediapipe as mp
import numpy as np
import json
import argparse
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
from scipy.signal import butter, filtfilt, find_peaks
from scipy import stats

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

@dataclass
class ImprovedAccuracyMetrics:
    """Enhanced accuracy metrics including visual quality measures"""
    # Overall metrics
    total_frames: int
    valid_comparison_frames: int
    overall_accuracy_score: float  # 0-100
    
    # Joint angle accuracy
    mean_angle_error: float  # Average angular error in degrees
    max_angle_error: float
    angle_error_std: float
    joint_angle_errors: Dict[str, float]  # Per-joint angle error
    
    # Relative position accuracy (hip-relative)
    mean_relative_position_error: float
    max_relative_position_error: float
    relative_position_errors: Dict[str, float]
    
    # Bone length consistency
    mean_bone_length_variation: float  # How much bone lengths vary
    bone_length_consistency: Dict[str, float]  # Per-bone consistency score
    
    # Motion pattern matching
    gait_cycle_correlation: float  # How well gait patterns match
    motion_smoothness_score: float  # Jerk metric
    acceleration_consistency: float
    
    # Temporal metrics
    position_jitter_score: float  # Frame-to-frame smoothness
    angular_velocity_correlation: float  # How well rotational speeds match
    
    # Symmetry metrics
    left_right_symmetry: float  # How symmetric the motion is
    symmetry_naturalness: float  # Whether symmetry is natural (not too perfect)
    
    # NEW: Visual quality metrics
    visual_naturalness_score: float  # Overall visual quality estimate
    temporal_drift_score: float  # Measures accumulating position drift
    over_smoothing_score: float  # Detects artificial smoothing
    ground_contact_score: float  # Foot sliding and ground adherence
    motion_dynamics_score: float  # Natural acceleration/deceleration patterns
    
    # Problem areas
    worst_frames: List[int]
    worst_joints: List[str]
    confidence_scores: Dict[str, float]  # Confidence in each metric
    quality_warnings: List[str]  # Warnings about detected issues


class ImprovedBVHAccuracyAnalyzer:
    """Enhanced analyzer with visual quality metrics"""
    
    def __init__(self, confidence_threshold=0.5):
        self.confidence_threshold = confidence_threshold
        self.pose = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Define joint mappings and bone connections
        self.joint_mappings = self._create_joint_mappings()
        self.bone_connections = self._create_bone_connections()
        
    def _create_joint_mappings(self):
        """Create mapping between MediaPipe landmarks and BVH joints"""
        return {
            'Hips': [mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.RIGHT_HIP],
            'Spine': [mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.RIGHT_HIP],
            'Chest': [mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.RIGHT_SHOULDER],
            'Neck': [mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.RIGHT_SHOULDER],
            'Head': [mp_pose.PoseLandmark.NOSE],
            'LeftShoulder': [mp_pose.PoseLandmark.LEFT_SHOULDER],
            'LeftArm': [mp_pose.PoseLandmark.LEFT_ELBOW],
            'LeftForeArm': [mp_pose.PoseLandmark.LEFT_WRIST],
            'LeftHand': [mp_pose.PoseLandmark.LEFT_WRIST],
            'RightShoulder': [mp_pose.PoseLandmark.RIGHT_SHOULDER],
            'RightArm': [mp_pose.PoseLandmark.RIGHT_ELBOW],
            'RightForeArm': [mp_pose.PoseLandmark.RIGHT_WRIST],
            'RightHand': [mp_pose.PoseLandmark.RIGHT_WRIST],
            'LeftUpLeg': [mp_pose.PoseLandmark.LEFT_HIP],
            'LeftLeg': [mp_pose.PoseLandmark.LEFT_KNEE],
            'LeftFoot': [mp_pose.PoseLandmark.LEFT_ANKLE],
            'RightUpLeg': [mp_pose.PoseLandmark.RIGHT_HIP],
            'RightLeg': [mp_pose.PoseLandmark.RIGHT_KNEE],
            'RightFoot': [mp_pose.PoseLandmark.RIGHT_ANKLE]
        }
    
    def _create_bone_connections(self):
        """Define bone connections for angle calculations"""
        return [
            # Arms
            ('LeftShoulder', 'LeftArm', 'LeftForeArm'),
            ('RightShoulder', 'RightArm', 'RightForeArm'),
            ('LeftArm', 'LeftForeArm', 'LeftHand'),
            ('RightArm', 'RightForeArm', 'RightHand'),
            # Legs
            ('LeftUpLeg', 'LeftLeg', 'LeftFoot'),
            ('RightUpLeg', 'RightLeg', 'RightFoot'),
            ('Hips', 'LeftUpLeg', 'LeftLeg'),
            ('Hips', 'RightUpLeg', 'RightLeg'),
            # Spine
            ('Hips', 'Spine', 'Chest'),
            ('Spine', 'Chest', 'Neck'),
            ('Chest', 'Neck', 'Head'),
        ]
    
    def parse_bvh(self, bvh_path: str) -> Dict:
        """Parse BVH file and extract motion data with forward kinematics"""
        with open(bvh_path, 'r') as f:
            content = f.read()
        
        if 'MOTION' not in content:
            raise ValueError("No MOTION section in BVH file")
        
        hierarchy, motion = content.split('MOTION', 1)
        
        # Parse hierarchy to build skeleton structure
        skeleton = self._parse_hierarchy(hierarchy)
        
        # Parse motion data
        lines = motion.strip().split('\n')
        num_frames = None
        frame_time = None
        motion_data = []
        
        for line in lines:
            if 'Frames:' in line:
                num_frames = int(line.split(':')[1].strip())
            elif 'Frame Time:' in line:
                frame_time = float(line.split(':')[1].strip())
            else:
                try:
                    values = [float(x) for x in line.strip().split()]
                    if values:
                        motion_data.append(values)
                except ValueError:
                    continue
        
        # Calculate joint positions and rotations for each frame
        joint_positions = []
        joint_rotations = []
        for frame_data in motion_data:
            positions, rotations = self._forward_kinematics_with_rotations(skeleton, frame_data)
            joint_positions.append(positions)
            joint_rotations.append(rotations)
        
        return {
            'num_frames': num_frames,
            'frame_time': frame_time,
            'motion_data': motion_data,
            'skeleton': skeleton,
            'joint_positions': joint_positions,
            'joint_rotations': joint_rotations
        }
    
    def _parse_hierarchy(self, hierarchy: str) -> Dict:
        """Parse BVH hierarchy section"""
        skeleton = {'joints': {}, 'root': None, 'joint_order': []}
        lines = hierarchy.strip().split('\n')
        
        joint_stack = []
        current_joint = None
        
        for line in lines:
            line = line.strip()
            
            if 'ROOT' in line:
                name = line.split('ROOT')[1].strip()
                skeleton['root'] = name
                skeleton['joints'][name] = {
                    'parent': None,
                    'offset': [0, 0, 0],
                    'channels': [],
                    'children': []
                }
                current_joint = name
                skeleton['joint_order'].append(name)
                joint_stack.append(name)
                
            elif 'JOINT' in line:
                name = line.split('JOINT')[1].strip()
                parent = joint_stack[-1] if joint_stack else None
                skeleton['joints'][name] = {
                    'parent': parent,
                    'offset': [0, 0, 0],
                    'channels': [],
                    'children': []
                }
                if parent:
                    skeleton['joints'][parent]['children'].append(name)
                current_joint = name
                skeleton['joint_order'].append(name)
                joint_stack.append(name)
                
            elif 'OFFSET' in line:
                if current_joint:
                    offset = [float(x) for x in line.split('OFFSET')[1].strip().split()]
                    skeleton['joints'][current_joint]['offset'] = offset
                    
            elif 'CHANNELS' in line:
                if current_joint:
                    parts = line.split()
                    num_channels = int(parts[1])
                    channels = parts[2:2+num_channels]
                    skeleton['joints'][current_joint]['channels'] = channels
                    
            elif '}' in line:
                if joint_stack:
                    joint_stack.pop()
                    current_joint = joint_stack[-1] if joint_stack else None
        
        return skeleton
    
    def _forward_kinematics_with_rotations(self, skeleton: Dict, frame_data: List[float]) -> Tuple[Dict, Dict]:
        """Calculate joint positions and rotations using forward kinematics"""
        positions = {}
        rotations = {}
        channel_index = 0
        
        def process_joint(joint_name, parent_transform=np.eye(4)):
            nonlocal channel_index
            
            joint = skeleton['joints'][joint_name]
            local_transform = np.eye(4)
            
            # Apply channels (position for root, rotation for all)
            if joint['parent'] is None:  # Root joint
                # Handle position channels
                for channel in joint['channels']:
                    if 'position' in channel.lower():
                        if 'x' in channel.lower():
                            local_transform[0, 3] = frame_data[channel_index]
                        elif 'y' in channel.lower():
                            local_transform[1, 3] = frame_data[channel_index]
                        elif 'z' in channel.lower():
                            local_transform[2, 3] = frame_data[channel_index]
                        channel_index += 1
            
            # Handle rotation channels
            rotation = np.eye(3)
            euler_angles = [0, 0, 0]  # XYZ
            for channel in joint['channels']:
                if 'rotation' in channel.lower():
                    angle = frame_data[channel_index]
                    if 'x' in channel.lower():
                        euler_angles[0] = angle
                    elif 'y' in channel.lower():
                        euler_angles[1] = angle
                    elif 'z' in channel.lower():
                        euler_angles[2] = angle
                    channel_index += 1
            
            # Convert Euler angles to rotation matrix
            rotation = R.from_euler('xyz', euler_angles, degrees=True).as_matrix()
            
            # Apply rotation to transform
            local_transform[:3, :3] = rotation
            
            # Apply offset
            offset_transform = np.eye(4)
            offset_transform[:3, 3] = joint['offset']
            
            # Combine transforms
            global_transform = parent_transform @ local_transform @ offset_transform
            
            # Store position and rotation
            positions[joint_name] = global_transform[:3, 3].copy()
            rotations[joint_name] = global_transform[:3, :3].copy()
            
            # Process children
            for child in joint['children']:
                process_joint(child, global_transform)
        
        # Start from root
        if skeleton['root']:
            process_joint(skeleton['root'])
        
        return positions, rotations
    
    def extract_mediapipe_data(self, video_path: str, sample_rate: int = 1) -> Tuple[List[Dict], List[Dict]]:
        """Extract MediaPipe joint positions and calculate rotations from video"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        positions_per_frame = []
        rotations_per_frame = []
        frame_idx = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_idx % sample_rate == 0:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.pose.process(frame_rgb)
                
                if results.pose_world_landmarks:
                    # Convert landmarks to joint positions
                    joint_positions = {}
                    for joint_name, landmark_indices in self.joint_mappings.items():
                        positions = []
                        for idx in landmark_indices:
                            lm = results.pose_world_landmarks.landmark[idx]
                            positions.append(np.array([lm.x, lm.y, lm.z]))
                        
                        # Average if multiple landmarks
                        if positions:
                            joint_positions[joint_name] = np.mean(positions, axis=0) * 100  # Scale to match BVH
                    
                    # Calculate rotations from positions
                    joint_rotations = self._calculate_rotations_from_positions(joint_positions)
                    
                    positions_per_frame.append(joint_positions)
                    rotations_per_frame.append(joint_rotations)
                else:
                    positions_per_frame.append({})
                    rotations_per_frame.append({})
            
            frame_idx += 1
        
        cap.release()
        return positions_per_frame, rotations_per_frame
    
    def _calculate_rotations_from_positions(self, positions: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Calculate joint rotations from positions"""
        rotations = {}
        
        # For each bone connection, calculate the rotation
        for joint_name in positions.keys():
            # Simple approximation: use direction vectors to estimate rotation
            # This is a simplified version - a full implementation would use inverse kinematics
            rotations[joint_name] = np.eye(3)  # Identity for now
        
        return rotations
    
    def _convert_to_hip_relative(self, positions: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Convert all positions to be relative to hip position"""
        if 'Hips' not in positions:
            return positions
        
        hip_pos = positions['Hips']
        relative_positions = {}
        for joint_name, pos in positions.items():
            relative_positions[joint_name] = pos - hip_pos
        
        return relative_positions
    
    def _calculate_joint_angle(self, parent_pos: np.ndarray, joint_pos: np.ndarray, 
                               child_pos: np.ndarray) -> float:
        """Calculate angle at joint given three positions"""
        vec1 = parent_pos - joint_pos
        vec2 = child_pos - joint_pos
        
        # Normalize vectors
        vec1_norm = vec1 / (np.linalg.norm(vec1) + 1e-6)
        vec2_norm = vec2 / (np.linalg.norm(vec2) + 1e-6)
        
        # Calculate angle
        cos_angle = np.clip(np.dot(vec1_norm, vec2_norm), -1.0, 1.0)
        angle = np.degrees(np.arccos(cos_angle))
        
        return angle
    
    def _calculate_bone_length(self, pos1: np.ndarray, pos2: np.ndarray) -> float:
        """Calculate distance between two joint positions"""
        return np.linalg.norm(pos2 - pos1)
    
    def _calculate_jerk(self, positions_over_time: List[np.ndarray], dt: float = 1/30.0) -> float:
        """Calculate jerk (rate of change of acceleration) for smoothness metric"""
        if len(positions_over_time) < 4:
            return 0.0

        # Convert to numpy array and normalize by position scale
        positions = np.array(positions_over_time)
        # Normalize positions to reasonable scale (BVH uses cm-like units)
        positions = positions / 100.0  # Convert to meter-like units

        # Calculate velocities
        velocities = np.diff(positions, axis=0) / dt

        # Calculate accelerations
        accelerations = np.diff(velocities, axis=0) / dt

        # Calculate jerk
        jerk = np.diff(accelerations, axis=0) / dt

        # Return mean magnitude of jerk
        return np.mean(np.linalg.norm(jerk, axis=1))
    
    def _calculate_visual_naturalness(self, bvh_positions: List[Dict], mediapipe_positions: List[Dict]) -> float:
        """Calculate visual naturalness score based on motion characteristics"""
        naturalness_scores = []
        
        # Check for natural motion ranges
        for frame_idx in range(1, min(len(bvh_positions), len(mediapipe_positions))):
            frame_score = 100.0
            
            # Check joint velocity consistency
            for joint_name in self.joint_mappings.keys():
                if joint_name in bvh_positions[frame_idx] and joint_name in bvh_positions[frame_idx-1]:
                    velocity = np.linalg.norm(
                        bvh_positions[frame_idx][joint_name] - bvh_positions[frame_idx-1][joint_name]
                    )
                    
                    # Penalize unrealistic velocities
                    if velocity > 50:  # Too fast
                        frame_score -= 10
                    elif velocity < 0.01 and 'Foot' not in joint_name:  # Too static (except feet)
                        frame_score -= 5
            
            naturalness_scores.append(max(0, frame_score))
        
        return np.mean(naturalness_scores) if naturalness_scores else 50.0
    
    def _calculate_temporal_drift(self, bvh_positions: List[Dict], mediapipe_positions: List[Dict]) -> float:
        """Measure accumulating position drift over time"""
        if not bvh_positions or not mediapipe_positions:
            return 0.0
        
        drift_scores = []
        
        # Calculate drift accumulation
        for joint_name in ['Hips', 'Chest', 'Head']:
            if joint_name not in bvh_positions[0] or joint_name not in mediapipe_positions[0]:
                continue
                
            bvh_trajectory = [frame.get(joint_name, np.zeros(3)) for frame in bvh_positions]
            mp_trajectory = [frame.get(joint_name, np.zeros(3)) for frame in mediapipe_positions]
            
            if len(bvh_trajectory) > 10 and len(mp_trajectory) > 10:
                # Compare overall trajectory drift
                bvh_end_to_end = np.linalg.norm(bvh_trajectory[-1] - bvh_trajectory[0])
                mp_end_to_end = np.linalg.norm(mp_trajectory[-1] - mp_trajectory[0])
                
                drift = abs(bvh_end_to_end - mp_end_to_end)
                drift_scores.append(100 * np.exp(-drift / 100))  # Exponential penalty
        
        return np.mean(drift_scores) if drift_scores else 50.0
    
    def _detect_over_smoothing(self, positions_over_time: Dict[str, List[np.ndarray]]) -> float:
        """Detect artificial over-smoothing in motion"""
        smoothness_scores = []
        
        for joint_name, trajectory in positions_over_time.items():
            if len(trajectory) < 10:
                continue
            
            # Calculate acceleration changes
            velocities = np.diff(trajectory, axis=0)
            accelerations = np.diff(velocities, axis=0)
            
            if len(accelerations) > 0:
                # Check for unnaturally constant acceleration (too smooth)
                acc_variance = np.var(np.linalg.norm(accelerations, axis=1))
                
                # Natural motion has varied acceleration
                if acc_variance < 0.001:  # Too smooth
                    smoothness_scores.append(0)
                elif acc_variance > 10:  # Too jerky
                    smoothness_scores.append(50)
                else:
                    # Good range
                    smoothness_scores.append(100)
        
        return np.mean(smoothness_scores) if smoothness_scores else 50.0
    
    def _calculate_ground_contact(self, foot_positions: Dict[str, List[np.ndarray]]) -> float:
        """Evaluate foot sliding and ground contact realism"""
        ground_scores = []
        
        for foot_name in ['LeftFoot', 'RightFoot']:
            if foot_name not in foot_positions or len(foot_positions[foot_name]) < 2:
                continue
            
            trajectory = foot_positions[foot_name]
            
            # Detect potential ground contact frames (low Y values)
            y_values = [pos[1] for pos in trajectory]
            min_y = min(y_values)
            ground_threshold = min_y + 5  # 5cm above minimum
            
            # Check for sliding during ground contact
            sliding_penalties = 0
            for i in range(1, len(trajectory)):
                if y_values[i] < ground_threshold and y_values[i-1] < ground_threshold:
                    # Both frames are near ground - check for sliding
                    horizontal_movement = np.sqrt(
                        (trajectory[i][0] - trajectory[i-1][0])**2 + 
                        (trajectory[i][2] - trajectory[i-1][2])**2
                    )
                    
                    if horizontal_movement > 2:  # More than 2cm movement when on ground
                        sliding_penalties += 1
            
            # Calculate score based on sliding
            if len(trajectory) > 0:
                slide_ratio = sliding_penalties / len(trajectory)
                ground_scores.append(100 * (1 - min(1, slide_ratio * 10)))
        
        return np.mean(ground_scores) if ground_scores else 50.0
    
    def _evaluate_symmetry_naturalness(self, symmetry_score: float) -> Tuple[float, List[str]]:
        """Evaluate if symmetry is natural or artificially perfect"""
        warnings = []
        naturalness = 100.0
        
        if symmetry_score > 95:
            warnings.append("Symmetry too perfect (>95%) - possible over-smoothing")
            naturalness = 50.0  # Penalize perfect symmetry
        elif symmetry_score > 90:
            warnings.append("Very high symmetry (>90%) - may be artificial")
            naturalness = 75.0
        elif symmetry_score < 30:
            warnings.append("Very low symmetry (<30%) - possible tracking issues")
            naturalness = 60.0
        elif 40 <= symmetry_score <= 70:
            # Natural range for human motion
            naturalness = 100.0
        else:
            naturalness = 85.0
        
        return naturalness, warnings
    
    def compare_motion_improved(self, bvh_data: Dict, mediapipe_positions: List[Dict], 
                                mediapipe_rotations: List[Dict]) -> ImprovedAccuracyMetrics:
        """Enhanced motion comparison with visual quality metrics"""
        
        bvh_positions = bvh_data['joint_positions']
        bvh_rotations = bvh_data['joint_rotations']
        
        # Ensure we have the same number of frames
        min_frames = min(len(bvh_positions), len(mediapipe_positions))
        
        # Initialize metric storage
        angle_errors = []
        joint_angle_errors = {joint: [] for joint in self.joint_mappings.keys()}
        relative_position_errors = []
        joint_relative_errors = {joint: [] for joint in self.joint_mappings.keys()}
        bone_length_variations = {bone: [] for bone in self.bone_connections}
        
        # Collect time series data for analysis
        all_bvh_positions = {joint: [] for joint in self.joint_mappings.keys()}
        all_mp_positions = {joint: [] for joint in self.joint_mappings.keys()}
        foot_positions = {'LeftFoot': [], 'RightFoot': []}
        
        # Process each frame
        for frame_idx in range(min_frames):
            bvh_frame = bvh_positions[frame_idx]
            mp_frame = mediapipe_positions[frame_idx]
            
            # Convert to hip-relative coordinates
            bvh_relative = self._convert_to_hip_relative(bvh_frame)
            mp_relative = self._convert_to_hip_relative(mp_frame)
            
            # Calculate relative position errors
            frame_relative_errors = []
            for joint_name in self.joint_mappings.keys():
                if joint_name in bvh_relative and joint_name in mp_relative:
                    error = np.linalg.norm(bvh_relative[joint_name] - mp_relative[joint_name])
                    frame_relative_errors.append(error)
                    joint_relative_errors[joint_name].append(error)
                    
                    # Store for analysis
                    all_bvh_positions[joint_name].append(bvh_frame[joint_name])
                    all_mp_positions[joint_name].append(mp_frame[joint_name])
                    
                    # Track foot positions
                    if joint_name in foot_positions and joint_name in bvh_frame:
                        foot_positions[joint_name].append(bvh_frame[joint_name])
            
            if frame_relative_errors:
                relative_position_errors.append(np.mean(frame_relative_errors))
            
            # Calculate joint angle errors
            frame_angle_errors = []
            for bone_chain in self.bone_connections:
                if len(bone_chain) == 3:
                    parent, joint, child = bone_chain
                    if all(j in bvh_frame and j in mp_frame for j in [parent, joint, child]):
                        bvh_angle = self._calculate_joint_angle(
                            bvh_frame[parent], bvh_frame[joint], bvh_frame[child]
                        )
                        mp_angle = self._calculate_joint_angle(
                            mp_frame[parent], mp_frame[joint], mp_frame[child]
                        )
                        angle_error = abs(bvh_angle - mp_angle)
                        frame_angle_errors.append(angle_error)
                        if joint in joint_angle_errors:
                            joint_angle_errors[joint].append(angle_error)
            
            if frame_angle_errors:
                angle_errors.append(np.mean(frame_angle_errors))
            
            # Calculate bone lengths
            for bone_chain in self.bone_connections:
                if len(bone_chain) >= 2:
                    for i in range(len(bone_chain) - 1):
                        bone_name = f"{bone_chain[i]}_{bone_chain[i+1]}"
                        if bone_chain[i] in bvh_frame and bone_chain[i+1] in bvh_frame:
                            length = self._calculate_bone_length(
                                bvh_frame[bone_chain[i]], bvh_frame[bone_chain[i+1]]
                            )
                            if bone_name not in bone_length_variations:
                                bone_length_variations[bone_name] = []
                            bone_length_variations[bone_name].append(length)
        
        # Calculate standard metrics
        position_jitter_scores = []
        for joint_name in self.joint_mappings.keys():
            if all_bvh_positions[joint_name]:
                jerk = self._calculate_jerk(all_bvh_positions[joint_name])
                position_jitter_scores.append(jerk)
        
        # Calculate bone length consistency
        bone_consistency_scores = {}
        mean_bone_variation = []
        for bone_name, lengths in bone_length_variations.items():
            if lengths:
                std_dev = np.std(lengths)
                mean_length = np.mean(lengths)
                consistency = 100 * (1 - std_dev / (mean_length + 1e-6))
                bone_consistency_scores[bone_name] = max(0, consistency)
                mean_bone_variation.append(std_dev)
        
        # Calculate symmetry
        symmetry_score = self._calculate_symmetry(joint_relative_errors)
        
        # NEW: Calculate visual quality metrics
        visual_naturalness = self._calculate_visual_naturalness(bvh_positions, mediapipe_positions)
        temporal_drift = self._calculate_temporal_drift(bvh_positions, mediapipe_positions)
        over_smoothing = self._detect_over_smoothing(all_bvh_positions)
        ground_contact = self._calculate_ground_contact(foot_positions)
        
        # Evaluate symmetry naturalness
        symmetry_naturalness, symmetry_warnings = self._evaluate_symmetry_naturalness(symmetry_score)
        
        # Calculate motion dynamics score
        motion_dynamics = (visual_naturalness + over_smoothing) / 2
        
        # Collect quality warnings
        quality_warnings = symmetry_warnings.copy()
        if temporal_drift < 30:
            quality_warnings.append(f"High temporal drift detected (score: {temporal_drift:.1f})")
        if over_smoothing < 30:
            quality_warnings.append(f"Over-smoothing detected (score: {over_smoothing:.1f})")
        if ground_contact < 40:
            quality_warnings.append(f"Foot sliding detected (score: {ground_contact:.1f})")
        
        # Calculate gait correlation
        gait_correlation = self._calculate_gait_correlation(all_bvh_positions, all_mp_positions)
        
        # Aggregate metrics
        angle_errors_array = np.array(angle_errors) if angle_errors else np.array([0])
        relative_errors_array = np.array(relative_position_errors) if relative_position_errors else np.array([0])
        
        # Calculate per-joint averages
        joint_angle_avg = {}
        joint_relative_avg = {}
        worst_joints = []
        
        for joint_name in self.joint_mappings.keys():
            if joint_angle_errors[joint_name]:
                joint_angle_avg[joint_name] = float(np.mean(joint_angle_errors[joint_name]))
            else:
                joint_angle_avg[joint_name] = 0.0
                
            if joint_relative_errors[joint_name]:
                joint_relative_avg[joint_name] = float(np.mean(joint_relative_errors[joint_name]))
                worst_joints.append((joint_name, joint_relative_avg[joint_name]))
            else:
                joint_relative_avg[joint_name] = 0.0
        
        worst_joints.sort(key=lambda x: x[1], reverse=True)
        worst_joints = [j[0] for j in worst_joints[:5]]
        
        # Find worst frames
        worst_frame_indices = []
        if len(angle_errors) > 0:
            worst_frame_indices = np.argsort(angle_errors)[-10:].tolist()
        
        # Calculate improved overall accuracy score
        angle_score = max(0, 100 - np.mean(angle_errors_array))  # 25% weight
        relative_pos_score = max(0, 100 - np.mean(relative_errors_array) / 2)  # 15% weight
        consistency_score = np.mean(list(bone_consistency_scores.values())) if bone_consistency_scores else 0  # 15% weight
        # Fix smoothness score calculation - scale jerk values appropriately
        # Typical jerk values can be very high, so we need better scaling
        if position_jitter_scores:
            mean_jerk = np.mean(position_jitter_scores)
            # Debug output to understand jerk values (disabled)
            # print(f"DEBUG: Jerk values - Min: {np.min(position_jitter_scores):.3f}, Max: {np.max(position_jitter_scores):.3f}, Mean: {mean_jerk:.3f}")

            # Use logarithmic scaling for more reasonable scores
            # Jerk values are typically in range 1-10000, so we need aggressive scaling
            if mean_jerk > 0:
                # Map jerk to score: 10 -> 90, 100 -> 70, 1000 -> 50, 10000 -> 30
                smoothness_score = max(0, min(100, 110 - 20 * np.log10(mean_jerk)))
            else:
                smoothness_score = 100  # Perfect smoothness if no jerk
        else:
            smoothness_score = 50  # Default if no data
        
        # NEW: Include visual quality in overall score
        visual_quality_score = (
            visual_naturalness * 0.10 +  # 10% weight
            temporal_drift * 0.10 +       # 10% weight
            over_smoothing * 0.05 +       # 5% weight
            ground_contact * 0.10         # 10% weight
        )
        
        overall_score = (
            angle_score * 0.25 +
            relative_pos_score * 0.15 +
            consistency_score * 0.15 +
            smoothness_score * 0.10 +
            visual_quality_score +  # 35% total weight for visual quality
            symmetry_naturalness * 0.00  # Don't include raw symmetry, use naturalness evaluation
        )
        
        # Calculate confidence scores
        confidence_scores = {
            'angle_confidence': min(100, len(angle_errors) / min_frames * 100) if min_frames > 0 else 0,
            'position_confidence': min(100, len(relative_position_errors) / min_frames * 100) if min_frames > 0 else 0,
            'consistency_confidence': min(100, len(bone_consistency_scores) * 10) if bone_consistency_scores else 0,
            'temporal_confidence': 100 if min_frames > 30 else min_frames / 30 * 100,
            'visual_confidence': 80 if min_frames > 60 else min_frames / 60 * 80
        }
        
        return ImprovedAccuracyMetrics(
            total_frames=max(len(bvh_positions), len(mediapipe_positions)),
            valid_comparison_frames=min_frames,
            overall_accuracy_score=float(overall_score),
            mean_angle_error=float(np.mean(angle_errors_array)),
            max_angle_error=float(np.max(angle_errors_array)) if len(angle_errors_array) > 0 else 0,
            angle_error_std=float(np.std(angle_errors_array)),
            joint_angle_errors=joint_angle_avg,
            mean_relative_position_error=float(np.mean(relative_errors_array)),
            max_relative_position_error=float(np.max(relative_errors_array)) if len(relative_errors_array) > 0 else 0,
            relative_position_errors=joint_relative_avg,
            mean_bone_length_variation=float(np.mean(mean_bone_variation)) if mean_bone_variation else 0,
            bone_length_consistency=bone_consistency_scores,
            gait_cycle_correlation=float(gait_correlation),
            motion_smoothness_score=float(smoothness_score),
            acceleration_consistency=0.0,  # TODO: Implement
            position_jitter_score=float(np.mean(position_jitter_scores)) if position_jitter_scores else 0,
            angular_velocity_correlation=0.0,  # TODO: Implement
            left_right_symmetry=float(symmetry_score),
            symmetry_naturalness=float(symmetry_naturalness),
            visual_naturalness_score=float(visual_naturalness),
            temporal_drift_score=float(temporal_drift),
            over_smoothing_score=float(over_smoothing),
            ground_contact_score=float(ground_contact),
            motion_dynamics_score=float(motion_dynamics),
            worst_frames=worst_frame_indices,
            worst_joints=worst_joints,
            confidence_scores=confidence_scores,
            quality_warnings=quality_warnings
        )
    
    def _calculate_symmetry(self, joint_errors: Dict[str, List[float]]) -> float:
        """Calculate left-right symmetry score"""
        left_errors = []
        right_errors = []
        
        for joint_name, errors in joint_errors.items():
            if errors:
                avg_error = np.mean(errors)
                if 'Left' in joint_name:
                    left_errors.append(avg_error)
                elif 'Right' in joint_name:
                    right_errors.append(avg_error)
        
        if left_errors and right_errors:
            # Symmetry score: closer error values = higher symmetry
            diff = abs(np.mean(left_errors) - np.mean(right_errors))
            max_error = max(np.mean(left_errors), np.mean(right_errors))
            if max_error > 0:
                symmetry = 100 * (1 - diff / max_error)
                return max(0, symmetry)
        
        return 50.0  # Default neutral symmetry
    
    def _calculate_gait_correlation(self, bvh_positions: Dict, mp_positions: Dict) -> float:
        """Calculate correlation between gait cycles"""
        # Focus on foot positions for gait analysis
        gait_joints = ['LeftFoot', 'RightFoot']
        correlations = []
        
        for joint in gait_joints:
            if bvh_positions[joint] and mp_positions[joint]:
                # Extract Y-coordinates (vertical movement)
                bvh_y = [pos[1] for pos in bvh_positions[joint]]
                mp_y = [pos[1] for pos in mp_positions[joint]]
                
                if len(bvh_y) == len(mp_y) and len(bvh_y) > 10:
                    # Apply low-pass filter to extract gait cycle
                    b, a = butter(3, 0.1)
                    bvh_y_filtered = filtfilt(b, a, bvh_y)
                    mp_y_filtered = filtfilt(b, a, mp_y)
                    
                    # Calculate correlation
                    corr = np.corrcoef(bvh_y_filtered, mp_y_filtered)[0, 1]
                    if not np.isnan(corr):
                        correlations.append(corr)
        
        return np.mean(correlations) if correlations else 0.0
    
    def run_improved_accuracy_test(self, video_path: str, bvh_path: str, output_dir: str = "accuracy_tests") -> ImprovedAccuracyMetrics:
        """Run the improved accuracy test with visual quality metrics"""
        print(f"\n{'='*60}")
        print("IMPROVED BVH ACCURACY TEST")
        print(f"{'='*60}")
        print(f"Video: {video_path}")
        print(f"BVH: {bvh_path}")
        
        # Parse BVH file
        print("\nParsing BVH file...")
        bvh_data = self.parse_bvh(bvh_path)
        print(f"  Frames: {bvh_data['num_frames']}")
        print(f"  Joints: {len(bvh_data['skeleton']['joints'])}")
        
        # Extract MediaPipe data
        # Use sample_rate=2 to match BVH converter's default sampling
        print("\nExtracting MediaPipe data from video...")
        mediapipe_positions, mediapipe_rotations = self.extract_mediapipe_data(video_path, sample_rate=2)
        print(f"  Frames analyzed: {len(mediapipe_positions)}")
        
        # Compare motion with improved metrics
        print("\nAnalyzing motion quality...")
        metrics = self.compare_motion_improved(bvh_data, mediapipe_positions, mediapipe_rotations)
        
        # Save results
        Path(output_dir).mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save JSON report
        json_path = Path(output_dir) / f"improved_accuracy_test_{timestamp}.json"
        with open(json_path, 'w') as f:
            json.dump(asdict(metrics), f, indent=2)
        
        # Save text report
        text_path = Path(output_dir) / f"improved_accuracy_test_{timestamp}.txt"
        self._write_text_report(metrics, text_path)
        
        # Generate visualization
        self._generate_accuracy_plots(metrics, output_dir, timestamp)
        
        print(f"\nResults saved to {output_dir}")
        print(f"\n{'='*60}")
        print("IMPROVED ACCURACY SUMMARY")
        print(f"{'='*60}")
        print(f"Overall Accuracy Score: {metrics.overall_accuracy_score:.1f}/100")
        
        print(f"\n📊 Standard Metrics:")
        print(f"  Mean Angle Error: {metrics.mean_angle_error:.1f}°")
        print(f"  Relative Position Error: {metrics.mean_relative_position_error:.2f} units")
        print(f"  Bone Length Consistency: {np.mean(list(metrics.bone_length_consistency.values())):.1f}%")
        print(f"  Motion Smoothness: {metrics.motion_smoothness_score:.1f}/100")
        print(f"  Left-Right Symmetry: {metrics.left_right_symmetry:.1f}%")
        
        print(f"\n✨ Visual Quality Metrics:")
        print(f"  Visual Naturalness: {metrics.visual_naturalness_score:.1f}/100")
        print(f"  Temporal Drift: {metrics.temporal_drift_score:.1f}/100")
        print(f"  Over-smoothing Detection: {metrics.over_smoothing_score:.1f}/100")
        print(f"  Ground Contact Realism: {metrics.ground_contact_score:.1f}/100")
        print(f"  Motion Dynamics: {metrics.motion_dynamics_score:.1f}/100")
        
        if metrics.quality_warnings:
            print(f"\n⚠️  Quality Warnings:")
            for warning in metrics.quality_warnings:
                print(f"  - {warning}")
        
        print(f"\nWorst Performing Joints:")
        for joint in metrics.worst_joints[:3]:
            print(f"  - {joint}: {metrics.joint_angle_errors.get(joint, 0):.1f}° angle error")
        
        return metrics
    
    def _write_text_report(self, metrics: ImprovedAccuracyMetrics, output_path: Path):
        """Write detailed text report for improved metrics"""
        with open(output_path, 'w') as f:
            f.write("IMPROVED BVH ACCURACY TEST REPORT\n")
            f.write("="*60 + "\n\n")
            
            f.write(f"Overall Accuracy Score: {metrics.overall_accuracy_score:.1f}/100\n")
            f.write(f"Frames Analyzed: {metrics.valid_comparison_frames}/{metrics.total_frames}\n\n")
            
            f.write("STANDARD METRICS\n")
            f.write("-"*40 + "\n")
            f.write(f"Mean Angle Error: {metrics.mean_angle_error:.2f}°\n")
            f.write(f"Relative Position Error: {metrics.mean_relative_position_error:.3f} units\n")
            f.write(f"Bone Length Consistency: {np.mean(list(metrics.bone_length_consistency.values())):.1f}%\n")
            f.write(f"Motion Smoothness: {metrics.motion_smoothness_score:.1f}/100\n")
            f.write(f"Left-Right Symmetry: {metrics.left_right_symmetry:.1f}%\n")
            f.write(f"Symmetry Naturalness: {metrics.symmetry_naturalness:.1f}/100\n\n")
            
            f.write("VISUAL QUALITY METRICS\n")
            f.write("-"*40 + "\n")
            f.write(f"Visual Naturalness: {metrics.visual_naturalness_score:.1f}/100\n")
            f.write(f"Temporal Drift: {metrics.temporal_drift_score:.1f}/100\n")
            f.write(f"Over-smoothing Detection: {metrics.over_smoothing_score:.1f}/100\n")
            f.write(f"Ground Contact Realism: {metrics.ground_contact_score:.1f}/100\n")
            f.write(f"Motion Dynamics: {metrics.motion_dynamics_score:.1f}/100\n\n")
            
            if metrics.quality_warnings:
                f.write("QUALITY WARNINGS\n")
                f.write("-"*40 + "\n")
                for warning in metrics.quality_warnings:
                    f.write(f"• {warning}\n")
                f.write("\n")
            
            f.write("WORST PERFORMING JOINTS\n")
            f.write("-"*40 + "\n")
            for joint in metrics.worst_joints:
                angle_error = metrics.joint_angle_errors.get(joint, 0)
                pos_error = metrics.relative_position_errors.get(joint, 0)
                f.write(f"{joint}:\n")
                f.write(f"  Angle Error: {angle_error:.2f}°\n")
                f.write(f"  Position Error: {pos_error:.2f} units\n")
            
            f.write("\nCONFIDENCE SCORES\n")
            f.write("-"*40 + "\n")
            for metric, confidence in metrics.confidence_scores.items():
                f.write(f"{metric}: {confidence:.1f}%\n")
    
    def _generate_accuracy_plots(self, metrics: ImprovedAccuracyMetrics, output_dir: str, timestamp: str):
        """Generate enhanced visualization plots"""
        fig, axes = plt.subplots(3, 3, figsize=(18, 14))
        
        # Plot 1: Joint angle errors
        ax = axes[0, 0]
        joints = list(metrics.joint_angle_errors.keys())
        angle_errors = list(metrics.joint_angle_errors.values())
        colors = ['red' if j in metrics.worst_joints else 'blue' for j in joints]
        ax.bar(range(len(joints)), angle_errors, color=colors)
        ax.set_xticks(range(len(joints)))
        ax.set_xticklabels(joints, rotation=45, ha='right', fontsize=8)
        ax.set_title('Joint Angle Errors')
        ax.set_ylabel('Mean Error (degrees)')
        ax.axhline(y=metrics.mean_angle_error, color='green', linestyle='--', label='Overall Mean')
        ax.legend()
        
        # Plot 2: Visual Quality Scores
        ax = axes[0, 1]
        quality_metrics = ['Visual\nNaturalness', 'Temporal\nDrift', 'Over-smoothing\nDetection', 
                          'Ground\nContact', 'Motion\nDynamics']
        quality_scores = [
            metrics.visual_naturalness_score,
            metrics.temporal_drift_score,
            metrics.over_smoothing_score,
            metrics.ground_contact_score,
            metrics.motion_dynamics_score
        ]
        colors = ['green' if s >= 70 else 'orange' if s >= 40 else 'red' for s in quality_scores]
        ax.bar(quality_metrics, quality_scores, color=colors)
        ax.set_ylim(0, 100)
        ax.set_title('Visual Quality Metrics')
        ax.set_ylabel('Score (0-100)')
        ax.axhline(y=70, color='green', linestyle='--', alpha=0.3)
        ax.axhline(y=40, color='orange', linestyle='--', alpha=0.3)
        
        # Plot 3: Bone length consistency
        ax = axes[0, 2]
        if metrics.bone_length_consistency:
            bones = list(metrics.bone_length_consistency.keys())[:10]
            consistency = [metrics.bone_length_consistency[b] for b in bones]
            ax.barh(range(len(bones)), consistency)
            ax.set_yticks(range(len(bones)))
            ax.set_yticklabels(bones, fontsize=8)
            ax.set_title('Bone Length Consistency')
            ax.set_xlabel('Consistency Score (%)')
            ax.axvline(x=90, color='green', linestyle='--', alpha=0.3)
        
        # Plot 4: Standard vs Visual Metrics Comparison
        ax = axes[1, 0]
        categories = ['Standard\nAccuracy', 'Visual\nQuality']
        standard_score = (metrics.mean_angle_error + metrics.mean_relative_position_error) / 2
        standard_score = max(0, 100 - standard_score)
        visual_score = (metrics.visual_naturalness_score + metrics.temporal_drift_score + 
                       metrics.over_smoothing_score + metrics.ground_contact_score) / 4
        scores = [standard_score, visual_score]
        colors = ['blue', 'purple']
        ax.bar(categories, scores, color=colors)
        ax.set_ylim(0, 100)
        ax.set_title('Standard vs Visual Quality')
        ax.set_ylabel('Score (0-100)')
        
        # Plot 5: Symmetry Analysis
        ax = axes[1, 1]
        symmetry_data = {
            'Raw Symmetry': metrics.left_right_symmetry,
            'Symmetry\nNaturalness': metrics.symmetry_naturalness
        }
        ax.bar(symmetry_data.keys(), symmetry_data.values(), 
               color=['orange' if metrics.left_right_symmetry > 90 else 'blue', 'green'])
        ax.set_ylim(0, 100)
        ax.set_title('Symmetry Analysis')
        ax.set_ylabel('Score (%)')
        ax.axhline(y=90, color='red', linestyle='--', alpha=0.3, label='Over-smooth threshold')
        ax.legend()
        
        # Plot 6: Quality Warnings
        ax = axes[1, 2]
        ax.axis('off')
        if metrics.quality_warnings:
            warning_text = "⚠️ QUALITY WARNINGS:\n\n"
            for i, warning in enumerate(metrics.quality_warnings[:5], 1):
                warning_text += f"{i}. {warning}\n"
        else:
            warning_text = "✅ No quality warnings detected"
        ax.text(0.1, 0.5, warning_text, fontsize=10, verticalalignment='center',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow" if metrics.quality_warnings else "lightgreen", alpha=0.3))
        
        # Plot 7: Confidence scores
        ax = axes[2, 0]
        conf_names = list(metrics.confidence_scores.keys())
        conf_values = list(metrics.confidence_scores.values())
        ax.bar(conf_names, conf_values, color='purple')
        ax.set_ylim(0, 100)
        ax.set_title('Metric Confidence Scores')
        ax.set_ylabel('Confidence (%)')
        ax.set_xticklabels(conf_names, rotation=45, ha='right', fontsize=8)
        
        # Plot 8: Score Breakdown
        ax = axes[2, 1]
        components = {
            'Angles (25%)': max(0, 100 - metrics.mean_angle_error) * 0.25,
            'Position (15%)': max(0, 100 - metrics.mean_relative_position_error / 2) * 0.15,
            'Consistency (15%)': np.mean(list(metrics.bone_length_consistency.values())) * 0.15 if metrics.bone_length_consistency else 0,
            'Smoothness (10%)': metrics.motion_smoothness_score * 0.10,
            'Visual (35%)': (metrics.visual_naturalness_score * 0.10 + 
                           metrics.temporal_drift_score * 0.10 +
                           metrics.over_smoothing_score * 0.05 +
                           metrics.ground_contact_score * 0.10)
        }
        ax.pie(components.values(), labels=components.keys(), autopct='%1.1f%%', startangle=90)
        ax.set_title('Score Component Breakdown')
        
        # Plot 9: Summary
        ax = axes[2, 2]
        ax.axis('off')
        summary_text = f"""
IMPROVED ACCURACY SUMMARY

Overall Score: {metrics.overall_accuracy_score:.1f}/100

Standard Metrics:
• Angle Error: {metrics.mean_angle_error:.1f}°
• Position Error: {metrics.mean_relative_position_error:.1f}
• Symmetry: {metrics.left_right_symmetry:.1f}%

Visual Quality:
• Naturalness: {metrics.visual_naturalness_score:.1f}
• Drift Control: {metrics.temporal_drift_score:.1f}
• Ground Contact: {metrics.ground_contact_score:.1f}

Key Insights:
• {"Over-smoothed" if metrics.left_right_symmetry > 90 else "Natural"} symmetry
• {"Good" if metrics.temporal_drift_score > 70 else "Poor"} drift control
• {"Realistic" if metrics.ground_contact_score > 60 else "Unrealistic"} foot contact
"""
        ax.text(0.1, 0.5, summary_text, fontsize=9, verticalalignment='center')
        
        plt.suptitle('Improved BVH Accuracy Analysis', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plot_path = Path(output_dir) / f"improved_accuracy_plot_{timestamp}.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()


def main():
    parser = argparse.ArgumentParser(description="Improved BVH accuracy testing with visual quality metrics")
    parser.add_argument("--video", required=True, help="Path to input video")
    parser.add_argument("--bvh", required=True, help="Path to BVH file")
    parser.add_argument("--output-dir", default="accuracy_tests", help="Output directory")
    
    args = parser.parse_args()
    
    # Run improved accuracy test
    analyzer = ImprovedBVHAccuracyAnalyzer()
    analyzer.run_improved_accuracy_test(args.video, args.bvh, args.output_dir)


if __name__ == "__main__":
    main()