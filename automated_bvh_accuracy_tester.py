#!/usr/bin/env python3
"""
Improved BVH Accuracy Testing System

This version includes additional metrics to capture visual quality:
- Visual naturalness score
- Temporal drift measurement
- Over-smoothing detection
- Ground contact realism
- Better symmetry interpretation

Head/Neck additions:
- Head/Neck hip-relative position error
- Head/Neck direction error (proxy for orientation) derived from positions
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
from bvh_converter import ImprovedBVHConverter
# from tqdm import tqdm
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
    knee_stability_score: float   # Knee popping/jitter detection
    trajectory_score: float       # Global path accuracy

    # Head/Neck metrics
    head_position_error: float
    neck_position_error: float
    head_direction_error_deg: float
    neck_direction_error_deg: float

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
            ('LeftShoulder', 'LeftArm', 'LeftForeArm'),
            ('RightShoulder', 'RightArm', 'RightForeArm'),
            ('LeftArm', 'LeftForeArm', 'LeftHand'),
            ('RightArm', 'RightForeArm', 'RightHand'),
            ('LeftUpLeg', 'LeftLeg', 'LeftFoot'),
            ('RightUpLeg', 'RightLeg', 'RightFoot'),
            ('Hips', 'LeftUpLeg', 'LeftLeg'),
            ('Hips', 'RightUpLeg', 'RightLeg'),
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
        skeleton = self._parse_hierarchy(hierarchy)

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
                parent = None
                for item in reversed(joint_stack):
                    if item is not None:
                        parent = item
                        break

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

            elif 'End Site' in line:
                joint_stack.append(None)

            elif 'OFFSET' in line:
                if joint_stack and joint_stack[-1] is not None:
                    current_joint = joint_stack[-1]
                    offset = [float(x) for x in line.split('OFFSET')[1].strip().split()]
                    skeleton['joints'][current_joint]['offset'] = offset

            elif 'CHANNELS' in line:
                if joint_stack and joint_stack[-1] is not None:
                    current_joint = joint_stack[-1]
                    parts = line.split()
                    num_channels = int(parts[1])
                    channels = parts[2:2+num_channels]
                    skeleton['joints'][current_joint]['channels'] = channels

            elif '}' in line:
                if joint_stack:
                    joint_stack.pop()
                    current_joint = None
                    for item in reversed(joint_stack):
                        if item is not None:
                            current_joint = item
                            break

        return skeleton

    def _forward_kinematics_with_rotations(self, skeleton: Dict, frame_data: List[float]) -> Tuple[Dict, Dict]:
        """Calculate joint positions and rotations using forward kinematics"""
        positions = {}
        rotations = {}
        channel_index = 0

        def process_joint(joint_name, parent_transform=np.eye(4)):
            nonlocal channel_index

            joint = skeleton['joints'][joint_name]

            offset_transform = np.eye(4)
            offset_transform[:3, 3] = joint['offset']

            root_pos = [0.0, 0.0, 0.0]
            euler_angles = [0.0, 0.0, 0.0]

            for channel in joint['channels']:
                val = frame_data[channel_index]
                channel_index += 1

                if 'position' in channel.lower():
                    if 'x' in channel.lower():
                        root_pos[0] = val
                    elif 'y' in channel.lower():
                        root_pos[1] = val
                    elif 'z' in channel.lower():
                        root_pos[2] = val
                elif 'rotation' in channel.lower():
                    if 'x' in channel.lower():
                        euler_angles[0] = val
                    elif 'y' in channel.lower():
                        euler_angles[1] = val
                    elif 'z' in channel.lower():
                        euler_angles[2] = val

            root_translation = np.eye(4)
            root_translation[:3, 3] = root_pos

            rotation_matrix = np.eye(4)
            rotation_matrix[:3, :3] = R.from_euler('xyz', euler_angles, degrees=True).as_matrix()

            local_transform = offset_transform @ root_translation @ rotation_matrix
            global_transform = parent_transform @ local_transform

            positions[joint_name] = global_transform[:3, 3].copy()
            rotations[joint_name] = global_transform[:3, :3].copy()

            for child in joint['children']:
                process_joint(child, global_transform)

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
                    joint_positions = {}
                    for joint_name, landmark_indices in self.joint_mappings.items():
                        positions = []
                        for idx in landmark_indices:
                            lm = results.pose_world_landmarks.landmark[idx]
                            positions.append(np.array([lm.x, lm.y, lm.z]))

                        if positions:
                            joint_positions[joint_name] = np.mean(positions, axis=0) * 100

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
        """Calculate joint rotations from positions (placeholder)"""
        rotations = {}
        for joint_name in positions.keys():
            rotations[joint_name] = np.eye(3)
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

    @staticmethod
    def _safe_normalize(v: np.ndarray, eps: float = 1e-10) -> Optional[np.ndarray]:
        n = float(np.linalg.norm(v))
        if n < eps:
            return None
        return v / n

    @staticmethod
    def _angle_between(v1: np.ndarray, v2: np.ndarray) -> Optional[float]:
        v1n = ImprovedBVHAccuracyAnalyzer._safe_normalize(v1)
        v2n = ImprovedBVHAccuracyAnalyzer._safe_normalize(v2)
        if v1n is None or v2n is None:
            return None
        c = float(np.clip(np.dot(v1n, v2n), -1.0, 1.0))
        return float(np.degrees(np.arccos(c)))

    def _calculate_joint_angle(self, parent_pos: np.ndarray, joint_pos: np.ndarray,
                               child_pos: np.ndarray) -> float:
        vec1 = parent_pos - joint_pos
        vec2 = child_pos - joint_pos

        vec1_norm = vec1 / (np.linalg.norm(vec1) + 1e-6)
        vec2_norm = vec2 / (np.linalg.norm(vec2) + 1e-6)

        cos_angle = np.clip(np.dot(vec1_norm, vec2_norm), -1.0, 1.0)
        angle = np.degrees(np.arccos(cos_angle))

        return angle

    def _calculate_bone_length(self, pos1: np.ndarray, pos2: np.ndarray) -> float:
        return np.linalg.norm(pos2 - pos1)

    def _calculate_jerk(self, positions_over_time: List[np.ndarray], dt: float = 1/30.0) -> float:
        if len(positions_over_time) < 4:
            return 0.0

        positions = np.array(positions_over_time)
        positions = positions / 100.0

        velocities = np.diff(positions, axis=0) / dt
        accelerations = np.diff(velocities, axis=0) / dt
        jerk = np.diff(accelerations, axis=0) / dt

        return float(np.mean(np.linalg.norm(jerk, axis=1)))

    def _calculate_visual_naturalness(self, bvh_positions: List[Dict], mediapipe_positions: List[Dict]) -> float:
        naturalness_scores = []

        for frame_idx in range(1, min(len(bvh_positions), len(mediapipe_positions))):
            frame_score = 100.0

            for joint_name in self.joint_mappings.keys():
                if joint_name in bvh_positions[frame_idx] and joint_name in bvh_positions[frame_idx-1]:
                    velocity = np.linalg.norm(
                        bvh_positions[frame_idx][joint_name] - bvh_positions[frame_idx-1][joint_name]
                    )

                    if velocity > 50:
                        frame_score -= 10
                    elif velocity < 0.01 and 'Foot' not in joint_name:
                        frame_score -= 5

            naturalness_scores.append(max(0, frame_score))

        return float(np.mean(naturalness_scores)) if naturalness_scores else 50.0

    def _calculate_temporal_drift(self, bvh_positions: List[Dict], mediapipe_positions: List[Dict]) -> float:
        if not bvh_positions or not mediapipe_positions:
            return 0.0

        drift_scores = []

        for joint_name in ['Hips', 'Chest', 'Head']:
            if joint_name not in bvh_positions[0] or joint_name not in mediapipe_positions[0]:
                continue

            bvh_trajectory = [frame.get(joint_name, np.zeros(3)) for frame in bvh_positions]
            mp_trajectory = [frame.get(joint_name, np.zeros(3)) for frame in mediapipe_positions]

            if len(bvh_trajectory) > 10 and len(mp_trajectory) > 10:
                bvh_end_to_end = np.linalg.norm(bvh_trajectory[-1] - bvh_trajectory[0])
                mp_end_to_end = np.linalg.norm(mp_trajectory[-1] - mp_trajectory[0])

                drift = abs(bvh_end_to_end - mp_end_to_end)
                drift_scores.append(100 * np.exp(-drift / 100))

        return float(np.mean(drift_scores)) if drift_scores else 50.0

    def _detect_over_smoothing(self, positions_over_time: Dict[str, List[np.ndarray]]) -> float:
        smoothness_scores = []

        for joint_name, trajectory in positions_over_time.items():
            if len(trajectory) < 10:
                continue

            velocities = np.diff(trajectory, axis=0)
            accelerations = np.diff(velocities, axis=0)

            if len(accelerations) > 0:
                acc_variance = np.var(np.linalg.norm(accelerations, axis=1))

                if acc_variance < 0.001:
                    smoothness_scores.append(0)
                elif acc_variance > 10:
                    smoothness_scores.append(50)
                else:
                    smoothness_scores.append(100)

        return float(np.mean(smoothness_scores)) if smoothness_scores else 50.0

    def _calculate_ground_contact(self, foot_positions: Dict[str, List[np.ndarray]]) -> float:
        ground_scores = []

        for foot_name in ['LeftFoot', 'RightFoot']:
            if foot_name not in foot_positions or len(foot_positions[foot_name]) < 2:
                continue

            trajectory = foot_positions[foot_name]

            y_values = [pos[1] for pos in trajectory]
            min_y = min(y_values)
            ground_threshold = min_y + 5

            sliding_penalties = 0
            for i in range(1, len(trajectory)):
                if y_values[i] < ground_threshold and y_values[i-1] < ground_threshold:
                    horizontal_movement = np.sqrt(
                        (trajectory[i][0] - trajectory[i-1][0])**2 +
                        (trajectory[i][2] - trajectory[i-1][2])**2
                    )

                    if horizontal_movement > 2:
                        if horizontal_movement < 15:
                            sliding_penalties += 1

            if len(trajectory) > 0:
                slide_ratio = sliding_penalties / len(trajectory)
                score = 100 * (1 - min(1, slide_ratio * 2.5))
                ground_scores.append(score)

        return float(np.mean(ground_scores)) if ground_scores else 50.0

    def _calculate_knee_stability(self, bvh_positions: List[Dict]) -> float:
        stability_scores = []
        for side in ['Left', 'Right']:
            hip = f'{side}UpLeg'
            knee = f'{side}Leg'
            ankle = f'{side}Foot'

            angles = []
            for frame in bvh_positions:
                if all(k in frame for k in [hip, knee, ankle]):
                    angle = self._calculate_joint_angle(frame[hip], frame[knee], frame[ankle])
                    angles.append(angle)

            if len(angles) > 2:
                velocities = np.diff(angles)
                accelerations = np.diff(velocities)

                mean_acc = np.mean(np.abs(accelerations))
                score = max(0, 100 - mean_acc * 20)
                stability_scores.append(score)

        return float(np.mean(stability_scores)) if stability_scores else 50.0

    def _calculate_trajectory_similarity(self, bvh_positions: List[Dict], mediapipe_positions: List[Dict]) -> float:
        if not bvh_positions or not mediapipe_positions:
            return 0.0

        bvh_path = []
        mp_path = []

        for i in range(min(len(bvh_positions), len(mediapipe_positions))):
            if 'Hips' in bvh_positions[i] and 'Hips' in mediapipe_positions[i]:
                bvh_path.append(bvh_positions[i]['Hips'])
                mp_path.append(mediapipe_positions[i]['Hips'])

        if len(bvh_path) < 10:
            return 0.0

        bvh_path = np.array(bvh_path)
        mp_path = np.array(mp_path)

        bvh_path = bvh_path - bvh_path[0]
        mp_path = mp_path - mp_path[0]

        bvh_len = np.sum(np.linalg.norm(np.diff(bvh_path, axis=0), axis=1))
        mp_len = np.sum(np.linalg.norm(np.diff(mp_path, axis=0), axis=1))

        length_ratio = (min(bvh_len, mp_len) / (max(bvh_len, mp_len) + 1e-6)) * 100

        distances = np.linalg.norm(bvh_path - mp_path, axis=1)
        mean_dist = np.mean(distances)

        normalized_error = mean_dist / (mp_len + 1.0)
        shape_score = 100 * np.exp(-normalized_error * 5)

        return float(length_ratio * 0.3 + shape_score * 0.7)

    def plot_trajectory_comparison(self, bvh_positions: List[Dict], mediapipe_positions: List[Dict], output_path: str = "trajectory_comparison.png"):
        if not bvh_positions or not mediapipe_positions:
            return

        bvh_path = []
        mp_path = []

        for i in range(min(len(bvh_positions), len(mediapipe_positions))):
            if 'Hips' in bvh_positions[i] and 'Hips' in mediapipe_positions[i]:
                bvh_path.append(bvh_positions[i]['Hips'])
                mp_path.append(mediapipe_positions[i]['Hips'])

        if not bvh_path:
            return

        bvh_path = np.array(bvh_path)
        mp_path = np.array(mp_path)

        bvh_path = bvh_path - bvh_path[0]
        mp_path = mp_path - mp_path[0]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        ax1.plot(bvh_path[:, 0], bvh_path[:, 2], label='BVH (Generated)', alpha=0.7)
        ax1.plot(mp_path[:, 0], mp_path[:, 2], label='MediaPipe (Reference)', alpha=0.7, linestyle='--')
        ax1.set_title('Top-Down View (X-Z Plane)')
        ax1.set_xlabel('X (Right/Left)')
        ax1.set_ylabel('Z (Forward/Back)')
        ax1.legend()
        ax1.grid(True)
        ax1.axis('equal')

        ax2.plot(bvh_path[:, 2], bvh_path[:, 1], label='BVH (Generated)', alpha=0.7)
        ax2.plot(mp_path[:, 2], mp_path[:, 1], label='MediaPipe (Reference)', alpha=0.7, linestyle='--')
        ax2.set_title('Side View (Z-Y Plane)')
        ax2.set_xlabel('Z (Forward/Back)')
        ax2.set_ylabel('Y (Up/Down)')
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path)
        plt.close()

    def _evaluate_symmetry_naturalness(self, symmetry_score: float) -> Tuple[float, List[str]]:
        warnings = []
        naturalness = 100.0

        if symmetry_score > 95:
            warnings.append("Symmetry too perfect (>95%) - possible over-smoothing")
            naturalness = 50.0
        elif symmetry_score > 90:
            warnings.append("Very high symmetry (>90%) - may be artificial")
            naturalness = 75.0
        elif symmetry_score < 30:
            warnings.append("Very low symmetry (<30%) - possible tracking issues")
            naturalness = 60.0
        elif 40 <= symmetry_score <= 70:
            naturalness = 100.0
        else:
            naturalness = 85.0

        return naturalness, warnings

    def _compute_head_neck_metrics(self, bvh_positions: List[Dict], mp_positions: List[Dict]) -> Tuple[float, float, float, float]:
        """
        Returns:
          head_pos_err, neck_pos_err (hip-relative mean)
          head_dir_err_deg, neck_dir_err_deg (mean)
        """
        head_pos_errs = []
        neck_pos_errs = []
        head_dir_errs = []
        neck_dir_errs = []

        n = min(len(bvh_positions), len(mp_positions))
        for i in range(n):
            b = bvh_positions[i]
            m = mp_positions[i]
            if not b or not m:
                continue
            if 'Hips' not in b or 'Hips' not in m:
                continue

            b_rel = {k: (v - b['Hips']) for k, v in b.items()}
            m_rel = {k: (v - m['Hips']) for k, v in m.items()}

            if 'Head' in b_rel and 'Head' in m_rel:
                head_pos_errs.append(float(np.linalg.norm(b_rel['Head'] - m_rel['Head'])))
            if 'Neck' in b_rel and 'Neck' in m_rel:
                neck_pos_errs.append(float(np.linalg.norm(b_rel['Neck'] - m_rel['Neck'])))

            if all(k in b for k in ['Head', 'Neck']) and all(k in m for k in ['Head', 'Neck']):
                ang = self._angle_between(b['Head'] - b['Neck'], m['Head'] - m['Neck'])
                if ang is not None:
                    head_dir_errs.append(ang)

            if all(k in b for k in ['Neck', 'Chest']) and all(k in m for k in ['Neck', 'Chest']):
                ang = self._angle_between(b['Neck'] - b['Chest'], m['Neck'] - m['Chest'])
                if ang is not None:
                    neck_dir_errs.append(ang)

        head_pos = float(np.mean(head_pos_errs)) if head_pos_errs else 0.0
        neck_pos = float(np.mean(neck_pos_errs)) if neck_pos_errs else 0.0
        head_dir = float(np.mean(head_dir_errs)) if head_dir_errs else 0.0
        neck_dir = float(np.mean(neck_dir_errs)) if neck_dir_errs else 0.0
        return head_pos, neck_pos, head_dir, neck_dir

    def compare_motion_improved(self, bvh_data: Dict, mediapipe_positions: List[Dict],
                                mediapipe_rotations: List[Dict]) -> ImprovedAccuracyMetrics:
        bvh_positions = bvh_data['joint_positions']

        min_frames = min(len(bvh_positions), len(mediapipe_positions))

        angle_errors = []
        joint_angle_errors = {joint: [] for joint in self.joint_mappings.keys()}
        relative_position_errors = []
        joint_relative_errors = {joint: [] for joint in self.joint_mappings.keys()}
        bone_length_variations = {bone: [] for bone in self.bone_connections}

        all_bvh_positions = {joint: [] for joint in self.joint_mappings.keys()}
        all_mp_positions = {joint: [] for joint in self.joint_mappings.keys()}
        foot_positions = {'LeftFoot': [], 'RightFoot': []}

        for frame_idx in range(min_frames):
            bvh_frame = bvh_positions[frame_idx]
            mp_frame = mediapipe_positions[frame_idx]

            bvh_relative = self._convert_to_hip_relative(bvh_frame)
            mp_relative = self._convert_to_hip_relative(mp_frame)

            frame_relative_errors = []
            for joint_name in self.joint_mappings.keys():
                if joint_name in bvh_relative and joint_name in mp_relative:
                    error = np.linalg.norm(bvh_relative[joint_name] - mp_relative[joint_name])
                    frame_relative_errors.append(error)
                    joint_relative_errors[joint_name].append(error)

                    all_bvh_positions[joint_name].append(bvh_frame[joint_name])
                    all_mp_positions[joint_name].append(mp_frame[joint_name])

                    if joint_name in foot_positions and joint_name in bvh_frame:
                        foot_positions[joint_name].append(bvh_frame[joint_name])

            if frame_relative_errors:
                relative_position_errors.append(np.mean(frame_relative_errors))

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

        position_jitter_scores = []
        for joint_name in self.joint_mappings.keys():
            if all_bvh_positions[joint_name]:
                jerk = self._calculate_jerk(all_bvh_positions[joint_name])
                position_jitter_scores.append(jerk)

        bone_consistency_scores = {}
        mean_bone_variation = []
        for bone_name, lengths in bone_length_variations.items():
            if lengths:
                std_dev = np.std(lengths)
                mean_length = np.mean(lengths)
                consistency = 100 * (1 - std_dev / (mean_length + 1e-6))
                bone_consistency_scores[bone_name] = max(0, float(consistency))
                mean_bone_variation.append(float(std_dev))

        symmetry_score = self._calculate_symmetry(joint_relative_errors)

        visual_naturalness = self._calculate_visual_naturalness(bvh_positions, mediapipe_positions)
        temporal_drift = self._calculate_temporal_drift(bvh_positions, mediapipe_positions)
        over_smoothing = self._detect_over_smoothing(all_bvh_positions)
        ground_contact = self._calculate_ground_contact(foot_positions)
        knee_stability = self._calculate_knee_stability(bvh_positions)
        trajectory_score = self._calculate_trajectory_similarity(bvh_positions, mediapipe_positions)

        self.plot_trajectory_comparison(bvh_positions, mediapipe_positions, "test_output/trajectory_comparison.png")

        symmetry_naturalness, symmetry_warnings = self._evaluate_symmetry_naturalness(symmetry_score)
        motion_dynamics = (visual_naturalness + over_smoothing) / 2

        head_pos_err, neck_pos_err, head_dir_err, neck_dir_err = self._compute_head_neck_metrics(bvh_positions, mediapipe_positions)

        quality_warnings = symmetry_warnings.copy()
        if temporal_drift < 30:
            quality_warnings.append(f"High temporal drift detected (score: {temporal_drift:.1f})")
        if over_smoothing < 30:
            quality_warnings.append(f"Over-smoothing detected (score: {over_smoothing:.1f})")
        if ground_contact < 40:
            quality_warnings.append(f"Foot sliding detected (score: {ground_contact:.1f})")
        if knee_stability < 60:
            quality_warnings.append(f"Knee instability/popping detected (score: {knee_stability:.1f})")
        if head_pos_err > 15:
            quality_warnings.append(f"Head position mismatch high (mean: {head_pos_err:.2f} units)")
        if neck_pos_err > 12:
            quality_warnings.append(f"Neck position mismatch high (mean: {neck_pos_err:.2f} units)")
        if head_dir_err > 25:
            quality_warnings.append(f"Head direction mismatch high (mean: {head_dir_err:.1f}°)")
        if neck_dir_err > 25:
            quality_warnings.append(f"Neck direction mismatch high (mean: {neck_dir_err:.1f}°)")

        gait_correlation = self._calculate_gait_correlation(all_bvh_positions, all_mp_positions)

        angle_errors_array = np.array(angle_errors) if angle_errors else np.array([0])
        relative_errors_array = np.array(relative_position_errors) if relative_position_errors else np.array([0])

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

        worst_frame_indices = []
        if len(angle_errors) > 0:
            worst_frame_indices = np.argsort(angle_errors)[-10:].tolist()

        angle_score = max(0, 100 - float(np.mean(angle_errors_array)))
        relative_pos_score = max(0, 100 - float(np.mean(relative_errors_array)) / 2)
        consistency_score = float(np.mean(list(bone_consistency_scores.values()))) if bone_consistency_scores else 0.0

        if position_jitter_scores:
            mean_jerk = float(np.mean(position_jitter_scores))
            if mean_jerk > 0:
                smoothness_score = max(0, min(100, 110 - 20 * np.log10(mean_jerk)))
            else:
                smoothness_score = 100.0
        else:
            smoothness_score = 50.0

        headneck_dir_score = float(100 * np.exp(-(head_dir_err + neck_dir_err) / 60.0)) if (head_dir_err or neck_dir_err) else 100.0

        visual_quality_score = (
            visual_naturalness * 0.10 +
            temporal_drift * 0.10 +
            over_smoothing * 0.05 +
            ground_contact * 0.10 +
            knee_stability * 0.10 +
            trajectory_score * 0.10
        )

        overall_score = (
            angle_score * 0.25 +
            relative_pos_score * 0.15 +
            consistency_score * 0.15 +
            smoothness_score * 0.10 +
            visual_quality_score +
            headneck_dir_score * 0.01
        )

        confidence_scores = {
            'angle_confidence': min(100, len(angle_errors) / min_frames * 100) if min_frames > 0 else 0,
            'position_confidence': min(100, len(relative_position_errors) / min_frames * 100) if min_frames > 0 else 0,
            'consistency_confidence': min(100, len(bone_consistency_scores) * 10) if bone_consistency_scores else 0,
            'temporal_confidence': 100 if min_frames > 30 else min_frames / 30 * 100,
            'visual_confidence': 80 if min_frames > 60 else min_frames / 60 * 80,
            'headneck_confidence': 100 if (head_pos_err or neck_pos_err) else 50
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
            acceleration_consistency=0.0,
            position_jitter_score=float(np.mean(position_jitter_scores)) if position_jitter_scores else 0,
            angular_velocity_correlation=0.0,
            left_right_symmetry=float(symmetry_score),
            symmetry_naturalness=float(symmetry_naturalness),
            visual_naturalness_score=float(visual_naturalness),
            temporal_drift_score=float(temporal_drift),
            over_smoothing_score=float(over_smoothing),
            ground_contact_score=float(ground_contact),
            motion_dynamics_score=float(motion_dynamics),
            knee_stability_score=float(knee_stability),
            trajectory_score=float(trajectory_score),
            head_position_error=float(head_pos_err),
            neck_position_error=float(neck_pos_err),
            head_direction_error_deg=float(head_dir_err),
            neck_direction_error_deg=float(neck_dir_err),
            worst_frames=worst_frame_indices,
            worst_joints=worst_joints,
            confidence_scores=confidence_scores,
            quality_warnings=quality_warnings
        )

    def _calculate_symmetry(self, joint_errors: Dict[str, List[float]]) -> float:
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
            diff = abs(np.mean(left_errors) - np.mean(right_errors))
            max_error = max(np.mean(left_errors), np.mean(right_errors))
            if max_error > 0:
                symmetry = 100 * (1 - diff / max_error)
                return max(0, float(symmetry))

        return 50.0

    def _calculate_gait_correlation(self, bvh_positions: Dict, mp_positions: Dict) -> float:
        gait_joints = ['LeftFoot', 'RightFoot']
        correlations = []

        for joint in gait_joints:
            if bvh_positions[joint] and mp_positions[joint]:
                bvh_y = [pos[1] for pos in bvh_positions[joint]]
                mp_y = [pos[1] for pos in mp_positions[joint]]

                if len(bvh_y) == len(mp_y) and len(bvh_y) > 10:
                    b, a = butter(3, 0.1)
                    bvh_y_filtered = filtfilt(b, a, bvh_y)
                    mp_y_filtered = filtfilt(b, a, mp_y)

                    corr = np.corrcoef(bvh_y_filtered, mp_y_filtered)[0, 1]
                    if not np.isnan(corr):
                        correlations.append(corr)

        return float(np.mean(correlations)) if correlations else 0.0

    def run_improved_accuracy_test(self, video_path: str, bvh_path: str, output_dir: str = "accuracy_tests") -> ImprovedAccuracyMetrics:
        print(f"\n{'='*60}")
        print("IMPROVED BVH ACCURACY TEST")
        print(f"{'='*60}")
        print(f"Video: {video_path}")
        print(f"BVH: {bvh_path}")

        print("\nParsing BVH file...")
        bvh_data = self.parse_bvh(bvh_path)
        print(f"  Frames: {bvh_data['num_frames']}")
        print(f"  Joints: {len(bvh_data['skeleton']['joints'])}")

        print("\nExtracting MediaPipe data from video...")
        mediapipe_positions, mediapipe_rotations = self.extract_mediapipe_data(video_path, sample_rate=2)
        print(f"  Frames analyzed: {len(mediapipe_positions)}")

        print("\nAnalyzing motion quality...")
        metrics = self.compare_motion_improved(bvh_data, mediapipe_positions, mediapipe_rotations)

        Path(output_dir).mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        json_path = Path(output_dir) / f"improved_accuracy_test_{timestamp}.json"
        with open(json_path, 'w') as f:
            json.dump(asdict(metrics), f, indent=2)

        text_path = Path(output_dir) / f"improved_accuracy_test_{timestamp}.txt"
        self._write_text_report(metrics, text_path)

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

        print(f"\n🧠 Head/Neck Metrics:")
        print(f"  Head position error: {metrics.head_position_error:.2f} units")
        print(f"  Neck position error: {metrics.neck_position_error:.2f} units")
        print(f"  Head direction error: {metrics.head_direction_error_deg:.1f}°")
        print(f"  Neck direction error: {metrics.neck_direction_error_deg:.1f}°")

        if metrics.quality_warnings:
            print(f"\n⚠️  Quality Warnings:")
            for warning in metrics.quality_warnings:
                print(f"  - {warning}")

        print(f"\nWorst Performing Joints:")
        for joint in metrics.worst_joints[:3]:
            print(f"  - {joint}: {metrics.joint_angle_errors.get(joint, 0):.1f}° angle error")

        return metrics

    def _write_text_report(self, metrics: ImprovedAccuracyMetrics, output_path: Path):
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

            f.write("HEAD/NECK METRICS\n")
            f.write("-"*40 + "\n")
            f.write(f"Head position error (hip-relative): {metrics.head_position_error:.2f} units\n")
            f.write(f"Neck position error (hip-relative): {metrics.neck_position_error:.2f} units\n")
            f.write(f"Head direction error: {metrics.head_direction_error_deg:.2f}°\n")
            f.write(f"Neck direction error: {metrics.neck_direction_error_deg:.2f}°\n\n")

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

    def _generate_accuracy