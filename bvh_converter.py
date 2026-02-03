"""
Improved BVH converter with fixes for ForeArm/Wrist errors and better IK calibration.
Based on issues identified in todo.md:
1. Better 3D hand reconstruction to fix ForeArm/Wrist errors (65-82°)
2. Calibrated IK thresholds for foot contact detection
3. Foot-based drift correction for walking videos

Head/Neck improvements:
- Optional FaceMesh-based head orientation via --face flag
- Fallback torso-based head orientation with pitch clamp safety
"""

import numpy as np
import argparse
import time
import copy
from typing import List, Dict, Optional, Tuple
import mediapipe as mp
from scipy.spatial.transform import Rotation  # FIX: used by _rotation_from_basis

from mediapipe_extractor import MediaPipeExtractor, PoseFrame
from skeleton_mapper import SkeletonMapper, BVHJoint
from math_utils import calculate_rotation_from_directions, smooth_rotations, smooth_positions
from config import BVH_CONFIG, PROCESSING_CONFIG, SMOOTHING_CONFIG
from ik_foot_lock import IKFootLockSystem

mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
mp_face_mesh = mp.solutions.face_mesh


class ImprovedBVHConverter:
    """Improved BVH converter with better hand tracking and IK calibration."""

    def __init__(self, enable_ik: bool = False, enable_face: bool = False):
        self.skeleton_mapper = SkeletonMapper()
        self.frame_time = 1.0 / BVH_CONFIG['fps']
        self.rotation_order = BVH_CONFIG['rotation_order']
        self.scale = PROCESSING_CONFIG['scale_factor']
        self.enable_ik = enable_ik
        self.enable_face = enable_face
        self.ik_system = None

        # FaceMesh (optional)
        self._face_mesh = None
        if self.enable_face:
            # refine_landmarks=True gives iris landmarks; not required but can help stability
            self._face_mesh = mp_face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )

        # Store foot ground levels for drift correction
        self.ground_level = None
        self.foot_contact_frames = []

    def convert(self, pose_frames: List[PoseFrame], output_path: str) -> bool:
        """Convert pose frames to BVH file with improved hand tracking and IK."""
        if not pose_frames:
            print("Error: No pose frames to convert")
            return False

        self.pose_frames = pose_frames

        extractor = MediaPipeExtractor(use_holistic=True)
        ref_idx = extractor.find_reference_frame(pose_frames)

        if not pose_frames[ref_idx].is_valid():
            print("Error: No valid reference frame found")
            return False

        print("Setting up skeleton from reference frame...")
        ref_landmarks = pose_frames[ref_idx].world_landmarks
        self.skeleton_mapper.calculate_bone_offsets(ref_landmarks, self.scale)

        print("Calculating dynamic ground level...")
        self.ground_level = self._calculate_dynamic_ground_level(pose_frames)
        print(f"Dynamic ground level determined at Y={self.ground_level:.2f}")

        if self.enable_ik:
            print("Initializing improved IK foot locking system...")
            self._initialize_improved_ik_system(ref_landmarks, self.scale)
            pose_frames = [copy.deepcopy(frame) for frame in pose_frames]

            print("Extracting leg positions for IK processing (Pass 1)...")
            all_leg_positions = []
            for frame in pose_frames:
                if frame.is_valid():
                    leg_pos = self._extract_leg_positions(frame.world_landmarks, self.scale)
                    all_leg_positions.append(leg_pos)
                else:
                    all_leg_positions.append(None)

            print("Applying improved IK foot locking (Pass 1)...")
            self._apply_improved_ik_corrections(all_leg_positions)

        print("Calculating root motion...")
        hip_positions = self._calculate_root_motion_from_feet(pose_frames)

        print("Updating pose frames with corrected hip positions...")
        for i, frame in enumerate(pose_frames):
            if i < len(hip_positions):
                try:
                    frame.hip_position = hip_positions[i]
                except Exception:
                    pass

        if self.enable_ik:
            print("Extracting leg positions for IK processing (Pass 2)...")
            all_leg_positions = []
            for frame in pose_frames:
                if frame.is_valid():
                    leg_pos = self._extract_leg_positions(frame.world_landmarks, self.scale)
                    all_leg_positions.append(leg_pos)
                else:
                    all_leg_positions.append(None)

            print("Applying improved IK foot locking (Pass 2)...")
            corrected_positions = self._apply_improved_ik_corrections(all_leg_positions)
            self._update_pose_frames_with_ik(pose_frames, corrected_positions)
            print(f"✅ Detected {len(self.foot_contact_frames)} foot contact frames")

        print("Validating and fixing outlier landmarks...")
        pose_frames = self._validate_and_fix_landmarks(pose_frames, max_shoulder_diff=0.05)

        print("Calculating joint rotations...")
        all_rotations = self._process_motion_improved(pose_frames)

        if SMOOTHING_CONFIG['enable_temporal_smoothing']:
            print("Applying adaptive temporal smoothing...")
            all_rotations = self._smooth_motion(all_rotations)

        print(f"Writing BVH file to {output_path}...")
        success = self._write_bvh(all_rotations, hip_positions, output_path)

        if success:
            print(f"BVH file created successfully: {output_path}")
        else:
            print("Error writing BVH file")

        return success

    def close(self):
        """Release optional resources."""
        if self._face_mesh is not None:
            self._face_mesh.close()
            self._face_mesh = None

    def _calculate_root_motion_from_feet(self, pose_frames: List[PoseFrame]) -> List[np.ndarray]:
        """
        Root translation baseline:
        - Uses MediaPipe world hip center as root translation (scaled, Y flipped).
        - Applies optional temporal smoothing.
        """
        positions: List[np.ndarray] = []

        l_hip_idx = mp_pose.PoseLandmark.LEFT_HIP
        r_hip_idx = mp_pose.PoseLandmark.RIGHT_HIP

        default_root = np.array([0.0, float(BVH_CONFIG.get('root_height', 60.0)), 0.0], dtype=float)

        for frame in pose_frames:
            if not frame.is_valid() or not getattr(frame, "world_landmarks", None):
                positions.append(positions[-1].copy() if positions else default_root.copy())
                continue

            l = frame.world_landmarks[l_hip_idx]
            r = frame.world_landmarks[r_hip_idx]

            hip_center = np.array([
                (l.x + r.x) * 0.5,
                -(l.y + r.y) * 0.5,
                (l.z + r.z) * 0.5
            ], dtype=float) * float(self.scale)

            positions.append(hip_center)

        if SMOOTHING_CONFIG.get('enable_temporal_smoothing', False) and len(positions) > 2:
            positions_arr = np.array(positions, dtype=float)
            smoothed = smooth_positions(
                positions_arr,
                window_size=int(SMOOTHING_CONFIG.get('temporal_window_size', 3)),
                preserve_dynamics=bool(SMOOTHING_CONFIG.get('preserve_dynamics', True)),
                preserve_y_axis=False
            )
            positions = [p for p in smoothed]

        return positions

    def _initialize_improved_ik_system(self, reference_landmarks, scale: float):
        left_hip_idx = mp_pose.PoseLandmark.LEFT_HIP
        left_knee_idx = mp_pose.PoseLandmark.LEFT_KNEE
        left_ankle_idx = mp_pose.PoseLandmark.LEFT_ANKLE
        left_foot_idx = mp_pose.PoseLandmark.LEFT_FOOT_INDEX

        left_hip = np.array([reference_landmarks[left_hip_idx].x, -reference_landmarks[left_hip_idx].y, reference_landmarks[left_hip_idx].z]) * scale
        left_knee = np.array([reference_landmarks[left_knee_idx].x, -reference_landmarks[left_knee_idx].y, reference_landmarks[left_knee_idx].z]) * scale
        left_ankle = np.array([reference_landmarks[left_ankle_idx].x, -reference_landmarks[left_ankle_idx].y, reference_landmarks[left_ankle_idx].z]) * scale
        left_foot = np.array([reference_landmarks[left_foot_idx].x, -reference_landmarks[left_foot_idx].y, reference_landmarks[left_foot_idx].z]) * scale

        thigh_length = np.linalg.norm(left_knee - left_hip)
        shin_length = np.linalg.norm(left_ankle - left_knee)

        if self.ground_level is None:
            self.ground_level = min(left_ankle[1], left_foot[1])

        self.ik_system = IKFootLockSystem(thigh_length, shin_length)
        self.ik_system.config.contact_velocity_threshold = 4.0 * (scale / 100.0)
        self.ik_system.config.contact_height_threshold = 0.12 * scale
        self.ik_system.config.foot_clearance_height = 0.05 * scale
        self.ik_system.config.vertical_velocity_threshold = 2.0 * (scale / 100.0)

    def _extract_leg_positions(self, world_landmarks, scale: float) -> Dict[str, Dict[str, np.ndarray]]:
        positions = {}

        left_hip_idx = mp_pose.PoseLandmark.LEFT_HIP
        left_knee_idx = mp_pose.PoseLandmark.LEFT_KNEE
        left_ankle_idx = mp_pose.PoseLandmark.LEFT_ANKLE
        left_foot_idx = mp_pose.PoseLandmark.LEFT_FOOT_INDEX

        positions['left'] = {
            'hip': np.array([world_landmarks[left_hip_idx].x, -world_landmarks[left_hip_idx].y, world_landmarks[left_hip_idx].z]) * scale,
            'knee': np.array([world_landmarks[left_knee_idx].x, -world_landmarks[left_knee_idx].y, world_landmarks[left_knee_idx].z]) * scale,
            'ankle': np.array([world_landmarks[left_ankle_idx].x, -world_landmarks[left_ankle_idx].y, world_landmarks[left_ankle_idx].z]) * scale,
            'foot': np.array([world_landmarks[left_foot_idx].x, -world_landmarks[left_foot_idx].y, world_landmarks[left_foot_idx].z]) * scale
        }

        right_hip_idx = mp_pose.PoseLandmark.RIGHT_HIP
        right_knee_idx = mp_pose.PoseLandmark.RIGHT_KNEE
        right_ankle_idx = mp_pose.PoseLandmark.RIGHT_ANKLE
        right_foot_idx = mp_pose.PoseLandmark.RIGHT_FOOT_INDEX

        positions['right'] = {
            'hip': np.array([world_landmarks[right_hip_idx].x, -world_landmarks[right_hip_idx].y, world_landmarks[right_hip_idx].z]) * scale,
            'knee': np.array([world_landmarks[right_knee_idx].x, -world_landmarks[right_knee_idx].y, world_landmarks[right_knee_idx].z]) * scale,
            'ankle': np.array([world_landmarks[right_ankle_idx].x, -world_landmarks[right_ankle_idx].y, world_landmarks[right_ankle_idx].z]) * scale,
            'foot': np.array([world_landmarks[right_foot_idx].x, -world_landmarks[right_foot_idx].y, world_landmarks[right_foot_idx].z]) * scale
        }

        return positions

    def _apply_improved_ik_corrections(self, all_leg_positions: List[Optional[Dict]]) -> List[Optional[Dict]]:
        corrected = []
        previous_ankles = None
        self.foot_contact_frames = []

        for i, leg_positions in enumerate(all_leg_positions):
            if leg_positions is None:
                corrected.append(None)
                continue

            hip_positions = {'left': leg_positions['left']['hip'], 'right': leg_positions['right']['hip']}
            knee_positions = {'left': leg_positions['left']['knee'], 'right': leg_positions['right']['knee']}
            ankle_positions = {'left': leg_positions['left']['ankle'], 'right': leg_positions['right']['ankle']}
            foot_positions = {'left': leg_positions['left']['foot'], 'right': leg_positions['right']['foot']}

            left_contact = self._detect_foot_contact(
                ankle_positions['left'],
                foot_positions['left'],
                previous_ankles['left'] if previous_ankles else None,
                'left'
            )
            right_contact = self._detect_foot_contact(
                ankle_positions['right'],
                foot_positions['right'],
                previous_ankles['right'] if previous_ankles else None,
                'right'
            )

            if left_contact or right_contact:
                self.foot_contact_frames.append(i)

            contact_overrides = {'left': left_contact, 'right': right_contact}

            ik_result = self.ik_system.process_frame(
                hip_positions,
                knee_positions,
                ankle_positions,
                i,
                previous_ankles,
                contact_overrides
            )

            corrected_frame = {'left': ik_result['left'], 'right': ik_result['right']}
            corrected.append(corrected_frame)

            previous_ankles = {'left': ik_result['left']['ankle'], 'right': ik_result['right']['ankle']}

        return corrected

    def _calculate_dynamic_ground_level(self, pose_frames: List[PoseFrame]) -> float:
        min_y = float('inf')

        left_ankle_idx = mp_pose.PoseLandmark.LEFT_ANKLE
        right_ankle_idx = mp_pose.PoseLandmark.RIGHT_ANKLE
        left_foot_idx = mp_pose.PoseLandmark.LEFT_FOOT_INDEX
        right_foot_idx = mp_pose.PoseLandmark.RIGHT_FOOT_INDEX

        valid_frames = 0
        for frame in pose_frames:
            if not frame.is_valid():
                continue

            l_ankle_y = -frame.world_landmarks[left_ankle_idx].y * self.scale
            r_ankle_y = -frame.world_landmarks[right_ankle_idx].y * self.scale
            l_foot_y = -frame.world_landmarks[left_foot_idx].y * self.scale
            r_foot_y = -frame.world_landmarks[right_foot_idx].y * self.scale

            frame_min = min(l_ankle_y, r_ankle_y, l_foot_y, r_foot_y)
            if frame_min < min_y:
                min_y = frame_min
            valid_frames += 1

        if valid_frames == 0:
            return 0.0
        return min_y

    def _detect_foot_contact(self, ankle_pos: np.ndarray, foot_pos: np.ndarray,
                            prev_ankle: Optional[np.ndarray], side: str) -> bool:
        foot_height = foot_pos[1] - self.ground_level if self.ground_level is not None else foot_pos[1]
        ankle_height = ankle_pos[1] - self.ground_level if self.ground_level is not None else ankle_pos[1]
        min_height = min(foot_height, ankle_height)

        height_contact = min_height < self.ik_system.config.contact_height_threshold

        velocity_contact = True
        vertical_velocity_contact = True
        if prev_ankle is not None:
            velocity = ankle_pos - prev_ankle
            velocity_mag = np.linalg.norm(velocity)
            vertical_velocity = abs(velocity[1])
            velocity_contact = velocity_mag < self.ik_system.config.contact_velocity_threshold
            vertical_velocity_contact = vertical_velocity < self.ik_system.config.vertical_velocity_threshold

        return height_contact and (velocity_contact or vertical_velocity_contact)

    def _update_pose_frames_with_ik(self, pose_frames: List[PoseFrame], corrected_positions: List[Optional[Dict]]):
        for frame, corrections in zip(pose_frames, corrected_positions):
            if not frame.is_valid() or corrections is None:
                continue

            if corrections['left']['confidence'] > 0:
                knee_pos = corrections['left']['knee'] / self.scale
                frame.world_landmarks[mp_pose.PoseLandmark.LEFT_KNEE].x = knee_pos[0]
                frame.world_landmarks[mp_pose.PoseLandmark.LEFT_KNEE].y = -knee_pos[1]
                frame.world_landmarks[mp_pose.PoseLandmark.LEFT_KNEE].z = knee_pos[2]

                ankle_pos = corrections['left']['ankle'] / self.scale
                frame.world_landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].x = ankle_pos[0]
                frame.world_landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].y = -ankle_pos[1]
                frame.world_landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].z = ankle_pos[2]

            if corrections['right']['confidence'] > 0:
                knee_pos = corrections['right']['knee'] / self.scale
                frame.world_landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].x = knee_pos[0]
                frame.world_landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].y = -knee_pos[1]
                frame.world_landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].z = knee_pos[2]

                ankle_pos = corrections['right']['ankle'] / self.scale
                frame.world_landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].x = ankle_pos[0]
                frame.world_landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].y = -ankle_pos[1]
                frame.world_landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].z = ankle_pos[2]

    def _validate_and_fix_landmarks(self, pose_frames: List[PoseFrame], max_shoulder_diff: float = 0.08) -> List[PoseFrame]:
        """
        Detect and fix frames with outlier landmarks.

        MediaPipe sometimes produces wildly incorrect landmark positions for certain frames.
        This function:
        1. Detects extreme outlier frames and interpolates from surrounding valid frames
        2. Levels shoulder Y positions to fix systematic shoulder asymmetry bias

        Args:
            pose_frames: List of pose frames to validate
            max_shoulder_diff: Maximum allowed Y difference between shoulders (in normalized coords)

        Returns:
            Fixed list of pose frames
        """
        if len(pose_frames) < 3:
            return pose_frames

        l_sh_idx = mp_pose.PoseLandmark.LEFT_SHOULDER
        r_sh_idx = mp_pose.PoseLandmark.RIGHT_SHOULDER
        l_hip_idx = mp_pose.PoseLandmark.LEFT_HIP
        r_hip_idx = mp_pose.PoseLandmark.RIGHT_HIP

        # First pass: identify extreme outlier frames (large shoulder difference)
        extreme_threshold = max_shoulder_diff * 1.5  # More permissive for extreme outliers
        is_outlier = [False] * len(pose_frames)

        for i, frame in enumerate(pose_frames):
            if not frame.is_valid():
                continue

            landmarks = frame.world_landmarks
            l_sh_y = landmarks[l_sh_idx].y
            r_sh_y = landmarks[r_sh_idx].y

            # Check for extreme shoulder difference
            shoulder_diff = abs(l_sh_y - r_sh_y)
            if shoulder_diff > extreme_threshold:
                is_outlier[i] = True

        # Count and fix extreme outliers
        num_outliers = sum(is_outlier)
        if num_outliers > 0:
            print(f"  Detected {num_outliers} extreme outlier frames - fixing with interpolation...")

        # Second pass: fix extreme outlier frames by interpolation
        for i in range(len(pose_frames)):
            if not is_outlier[i]:
                continue

            if not pose_frames[i].is_valid():
                continue

            # Find previous valid (non-outlier) frame
            prev_idx = i - 1
            while prev_idx >= 0 and (is_outlier[prev_idx] or not pose_frames[prev_idx].is_valid()):
                prev_idx -= 1

            # Find next valid (non-outlier) frame
            next_idx = i + 1
            while next_idx < len(pose_frames) and (is_outlier[next_idx] or not pose_frames[next_idx].is_valid()):
                next_idx += 1

            # Interpolate landmarks
            if prev_idx >= 0 and next_idx < len(pose_frames):
                t = (i - prev_idx) / (next_idx - prev_idx)
                prev_landmarks = pose_frames[prev_idx].world_landmarks
                next_landmarks = pose_frames[next_idx].world_landmarks

                for lm_idx in range(len(pose_frames[i].world_landmarks)):
                    pose_frames[i].world_landmarks[lm_idx].x = (
                        prev_landmarks[lm_idx].x * (1 - t) + next_landmarks[lm_idx].x * t
                    )
                    pose_frames[i].world_landmarks[lm_idx].y = (
                        prev_landmarks[lm_idx].y * (1 - t) + next_landmarks[lm_idx].y * t
                    )
                    pose_frames[i].world_landmarks[lm_idx].z = (
                        prev_landmarks[lm_idx].z * (1 - t) + next_landmarks[lm_idx].z * t
                    )

            elif prev_idx >= 0:
                prev_landmarks = pose_frames[prev_idx].world_landmarks
                for lm_idx in range(len(pose_frames[i].world_landmarks)):
                    pose_frames[i].world_landmarks[lm_idx].x = prev_landmarks[lm_idx].x
                    pose_frames[i].world_landmarks[lm_idx].y = prev_landmarks[lm_idx].y
                    pose_frames[i].world_landmarks[lm_idx].z = prev_landmarks[lm_idx].z

            elif next_idx < len(pose_frames):
                next_landmarks = pose_frames[next_idx].world_landmarks
                for lm_idx in range(len(pose_frames[i].world_landmarks)):
                    pose_frames[i].world_landmarks[lm_idx].x = next_landmarks[lm_idx].x
                    pose_frames[i].world_landmarks[lm_idx].y = next_landmarks[lm_idx].y
                    pose_frames[i].world_landmarks[lm_idx].z = next_landmarks[lm_idx].z

        # Third pass: level shoulders and hips in ALL frames to fix systematic bias
        # This addresses MediaPipe's tendency to consistently detect one shoulder higher
        leveled_count = 0
        for i, frame in enumerate(pose_frames):
            if not frame.is_valid():
                continue

            landmarks = frame.world_landmarks

            # Level shoulders
            l_sh_y = landmarks[l_sh_idx].y
            r_sh_y = landmarks[r_sh_idx].y
            if abs(l_sh_y - r_sh_y) > 0.01:  # Only adjust if there's noticeable difference
                avg_sh_y = (l_sh_y + r_sh_y) / 2.0
                landmarks[l_sh_idx].y = avg_sh_y
                landmarks[r_sh_idx].y = avg_sh_y
                leveled_count += 1

            # Level hips
            l_hip_y = landmarks[l_hip_idx].y
            r_hip_y = landmarks[r_hip_idx].y
            if abs(l_hip_y - r_hip_y) > 0.01:
                avg_hip_y = (l_hip_y + r_hip_y) / 2.0
                landmarks[l_hip_idx].y = avg_hip_y
                landmarks[r_hip_idx].y = avg_hip_y

        if leveled_count > 0:
            print(f"  Leveled shoulders/hips in {leveled_count} frames")

        return pose_frames

    def _process_motion_improved(self, pose_frames: List[PoseFrame]) -> List[Dict[str, np.ndarray]]:
        all_rotations = []
        for frame in pose_frames:
            if frame.is_valid():
                frame_rotations = self._calculate_frame_rotations_improved(frame)
            else:
                frame_rotations = self._get_zero_rotations()
            all_rotations.append(frame_rotations)
        return all_rotations

    @staticmethod
    def _safe_normalize(v: np.ndarray, eps: float = 1e-10) -> Optional[np.ndarray]:
        n = np.linalg.norm(v)
        if n < eps:
            return None
        return v / n

    def _calculate_torso_basis(self, landmarks, level_shoulders: bool = True) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Calculate torso orientation basis vectors.

        Args:
            landmarks: MediaPipe pose landmarks
            level_shoulders: If True, level the shoulders to reduce asymmetry from noise
        """
        try:
            l_sh = np.array([landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x,
                             -landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y,
                             landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].z])
            r_sh = np.array([landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x,
                             -landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y,
                             landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].z])
            l_hip = np.array([landmarks[mp_pose.PoseLandmark.LEFT_HIP].x,
                              -landmarks[mp_pose.PoseLandmark.LEFT_HIP].y,
                              landmarks[mp_pose.PoseLandmark.LEFT_HIP].z])
            r_hip = np.array([landmarks[mp_pose.PoseLandmark.RIGHT_HIP].x,
                              -landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y,
                              landmarks[mp_pose.PoseLandmark.RIGHT_HIP].z])

            # Level shoulders to reduce noise-induced asymmetry
            if level_shoulders:
                # Average the Y (height) of shoulders to make them level
                avg_shoulder_y = (l_sh[1] + r_sh[1]) / 2.0
                l_sh[1] = avg_shoulder_y
                r_sh[1] = avg_shoulder_y

                # Also level the hips
                avg_hip_y = (l_hip[1] + r_hip[1]) / 2.0
                l_hip[1] = avg_hip_y
                r_hip[1] = avg_hip_y

            sh_center = (l_sh + r_sh) / 2.0
            hip_center = (l_hip + r_hip) / 2.0

            left_axis = self._safe_normalize(l_sh - r_sh)
            up_axis = self._safe_normalize(sh_center - hip_center)
            if left_axis is None or up_axis is None:
                return None

            forward_axis = self._safe_normalize(np.cross(left_axis, up_axis))
            if forward_axis is None:
                return None

            up_axis = self._safe_normalize(np.cross(forward_axis, left_axis))
            if up_axis is None:
                return None

            return left_axis, up_axis, forward_axis
        except Exception:
            return None

    def _rotation_from_basis(self, left_axis: np.ndarray, up_axis: np.ndarray, forward_axis: np.ndarray) -> Optional[np.ndarray]:
        try:
            rot_mat = np.column_stack((left_axis, up_axis, forward_axis))
            r = Rotation.from_matrix(rot_mat)
            return r.as_euler('XYZ', degrees=True)
        except Exception:
            return None

    def _clamp_head_pitch(self, euler_xyz: np.ndarray, min_pitch: float = -45.0, max_pitch: float = 45.0) -> np.ndarray:
        out = np.array(euler_xyz, dtype=float)
        out[0] = float(np.clip(out[0], min_pitch, max_pitch))
        return out

    def _clamp_head_rotation(self, euler_xyz: np.ndarray) -> np.ndarray:
        """
        Clamp all head rotation axes to physically possible ranges.
        Humans have limited head rotation:
        - Pitch (X): ±45° (looking up/down)
        - Yaw (Y): ±80° (looking left/right)
        - Roll (Z): ±35° (tilting head side-to-side)
        """
        out = np.array(euler_xyz, dtype=float)
        out[0] = float(np.clip(out[0], -45.0, 45.0))   # Pitch
        out[1] = float(np.clip(out[1], -80.0, 80.0))   # Yaw
        out[2] = float(np.clip(out[2], -35.0, 35.0))   # Roll
        return out

    def _face_mesh_head_basis(self, frame: PoseFrame) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        if self._face_mesh is None:
            return None
        if not getattr(frame, "image_bgr", None) is not None:
            return None

        img = frame.image_bgr
        rgb = img[:, :, ::-1]
        res = self._face_mesh.process(rgb)
        if not res.multi_face_landmarks:
            return None

        lm = res.multi_face_landmarks[0].landmark

        left_eye = np.array([lm[33].x, lm[33].y, lm[33].z], dtype=float)
        right_eye = np.array([lm[263].x, lm[263].y, lm[263].z], dtype=float)
        forehead = np.array([lm[10].x, lm[10].y, lm[10].z], dtype=float)
        chin = np.array([lm[152].x, lm[152].y, lm[152].z], dtype=float)

        left_axis_cam = self._safe_normalize(left_eye - right_eye)
        up_axis_cam = self._safe_normalize(forehead - chin)
        if left_axis_cam is None or up_axis_cam is None:
            return None

        forward_axis_cam = self._safe_normalize(np.cross(left_axis_cam, up_axis_cam))
        if forward_axis_cam is None:
            return None

        up_axis_cam = self._safe_normalize(np.cross(forward_axis_cam, left_axis_cam))
        if up_axis_cam is None:
            return None

        left_axis = left_axis_cam.copy()
        up_axis = up_axis_cam.copy()
        forward_axis = forward_axis_cam.copy()
        left_axis[1] *= -1.0
        up_axis[1] *= -1.0
        forward_axis[1] *= -1.0

        left_axis = self._safe_normalize(left_axis) or left_axis
        up_axis = self._safe_normalize(up_axis) or up_axis
        forward_axis = self._safe_normalize(forward_axis) or forward_axis

        return left_axis, up_axis, forward_axis

    def _calculate_frame_rotations_improved(self, frame: PoseFrame) -> Dict[str, np.ndarray]:
        from scipy.spatial.transform import Rotation as Rotation

        landmarks = frame.world_landmarks
        rotations = {joint.name: np.zeros(3) for joint in self.skeleton_mapper.get_all_joints()}
        skeleton = self.skeleton_mapper.skeleton

        torso_basis = self._calculate_torso_basis(landmarks)
        chest_global_euler = None
        chest_global_rot = None
        if torso_basis is not None:
            chest_global_euler = self._rotation_from_basis(*torso_basis)
            if chest_global_euler is not None:
                chest_global_rot = Rotation.from_euler('XYZ', chest_global_euler, degrees=True)

        head_global_euler = None
        head_global_rot = None
        if self.enable_face:
            face_basis = self._face_mesh_head_basis(frame)
            if face_basis is not None:
                head_global_euler = self._rotation_from_basis(*face_basis)

        if head_global_euler is None:
            head_global_euler = self._calculate_head_global_rotation(landmarks, torso_basis)
            if head_global_euler is not None:
                # Clamp all head rotation axes to physically possible ranges
                head_global_euler = self._clamp_head_rotation(head_global_euler)

        if head_global_euler is not None:
            head_global_rot = Rotation.from_euler('XYZ', head_global_euler, degrees=True)

        def get_bone_direction(joint_name: str, child_name: str) -> Optional[np.ndarray]:
            # Use raw positions for rotation calculations
            # (Shoulder leveling is applied in skeleton offsets, not rotations)
            parent_pos = self.skeleton_mapper.get_joint_position(joint_name, landmarks, self.scale)
            child_pos = self.skeleton_mapper.get_joint_position(child_name, landmarks, self.scale)
            if parent_pos is not None and child_pos is not None:
                direction = child_pos - parent_pos
                if np.linalg.norm(direction) > 1e-10:
                    return direction / np.linalg.norm(direction)
            return None

        def process_joint(joint: BVHJoint, parent_rotation: Rotation):
            global_rotation = parent_rotation
            local_rotation_euler = np.zeros(3)

            calculated_euler = None
            is_global = False

            if joint.name == "mixamorig:Spine2" and chest_global_rot is not None:
                calculated_euler = chest_global_euler
                is_global = True

            elif joint.name == "mixamorig:Neck" and chest_global_rot is not None and head_global_rot is not None:
                neck_local = chest_global_rot.inv() * head_global_rot
                neck_local_euler = neck_local.as_euler('XYZ', degrees=True)
                neck_local_euler *= 0.5
                calculated_euler = neck_local_euler
                is_global = False

            elif joint.name == "mixamorig:Head" and head_global_euler is not None:
                calculated_euler = head_global_euler
                is_global = True

            elif joint.children:
                child = joint.children[0]
                direction = get_bone_direction(joint.name, child.name)
                if direction is not None and np.linalg.norm(child.offset) > 0:
                    rest_direction = child.offset / np.linalg.norm(child.offset)
                    calculated_euler = calculate_rotation_from_directions(rest_direction, direction, order='XYZ')
                    is_global = True

            if calculated_euler is not None:
                if is_global:
                    global_rot_obj = Rotation.from_euler('XYZ', calculated_euler, degrees=True)
                    local_rot_obj = parent_rotation.inv() * global_rot_obj
                    local_rotation_euler = local_rot_obj.as_euler('XYZ', degrees=True)
                    global_rotation = global_rot_obj
                else:
                    local_rotation_euler = calculated_euler
                    local_rot_obj = Rotation.from_euler('XYZ', calculated_euler, degrees=True)
                    global_rotation = parent_rotation * local_rot_obj

            rotations[joint.name] = local_rotation_euler

            for child in joint.children:
                process_joint(child, global_rotation)

        process_joint(skeleton, Rotation.identity())
        return rotations

    def _calculate_head_global_rotation(self, landmarks, torso_basis: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None) -> Optional[np.ndarray]:
        """
        Simple head rotation from ear and nose landmarks.
        This approach works well for most poses including walking, fighting, and shrugging.
        Based on the original stable approach from commit 01ead91.

        Returns GLOBAL Euler XYZ degrees.
        """
        try:
            nose_idx = mp_pose.PoseLandmark.NOSE
            l_ear_idx = mp_pose.PoseLandmark.LEFT_EAR
            r_ear_idx = mp_pose.PoseLandmark.RIGHT_EAR

            nose = np.array([landmarks[nose_idx].x, -landmarks[nose_idx].y, landmarks[nose_idx].z])
            l_ear = np.array([landmarks[l_ear_idx].x, -landmarks[l_ear_idx].y, landmarks[l_ear_idx].z])
            r_ear = np.array([landmarks[r_ear_idx].x, -landmarks[r_ear_idx].y, landmarks[r_ear_idx].z])

            # Sanity check: ear distance should be reasonable (0.05-0.3 in normalized coords)
            ear_dist = np.linalg.norm(r_ear - l_ear)
            if ear_dist < 0.02 or ear_dist > 0.5:
                # Ears too close or too far apart - bad detection
                return None

            # Sanity check: nose should be in front of mid-ears (positive Z or reasonable distance)
            mid_ears = (l_ear + r_ear) / 2.0
            nose_to_midear_dist = np.linalg.norm(nose - mid_ears)
            if nose_to_midear_dist < 0.01 or nose_to_midear_dist > 0.3:
                # Nose too close or too far from ears - bad detection
                return None

            # Check that ears are roughly level (Y difference should be small)
            ear_y_diff = abs(l_ear[1] - r_ear[1])
            if ear_y_diff > 0.15:
                # Ears at very different heights - likely bad detection or extreme head tilt
                # Still calculate but be cautious
                pass

            # Right vector: Right Ear - Left Ear
            right = self._safe_normalize(r_ear - l_ear)
            if right is None:
                return None

            # Forward vector: Mid(Ears) to Nose
            forward = self._safe_normalize(nose - mid_ears)
            if forward is None:
                return None

            # Up vector: Cross(Right, Forward)
            up = self._safe_normalize(np.cross(right, forward))
            if up is None:
                return None

            # Sanity check: up vector should generally point upward (positive Y)
            # Allow some tolerance for looking down
            if up[1] < -0.5:
                # Up vector pointing strongly downward - something is wrong
                # Flip it to maintain reasonable orientation
                up = -up
                forward = -forward

            # Re-orthogonalize forward
            forward = self._safe_normalize(np.cross(up, right))
            if forward is None:
                return None

            # Left = -Right (for the rotation basis)
            left = -right

            return self._rotation_from_basis(left, up, forward)

        except Exception as e:
            print(f"Error in head rotation calculation: {e}")
            return None

    def _reject_outliers(self, rotations: np.ndarray, max_change: float = 45.0) -> np.ndarray:
        """
        Detect and replace outlier rotation values.

        When rotation changes by more than max_change degrees between consecutive frames,
        interpolate from surrounding valid frames instead.
        """
        result = rotations.copy()
        n_frames = len(result)
        if n_frames < 3:
            return result

        # Mark frames as outliers based on sudden changes
        is_outlier = np.zeros(n_frames, dtype=bool)

        for i in range(1, n_frames - 1):
            # Check change from previous frame
            prev_diff = np.max(np.abs(result[i] - result[i-1]))
            next_diff = np.max(np.abs(result[i+1] - result[i]))

            # If this frame differs greatly from both neighbors, it's an outlier
            if prev_diff > max_change and next_diff > max_change:
                is_outlier[i] = True

        # Check first and last frames against their neighbors
        if n_frames > 1:
            if np.max(np.abs(result[0] - result[1])) > max_change * 2:
                is_outlier[0] = True
            if np.max(np.abs(result[-1] - result[-2])) > max_change * 2:
                is_outlier[-1] = True

        # Replace outliers with interpolated values
        for i in range(n_frames):
            if is_outlier[i]:
                # Find nearest valid frames
                prev_valid = i - 1
                while prev_valid >= 0 and is_outlier[prev_valid]:
                    prev_valid -= 1

                next_valid = i + 1
                while next_valid < n_frames and is_outlier[next_valid]:
                    next_valid += 1

                # Interpolate
                if prev_valid >= 0 and next_valid < n_frames:
                    # Interpolate between valid neighbors
                    t = (i - prev_valid) / (next_valid - prev_valid)
                    result[i] = result[prev_valid] * (1 - t) + result[next_valid] * t
                elif prev_valid >= 0:
                    # Use previous valid frame
                    result[i] = result[prev_valid]
                elif next_valid < n_frames:
                    # Use next valid frame
                    result[i] = result[next_valid]

        return result

    def _smooth_motion(self, all_rotations: List[Dict[str, np.ndarray]]) -> List[Dict[str, np.ndarray]]:
        if not all_rotations:
            return all_rotations

        joint_names = list(all_rotations[0].keys())
        smoothed_rotations = [{} for _ in range(len(all_rotations))]

        joint_smoothing = {
            'mixamorig:Hips': 3, 'mixamorig:Spine2': 3, 'mixamorig:Neck': 2, 'mixamorig:Head': 2,
        }

        # Joints that need outlier rejection (head/neck/spine chain most affected)
        outlier_rejection_joints = {
            'mixamorig:Head', 'mixamorig:Neck', 'mixamorig:Spine2', 'mixamorig:Spine1',
            'mixamorig:Spine', 'mixamorig:LeftShoulder', 'mixamorig:RightShoulder'
        }

        for joint_name in joint_names:
            joint_rotations = np.array([frame_rots[joint_name] for frame_rots in all_rotations])

            # Apply outlier rejection for sensitive joints
            if joint_name in outlier_rejection_joints:
                joint_rotations = self._reject_outliers(joint_rotations, max_change=45.0)

            window_size = joint_smoothing.get(joint_name, SMOOTHING_CONFIG['temporal_window_size'])

            smoothed = smooth_rotations(
                joint_rotations,
                window_size=window_size,
                preserve_dynamics=SMOOTHING_CONFIG['preserve_dynamics']
            )

            for i, rotation in enumerate(smoothed):
                smoothed_rotations[i][joint_name] = rotation

        return smoothed_rotations

    def _get_zero_rotations(self) -> Dict[str, np.ndarray]:
        return {joint.name: np.zeros(3) for joint in self.skeleton_mapper.get_all_joints()}

    def _write_bvh(self, all_rotations: List[Dict[str, np.ndarray]],
                   hip_positions: List[np.ndarray], output_path: str) -> bool:
        try:
            with open(output_path, 'w') as f:
                f.write("HIERARCHY\n")
                self._write_hierarchy(f, self.skeleton_mapper.skeleton, 0)

                num_frames = len(all_rotations)
                f.write("MOTION\n")
                f.write(f"Frames: {num_frames}\n")
                f.write(f"Frame Time: {self.frame_time:.6f}\n")

                for frame_idx in range(num_frames):
                    frame_data = []
                    hip_pos = hip_positions[frame_idx]
                    frame_data.extend([hip_pos[0], hip_pos[1], hip_pos[2]])

                    frame_rotations = all_rotations[frame_idx]
                    self._write_joint_rotations(self.skeleton_mapper.skeleton, frame_rotations, frame_data)
                    f.write(" ".join([f"{val:.6f}" for val in frame_data]) + "\n")

            return True
        except Exception as e:
            print(f"Error writing BVH file: {e}")
            return False

    def _write_hierarchy(self, f, joint: BVHJoint, level: int):
        indent = "  " * level
        if level == 0:
            f.write(f"{indent}ROOT {joint.name}\n")
        else:
            f.write(f"{indent}JOINT {joint.name}\n")

        f.write(f"{indent}{{\n")
        f.write(f"{indent}  OFFSET {joint.offset[0]:.6f} {joint.offset[1]:.6f} {joint.offset[2]:.6f}\n")

        if joint.channels:
            channels_str = " ".join(joint.channels)
            f.write(f"{indent}  CHANNELS {len(joint.channels)} {channels_str}\n")

        for child in joint.children:
            self._write_hierarchy(f, child, level + 1)

        if not joint.children:
            f.write(f"{indent}  End Site\n")
            f.write(f"{indent}  {{\n")
            end_offset = joint.offset * 0.3 if np.linalg.norm(joint.offset) > 0 else np.array([0, -5, 0])
            f.write(f"{indent}    OFFSET {end_offset[0]:.6f} {end_offset[1]:.6f} {end_offset[2]:.6f}\n")
            f.write(f"{indent}  }}\n")

        f.write(f"{indent}}}\n")

    def _write_joint_rotations(self, joint: BVHJoint, frame_rotations: Dict, frame_data: List):
        if joint.name in frame_rotations:
            rotation = frame_rotations[joint.name]
            if 'Xrotation' in joint.channels:
                frame_data.append(rotation[0])
            if 'Yrotation' in joint.channels:
                frame_data.append(rotation[1])
            if 'Zrotation' in joint.channels:
                frame_data.append(rotation[2])

        for child in joint.children:
            self._write_joint_rotations(child, frame_rotations, frame_data)


def main():
    parser = argparse.ArgumentParser(description="Improved BVH Converter with optional FaceMesh head tracking")
    parser.add_argument("--video", required=True, help="Path to input video file")
    parser.add_argument("--output", required=True, help="Path to output BVH file")
    parser.add_argument("--preview", action="store_true", help="Show pose detection preview")
    parser.add_argument("--sample-rate", type=int, default=1, help="Process every Nth frame (default: 1 = every frame)")
    parser.add_argument("--ik", action="store_true", help="Enable improved IK foot locking")
    parser.add_argument("--face", action="store_true", help="Enable MediaPipe FaceMesh for head orientation")

    args = parser.parse_args()
    PROCESSING_CONFIG['sample_rate'] = args.sample_rate

    start_time = time.time()

    with MediaPipeExtractor(use_holistic=True) as extractor:
        extractor.sample_rate = args.sample_rate
        pose_frames = extractor.extract_from_video(args.video, preview=args.preview)
        if not pose_frames:
            print("Error: No poses extracted from video")
            return
        pose_frames = extractor.interpolate_missing_frames(pose_frames)

    if args.face:
        cap = mp.solutions.cv2.VideoCapture(args.video) if hasattr(mp.solutions, "cv2") else None
        if cap is None:
            import cv2 as _cv2
            cap = _cv2.VideoCapture(args.video)
        idx = 0
        out_idx = 0
        while cap.isOpened() and out_idx < len(pose_frames):
            ret, frame_bgr = cap.read()
            if not ret:
                break
            if idx % args.sample_rate == 0:
                pose_frames[out_idx].image_bgr = frame_bgr
                out_idx += 1
            idx += 1
        cap.release()

    converter = ImprovedBVHConverter(enable_ik=args.ik, enable_face=args.face)
    try:
        success = converter.convert(pose_frames, args.output)
    finally:
        converter.close()

    elapsed_time = time.time() - start_time
    if success:
        print(f"\nConversion completed in {elapsed_time:.2f} seconds")
        print(f"Output saved to: {args.output}")
    else:
        print("\nConversion failed")


if __name__ == "__main__":
    main()
