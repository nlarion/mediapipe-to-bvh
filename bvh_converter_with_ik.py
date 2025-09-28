"""
Enhanced BVH converter with IK foot locking to prevent sliding.
"""

import numpy as np
import argparse
import copy
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from tqdm import tqdm
import mediapipe as mp

from mediapipe_extractor import MediaPipeExtractor, PoseFrame
from skeleton_mapper import SkeletonMapper, BVHJoint
from math_utils import calculate_rotation_from_directions, smooth_rotations, smooth_positions
from config import BVH_CONFIG, PROCESSING_CONFIG, SMOOTHING_CONFIG
from ik_foot_lock import IKFootLockSystem, IKChainConfig
from bvh_converter import BVHConverter

mp_pose = mp.solutions.pose


class BVHConverterWithIK(BVHConverter):
    """BVH converter with integrated IK foot locking system."""

    def __init__(self, enable_ik: bool = True):
        super().__init__()
        self.enable_ik = enable_ik
        self.ik_system = None

        # Store landmark positions for IK processing
        self.hip_positions = {'left': [], 'right': []}
        self.knee_positions = {'left': [], 'right': []}
        self.ankle_positions = {'left': [], 'right': []}

    def _initialize_ik_system(self, reference_landmarks, scale: float):
        """Initialize IK system with bone lengths from reference frame."""

        # Get joint positions from reference
        left_hip_idx = mp_pose.PoseLandmark.LEFT_HIP
        left_knee_idx = mp_pose.PoseLandmark.LEFT_KNEE
        left_ankle_idx = mp_pose.PoseLandmark.LEFT_ANKLE

        right_hip_idx = mp_pose.PoseLandmark.RIGHT_HIP
        right_knee_idx = mp_pose.PoseLandmark.RIGHT_KNEE
        right_ankle_idx = mp_pose.PoseLandmark.RIGHT_ANKLE

        # Calculate bone lengths from reference pose
        left_hip = np.array([
            reference_landmarks[left_hip_idx].x,
            -reference_landmarks[left_hip_idx].y,  # Flip Y
            reference_landmarks[left_hip_idx].z
        ]) * scale

        left_knee = np.array([
            reference_landmarks[left_knee_idx].x,
            -reference_landmarks[left_knee_idx].y,
            reference_landmarks[left_knee_idx].z
        ]) * scale

        left_ankle = np.array([
            reference_landmarks[left_ankle_idx].x,
            -reference_landmarks[left_ankle_idx].y,
            reference_landmarks[left_ankle_idx].z
        ]) * scale

        # Calculate thigh and shin lengths (use left leg as reference)
        thigh_length = np.linalg.norm(left_knee - left_hip)
        shin_length = np.linalg.norm(left_ankle - left_knee)

        print(f"IK System initialized with bone lengths:")
        print(f"  Thigh: {thigh_length:.2f} units")
        print(f"  Shin: {shin_length:.2f} units")

        # Create IK system
        self.ik_system = IKFootLockSystem(thigh_length, shin_length)

        # Calibrate thresholds based on scale and MediaPipe coordinate system
        # These values are tuned for better contact detection
        self.ik_system.config.contact_velocity_threshold = 3.0 * (scale / 100.0)  # Lowered from 5.0
        self.ik_system.config.contact_height_threshold = 8.0 * (scale / 100.0)   # Lowered from 10.0
        self.ik_system.config.foot_clearance_height = 3.0 * (scale / 100.0)      # Lowered from 5.0

    def _extract_leg_positions(self, world_landmarks, scale: float) -> Dict[str, Dict[str, np.ndarray]]:
        """Extract hip, knee, and ankle positions from landmarks."""

        positions = {}

        # Left leg
        left_hip_idx = mp_pose.PoseLandmark.LEFT_HIP
        left_knee_idx = mp_pose.PoseLandmark.LEFT_KNEE
        left_ankle_idx = mp_pose.PoseLandmark.LEFT_ANKLE

        positions['left'] = {
            'hip': np.array([
                world_landmarks[left_hip_idx].x,
                -world_landmarks[left_hip_idx].y,
                world_landmarks[left_hip_idx].z
            ]) * scale,
            'knee': np.array([
                world_landmarks[left_knee_idx].x,
                -world_landmarks[left_knee_idx].y,
                world_landmarks[left_knee_idx].z
            ]) * scale,
            'ankle': np.array([
                world_landmarks[left_ankle_idx].x,
                -world_landmarks[left_ankle_idx].y,
                world_landmarks[left_ankle_idx].z
            ]) * scale
        }

        # Right leg
        right_hip_idx = mp_pose.PoseLandmark.RIGHT_HIP
        right_knee_idx = mp_pose.PoseLandmark.RIGHT_KNEE
        right_ankle_idx = mp_pose.PoseLandmark.RIGHT_ANKLE

        positions['right'] = {
            'hip': np.array([
                world_landmarks[right_hip_idx].x,
                -world_landmarks[right_hip_idx].y,
                world_landmarks[right_hip_idx].z
            ]) * scale,
            'knee': np.array([
                world_landmarks[right_knee_idx].x,
                -world_landmarks[right_knee_idx].y,
                world_landmarks[right_knee_idx].z
            ]) * scale,
            'ankle': np.array([
                world_landmarks[right_ankle_idx].x,
                -world_landmarks[right_ankle_idx].y,
                world_landmarks[right_ankle_idx].z
            ]) * scale
        }

        return positions

    def convert(self, pose_frames: List[PoseFrame], output_path: str) -> bool:
        """Convert pose frames to BVH file with optional IK foot locking.

        FIXED: IK corrections are now applied BEFORE rotation calculation.
        """

        if not pose_frames:
            print("Error: No pose frames to convert")
            return False

        # Store original pose_frames for hip position calculation
        self.pose_frames = pose_frames

        # Find reference frame for skeleton setup
        extractor = MediaPipeExtractor(use_holistic=True)
        ref_idx = extractor.find_reference_frame(pose_frames)

        if not pose_frames[ref_idx].is_valid():
            print("Error: No valid reference frame found")
            return False

        # Calculate bone offsets from reference frame
        print("Setting up skeleton from reference frame...")
        ref_landmarks = pose_frames[ref_idx].world_landmarks
        self.skeleton_mapper.calculate_bone_offsets(ref_landmarks, self.scale)

        # CRITICAL FIX: Apply IK corrections BEFORE calculating rotations
        if self.enable_ik:
            print("Initializing IK foot locking system...")
            self._initialize_ik_system(ref_landmarks, self.scale)

            # Create deep copies of frames to modify
            pose_frames = [copy.deepcopy(frame) for frame in pose_frames]

            # Pre-process all frames to extract leg positions
            print("Extracting leg positions for IK processing...")
            all_leg_positions = []
            for frame in pose_frames:
                if frame.is_valid():
                    leg_pos = self._extract_leg_positions(frame.world_landmarks, self.scale)
                    all_leg_positions.append(leg_pos)
                else:
                    all_leg_positions.append(None)

            # Apply IK corrections
            print("Applying IK foot locking...")
            corrected_positions = self._apply_ik_corrections(all_leg_positions)

            # Update pose frames with corrected positions BEFORE rotation calculation
            self._update_pose_frames_with_ik(pose_frames, corrected_positions)

        # Process all frames to calculate rotations FROM CORRECTED POSITIONS
        print("Calculating joint rotations from IK-corrected positions...")
        all_rotations = self._process_motion(pose_frames)

        # Apply smoothing if enabled
        if SMOOTHING_CONFIG['enable_temporal_smoothing']:
            print("Applying temporal smoothing...")
            all_rotations = self._smooth_motion(all_rotations)

        # Calculate hip positions from actual landmark data
        print("Calculating hip movement through 3D space...")
        hip_positions = self._calculate_hip_positions(pose_frames)

        # Write BVH file
        print(f"Writing BVH file to {output_path}...")
        success = self._write_bvh(all_rotations, hip_positions, output_path)

        if success:
            print(f"BVH file created successfully: {output_path}")
            if self.enable_ik:
                print("✅ IK foot locking was properly applied (fixed pipeline)")
        else:
            print("Error writing BVH file")

        return success

    def _apply_ik_corrections(self, all_leg_positions: List[Optional[Dict]]) -> List[Optional[Dict]]:
        """Apply IK corrections to all frames."""

        corrected = []
        previous_ankles = None

        for i, leg_positions in enumerate(tqdm(all_leg_positions, desc="Applying IK")):
            if leg_positions is None:
                corrected.append(None)
                continue

            # Extract positions for this frame
            hip_positions = {
                'left': leg_positions['left']['hip'],
                'right': leg_positions['right']['hip']
            }
            knee_positions = {
                'left': leg_positions['left']['knee'],
                'right': leg_positions['right']['knee']
            }
            ankle_positions = {
                'left': leg_positions['left']['ankle'],
                'right': leg_positions['right']['ankle']
            }

            # Apply IK correction
            ik_result = self.ik_system.process_frame(
                hip_positions,
                knee_positions,
                ankle_positions,
                i,
                previous_ankles
            )

            # Store corrected positions
            corrected_frame = {
                'left': ik_result['left'],
                'right': ik_result['right']
            }
            corrected.append(corrected_frame)

            # Update previous ankles for velocity calculation
            previous_ankles = {
                'left': ik_result['left']['ankle'],
                'right': ik_result['right']['ankle']
            }

        # Print statistics
        planted_frames = sum(
            1 for frame in corrected
            if frame and (frame['left']['confidence'] > 0.5 or frame['right']['confidence'] > 0.5)
        )
        total_frames = len([f for f in corrected if f is not None])

        if total_frames > 0:
            print(f"IK Statistics:")
            print(f"  Frames with foot contact: {planted_frames}/{total_frames} ({100*planted_frames/total_frames:.1f}%)")

        return corrected

    def _update_pose_frames_with_ik(self, pose_frames: List[PoseFrame], corrected_positions: List[Optional[Dict]]):
        """Update pose frame landmarks with IK-corrected positions."""

        for i, (frame, corrections) in enumerate(zip(pose_frames, corrected_positions)):
            if not frame.is_valid() or corrections is None:
                continue

            # Update knee and ankle positions in world landmarks
            # Note: We need to convert back to MediaPipe coordinate system (Y down)

            # Left leg
            if corrections['left']['confidence'] > 0:
                # Update left knee
                knee_pos = corrections['left']['knee'] / self.scale
                frame.world_landmarks[mp_pose.PoseLandmark.LEFT_KNEE].x = knee_pos[0]
                frame.world_landmarks[mp_pose.PoseLandmark.LEFT_KNEE].y = -knee_pos[1]  # Flip Y back
                frame.world_landmarks[mp_pose.PoseLandmark.LEFT_KNEE].z = knee_pos[2]

                # Update left ankle
                ankle_pos = corrections['left']['ankle'] / self.scale
                frame.world_landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].x = ankle_pos[0]
                frame.world_landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].y = -ankle_pos[1]
                frame.world_landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].z = ankle_pos[2]

            # Right leg
            if corrections['right']['confidence'] > 0:
                # Update right knee
                knee_pos = corrections['right']['knee'] / self.scale
                frame.world_landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].x = knee_pos[0]
                frame.world_landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].y = -knee_pos[1]
                frame.world_landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].z = knee_pos[2]

                # Update right ankle
                ankle_pos = corrections['right']['ankle'] / self.scale
                frame.world_landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].x = ankle_pos[0]
                frame.world_landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].y = -ankle_pos[1]
                frame.world_landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].z = ankle_pos[2]


def main():
    parser = argparse.ArgumentParser(description='Convert video to BVH with IK foot locking')
    parser.add_argument('--video', type=str, required=True, help='Path to input video file')
    parser.add_argument('--output', type=str, required=True, help='Path to output BVH file')
    parser.add_argument('--holistic', action='store_true', help='Use MediaPipe Holistic model')
    parser.add_argument('--no-ik', action='store_true', help='Disable IK foot locking')
    parser.add_argument('--confidence', type=float, default=0.5,
                       help='Minimum confidence threshold for landmarks')

    args = parser.parse_args()

    # Extract pose data
    print(f"Processing video: {args.video}")

    # Create extractor with proper context
    with MediaPipeExtractor(use_holistic=args.holistic) as extractor:
        # Extract frames
        pose_frames = extractor.extract_from_video(args.video)

    if not pose_frames:
        print("Error: No valid pose data extracted from video")
        return

    print(f"Extracted {len(pose_frames)} frames")

    # Convert to BVH with IK
    converter = BVHConverterWithIK(enable_ik=not args.no_ik)
    success = converter.convert(pose_frames, args.output)

    if success:
        print("\nConversion complete!")
        if not args.no_ik:
            print("IK foot locking was applied to reduce foot sliding")
    else:
        print("\nConversion failed")

    return success


if __name__ == "__main__":
    main()