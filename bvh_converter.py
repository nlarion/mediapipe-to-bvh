"""
Improved BVH converter with fixes for ForeArm/Wrist errors and better IK calibration.
Based on issues identified in todo.md:
1. Better 3D hand reconstruction to fix ForeArm/Wrist errors (65-82°)
2. Calibrated IK thresholds for foot contact detection
3. Foot-based drift correction for walking videos
"""

import numpy as np
import argparse
import time
import copy
from pathlib import Path
from typing import List, Dict, Optional, Tuple
# from tqdm import tqdm
import mediapipe as mp

from mediapipe_extractor import MediaPipeExtractor, PoseFrame
from skeleton_mapper import SkeletonMapper, BVHJoint
from math_utils import calculate_rotation_from_directions, smooth_rotations, smooth_positions, calculate_depth_from_projected_length
from config import BVH_CONFIG, PROCESSING_CONFIG, SMOOTHING_CONFIG
from ik_foot_lock import IKFootLockSystem, IKChainConfig

mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands


class ImprovedBVHConverter:
    """Improved BVH converter with better hand tracking and IK calibration."""

    def __init__(self, enable_ik: bool = False):
        self.skeleton_mapper = SkeletonMapper()
        self.frame_time = 1.0 / BVH_CONFIG['fps']
        self.rotation_order = BVH_CONFIG['rotation_order']
        self.scale = PROCESSING_CONFIG['scale_factor']
        self.enable_ik = enable_ik
        self.ik_system = None

        # NEW: Store foot ground levels for drift correction
        self.ground_level = None
        self.foot_contact_frames = []

    def convert(self, pose_frames: List[PoseFrame], output_path: str) -> bool:
        """Convert pose frames to BVH file with improved hand tracking and IK."""
        if not pose_frames:
            print("Error: No pose frames to convert")
            return False

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

        # Calculate dynamic ground level
        print("Calculating dynamic ground level...")
        self.ground_level = self._calculate_dynamic_ground_level(pose_frames)
        print(f"Dynamic ground level determined at Y={self.ground_level:.2f}")

        # Initialize IK if enabled
        if self.enable_ik:
            print("Initializing improved IK foot locking system...")
            self._initialize_improved_ik_system(ref_landmarks, self.scale)
            pose_frames = [copy.deepcopy(frame) for frame in pose_frames]

            # PASS 1: Run IK to detect foot contacts
            print("Extracting leg positions for IK processing (Pass 1)...")
            all_leg_positions = []
            for frame in pose_frames:
                if frame.is_valid():
                    leg_pos = self._extract_leg_positions(frame.world_landmarks, self.scale)
                    all_leg_positions.append(leg_pos)
                else:
                    all_leg_positions.append(None)

            print("Applying improved IK foot locking (Pass 1)...")
            # This populates self.foot_contact_frames
            self._apply_improved_ik_corrections(all_leg_positions)

        # 4. Calculate hip positions (root) with drift correction
        # This is done AFTER IK to ensure we have stable foot contacts for drift correction
        print("Calculating root motion from foot lock...")
        hip_positions = self._calculate_root_motion_from_feet(pose_frames)
        
        # 5. Update pose frames with corrected hip positions
        print("Updating pose frames with corrected hip positions...")
        for i, frame in enumerate(pose_frames):
            if i < len(hip_positions):
                frame.hip_position = hip_positions[i]
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

        # Process all frames to calculate rotations (using updated hips and legs)
        print("Calculating joint rotations with improved hand tracking...")
        all_rotations = self._process_motion_improved(pose_frames)

        # Apply smoothing if enabled
        if SMOOTHING_CONFIG['enable_temporal_smoothing']:
            print("Applying adaptive temporal smoothing...")
            all_rotations = self._smooth_motion(all_rotations)

        # Write BVH file
        print(f"Writing BVH file to {output_path}...")
        success = self._write_bvh(all_rotations, hip_positions, output_path)

        if success:
            print(f"BVH file created successfully: {output_path}")
            if self.enable_ik:
                print("✅ Improved IK foot locking applied")
                print(f"✅ Detected {len(self.foot_contact_frames)} foot contact frames")
        else:
            print("Error writing BVH file")

        return success

    def _initialize_improved_ik_system(self, reference_landmarks, scale: float):
        """Initialize IK system with calibrated thresholds for MediaPipe."""

        # Get joint positions from reference
        left_hip_idx = mp_pose.PoseLandmark.LEFT_HIP
        left_knee_idx = mp_pose.PoseLandmark.LEFT_KNEE
        left_ankle_idx = mp_pose.PoseLandmark.LEFT_ANKLE
        left_foot_idx = mp_pose.PoseLandmark.LEFT_FOOT_INDEX

        # Calculate bone lengths from reference pose
        left_hip = np.array([
            reference_landmarks[left_hip_idx].x,
            -reference_landmarks[left_hip_idx].y,
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

        # NEW: Use foot landmark to establish ground level
        left_foot = np.array([
            reference_landmarks[left_foot_idx].x,
            -reference_landmarks[left_foot_idx].y,
            reference_landmarks[left_foot_idx].z
        ]) * scale

        # Calculate thigh and shin lengths
        thigh_length = np.linalg.norm(left_knee - left_hip)
        shin_length = np.linalg.norm(left_ankle - left_knee)

        # NEW: Establish ground level from reference frame (Fallback if dynamic failed)
        if self.ground_level is None:
            self.ground_level = min(left_ankle[1], left_foot[1])

        print(f"IK System initialized with bone lengths:")
        print(f"  Thigh: {thigh_length:.2f} units")
        print(f"  Shin: {shin_length:.2f} units")
        print(f"  Ground level: {self.ground_level:.2f} units")

        # Create IK system
        self.ik_system = IKFootLockSystem(thigh_length, shin_length)

        # IMPROVED: Calibrated thresholds based on MediaPipe coordinate system
        # MediaPipe world coordinates are in meters, scale factor is typically 100
        
        # Velocity threshold: Increased to tolerate noise
        # 2.0 * (scale/100) -> ~2.0 units/frame if scale is 100
        self.ik_system.config.contact_velocity_threshold = 4.0 * (scale / 100.0)

        # Height threshold: Increased to catch feet earlier
        # 0.08 * scale -> 8cm if scale is 100
        self.ik_system.config.contact_height_threshold = 0.12 * scale

        # Foot clearance: minimum height for foot to be considered off ground
        self.ik_system.config.foot_clearance_height = 0.05 * scale
        
        # Vertical velocity threshold (NEW)
        self.ik_system.config.vertical_velocity_threshold = 2.0 * (scale / 100.0)

        print(f"Calibrated IK thresholds:")
        print(f"  Velocity threshold: {self.ik_system.config.contact_velocity_threshold:.3f}")
        print(f"  Height threshold: {self.ik_system.config.contact_height_threshold:.3f}")
        print(f"  Vertical vel threshold: {self.ik_system.config.vertical_velocity_threshold:.3f}")
        print(f"  Clearance height: {self.ik_system.config.foot_clearance_height:.3f}")

    def _extract_leg_positions(self, world_landmarks, scale: float) -> Dict[str, Dict[str, np.ndarray]]:
        """Extract hip, knee, and ankle positions from landmarks."""

        positions = {}

        # Left leg
        left_hip_idx = mp_pose.PoseLandmark.LEFT_HIP
        left_knee_idx = mp_pose.PoseLandmark.LEFT_KNEE
        left_ankle_idx = mp_pose.PoseLandmark.LEFT_ANKLE
        left_foot_idx = mp_pose.PoseLandmark.LEFT_FOOT_INDEX

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
            ]) * scale,
            'foot': np.array([
                world_landmarks[left_foot_idx].x,
                -world_landmarks[left_foot_idx].y,
                world_landmarks[left_foot_idx].z
            ]) * scale
        }

        # Right leg
        right_hip_idx = mp_pose.PoseLandmark.RIGHT_HIP
        right_knee_idx = mp_pose.PoseLandmark.RIGHT_KNEE
        right_ankle_idx = mp_pose.PoseLandmark.RIGHT_ANKLE
        right_foot_idx = mp_pose.PoseLandmark.RIGHT_FOOT_INDEX

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
            ]) * scale,
            'foot': np.array([
                world_landmarks[right_foot_idx].x,
                -world_landmarks[right_foot_idx].y,
                world_landmarks[right_foot_idx].z
            ]) * scale
        }

        return positions

    def _apply_improved_ik_corrections(self, all_leg_positions: List[Optional[Dict]]) -> List[Optional[Dict]]:
        """Apply improved IK corrections with better foot contact detection."""

        corrected = []
        previous_ankles = None
        self.foot_contact_frames = []

        for i, leg_positions in enumerate(all_leg_positions):
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

            # NEW: Use foot positions for better ground detection
            foot_positions = {
                'left': leg_positions['left']['foot'],
                'right': leg_positions['right']['foot']
            }

            # IMPROVED: Better foot contact detection
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

            # Track contact frames for drift correction
            if left_contact or right_contact:
                self.foot_contact_frames.append(i)


            # Create overrides dictionary
            contact_overrides = {
                'left': left_contact,
                'right': right_contact
            }

            # Apply IK correction with improved contact detection
            ik_result = self.ik_system.process_frame(
                hip_positions,
                knee_positions,
                ankle_positions,
                i,
                previous_ankles,
                contact_overrides
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
            print(f"Improved IK Statistics:")
            print(f"  Frames with foot contact: {planted_frames}/{total_frames} ({100*planted_frames/total_frames:.1f}%)")

        return corrected

    def _calculate_dynamic_ground_level(self, pose_frames: List[PoseFrame]) -> float:
        """Calculate ground level dynamically from the lowest foot positions."""
        min_y = float('inf')
        
        left_ankle_idx = mp_pose.PoseLandmark.LEFT_ANKLE
        right_ankle_idx = mp_pose.PoseLandmark.RIGHT_ANKLE
        left_foot_idx = mp_pose.PoseLandmark.LEFT_FOOT_INDEX
        right_foot_idx = mp_pose.PoseLandmark.RIGHT_FOOT_INDEX
        
        valid_frames = 0
        
        for frame in pose_frames:
            if not frame.is_valid():
                continue
                
            # Get foot Y positions (remember MediaPipe Y is inverted relative to BVH usually, 
            # but here we are working with world landmarks where Y is down? 
            # Actually in _extract_leg_positions we do -y.
            # Let's look at raw landmarks first.
            # In _extract_leg_positions: -world_landmarks[idx].y * scale
            
            # So we should look for the MINIMUM Y value (which corresponds to the lowest point 
            # if we are using the inverted Y, or MAXIMUM if we are using raw Y).
            
            # Let's use the same transformation as _extract_leg_positions to be safe
            l_ankle_y = -frame.world_landmarks[left_ankle_idx].y * self.scale
            r_ankle_y = -frame.world_landmarks[right_ankle_idx].y * self.scale
            l_foot_y = -frame.world_landmarks[left_foot_idx].y * self.scale
            r_foot_y = -frame.world_landmarks[right_foot_idx].y * self.scale
            
            # Find lowest point in this frame
            frame_min = min(l_ankle_y, r_ankle_y, l_foot_y, r_foot_y)
            
            if frame_min < min_y:
                min_y = frame_min
            
            valid_frames += 1
            
        if valid_frames == 0:
            return 0.0
            
        return min_y

    def _detect_foot_contact(self, ankle_pos: np.ndarray, foot_pos: np.ndarray,
                            prev_ankle: Optional[np.ndarray], side: str) -> bool:
        """Improved foot contact detection using multiple signals."""

        # Check height relative to ground
        foot_height = foot_pos[1] - self.ground_level if self.ground_level is not None else foot_pos[1]
        ankle_height = ankle_pos[1] - self.ground_level if self.ground_level is not None else ankle_pos[1]
        
        # Use the lowest point
        min_height = min(foot_height, ankle_height)

        # Foot is on ground if close to ground level
        height_contact = min_height < self.ik_system.config.contact_height_threshold

        # Check velocity if we have previous frame
        velocity_contact = True
        vertical_velocity_contact = True
        
        if prev_ankle is not None:
            velocity = ankle_pos - prev_ankle
            velocity_mag = np.linalg.norm(velocity)
            vertical_velocity = abs(velocity[1])
            
            velocity_contact = velocity_mag < self.ik_system.config.contact_velocity_threshold
            vertical_velocity_contact = vertical_velocity < self.ik_system.config.vertical_velocity_threshold

        # Combined check - slightly looser than the strict AND to allow for some noise
        # But we rely on the IK system's internal hysteresis for stability
        return height_contact and (velocity_contact or vertical_velocity_contact)

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

    def _update_pose_frames_with_new_hips(self, pose_frames: List[PoseFrame], hip_positions: List[np.ndarray]):
        """Update pose frame hip landmarks and upper body to match calculated hip positions."""
        
        # Indices to shift (Upper Body + Hips)
        # 0-24 (Nose to Hips)
        # 25-32 are Legs (Knee, Ankle, Heel, Toe) - DO NOT SHIFT (IK will handle)
        shift_indices = list(range(25)) 
        
        for i, frame in enumerate(pose_frames):
            if not frame.is_valid() or i >= len(hip_positions):
                continue
                
            target_hip = hip_positions[i] # [X, Y, Z] in BVH space
            
            # Get current hip center in BVH space
            l_hip = frame.world_landmarks[mp_pose.PoseLandmark.LEFT_HIP]
            r_hip = frame.world_landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
            
            current_center = np.array([
                (l_hip.x + r_hip.x) / 2 * self.scale,
                -(l_hip.y + r_hip.y) / 2 * self.scale,
                (l_hip.z + r_hip.z) / 2 * self.scale # Approximate Z
            ])
            
            # Calculate delta in BVH space
            delta = target_hip - current_center
            
            # Convert delta to MP space
            # MP Space: X = X_bvh / scale, Y = -Y_bvh / scale, Z = Z_bvh / scale
            delta_mp_x = delta[0] / self.scale
            delta_mp_y = -delta[1] / self.scale
            delta_mp_z = delta[2] / self.scale
            
            # Apply delta to upper body landmarks
            for idx in shift_indices:
                if idx < len(frame.world_landmarks):
                    landmark = frame.world_landmarks[idx]
                    landmark.x += delta_mp_x
                    landmark.y += delta_mp_y
                    landmark.z += delta_mp_z

    def _process_motion_improved(self, pose_frames: List[PoseFrame]) -> List[Dict[str, np.ndarray]]:
        """Process all frames with improved hand tracking."""
        all_rotations = []

        for frame in pose_frames:
            if frame.is_valid():
                frame_rotations = self._calculate_frame_rotations_improved(
                    frame.world_landmarks,
                    left_hand_landmarks=frame.left_hand_landmarks,
                    right_hand_landmarks=frame.right_hand_landmarks
                )
            else:
                frame_rotations = self._get_zero_rotations()

            all_rotations.append(frame_rotations)

        return all_rotations

    def _calculate_frame_rotations_improved(self, landmarks,
                                          left_hand_landmarks=None,
                                          right_hand_landmarks=None) -> Dict[str, np.ndarray]:
        """Calculate rotations with improved hand tracking."""
        
        # Import Rotation here to avoid circular imports
        from scipy.spatial.transform import Rotation as R

        # Start with all zero rotations
        rotations = {joint.name: np.zeros(3) for joint in self.skeleton_mapper.get_all_joints()}
        skeleton = self.skeleton_mapper.skeleton

        def get_bone_direction(joint_name: str, child_name: str) -> Optional[np.ndarray]:
            """Calculate normalized direction from parent to child."""
            parent_pos = self.skeleton_mapper.get_joint_position(
                joint_name, landmarks, self.scale
            )
            child_pos = self.skeleton_mapper.get_joint_position(
                child_name, landmarks, self.scale
            )

            if parent_pos is not None and child_pos is not None:
                direction = child_pos - parent_pos
                if np.linalg.norm(direction) > 1e-10:
                    return direction / np.linalg.norm(direction)

            return None

        def process_joint(joint: BVHJoint, parent_rotation: R):
            """Recursively process joint hierarchy."""
            
            # Default: inherit parent rotation (local rotation is identity)
            global_rotation = parent_rotation
            local_rotation_euler = np.zeros(3)
            
            calculated_euler = None
            is_global = False

            # Special handling for specific joints
            if joint.name == "Head":
                # IMPROVED: Calculate head rotation from landmarks
                head_rotation = self._calculate_head_rotation(landmarks)
                if head_rotation is not None:
                    calculated_euler = head_rotation
                    is_global = True
                else:
                    calculated_euler = np.array([0.0, 0.0, 0.0])
                    is_global = False

            elif joint.name in ["Chest", "Neck"]:
                if joint.children:
                    child = joint.children[0]
                    direction = get_bone_direction(joint.name, child.name)

                    if direction is not None and np.linalg.norm(child.offset) > 0:
                        rest_direction = child.offset / np.linalg.norm(child.offset)
                        euler_angles = calculate_rotation_from_directions(
                            rest_direction, direction, order='XYZ'
                        )
                        calculated_euler = euler_angles * 0.3
                        is_global = True # Calculated from global vectors
            
            elif joint.name in ["LeftShoulder", "RightShoulder"]:
                shoulder_pos = self.skeleton_mapper.get_joint_position(joint.name, landmarks, self.scale)

                if joint.children and shoulder_pos is not None:
                    child = joint.children[0]  # Arm
                    arm_pos = self.skeleton_mapper.get_joint_position(child.name, landmarks, self.scale)

                    if arm_pos is not None:
                        direction = arm_pos - shoulder_pos
                        if np.linalg.norm(direction) > 1e-10 and np.linalg.norm(child.offset) > 0:
                            direction = direction / np.linalg.norm(direction)
                            rest_direction = child.offset / np.linalg.norm(child.offset)

                            calculated_euler = calculate_rotation_from_directions(
                                rest_direction, direction, order='XYZ'
                            )
                            is_global = True

            elif joint.name in ["LeftArm", "RightArm"]:
                parent = joint.parent
                if parent and joint.children:
                    shoulder_pos = self.skeleton_mapper.get_joint_position(parent.name, landmarks, self.scale)
                    elbow_pos = self.skeleton_mapper.get_joint_position(joint.name, landmarks, self.scale)
                    child = joint.children[0]  # ForeArm
                    wrist_pos = self.skeleton_mapper.get_joint_position(child.name, landmarks, self.scale)

                    if shoulder_pos is not None and elbow_pos is not None and wrist_pos is not None:
                        upper_arm = elbow_pos - shoulder_pos
                        forearm = wrist_pos - elbow_pos

                        if np.linalg.norm(forearm) > 1e-10 and np.linalg.norm(child.offset) > 0:
                            forearm_direction = forearm / np.linalg.norm(forearm)
                            rest_direction = child.offset / np.linalg.norm(child.offset)

                            euler_angles = calculate_rotation_from_directions(
                                rest_direction, forearm_direction, order='XYZ'
                            )

                            # Elbow constraints
                            euler_angles[1] *= 0.2  # Minimal Y rotation
                            euler_angles[2] *= 0.2  # Minimal Z rotation

                            calculated_euler = euler_angles
                            is_global = True

            elif joint.name in ["LeftForeArm", "RightForeArm"]:
                # IMPROVED: Better 3D hand reconstruction for ForeArm rotation
                hand_orientation = self._calculate_improved_hand_orientation(
                    landmarks,
                    left_hand_landmarks if joint.name == "LeftForeArm" else right_hand_landmarks,
                    joint.name == "LeftForeArm"
                )

                if hand_orientation is not None and joint.children:
                    child = joint.children[0]  # Hand
                    if np.linalg.norm(child.offset) > 0:
                        rest_direction = child.offset / np.linalg.norm(child.offset)

                        euler_angles = calculate_rotation_from_directions(
                            rest_direction, hand_orientation, order='XYZ'
                        )

                        # More natural wrist constraints
                        euler_angles[0] *= 0.9   # Good flexion/extension range
                        euler_angles[1] *= 0.7   # Limited radial/ulnar deviation
                        euler_angles[2] *= 0.8   # Moderate pronation/supination

                        calculated_euler = euler_angles
                        is_global = True

            elif joint.children:
                # Standard joint with children
                child = joint.children[0]
                direction = get_bone_direction(joint.name, child.name)

                if direction is not None:
                    if np.linalg.norm(child.offset) > 0:
                        rest_direction = child.offset / np.linalg.norm(child.offset)

                        if joint.name in ["LeftHand", "RightHand", "LeftFoot", "RightFoot"]:
                            if direction[2] < 0 and abs(direction[2]) > 0.1:
                                direction[2] = abs(direction[2])

                            if joint.name == "LeftHand" and direction[0] > 0:
                                direction[0] = -abs(direction[0])
                            elif joint.name == "RightHand" and direction[0] < 0:
                                direction[0] = abs(direction[0])

                        calculated_euler = calculate_rotation_from_directions(
                            rest_direction, direction, order='XYZ'
                        )
                        is_global = True
            
            # Process calculated rotation
            if calculated_euler is not None:
                if is_global:
                    # Convert Global Euler to Rotation Matrix
                    global_rot_obj = R.from_euler('XYZ', calculated_euler, degrees=True)
                    
                    # Calculate Local Rotation: R_local = inv(R_parent) * R_global
                    local_rot_obj = parent_rotation.inv() * global_rot_obj
                    local_rotation_euler = local_rot_obj.as_euler('XYZ', degrees=True)
                    
                    # Update global rotation for children
                    global_rotation = global_rot_obj
                else:
                    # It's already local
                    local_rotation_euler = calculated_euler
                    
                    # Update global rotation: R_global = R_parent * R_local
                    local_rot_obj = R.from_euler('XYZ', calculated_euler, degrees=True)
                    global_rotation = parent_rotation * local_rot_obj
            
            rotations[joint.name] = local_rotation_euler

            # Process children
            for child in joint.children:
                process_joint(child, global_rotation)

        # Start from root with Identity rotation
        process_joint(skeleton, R.identity())

        return rotations

    def _calculate_improved_hand_orientation(self, pose_landmarks, hand_landmarks, is_left: bool) -> Optional[np.ndarray]:
        """IMPROVED: Better 3D hand orientation calculation using world coordinates."""

        if not hand_landmarks or len(hand_landmarks) < 21:
            return None

        try:
            # Get wrist, elbow positions from pose world landmarks
            wrist_idx = mp_pose.PoseLandmark.LEFT_WRIST if is_left else mp_pose.PoseLandmark.RIGHT_WRIST
            elbow_idx = mp_pose.PoseLandmark.LEFT_ELBOW if is_left else mp_pose.PoseLandmark.RIGHT_ELBOW

            wrist_world = np.array([
                pose_landmarks[wrist_idx].x,
                -pose_landmarks[wrist_idx].y,
                pose_landmarks[wrist_idx].z
            ]) * self.scale

            elbow_world = np.array([
                pose_landmarks[elbow_idx].x,
                -pose_landmarks[elbow_idx].y,
                pose_landmarks[elbow_idx].z
            ]) * self.scale

            # Get forearm vector for depth reference
            forearm = wrist_world - elbow_world
            forearm_length = np.linalg.norm(forearm)

            # Hand landmarks are in normalized image space (0-1)
            # We need to transform them to world space using the wrist as anchor

            # Key hand landmarks
            wrist_2d = np.array([hand_landmarks[0].x, hand_landmarks[0].y])
            thumb_cmc = np.array([hand_landmarks[1].x, hand_landmarks[1].y])  # Thumb base
            index_mcp = np.array([hand_landmarks[5].x, hand_landmarks[5].y])
            middle_mcp = np.array([hand_landmarks[9].x, hand_landmarks[9].y])
            ring_mcp = np.array([hand_landmarks[13].x, hand_landmarks[13].y])
            pinky_mcp = np.array([hand_landmarks[17].x, hand_landmarks[17].y])
            middle_tip = np.array([hand_landmarks[12].x, hand_landmarks[12].y])

            # Calculate hand scale based on typical proportions
            # Hand length is approximately 0.4 * forearm length
            hand_scale = forearm_length * 0.4

            # Build hand coordinate system
            # Use MCP joints to define palm plane
            palm_center_2d = (index_mcp + middle_mcp + ring_mcp + pinky_mcp) / 4

            # Hand width vector (index to pinky)
            hand_width_2d = pinky_mcp - index_mcp
            # Hand length vector (wrist to middle finger)
            hand_length_2d = middle_tip - wrist_2d

            # Estimate 3D positions relative to wrist
            # The hand is roughly perpendicular to the forearm
            forearm_dir = forearm / forearm_length if forearm_length > 0 else np.array([0, 0, 1])

            # Create orthogonal basis for hand
            # Use cross product with up vector to get hand right vector
            up = np.array([0, 1, 0])
            hand_right = np.cross(forearm_dir, up)
            if np.linalg.norm(hand_right) < 0.1:  # Forearm is vertical
                hand_right = np.array([1, 0, 0])
            hand_right = hand_right / np.linalg.norm(hand_right)

            # Hand up vector
            hand_up = np.cross(hand_right, forearm_dir)
            hand_up = hand_up / np.linalg.norm(hand_up)

            # Transform 2D hand points to 3D using the hand basis
            # Map hand width to right axis, hand height to up axis
            width_3d = hand_width_2d[0] * hand_scale * hand_right
            height_3d = hand_length_2d[1] * hand_scale * hand_up

            # Calculate palm normal from MCP joints
            index_3d = wrist_world + (index_mcp[0] - wrist_2d[0]) * hand_scale * hand_right + \
                                    (index_mcp[1] - wrist_2d[1]) * hand_scale * hand_up

            pinky_3d = wrist_world + (pinky_mcp[0] - wrist_2d[0]) * hand_scale * hand_right + \
                                    (pinky_mcp[1] - wrist_2d[1]) * hand_scale * hand_up

            # Palm normal (perpendicular to palm)
            palm_normal = np.cross(v1, v2)
            if is_left:
                palm_normal = -palm_normal  # Flip for left hand

            if np.linalg.norm(palm_normal) > 1e-10:
                palm_normal = palm_normal / np.linalg.norm(palm_normal)

            # Hand forward direction (from wrist to middle finger)
            hand_forward = middle_3d - wrist_world
            if np.linalg.norm(hand_forward) > 1e-10:
                hand_forward = hand_forward / np.linalg.norm(hand_forward)

            # Combine for final hand orientation
            # Weight forward direction more than palm normal
            hand_orientation = hand_forward * 0.7 + palm_normal * 0.3
            hand_orientation = hand_orientation / np.linalg.norm(hand_orientation)

            return hand_orientation

        except Exception as e:
            # Fallback to simple direction if calculation fails
            return None

    def _calculate_head_rotation(self, landmarks) -> Optional[np.ndarray]:
        """Calculate head rotation from face landmarks."""
        try:
            nose_idx = mp_pose.PoseLandmark.NOSE
            l_ear_idx = mp_pose.PoseLandmark.LEFT_EAR
            r_ear_idx = mp_pose.PoseLandmark.RIGHT_EAR

            nose = np.array([landmarks[nose_idx].x, -landmarks[nose_idx].y, landmarks[nose_idx].z])
            l_ear = np.array([landmarks[l_ear_idx].x, -landmarks[l_ear_idx].y, landmarks[l_ear_idx].z])
            r_ear = np.array([landmarks[r_ear_idx].x, -landmarks[r_ear_idx].y, landmarks[r_ear_idx].z])

            # Calculate head basis vectors
            # Right vector: Right Ear - Left Ear (MP coordinates)
            # If +X is Left (based on offsets), then Right is -X.
            # L_Ear has higher X than R_Ear if +X is Left.
            # So R_Ear - L_Ear is negative X (Right).
            right = r_ear - l_ear
            right = right / np.linalg.norm(right)

            # Forward vector: Mid(Ears) to Nose
            mid_ears = (l_ear + r_ear) / 2
            forward = nose - mid_ears
            forward = forward / np.linalg.norm(forward)

            # Up vector: Cross(Right, Forward)
            up = np.cross(right, forward)
            up = up / np.linalg.norm(up)

            # Re-orthogonalize forward
            forward = np.cross(up, right)
            forward = forward / np.linalg.norm(forward)

            # We want to align the Head bone (Y-axis) with 'up'.
            # And Head bone (Z-axis) with 'forward'.
            # And Head bone (X-axis) with 'right' (or left?).
            
            # If X is Left, then Right is -X.
            # So Left = -Right.
            left = -right
            
            # Matrix columns: [Left, Up, Forward]
            # This assumes BVH local axes: X=Left, Y=Up, Z=Forward.
            rot_mat = np.column_stack((left, up, forward))
            
            from scipy.spatial.transform import Rotation as R
            r = R.from_matrix(rot_mat)
            euler = r.as_euler('XYZ', degrees=True)
            
            return euler

        except Exception:
            return None

    def _calculate_root_motion_from_feet(self, pose_frames: List[PoseFrame]) -> List[np.ndarray]:
        """
        Calculate root motion by tracking planted feet.
        When a foot is planted, the root moves in the opposite direction of the foot's relative movement.
        """
        positions = []
        
        # Initialize with first frame's hip center
        if not pose_frames or not pose_frames[0].world_landmarks:
            return [np.array([0.0, BVH_CONFIG['root_height'], 0.0])] * len(pose_frames)

        # Initial position (0,0,0) relative to start
        current_root_pos = np.array([0.0, BVH_CONFIG['root_height'], 0.0])
        positions.append(current_root_pos.copy())

        # Calculate leg length for depth estimation
        left_hip = np.array([pose_frames[0].world_landmarks[mp_pose.PoseLandmark.LEFT_HIP].x, pose_frames[0].world_landmarks[mp_pose.PoseLandmark.LEFT_HIP].y, pose_frames[0].world_landmarks[mp_pose.PoseLandmark.LEFT_HIP].z])
        left_knee = np.array([pose_frames[0].world_landmarks[mp_pose.PoseLandmark.LEFT_KNEE].x, pose_frames[0].world_landmarks[mp_pose.PoseLandmark.LEFT_KNEE].y, pose_frames[0].world_landmarks[mp_pose.PoseLandmark.LEFT_KNEE].z])
        left_ankle = np.array([pose_frames[0].world_landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].x, pose_frames[0].world_landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].y, pose_frames[0].world_landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].z])
        actual_leg_length = np.linalg.norm(left_knee - left_hip) + np.linalg.norm(left_ankle - left_knee)
        focal_length = PROCESSING_CONFIG.get('focal_length', 1.0)

        for i in range(1, len(pose_frames)):
            prev_frame = pose_frames[i-1]
            curr_frame = pose_frames[i]
            
            if not curr_frame.world_landmarks or not prev_frame.world_landmarks:
                positions.append(current_root_pos.copy())
                continue

            # Determine if feet are planted
            l_planted = False
            r_planted = False
            
            # Check if frame index is in detected contact frames
            # Note: self.foot_contact_frames is populated by _apply_improved_ik_corrections
            # We need to check if the *current* frame is a contact frame for either foot
            # But self.foot_contact_frames is a list of indices where ANY foot is in contact
            # We need more granular info. For now, let's re-evaluate contact or assume the list implies *at least one* foot.
            # Better approach: Check velocity of feet relative to hips in 2D to confirm which one is planted.
            
            # Get hip-relative foot positions (2D)
            l_foot_curr = np.array([curr_frame.landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].x, curr_frame.landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].y])
            r_foot_curr = np.array([curr_frame.landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].x, curr_frame.landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].y])
            l_foot_prev = np.array([prev_frame.landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].x, prev_frame.landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].y])
            r_foot_prev = np.array([prev_frame.landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].x, prev_frame.landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].y])
            
            l_hip_curr = np.array([curr_frame.landmarks[mp_pose.PoseLandmark.LEFT_HIP].x, curr_frame.landmarks[mp_pose.PoseLandmark.LEFT_HIP].y])
            r_hip_curr = np.array([curr_frame.landmarks[mp_pose.PoseLandmark.RIGHT_HIP].x, curr_frame.landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y])
            
            # Calculate foot velocity in screen space
            l_vel = np.linalg.norm(l_foot_curr - l_foot_prev)
            r_vel = np.linalg.norm(r_foot_curr - r_foot_prev)
            
            # Heuristic: Lower velocity = more likely planted
            # Also check height (Y)
            l_h = l_foot_curr[1]
            r_h = r_foot_curr[1]
            ground_y = max(l_h, r_h) # In MP, larger Y is lower (screen space)
            
            is_contact_frame = i in self.foot_contact_frames
            
            if is_contact_frame:
                if l_vel < r_vel:
                    l_planted = True
                else:
                    r_planted = True
                # If both are very slow, both might be planted
                if l_vel < 0.005 and r_vel < 0.005:
                    l_planted = True
                    r_planted = True
            
            # Calculate root delta
            root_delta = np.zeros(3)
            
            if l_planted or r_planted:
                # Calculate movement of planted foot relative to hip
                # If foot is planted, root moves opposite to foot's relative movement
                
                # We need 3D relative movement. Use World Landmarks but correct for root drift?
                # No, World Landmarks are root-centered (mostly).
                # If MP World Landmarks are root-centered, then:
                # Foot_World_Pos = Foot_Rel_Pos
                # If Foot is planted in world: Foot_World_Pos_t1 = Foot_World_Pos_t0
                # Root_World_Pos_t1 + Foot_Rel_Pos_t1 = Root_World_Pos_t0 + Foot_Rel_Pos_t0
                # Root_World_Pos_t1 - Root_World_Pos_t0 = Foot_Rel_Pos_t0 - Foot_Rel_Pos_t1
                # Delta_Root = -(Foot_Rel_Pos_t1 - Foot_Rel_Pos_t0)
                
                l_foot_rel_prev = np.array([prev_frame.world_landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].x, prev_frame.world_landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].y, prev_frame.world_landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].z])
                l_foot_rel_curr = np.array([curr_frame.world_landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].x, curr_frame.world_landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].y, curr_frame.world_landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].z])
                
                r_foot_rel_prev = np.array([prev_frame.world_landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].x, prev_frame.world_landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].y, prev_frame.world_landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].z])
                r_foot_rel_curr = np.array([curr_frame.world_landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].x, curr_frame.world_landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].y, curr_frame.world_landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].z])
                
                l_delta = -(l_foot_rel_curr - l_foot_rel_prev)
                r_delta = -(r_foot_rel_curr - r_foot_rel_prev)
                
                if l_planted and r_planted:
                    root_delta = (l_delta + r_delta) / 2.0
                elif l_planted:
                    root_delta = l_delta
                elif r_planted:
                    root_delta = r_delta
                    
                # Apply scale
                root_delta *= self.scale
                
                # Z-axis correction (MP Z is often unreliable, scale it down or use depth estimation)
                # Using depth estimation delta might be better for Z
                # Calculate observed leg lengths in 2D for depth
                l_obs_len_curr = np.linalg.norm(np.array([curr_frame.landmarks[mp_pose.PoseLandmark.LEFT_KNEE].x, curr_frame.landmarks[mp_pose.PoseLandmark.LEFT_KNEE].y]) - np.array([curr_frame.landmarks[mp_pose.PoseLandmark.LEFT_HIP].x, curr_frame.landmarks[mp_pose.PoseLandmark.LEFT_HIP].y])) + \
                                 np.linalg.norm(np.array([curr_frame.landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].x, curr_frame.landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].y]) - np.array([curr_frame.landmarks[mp_pose.PoseLandmark.LEFT_KNEE].x, curr_frame.landmarks[mp_pose.PoseLandmark.LEFT_KNEE].y]))
                l_depth_curr = calculate_depth_from_projected_length(l_obs_len_curr, actual_leg_length, focal_length)
                
                l_obs_len_prev = np.linalg.norm(np.array([prev_frame.landmarks[mp_pose.PoseLandmark.LEFT_KNEE].x, prev_frame.landmarks[mp_pose.PoseLandmark.LEFT_KNEE].y]) - np.array([prev_frame.landmarks[mp_pose.PoseLandmark.LEFT_HIP].x, prev_frame.landmarks[mp_pose.PoseLandmark.LEFT_HIP].y])) + \
                                 np.linalg.norm(np.array([prev_frame.landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].x, curr_frame.landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].y]) - np.array([prev_frame.landmarks[mp_pose.PoseLandmark.LEFT_KNEE].x, prev_frame.landmarks[mp_pose.PoseLandmark.LEFT_KNEE].y]))
                l_depth_prev = calculate_depth_from_projected_length(l_obs_len_prev, actual_leg_length, focal_length)
                
                depth_delta = l_depth_curr - l_depth_prev
                # Blend MP Z delta and calculated depth delta?
                # For now, stick to MP Z but scaled down if needed.
                
            else:
                # Flight phase - use MP trajectory (scaled) or depth estimation
                
                # Calculate hip center movement in screen space (X, Y)
                hip_curr_2d = (l_hip_curr + r_hip_curr) / 2.0
                hip_prev_2d = (np.array([prev_frame.landmarks[mp_pose.PoseLandmark.LEFT_HIP].x, prev_frame.landmarks[mp_pose.PoseLandmark.LEFT_HIP].y]) + 
                               np.array([prev_frame.landmarks[mp_pose.PoseLandmark.RIGHT_HIP].x, prev_frame.landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y])) / 2.0
                
                delta_2d = hip_curr_2d - hip_prev_2d
                root_delta[0] = delta_2d[0] * self.scale # X
                root_delta[1] = -delta_2d[1] * self.scale # Y (inverted)
                
                # Calculate depth change for Z
                # Use leg length projection to estimate depth change
                l_obs_len_curr = np.linalg.norm(np.array([curr_frame.landmarks[mp_pose.PoseLandmark.LEFT_KNEE].x, curr_frame.landmarks[mp_pose.PoseLandmark.LEFT_KNEE].y]) - np.array([curr_frame.landmarks[mp_pose.PoseLandmark.LEFT_HIP].x, curr_frame.landmarks[mp_pose.PoseLandmark.LEFT_HIP].y])) + \
                                 np.linalg.norm(np.array([curr_frame.landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].x, curr_frame.landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].y]) - np.array([curr_frame.landmarks[mp_pose.PoseLandmark.LEFT_KNEE].x, curr_frame.landmarks[mp_pose.PoseLandmark.LEFT_KNEE].y]))
                l_depth_curr = calculate_depth_from_projected_length(l_obs_len_curr, actual_leg_length, focal_length)
                
                l_obs_len_prev = np.linalg.norm(np.array([prev_frame.landmarks[mp_pose.PoseLandmark.LEFT_KNEE].x, prev_frame.landmarks[mp_pose.PoseLandmark.LEFT_KNEE].y]) - np.array([prev_frame.landmarks[mp_pose.PoseLandmark.LEFT_HIP].x, prev_frame.landmarks[mp_pose.PoseLandmark.LEFT_HIP].y])) + \
                                 np.linalg.norm(np.array([prev_frame.landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].x, curr_frame.landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].y]) - np.array([prev_frame.landmarks[mp_pose.PoseLandmark.LEFT_KNEE].x, prev_frame.landmarks[mp_pose.PoseLandmark.LEFT_KNEE].y]))
                l_depth_prev = calculate_depth_from_projected_length(l_obs_len_prev, actual_leg_length, focal_length)
                
                depth_delta = l_depth_curr - l_depth_prev
                
                # Apply depth delta to Z
                # Scale multiplier for depth to match X/Y scale roughly
                z_scale = PROCESSING_CONFIG.get('depth_scale_multiplier', 1.0) * 10.0 # Boost Z slightly
                root_delta[2] = depth_delta * z_scale

            # Update position
            current_root_pos += root_delta
            
            # Enforce ground floor constraint (Y shouldn't go too low)
            # But BVH Y is height.
            
            positions.append(current_root_pos.copy())

        # Apply smoothing
        if SMOOTHING_CONFIG['enable_temporal_smoothing']:
             positions_array = np.array(positions)
             smoothed_positions = smooth_positions(
                positions_array,
                window_size=SMOOTHING_CONFIG['temporal_window_size'],
                preserve_dynamics=SMOOTHING_CONFIG['preserve_dynamics'],
                preserve_y_axis=False
            )
             positions = [pos for pos in smoothed_positions]

        return positions

    def _smooth_motion(self, all_rotations: List[Dict[str, np.ndarray]]) -> List[Dict[str, np.ndarray]]:
        """Apply adaptive temporal smoothing to motion data."""
        if not all_rotations:
            return all_rotations

        joint_names = list(all_rotations[0].keys())
        smoothed_rotations = [{} for _ in range(len(all_rotations))]

        # Adaptive smoothing windows for different joint types
        joint_smoothing = {
            # Less smoothing for hands to preserve detail
            'LeftArm': 2, 'RightArm': 2,
            'LeftForeArm': 1, 'RightForeArm': 1,  # Minimal smoothing for ForeArms
            'LeftHand': 1, 'RightHand': 1,  # Minimal for hands

            # Moderate smoothing for torso
            'Hips': 3, 'Chest': 3, 'Neck': 3,

            # Standard smoothing for legs
            'LeftUpLeg': 2, 'RightUpLeg': 2,
            'LeftLeg': 2, 'RightLeg': 2,
            'LeftFoot': 2, 'RightFoot': 2,

            # Light smoothing for head and shoulders
            'Head': 2,
            'LeftShoulder': 2, 'RightShoulder': 2,
        }

        for joint_name in joint_names:
            joint_rotations = np.array([
                frame_rots[joint_name] for frame_rots in all_rotations
            ])

            if len(joint_rotations) > 1:
                velocity = np.diff(joint_rotations, axis=0)
                mean_velocity = np.mean(np.abs(velocity))

                base_window = joint_smoothing.get(joint_name, SMOOTHING_CONFIG['temporal_window_size'])

                # Adaptive window based on velocity
                if mean_velocity > 10.0:
                    window_size = max(1, base_window - 1)
                elif mean_velocity > 5.0:
                    window_size = base_window
                else:
                    window_size = min(5, base_window + 1)
            else:
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
        """Get zero rotations for all joints."""
        rotations = {}
        for joint in self.skeleton_mapper.get_all_joints():
            rotations[joint.name] = np.zeros(3)
        return rotations

    def _write_bvh(self, all_rotations: List[Dict[str, np.ndarray]],
                   hip_positions: List[np.ndarray], output_path: str) -> bool:
        """Write BVH file with motion data."""
        try:
            with open(output_path, 'w') as f:
                # Write hierarchy
                f.write("HIERARCHY\n")
                self._write_hierarchy(f, self.skeleton_mapper.skeleton, 0)

                # Write motion data
                num_frames = len(all_rotations)
                f.write("MOTION\n")
                f.write(f"Frames: {num_frames}\n")
                f.write(f"Frame Time: {self.frame_time:.6f}\n")

                # Write each frame
                for frame_idx in range(num_frames):
                    frame_data = []

                    # Root position from tracked hip movement
                    hip_pos = hip_positions[frame_idx]
                    frame_data.extend([hip_pos[0], hip_pos[1], hip_pos[2]])

                    # Write rotations for all joints
                    frame_rotations = all_rotations[frame_idx]
                    self._write_joint_rotations(
                        self.skeleton_mapper.skeleton,
                        frame_rotations,
                        frame_data
                    )

                    # Write frame line
                    f.write(" ".join([f"{val:.6f}" for val in frame_data]) + "\n")

            return True

        except Exception as e:
            print(f"Error writing BVH file: {e}")
            return False

    def _write_hierarchy(self, f, joint: BVHJoint, level: int):
        """Recursively write joint hierarchy."""
        indent = "  " * level

        if level == 0:
            f.write(f"{indent}ROOT {joint.name}\n")
        else:
            f.write(f"{indent}JOINT {joint.name}\n")

        f.write(f"{indent}{{\n")
        f.write(f"{indent}  OFFSET {joint.offset[0]:.6f} {joint.offset[1]:.6f} {joint.offset[2]:.6f}\n")

        # Write channels
        if joint.channels:
            channels_str = " ".join(joint.channels)
            f.write(f"{indent}  CHANNELS {len(joint.channels)} {channels_str}\n")

        # Write children
        for child in joint.children:
            self._write_hierarchy(f, child, level + 1)

        # Add end site for leaf joints
        if not joint.children:
            f.write(f"{indent}  End Site\n")
            f.write(f"{indent}  {{\n")
            end_offset = joint.offset * 0.3 if np.linalg.norm(joint.offset) > 0 else np.array([0, -5, 0])
            f.write(f"{indent}    OFFSET {end_offset[0]:.6f} {end_offset[1]:.6f} {end_offset[2]:.6f}\n")
            f.write(f"{indent}  }}\n")

        f.write(f"{indent}}}\n")

    def _write_joint_rotations(self, joint: BVHJoint, frame_rotations: Dict, frame_data: List):
        """Write rotation data for a joint and its children."""
        if joint.name in frame_rotations:
            rotation = frame_rotations[joint.name]
            # Write in XYZ order as specified in channels
            if 'Xrotation' in joint.channels:
                frame_data.append(rotation[0])
            if 'Yrotation' in joint.channels:
                frame_data.append(rotation[1])
            if 'Zrotation' in joint.channels:
                frame_data.append(rotation[2])

        # Process children
        for child in joint.children:
            self._write_joint_rotations(child, frame_rotations, frame_data)


def main():
    """Main entry point for the improved converter."""
    parser = argparse.ArgumentParser(description="Improved BVH Converter with better hand tracking and IK")
    parser.add_argument("--video", required=True, help="Path to input video file")
    parser.add_argument("--output", required=True, help="Path to output BVH file")
    parser.add_argument("--preview", action="store_true", help="Show pose detection preview")
    parser.add_argument("--sample-rate", type=int, default=2,
                       help="Process every Nth frame (default: 2)")
    parser.add_argument("--ik", action="store_true",
                       help="Enable improved IK foot locking")

    args = parser.parse_args()

    # Update config with command line arguments
    PROCESSING_CONFIG['sample_rate'] = args.sample_rate

    print("=" * 60)
    print("IMPROVED MediaPipe to BVH Converter")
    print("Fixes: Better hand tracking, calibrated IK, drift correction")
    print("=" * 60)

    start_time = time.time()

    # Extract poses from video
    with MediaPipeExtractor(use_holistic=True) as extractor:
        extractor.sample_rate = args.sample_rate
        print("Using MediaPipe Holistic model with improved hand reconstruction...")
        pose_frames = extractor.extract_from_video(args.video, preview=args.preview)

        if not pose_frames:
            print("Error: No poses extracted from video")
            return

        # Interpolate missing frames if needed
        pose_frames = extractor.interpolate_missing_frames(pose_frames)

    # Convert to BVH with improvements
    if args.ik:
        print("Using improved IK foot locking with calibrated thresholds...")
    converter = ImprovedBVHConverter(enable_ik=args.ik)
    success = converter.convert(pose_frames, args.output)

    elapsed_time = time.time() - start_time

    if success:
        print(f"\nConversion completed in {elapsed_time:.2f} seconds")
        print(f"Output saved to: {args.output}")
        print("\n✅ Improvements applied:")
        print("  - Better 3D hand reconstruction for ForeArm/Wrist")
        print("  - Calibrated IK thresholds for foot contact")
        if args.ik:
            print("  - Foot-based drift correction for walking")
    else:
        print("\nConversion failed")


if __name__ == "__main__":
    main()