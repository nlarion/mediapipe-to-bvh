"""
Inverse Kinematics (IK) Foot Locking System

This module implements IK-based foot contact detection and locking to prevent
foot sliding during ground contact phases in motion capture data.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from scipy.signal import butter, filtfilt
from enum import Enum


class FootState(Enum):
    """Foot contact state"""
    SWING = 0      # Foot in air, moving
    PLANTED = 1    # Foot on ground, locked
    TRANSITION = 2 # Transitioning between states


@dataclass
class FootContact:
    """Foot contact information for a single frame"""
    state: FootState
    confidence: float  # 0.0 to 1.0
    position: np.ndarray  # 3D position when planted
    frame_index: int
    velocity: float
    height: float


@dataclass
class IKChainConfig:
    """Configuration for IK leg chain"""
    thigh_length: float  # Distance from hip to knee
    shin_length: float   # Distance from knee to ankle
    foot_height_offset: float  # Ground plane offset

    # Thresholds from reference BVH analysis
    # Increased velocity threshold to tolerate MediaPipe noise
    contact_velocity_threshold: float = 8.0  # units/frame (adjustable)
    contact_height_threshold: float = 12.0   # units from ground
    foot_clearance_height: float = 5.0       # Expected foot clearance when walking
    sliding_threshold: float = 2.0           # Maximum allowed sliding distance
    vertical_velocity_threshold: float = 2.0 # Max vertical velocity for planted foot


class FootContactDetector:
    """Detects foot contact states from motion data"""

    def __init__(self, config: IKChainConfig):
        self.config = config
        self.contact_history: Dict[str, List[FootContact]] = {
            'left': [],
            'right': []
        }
        # Hysteresis parameters to prevent rapid state changes
        self.min_contact_frames = 3  # Minimum frames to consider foot planted
        self.min_swing_frames = 2    # Minimum frames to consider foot in swing

    def detect_contact(
        self,
        foot_position: np.ndarray,
        foot_velocity: np.ndarray,
        frame_index: int,
        foot_side: str = 'left',
        external_contact: Optional[bool] = None
    ) -> FootContact:
        """
        Detect foot contact state based on position and velocity.

        Args:
            foot_position: 3D position of foot
            foot_velocity: 3D velocity of foot
            frame_index: Current frame number
            foot_side: 'left' or 'right'
            external_contact: Optional override from external detector
        
        Returns:
            FootContact object with state and confidence
        """
        # Calculate velocity magnitude
        velocity_mag = np.linalg.norm(foot_velocity)
        vertical_velocity = abs(foot_velocity[1])

        # Get foot height (Y component)
        foot_height = foot_position[1]

        # Check thresholds
        is_low_velocity = velocity_mag < self.config.contact_velocity_threshold
        is_low_vertical_velocity = vertical_velocity < self.config.vertical_velocity_threshold
        is_near_ground = foot_height < self.config.contact_height_threshold

        # Calculate base confidence from velocity and height
        # Velocity confidence: Higher is better (lower velocity)
        velocity_confidence = max(0, 1.0 - velocity_mag / (self.config.contact_velocity_threshold * 1.5))
        
        # Height confidence: Higher is better (lower height)
        height_confidence = max(0, 1.0 - foot_height / (self.config.contact_height_threshold * 1.5))
        
        # Vertical velocity confidence: Planted feet shouldn't move up/down much
        vert_vel_confidence = max(0, 1.0 - vertical_velocity / (self.config.vertical_velocity_threshold * 2.0))

        # Combined confidence - Weighted sum
        # Height is the strongest indicator for "can be planted"
        # Velocity is the strongest indicator for "is planted"
        base_confidence = (
            velocity_confidence * 0.4 + 
            height_confidence * 0.4 + 
            vert_vel_confidence * 0.2
        )

        # Apply hysteresis using history
        history = self.contact_history[foot_side]
        if len(history) >= self.min_contact_frames:
            recent_states = [c.state for c in history[-self.min_contact_frames:]]

            # Check for consistent states
            if all(s == FootState.PLANTED for s in recent_states):
                # Bias toward staying planted
                base_confidence = min(1.0, base_confidence * 1.3)
            elif all(s == FootState.SWING for s in recent_states):
                # Bias toward staying in swing
                base_confidence = max(0.0, base_confidence * 0.8)

        # Determine state based on confidence
        # Stricter requirements for entering PLANTED state
        
        # Check external override first
        if external_contact is True:
            state = FootState.PLANTED
            confidence = 1.0
            # Force update base confidence for history consistency
            base_confidence = 1.0
        elif external_contact is False:
            state = FootState.SWING
            confidence = 1.0
            base_confidence = 0.0
        elif base_confidence > 0.65 and is_low_velocity and is_near_ground:
            state = FootState.PLANTED
            confidence = base_confidence
        elif base_confidence < 0.35 or not is_near_ground:
            state = FootState.SWING
            confidence = 1.0 - base_confidence
        else:
            state = FootState.TRANSITION
            confidence = 0.5

        # Create contact info
        # If we are planted and were planted previously, keep the original planted position
        # This prevents the "locked" position from drifting with the sliding input
        final_position = foot_position.copy()
        
        if state == FootState.PLANTED and len(history) > 0:
            last_contact = history[-1]
            if last_contact.state == FootState.PLANTED:
                final_position = last_contact.position.copy()

        contact = FootContact(
            state=state,
            confidence=confidence,
            position=final_position,
            frame_index=frame_index,
            velocity=velocity_mag,
            height=foot_height
        )

        # Update history
        history.append(contact)
        if len(history) > 10:  # Keep last 10 frames
            history.pop(0)

        return contact

    def get_planted_position(self, foot_side: str) -> Optional[np.ndarray]:
        """Get the last planted position for a foot"""
        history = self.contact_history[foot_side]
        for contact in reversed(history):
            if contact.state == FootState.PLANTED:
                return contact.position.copy()
        return None


class TwoBoneIKSolver:
    """
    Two-bone IK solver for leg chains using analytical solution.
    Solves for knee position given hip and ankle positions.
    """

    def __init__(self, config: IKChainConfig):
        self.config = config

    def solve(
        self,
        hip_position: np.ndarray,
        target_ankle_position: np.ndarray,
        knee_hint_vector: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, bool]:
        """
        Solve IK for knee position given hip and ankle.

        Args:
            hip_position: 3D position of hip joint
            target_ankle_position: 3D target position for ankle
            knee_hint_vector: Optional hint for knee bend direction

        Returns:
            Tuple of (knee_position, success)
        """
        # Vector from hip to target
        hip_to_target = target_ankle_position - hip_position
        distance = np.linalg.norm(hip_to_target)

        # Check if target is reachable
        max_reach = self.config.thigh_length + self.config.shin_length
        min_reach = abs(self.config.thigh_length - self.config.shin_length)

        if distance > max_reach or distance < min_reach:
            # Target unreachable - clamp to valid range
            if distance > max_reach:
                scale = (max_reach * 0.99) / distance
                target_ankle_position = hip_position + hip_to_target * scale
                hip_to_target = target_ankle_position - hip_position
                distance = np.linalg.norm(hip_to_target)
            else:
                # Too close - extend slightly
                scale = (min_reach * 1.01) / distance
                target_ankle_position = hip_position + hip_to_target * scale
                hip_to_target = target_ankle_position - hip_position
                distance = np.linalg.norm(hip_to_target)

        # Calculate knee angle using law of cosines
        # cos(knee_angle) = (thigh² + shin² - distance²) / (2 * thigh * shin)
        thigh_sq = self.config.thigh_length ** 2
        shin_sq = self.config.shin_length ** 2
        dist_sq = distance ** 2

        cos_knee = (thigh_sq + shin_sq - dist_sq) / (2 * self.config.thigh_length * self.config.shin_length)
        cos_knee = np.clip(cos_knee, -1.0, 1.0)
        knee_angle = np.arccos(cos_knee)

        # Calculate hip angle
        # cos(hip_angle) = (thigh² + distance² - shin²) / (2 * thigh * distance)
        cos_hip = (thigh_sq + dist_sq - shin_sq) / (2 * self.config.thigh_length * distance)
        cos_hip = np.clip(cos_hip, -1.0, 1.0)
        hip_angle = np.arccos(cos_hip)

        # Find knee position
        # First, get the forward direction
        forward = hip_to_target / distance

        # Determine knee bend direction
        if knee_hint_vector is not None:
            # Use hint vector (typically forward for natural knee bend)
            right = np.cross(forward, knee_hint_vector)
            right = right / (np.linalg.norm(right) + 1e-6)
            up = np.cross(right, forward)
        else:
            # Default: assume knee bends forward (positive Z in most setups)
            if abs(forward[1]) < 0.99:  # Not vertical
                right = np.cross(forward, np.array([0, 1, 0]))
            else:
                right = np.cross(forward, np.array([1, 0, 0]))
            right = right / (np.linalg.norm(right) + 1e-6)
            up = np.cross(right, forward)

        # Calculate knee position
        knee_forward = self.config.thigh_length * np.cos(hip_angle)
        knee_up = self.config.thigh_length * np.sin(hip_angle)

        knee_position = hip_position + forward * knee_forward + up * knee_up

        # Verify solution
        hip_to_knee_dist = np.linalg.norm(knee_position - hip_position)
        knee_to_ankle_dist = np.linalg.norm(target_ankle_position - knee_position)

        error_threshold = 0.1  # Allow small numerical errors
        success = (
            abs(hip_to_knee_dist - self.config.thigh_length) < error_threshold and
            abs(knee_to_ankle_dist - self.config.shin_length) < error_threshold
        )

        return knee_position, success


class IKFootLockSystem:
    """
    Complete IK foot locking system that integrates contact detection
    and IK solving with smooth blending.
    """

    def __init__(self, thigh_length: float, shin_length: float):
        """
        Initialize IK foot lock system.

        Args:
            thigh_length: Length of thigh bone
            shin_length: Length of shin bone
        """
        self.config = IKChainConfig(
            thigh_length=thigh_length,
            shin_length=shin_length,
            foot_height_offset=0.0,
            contact_velocity_threshold=5.0,
            contact_height_threshold=10.0
        )

        self.left_detector = FootContactDetector(self.config)
        self.right_detector = FootContactDetector(self.config)
        self.ik_solver = TwoBoneIKSolver(self.config)

        # Blending parameters
        self.blend_frames = 5  # Frames to blend over during transitions

    def process_frame(
        self,
        hip_positions: Dict[str, np.ndarray],  # 'left' and 'right' hip positions
        knee_positions: Dict[str, np.ndarray],
        ankle_positions: Dict[str, np.ndarray],
        frame_index: int,
        previous_ankle_positions: Optional[Dict[str, np.ndarray]] = None,
        contact_overrides: Optional[Dict[str, bool]] = None
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Process a single frame with IK foot locking.

        Args:
            hip_positions: Dictionary with 'left' and 'right' hip positions
            knee_positions: Original knee positions from tracking
            ankle_positions: Original ankle positions from tracking
            frame_index: Current frame number
            previous_ankle_positions: Ankle positions from previous frame
            contact_overrides: Optional dictionary of contact overrides ('left', 'right')

        Returns:
            Dictionary with corrected positions for each side
        """
        results = {}

        for side in ['left', 'right']:
            # Calculate velocity if previous frame exists
            if previous_ankle_positions and side in previous_ankle_positions:
                velocity = ankle_positions[side] - previous_ankle_positions[side]
            else:
                velocity = np.zeros(3)

            # Detect foot contact
            detector = self.left_detector if side == 'left' else self.right_detector
            
            external_contact = None
            if contact_overrides and side in contact_overrides:
                external_contact = contact_overrides[side]
                
            contact = detector.detect_contact(
                ankle_positions[side],
                velocity,
                frame_index,
                side,
                external_contact
            )

            # Apply IK if foot is planted
            if contact.state == FootState.PLANTED:
                # Get planted position or use current if first plant
                planted_pos = detector.get_planted_position(side)
                if planted_pos is None:
                    planted_pos = ankle_positions[side].copy()

                # Solve IK for locked ankle
                knee_hint = knee_positions[side] - hip_positions[side]
                new_knee, success = self.ik_solver.solve(
                    hip_positions[side],
                    planted_pos,
                    knee_hint
                )

                if success:
                    # Full IK solution
                    results[side] = {
                        'hip': hip_positions[side],
                        'knee': new_knee,
                        'ankle': planted_pos,
                        'confidence': contact.confidence
                    }
                else:
                    # Fallback to original
                    results[side] = {
                        'hip': hip_positions[side],
                        'knee': knee_positions[side],
                        'ankle': ankle_positions[side],
                        'confidence': 0.0
                    }

            elif contact.state == FootState.TRANSITION:
                # Blend between original and IK
                planted_pos = detector.get_planted_position(side)
                if planted_pos is not None:
                    # Solve IK for planted position
                    knee_hint = knee_positions[side] - hip_positions[side]
                    new_knee, success = self.ik_solver.solve(
                        hip_positions[side],
                        planted_pos,
                        knee_hint
                    )

                    if success:
                        # Blend based on confidence
                        blend_weight = contact.confidence
                        results[side] = {
                            'hip': hip_positions[side],
                            'knee': knee_positions[side] * (1 - blend_weight) + new_knee * blend_weight,
                            'ankle': ankle_positions[side] * (1 - blend_weight) + planted_pos * blend_weight,
                            'confidence': contact.confidence
                        }
                    else:
                        results[side] = {
                            'hip': hip_positions[side],
                            'knee': knee_positions[side],
                            'ankle': ankle_positions[side],
                            'confidence': 0.0
                        }
                else:
                    # No previous plant, use original
                    results[side] = {
                        'hip': hip_positions[side],
                        'knee': knee_positions[side],
                        'ankle': ankle_positions[side],
                        'confidence': 0.0
                    }

            else:  # SWING state
                # Use original positions
                results[side] = {
                    'hip': hip_positions[side],
                    'knee': knee_positions[side],
                    'ankle': ankle_positions[side],
                    'confidence': 0.0
                }

        return results


def smooth_foot_positions(positions: np.ndarray, window_size: int = 3) -> np.ndarray:
    """Apply smoothing to foot positions to reduce jitter"""
    if len(positions) < window_size:
        return positions

    # Use Butterworth filter for smoothing
    b, a = butter(2, 0.3)
    smoothed = np.zeros_like(positions)

    for i in range(3):  # For each axis
        smoothed[:, i] = filtfilt(b, a, positions[:, i])

    return smoothed


# Test function
def test_ik_solver():
    """Test the IK solver with known configurations"""
    print("Testing IK Solver...")

    config = IKChainConfig(
        thigh_length=40.0,
        shin_length=40.0,
        foot_height_offset=0.0
    )

    solver = TwoBoneIKSolver(config)

    # Test case 1: Straight leg
    hip = np.array([0, 100, 0])
    ankle_target = np.array([0, 20, 0])

    knee, success = solver.solve(hip, ankle_target)

    print(f"\nTest 1 - Straight leg:")
    print(f"  Hip: {hip}")
    print(f"  Target Ankle: {ankle_target}")
    print(f"  Calculated Knee: {knee}")
    print(f"  Success: {success}")

    if success:
        hip_knee_dist = np.linalg.norm(knee - hip)
        knee_ankle_dist = np.linalg.norm(ankle_target - knee)
        print(f"  Hip-Knee distance: {hip_knee_dist:.2f} (expected: {config.thigh_length})")
        print(f"  Knee-Ankle distance: {knee_ankle_dist:.2f} (expected: {config.shin_length})")

    # Test case 2: Bent knee
    ankle_target = np.array([30, 40, 20])
    knee_hint = np.array([1, 0, 1])  # Forward-ish direction

    knee, success = solver.solve(hip, ankle_target, knee_hint)

    print(f"\nTest 2 - Bent knee:")
    print(f"  Hip: {hip}")
    print(f"  Target Ankle: {ankle_target}")
    print(f"  Calculated Knee: {knee}")
    print(f"  Success: {success}")

    if success:
        hip_knee_dist = np.linalg.norm(knee - hip)
        knee_ankle_dist = np.linalg.norm(ankle_target - knee)
        print(f"  Hip-Knee distance: {hip_knee_dist:.2f} (expected: {config.thigh_length})")
        print(f"  Knee-Ankle distance: {knee_ankle_dist:.2f} (expected: {config.shin_length})")

    # Test case 3: Unreachable target (too far)
    ankle_target = np.array([100, 0, 0])

    knee, success = solver.solve(hip, ankle_target)

    print(f"\nTest 3 - Unreachable target:")
    print(f"  Hip: {hip}")
    print(f"  Target Ankle: {ankle_target}")
    print(f"  Calculated Knee: {knee}")
    print(f"  Success: {success}")
    print(f"  Note: Should clamp to maximum reach")

    return True


if __name__ == "__main__":
    # Run tests
    test_ik_solver()