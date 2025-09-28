#!/usr/bin/env python3
"""
BVH Reference Motion Analyzer

Analyzes reference BVH files to extract motion characteristics for comparison.
Used to evaluate the quality of generated BVH files against known good examples.
"""

import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from scipy.spatial.transform import Rotation as R
from scipy.signal import find_peaks, butter, filtfilt
from scipy import stats
import json

@dataclass
class ReferenceMotionProfile:
    """Profile of reference motion characteristics"""
    # Gait characteristics
    stride_length: float
    stride_frequency: float
    gait_symmetry: float

    # Joint angle ranges
    joint_angle_ranges: Dict[str, Tuple[float, float]]  # min, max for each joint
    joint_angle_velocities: Dict[str, float]  # average angular velocity

    # Timing patterns
    stance_phase_duration: float  # percentage of gait cycle
    swing_phase_duration: float
    double_support_time: float

    # Motion smoothness
    jerk_metric: float
    acceleration_profile: np.ndarray

    # Vertical oscillation
    center_of_mass_oscillation: float
    hip_vertical_displacement: float

    # Foot characteristics
    foot_clearance_height: float
    foot_contact_velocity: float
    foot_sliding_threshold: float

    # Arm swing
    arm_swing_amplitude: float
    arm_swing_phase_offset: float  # relative to legs

    # Spine dynamics
    spine_rotation_range: float
    spine_lateral_sway: float

    # Energy metrics
    kinetic_energy_pattern: np.ndarray
    potential_energy_pattern: np.ndarray


class BVHReferenceAnalyzer:
    """Analyzes reference BVH files to create motion profiles"""

    def __init__(self):
        self.joint_hierarchy = {}
        self.motion_data = []
        self.frame_time = 0.0

    def parse_bvh(self, filepath: Path) -> Dict:
        """Parse BVH file and extract motion data"""
        with open(filepath, 'r') as f:
            lines = f.readlines()

        # Parse hierarchy
        hierarchy_data = {}
        motion_section_start = 0
        current_joint = None
        joint_stack = []
        joint_channels = {}
        channel_index = 0

        for i, line in enumerate(lines):
            line = line.strip()

            if line.startswith('ROOT'):
                current_joint = line.split()[1]
                hierarchy_data[current_joint] = {
                    'parent': None,
                    'offset': None,
                    'channels': [],
                    'children': []
                }
                joint_stack = [current_joint]

            elif line.startswith('JOINT') or line.startswith('End Site'):
                if line.startswith('JOINT'):
                    joint_name = line.split()[1]
                    parent = joint_stack[-1] if joint_stack else None
                    hierarchy_data[joint_name] = {
                        'parent': parent,
                        'offset': None,
                        'channels': [],
                        'children': []
                    }
                    if parent:
                        hierarchy_data[parent]['children'].append(joint_name)
                    joint_stack.append(joint_name)
                    current_joint = joint_name

            elif line.startswith('OFFSET'):
                parts = line.split()
                offset = [float(parts[1]), float(parts[2]), float(parts[3])]
                if current_joint and current_joint in hierarchy_data:
                    hierarchy_data[current_joint]['offset'] = offset

            elif line.startswith('CHANNELS'):
                parts = line.split()
                num_channels = int(parts[1])
                channels = parts[2:2+num_channels]
                if current_joint and current_joint in hierarchy_data:
                    hierarchy_data[current_joint]['channels'] = channels
                    joint_channels[current_joint] = {
                        'channels': channels,
                        'index': channel_index
                    }
                    channel_index += num_channels

            elif line == '}':
                if joint_stack:
                    joint_stack.pop()
                    if joint_stack:
                        current_joint = joint_stack[-1]

            elif line == 'MOTION':
                motion_section_start = i
                break

        # Parse motion data
        frames = 0
        frame_time = 0.0
        motion_data = []

        for i in range(motion_section_start + 1, len(lines)):
            line = lines[i].strip()

            if line.startswith('Frames:'):
                frames = int(line.split()[1])
            elif line.startswith('Frame Time:'):
                frame_time = float(line.split()[2])
            elif line and not line.startswith('Frame'):
                values = [float(v) for v in line.split()]
                if values:
                    motion_data.append(values)

        return {
            'hierarchy': hierarchy_data,
            'joint_channels': joint_channels,
            'motion': np.array(motion_data),
            'frames': frames,
            'frame_time': frame_time
        }

    def extract_joint_angles(self, bvh_data: Dict) -> Dict[str, np.ndarray]:
        """Extract joint angles from BVH motion data"""
        joint_angles = {}
        joint_channels = bvh_data['joint_channels']
        motion = bvh_data['motion']

        for joint_name, channel_info in joint_channels.items():
            channels = channel_info['channels']
            start_idx = channel_info['index']

            # Extract rotation channels
            rotation_indices = []
            rotation_order = ''

            for i, channel in enumerate(channels):
                if 'rotation' in channel.lower():
                    rotation_indices.append(start_idx + i)
                    rotation_order += channel[0].upper()

            if rotation_indices:
                angles = motion[:, rotation_indices]
                joint_angles[joint_name] = angles

        return joint_angles

    def detect_gait_cycles(self, bvh_data: Dict) -> List[Tuple[int, int]]:
        """Detect gait cycles from foot motion"""
        motion = bvh_data['motion']
        hierarchy = bvh_data['hierarchy']

        # Find foot joints
        foot_joints = []
        for joint_name in hierarchy.keys():
            if any(foot_keyword in joint_name.lower()
                   for foot_keyword in ['foot', 'ankle', 'toe']):
                foot_joints.append(joint_name)

        if not foot_joints:
            # Fallback: use lowest joints in hierarchy
            foot_joints = self._find_leaf_joints(hierarchy)

        # Analyze vertical motion of feet to detect steps
        cycles = []

        if foot_joints and bvh_data['joint_channels']:
            # Get root position (usually has Y position)
            root_joint = None
            for joint_name in hierarchy.keys():
                if hierarchy[joint_name]['parent'] is None:
                    root_joint = joint_name
                    break

            if root_joint and root_joint in bvh_data['joint_channels']:
                channel_info = bvh_data['joint_channels'][root_joint]
                y_pos_idx = None

                for i, channel in enumerate(channel_info['channels']):
                    if channel.lower() == 'yposition':
                        y_pos_idx = channel_info['index'] + i
                        break

                if y_pos_idx is not None:
                    y_positions = motion[:, y_pos_idx]

                    # Smooth the signal
                    b, a = butter(4, 0.1)
                    y_smooth = filtfilt(b, a, y_positions)

                    # Find peaks (foot contacts)
                    peaks, _ = find_peaks(-y_smooth, distance=10)

                    # Group peaks into cycles
                    for i in range(len(peaks) - 1):
                        cycles.append((peaks[i], peaks[i + 1]))

        return cycles

    def _find_leaf_joints(self, hierarchy: Dict) -> List[str]:
        """Find leaf joints (no children) in the hierarchy"""
        leaf_joints = []
        for joint_name, joint_data in hierarchy.items():
            if not joint_data['children']:
                leaf_joints.append(joint_name)
        return leaf_joints

    def calculate_motion_smoothness(self, motion_data: np.ndarray, frame_time: float) -> float:
        """Calculate jerk metric for motion smoothness"""
        if len(motion_data) < 3:
            return 0.0

        # Calculate velocities and accelerations
        dt = frame_time
        velocity = np.diff(motion_data, axis=0) / dt
        acceleration = np.diff(velocity, axis=0) / dt
        jerk = np.diff(acceleration, axis=0) / dt

        # RMS jerk
        jerk_magnitude = np.sqrt(np.sum(jerk**2, axis=1))
        return float(np.mean(jerk_magnitude))

    def analyze_reference_motion(self, filepath: Path) -> ReferenceMotionProfile:
        """Create a complete motion profile from reference BVH"""
        bvh_data = self.parse_bvh(filepath)

        # Extract joint angles
        joint_angles = self.extract_joint_angles(bvh_data)

        # Calculate joint angle ranges and velocities
        joint_angle_ranges = {}
        joint_angle_velocities = {}

        for joint_name, angles in joint_angles.items():
            if len(angles) > 0:
                joint_angle_ranges[joint_name] = (
                    float(np.min(angles)),
                    float(np.max(angles))
                )

                if len(angles) > 1:
                    angular_velocity = np.diff(angles, axis=0) / bvh_data['frame_time']
                    joint_angle_velocities[joint_name] = float(np.mean(np.abs(angular_velocity)))
                else:
                    joint_angle_velocities[joint_name] = 0.0

        # Detect gait cycles
        gait_cycles = self.detect_gait_cycles(bvh_data)

        # Calculate gait characteristics
        if gait_cycles:
            cycle_lengths = [end - start for start, end in gait_cycles]
            avg_cycle_length = np.mean(cycle_lengths) * bvh_data['frame_time']
            stride_frequency = 1.0 / avg_cycle_length if avg_cycle_length > 0 else 0.0

            # Estimate stance/swing phases (simplified)
            stance_phase_duration = 0.6  # Typical value
            swing_phase_duration = 0.4
            double_support_time = 0.2
        else:
            stride_frequency = 0.0
            stance_phase_duration = 0.0
            swing_phase_duration = 0.0
            double_support_time = 0.0

        # Calculate motion smoothness
        jerk_metric = self.calculate_motion_smoothness(
            bvh_data['motion'],
            bvh_data['frame_time']
        )

        # Extract root motion for stride and oscillation analysis
        root_motion = bvh_data['motion'][:, :3] if len(bvh_data['motion'][0]) >= 3 else np.zeros((len(bvh_data['motion']), 3))

        # Calculate stride length (simplified from X-Z displacement)
        if len(root_motion) > 1:
            total_displacement = np.sqrt(
                (root_motion[-1, 0] - root_motion[0, 0])**2 +
                (root_motion[-1, 2] - root_motion[0, 2])**2
            )
            stride_length = total_displacement / max(len(gait_cycles), 1)
        else:
            stride_length = 0.0

        # Calculate vertical oscillation
        if len(root_motion) > 0:
            hip_vertical_displacement = float(np.ptp(root_motion[:, 1]))
            center_of_mass_oscillation = hip_vertical_displacement  # Simplified
        else:
            hip_vertical_displacement = 0.0
            center_of_mass_oscillation = 0.0

        # Calculate acceleration profile
        if len(root_motion) > 2:
            velocity = np.diff(root_motion, axis=0) / bvh_data['frame_time']
            acceleration = np.diff(velocity, axis=0) / bvh_data['frame_time']
            acceleration_profile = np.linalg.norm(acceleration, axis=1)
        else:
            acceleration_profile = np.array([])

        # Energy patterns (simplified)
        frames = len(bvh_data['motion'])
        kinetic_energy_pattern = np.zeros(frames)
        potential_energy_pattern = np.zeros(frames)

        if len(root_motion) > 1:
            velocity_magnitude = np.linalg.norm(
                np.diff(root_motion, axis=0) / bvh_data['frame_time'],
                axis=1
            )
            kinetic_energy_pattern[1:] = 0.5 * velocity_magnitude**2
            potential_energy_pattern = 9.8 * root_motion[:, 1]

        # Calculate gait symmetry (simplified)
        gait_symmetry = 0.9  # Default high symmetry for reference

        # Arm swing analysis (simplified estimates)
        arm_swing_amplitude = 30.0  # degrees
        arm_swing_phase_offset = 180.0  # opposite to legs

        # Spine dynamics (simplified estimates)
        spine_rotation_range = 10.0  # degrees
        spine_lateral_sway = 5.0  # degrees

        # Foot characteristics (simplified estimates)
        foot_clearance_height = 5.0  # cm
        foot_contact_velocity = 0.1  # m/s
        foot_sliding_threshold = 0.02  # m

        return ReferenceMotionProfile(
            stride_length=stride_length,
            stride_frequency=stride_frequency,
            gait_symmetry=gait_symmetry,
            joint_angle_ranges=joint_angle_ranges,
            joint_angle_velocities=joint_angle_velocities,
            stance_phase_duration=stance_phase_duration,
            swing_phase_duration=swing_phase_duration,
            double_support_time=double_support_time,
            jerk_metric=jerk_metric,
            acceleration_profile=acceleration_profile,
            center_of_mass_oscillation=center_of_mass_oscillation,
            hip_vertical_displacement=hip_vertical_displacement,
            foot_clearance_height=foot_clearance_height,
            foot_contact_velocity=foot_contact_velocity,
            foot_sliding_threshold=foot_sliding_threshold,
            arm_swing_amplitude=arm_swing_amplitude,
            arm_swing_phase_offset=arm_swing_phase_offset,
            spine_rotation_range=spine_rotation_range,
            spine_lateral_sway=spine_lateral_sway,
            kinetic_energy_pattern=kinetic_energy_pattern,
            potential_energy_pattern=potential_energy_pattern
        )

    def save_profile(self, profile: ReferenceMotionProfile, output_path: Path):
        """Save motion profile to JSON"""
        # Convert numpy arrays to lists for JSON serialization
        profile_dict = {
            'stride_length': profile.stride_length,
            'stride_frequency': profile.stride_frequency,
            'gait_symmetry': profile.gait_symmetry,
            'joint_angle_ranges': profile.joint_angle_ranges,
            'joint_angle_velocities': profile.joint_angle_velocities,
            'stance_phase_duration': profile.stance_phase_duration,
            'swing_phase_duration': profile.swing_phase_duration,
            'double_support_time': profile.double_support_time,
            'jerk_metric': profile.jerk_metric,
            'acceleration_profile': profile.acceleration_profile.tolist(),
            'center_of_mass_oscillation': profile.center_of_mass_oscillation,
            'hip_vertical_displacement': profile.hip_vertical_displacement,
            'foot_clearance_height': profile.foot_clearance_height,
            'foot_contact_velocity': profile.foot_contact_velocity,
            'foot_sliding_threshold': profile.foot_sliding_threshold,
            'arm_swing_amplitude': profile.arm_swing_amplitude,
            'arm_swing_phase_offset': profile.arm_swing_phase_offset,
            'spine_rotation_range': profile.spine_rotation_range,
            'spine_lateral_sway': profile.spine_lateral_sway,
            'kinetic_energy_pattern': profile.kinetic_energy_pattern.tolist(),
            'potential_energy_pattern': profile.potential_energy_pattern.tolist()
        }

        with open(output_path, 'w') as f:
            json.dump(profile_dict, f, indent=2)

    def load_profile(self, filepath: Path) -> ReferenceMotionProfile:
        """Load motion profile from JSON"""
        with open(filepath, 'r') as f:
            data = json.load(f)

        return ReferenceMotionProfile(
            stride_length=data['stride_length'],
            stride_frequency=data['stride_frequency'],
            gait_symmetry=data['gait_symmetry'],
            joint_angle_ranges=data['joint_angle_ranges'],
            joint_angle_velocities=data['joint_angle_velocities'],
            stance_phase_duration=data['stance_phase_duration'],
            swing_phase_duration=data['swing_phase_duration'],
            double_support_time=data['double_support_time'],
            jerk_metric=data['jerk_metric'],
            acceleration_profile=np.array(data['acceleration_profile']),
            center_of_mass_oscillation=data['center_of_mass_oscillation'],
            hip_vertical_displacement=data['hip_vertical_displacement'],
            foot_clearance_height=data['foot_clearance_height'],
            foot_contact_velocity=data['foot_contact_velocity'],
            foot_sliding_threshold=data['foot_sliding_threshold'],
            arm_swing_amplitude=data['arm_swing_amplitude'],
            arm_swing_phase_offset=data['arm_swing_phase_offset'],
            spine_rotation_range=data['spine_rotation_range'],
            spine_lateral_sway=data['spine_lateral_sway'],
            kinetic_energy_pattern=np.array(data['kinetic_energy_pattern']),
            potential_energy_pattern=np.array(data['potential_energy_pattern'])
        )


def main():
    """Analyze reference BVH files and save their profiles"""
    analyzer = BVHReferenceAnalyzer()

    # Analyze example BVH files
    example_files = [
        Path('bvh_examples/walk-through-spce.bvh'),
        Path('bvh_examples/walking-standing-still.bvh')
    ]

    for bvh_file in example_files:
        if bvh_file.exists():
            print(f"\nAnalyzing {bvh_file.name}...")
            profile = analyzer.analyze_reference_motion(bvh_file)

            # Save profile
            output_path = bvh_file.parent / f"{bvh_file.stem}_profile.json"
            analyzer.save_profile(profile, output_path)
            print(f"Saved profile to {output_path}")

            # Print key metrics
            print(f"  Stride length: {profile.stride_length:.2f}")
            print(f"  Stride frequency: {profile.stride_frequency:.2f} Hz")
            print(f"  Jerk metric: {profile.jerk_metric:.2f}")
            print(f"  Hip vertical displacement: {profile.hip_vertical_displacement:.2f}")
            print(f"  Number of joints: {len(profile.joint_angle_ranges)}")
        else:
            print(f"File not found: {bvh_file}")


if __name__ == "__main__":
    main()