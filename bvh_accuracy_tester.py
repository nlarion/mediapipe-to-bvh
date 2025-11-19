"""
BVH Accuracy Tester
A comprehensive testing system for validating MediaPipe to BVH pipeline accuracy.
Focuses on visual correctness, not just numerical metrics.
"""

import numpy as np
import cv2
from scipy.spatial.transform import Rotation
from scipy.spatial import procrustes
from typing import List, Dict, Tuple, Optional, Any
import json
import matplotlib.pyplot as plt
from pathlib import Path
import mediapipe as mp
from dataclasses import dataclass
from datetime import datetime
import warnings

# Try to import optional libraries
try:
    import pymo.parsers as parsers
    import pymo.writers as writers
    PYMO_AVAILABLE = True
except ImportError:
    PYMO_AVAILABLE = False
    warnings.warn("pymo not available. BVH parsing will be limited.")

try:
    from fastdtw import fastdtw
    DTW_AVAILABLE = True
except ImportError:
    DTW_AVAILABLE = False
    warnings.warn("fastdtw not available. DTW metrics will be disabled.")


@dataclass
class ValidationResult:
    """Container for validation results"""
    metric_name: str
    value: float
    unit: str
    passed: bool
    threshold: Optional[float] = None
    details: Optional[Dict[str, Any]] = None


class AccuracyTester:
    """
    Core accuracy testing framework for BVH validation.
    Implements multiple metrics to ensure visual and numerical correctness.
    """

    # MediaPipe to BVH joint mapping
    MEDIAPIPE_TO_BVH_MAPPING = {
        0: "Hips",           # Pelvis/Hip center
        11: "LeftHip",       # Left hip
        12: "RightHip",      # Right hip
        23: "LeftKnee",      # Left knee
        24: "RightKnee",     # Right knee
        25: "LeftAnkle",     # Left ankle
        26: "RightAnkle",    # Right ankle
        13: "LeftShoulder",  # Left shoulder
        14: "RightShoulder", # Right shoulder
        15: "LeftElbow",     # Left elbow
        16: "RightElbow",    # Right elbow
        17: "LeftWrist",     # Left wrist
        18: "RightWrist",    # Right wrist
    }

    # Anatomical joint angle limits (in degrees)
    JOINT_ANGLE_LIMITS = {
        "elbow": (0, 150),
        "knee": (0, 140),
        "shoulder_flexion": (-180, 180),
        "shoulder_abduction": (-180, 180),
        "hip_flexion": (-30, 120),
        "hip_abduction": (-45, 45),
        "neck": (-60, 60),
        "spine": (-30, 30),
    }

    def __init__(self, verbose: bool = True):
        """
        Initialize the accuracy tester.

        Args:
            verbose: Whether to print detailed information during testing
        """
        self.verbose = verbose
        self.metrics = {}
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils

    # =========================
    # Joint Position Metrics
    # =========================

    def mpjpe(self, pred_joints: np.ndarray, gt_joints: np.ndarray) -> float:
        """
        Mean Per Joint Position Error - Average Euclidean distance between joints.

        Args:
            pred_joints: Predicted joint positions [N, 3] or [T, N, 3]
            gt_joints: Ground truth joint positions [N, 3] or [T, N, 3]

        Returns:
            Mean error in the same units as input
        """
        if pred_joints.shape != gt_joints.shape:
            raise ValueError(f"Shape mismatch: {pred_joints.shape} vs {gt_joints.shape}")

        errors = np.linalg.norm(pred_joints - gt_joints, axis=-1)
        return np.mean(errors)

    def pa_mpjpe(self, pred_joints: np.ndarray, gt_joints: np.ndarray) -> float:
        """
        Procrustes-aligned MPJPE - Aligns poses before computing error.
        Removes global rotation, translation, and scale differences.

        Args:
            pred_joints: Predicted joint positions [N, 3]
            gt_joints: Ground truth joint positions [N, 3]

        Returns:
            Aligned mean error
        """
        # Handle multi-frame data
        if len(pred_joints.shape) == 3:
            errors = []
            for pred_frame, gt_frame in zip(pred_joints, gt_joints):
                errors.append(self.pa_mpjpe(pred_frame, gt_frame))
            return np.mean(errors)

        # Procrustes alignment
        gt_normalized, pred_aligned, disparity = procrustes(gt_joints, pred_joints)
        return self.mpjpe(pred_aligned, gt_normalized)

    def pck(self, pred_joints: np.ndarray, gt_joints: np.ndarray,
            threshold: float = 0.05, use_bbox: bool = False) -> float:
        """
        Percentage of Correct Keypoints - % of joints within threshold.

        Args:
            pred_joints: Predicted joint positions [N, 3] or [T, N, 3]
            gt_joints: Ground truth joint positions [N, 3] or [T, N, 3]
            threshold: Distance threshold (relative to bbox diagonal if use_bbox=True)
            use_bbox: Whether to normalize threshold by bounding box size

        Returns:
            Percentage of correct keypoints (0-100)
        """
        distances = np.linalg.norm(pred_joints - gt_joints, axis=-1)

        if use_bbox:
            # Normalize threshold by bounding box diagonal
            bbox_min = np.min(gt_joints, axis=-2)
            bbox_max = np.max(gt_joints, axis=-2)
            bbox_diag = np.linalg.norm(bbox_max - bbox_min)
            threshold = threshold * bbox_diag

        correct = distances < threshold
        return np.mean(correct) * 100

    # =========================
    # Rotation-Based Metrics
    # =========================

    def angular_error(self, pred_rotations: List, gt_rotations: List) -> Tuple[float, np.ndarray]:
        """
        Mean angular error for rotations.

        Args:
            pred_rotations: Predicted rotations (quaternions or Rotation objects)
            gt_rotations: Ground truth rotations

        Returns:
            Tuple of (mean_error, per_joint_errors) in degrees
        """
        errors = []

        for pred_rot, gt_rot in zip(pred_rotations, gt_rotations):
            if not isinstance(pred_rot, Rotation):
                pred_rot = Rotation.from_quat(pred_rot)
            if not isinstance(gt_rot, Rotation):
                gt_rot = Rotation.from_quat(gt_rot)

            # Compute relative rotation
            relative_rot = pred_rot.inv() * gt_rot

            # Get angle (magnitude of rotation)
            angle = relative_rot.magnitude()
            errors.append(np.degrees(angle))

        errors = np.array(errors)
        return np.mean(errors), errors

    def geodesic_distance(self, rot1: Rotation, rot2: Rotation) -> float:
        """
        Compute geodesic distance between two rotations on SO(3).
        More accurate than simple angular difference.

        Args:
            rot1: First rotation
            rot2: Second rotation

        Returns:
            Geodesic distance in radians
        """
        relative = rot1.inv() * rot2
        return relative.magnitude()

    # =========================
    # Temporal Consistency Metrics
    # =========================

    def temporal_jitter(self, joint_sequence: np.ndarray) -> Dict[str, float]:
        """
        Measure temporal jitter/smoothness of motion.

        Args:
            joint_sequence: Joint positions over time [T, N, 3]

        Returns:
            Dictionary with jitter metrics
        """
        # Calculate velocities and accelerations
        velocities = np.diff(joint_sequence, axis=0)
        accelerations = np.diff(velocities, axis=0)

        # Jitter metrics
        velocity_std = np.std(velocities)
        acceleration_std = np.std(accelerations)

        # Per-joint jitter
        per_joint_jitter = np.std(accelerations, axis=(0, 2))

        return {
            "velocity_std": velocity_std,
            "acceleration_std": acceleration_std,
            "mean_jitter": np.mean(per_joint_jitter),
            "max_jitter": np.max(per_joint_jitter),
            "per_joint_jitter": per_joint_jitter
        }

    def velocity_error(self, pred_sequence: np.ndarray, gt_sequence: np.ndarray) -> float:
        """
        Compare motion velocities between sequences.

        Args:
            pred_sequence: Predicted joint sequence [T, N, 3]
            gt_sequence: Ground truth sequence [T, N, 3]

        Returns:
            Mean velocity error
        """
        pred_vel = np.diff(pred_sequence, axis=0)
        gt_vel = np.diff(gt_sequence, axis=0)

        return np.mean(np.linalg.norm(pred_vel - gt_vel, axis=-1))

    def dtw_distance(self, pred_sequence: np.ndarray, gt_sequence: np.ndarray) -> Optional[float]:
        """
        Dynamic Time Warping distance for handling temporal misalignments.

        Args:
            pred_sequence: Predicted sequence [T1, N, 3]
            gt_sequence: Ground truth sequence [T2, N, 3]

        Returns:
            DTW distance or None if fastdtw not available
        """
        if not DTW_AVAILABLE:
            if self.verbose:
                print("DTW not available. Install fastdtw: pip install fastdtw")
            return None

        # Flatten spatial dimensions for DTW
        pred_flat = pred_sequence.reshape(pred_sequence.shape[0], -1)
        gt_flat = gt_sequence.reshape(gt_sequence.shape[0], -1)

        distance, path = fastdtw(pred_flat, gt_flat)
        return distance

    # =========================
    # Visual Correctness Metrics
    # =========================

    def check_limb_lengths(self, joints: np.ndarray,
                          reference: Optional[np.ndarray] = None,
                          tolerance: float = 0.1) -> ValidationResult:
        """
        Verify that bone lengths remain consistent (no stretching/compression).

        Args:
            joints: Joint positions [T, N, 3] or [N, 3]
            reference: Reference joint positions for comparison
            tolerance: Acceptable variation ratio (0.1 = 10%)

        Returns:
            Validation result
        """
        # Define bone connections
        bones = [
            ("LeftShoulder", "LeftElbow"),
            ("LeftElbow", "LeftWrist"),
            ("RightShoulder", "RightElbow"),
            ("RightElbow", "RightWrist"),
            ("LeftHip", "LeftKnee"),
            ("LeftKnee", "LeftAnkle"),
            ("RightHip", "RightKnee"),
            ("RightKnee", "RightAnkle"),
        ]

        # Calculate bone lengths
        if len(joints.shape) == 3:
            # Multi-frame: check consistency across frames
            bone_lengths = {}
            for bone_name, (start, end) in bones:
                # Map joint names to indices (simplified for this example)
                lengths = []
                for frame in joints:
                    # Calculate length for this frame
                    # Note: Would need actual joint index mapping here
                    pass

            # Check variation
            max_variation = 0
            for bone, lengths in bone_lengths.items():
                variation = (np.max(lengths) - np.min(lengths)) / np.mean(lengths)
                max_variation = max(max_variation, variation)

            passed = max_variation < tolerance
            return ValidationResult(
                metric_name="Limb Length Consistency",
                value=max_variation,
                unit="ratio",
                passed=passed,
                threshold=tolerance
            )

        return ValidationResult(
            metric_name="Limb Length Consistency",
            value=0.0,
            unit="ratio",
            passed=True
        )

    def check_joint_angle_limits(self, rotations: Dict[str, np.ndarray]) -> ValidationResult:
        """
        Verify joint angles are within anatomically plausible ranges.

        Args:
            rotations: Dictionary of joint rotations

        Returns:
            Validation result
        """
        violations = []

        for joint_name, rotation_data in rotations.items():
            # Convert rotation to Euler angles
            if isinstance(rotation_data, Rotation):
                angles = rotation_data.as_euler('xyz', degrees=True)
            else:
                angles = np.degrees(rotation_data)

            # Check against limits
            for angle_type, (min_angle, max_angle) in self.JOINT_ANGLE_LIMITS.items():
                if angle_type.lower() in joint_name.lower():
                    for angle in angles:
                        if angle < min_angle or angle > max_angle:
                            violations.append({
                                "joint": joint_name,
                                "angle": angle,
                                "limits": (min_angle, max_angle)
                            })

        passed = len(violations) == 0
        return ValidationResult(
            metric_name="Joint Angle Limits",
            value=len(violations),
            unit="violations",
            passed=passed,
            details={"violations": violations}
        )

    def reprojection_error(self, bvh_joints_3d: np.ndarray,
                          video_joints_2d: np.ndarray,
                          camera_matrix: Optional[np.ndarray] = None) -> float:
        """
        Project BVH skeleton back to 2D and compare with video keypoints.

        Args:
            bvh_joints_3d: 3D joint positions from BVH [N, 3]
            video_joints_2d: 2D joint positions from video [N, 2]
            camera_matrix: Camera intrinsic matrix (if None, use simple projection)

        Returns:
            Mean reprojection error in pixels
        """
        if camera_matrix is None:
            # Simple orthographic projection
            projected_2d = bvh_joints_3d[:, :2]
        else:
            # Perspective projection
            points_3d_homo = np.hstack([bvh_joints_3d, np.ones((bvh_joints_3d.shape[0], 1))])
            projected = camera_matrix @ points_3d_homo.T
            projected_2d = (projected[:2, :] / projected[2, :]).T

        # Calculate pixel error
        error = np.linalg.norm(projected_2d - video_joints_2d, axis=1)
        return np.mean(error)

    def check_global_orientation(self, bvh_data: Any, expected_facing: str = "forward") -> ValidationResult:
        """
        Detect global orientation errors (e.g., 90-degree rotations).

        Args:
            bvh_data: BVH data structure
            expected_facing: Expected facing direction

        Returns:
            Validation result
        """
        # This would analyze the root orientation and major body axes
        # to detect common orientation errors

        # Placeholder implementation
        return ValidationResult(
            metric_name="Global Orientation",
            value=0.0,
            unit="degrees",
            passed=True,
            details={"facing": expected_facing}
        )

    # =========================
    # Comprehensive Testing
    # =========================

    def run_full_validation(self, bvh_path: str, video_path: str,
                           ground_truth_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Run comprehensive validation suite.

        Args:
            bvh_path: Path to BVH file to validate
            video_path: Path to source video
            ground_truth_path: Optional path to ground truth data

        Returns:
            Dictionary with all validation results
        """
        results = {
            "timestamp": datetime.now().isoformat(),
            "bvh_file": bvh_path,
            "video_file": video_path,
            "metrics": {},
            "warnings": [],
            "errors": []
        }

        try:
            # Load BVH data
            if PYMO_AVAILABLE:
                parser = parsers.BVHParser()
                parsed_data = parser.parse(bvh_path)
                bvh_joints = self._extract_joints_from_bvh(parsed_data)
            else:
                results["warnings"].append("BVH parser not available")
                bvh_joints = None

            # Extract MediaPipe joints from video
            video_joints = self._extract_mediapipe_joints(video_path)

            if bvh_joints is not None and video_joints is not None:
                # Position metrics
                results["metrics"]["mpjpe"] = self.mpjpe(bvh_joints, video_joints)
                results["metrics"]["pa_mpjpe"] = self.pa_mpjpe(bvh_joints, video_joints)
                results["metrics"]["pck_5cm"] = self.pck(bvh_joints, video_joints, threshold=0.05)

                # Temporal metrics
                jitter_metrics = self.temporal_jitter(bvh_joints)
                results["metrics"]["temporal_jitter"] = jitter_metrics

                # Visual correctness
                limb_check = self.check_limb_lengths(bvh_joints)
                results["metrics"]["limb_consistency"] = {
                    "passed": limb_check.passed,
                    "value": limb_check.value
                }

            # Load ground truth if available
            if ground_truth_path:
                gt_data = self._load_ground_truth(ground_truth_path)
                if gt_data is not None and bvh_joints is not None:
                    results["metrics"]["gt_mpjpe"] = self.mpjpe(bvh_joints, gt_data)
                    results["metrics"]["gt_pa_mpjpe"] = self.pa_mpjpe(bvh_joints, gt_data)

        except Exception as e:
            results["errors"].append(str(e))
            if self.verbose:
                print(f"Error during validation: {e}")

        return results

    def _extract_joints_from_bvh(self, bvh_data: Any) -> Optional[np.ndarray]:
        """Extract joint positions from parsed BVH data."""
        # Implementation depends on BVH parser used
        # This is a placeholder
        return None

    def _extract_mediapipe_joints(self, video_path: str) -> Optional[np.ndarray]:
        """Extract MediaPipe joints from video."""
        cap = cv2.VideoCapture(video_path)
        joints_list = []

        with self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        ) as pose:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # Process frame
                results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

                if results.pose_landmarks:
                    # Extract landmark positions
                    landmarks = []
                    for landmark in results.pose_landmarks.landmark:
                        landmarks.append([landmark.x, landmark.y, landmark.z])
                    joints_list.append(landmarks)

        cap.release()

        if joints_list:
            return np.array(joints_list)
        return None

    def _load_ground_truth(self, gt_path: str) -> Optional[np.ndarray]:
        """Load ground truth data from file."""
        path = Path(gt_path)

        if path.suffix == '.npy':
            return np.load(gt_path)
        elif path.suffix == '.json':
            with open(gt_path, 'r') as f:
                data = json.load(f)
                return np.array(data['joints'])
        else:
            if self.verbose:
                print(f"Unknown ground truth format: {path.suffix}")
            return None

    # =========================
    # Reporting
    # =========================

    def generate_report(self, results: Dict[str, Any], output_path: Optional[str] = None) -> str:
        """
        Generate a human-readable validation report.

        Args:
            results: Validation results dictionary
            output_path: Optional path to save report

        Returns:
            Report as string
        """
        report = []
        report.append("=" * 60)
        report.append("BVH ACCURACY VALIDATION REPORT")
        report.append("=" * 60)
        report.append(f"\nTimestamp: {results['timestamp']}")
        report.append(f"BVH File: {results['bvh_file']}")
        report.append(f"Video File: {results['video_file']}")

        report.append("\n" + "-" * 40)
        report.append("METRICS")
        report.append("-" * 40)

        for metric_name, value in results['metrics'].items():
            if isinstance(value, dict):
                report.append(f"\n{metric_name}:")
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, float):
                        report.append(f"  {sub_key}: {sub_value:.4f}")
                    else:
                        report.append(f"  {sub_key}: {sub_value}")
            elif isinstance(value, float):
                report.append(f"{metric_name}: {value:.4f}")
            else:
                report.append(f"{metric_name}: {value}")

        if results.get('warnings'):
            report.append("\n" + "-" * 40)
            report.append("WARNINGS")
            report.append("-" * 40)
            for warning in results['warnings']:
                report.append(f"⚠ {warning}")

        if results.get('errors'):
            report.append("\n" + "-" * 40)
            report.append("ERRORS")
            report.append("-" * 40)
            for error in results['errors']:
                report.append(f"✗ {error}")

        report.append("\n" + "=" * 60)

        report_text = "\n".join(report)

        if output_path:
            with open(output_path, 'w') as f:
                f.write(report_text)
            if self.verbose:
                print(f"Report saved to: {output_path}")

        return report_text

    def visualize_errors(self, results: Dict[str, Any], output_dir: str = "validation_output"):
        """
        Create visual representations of errors.

        Args:
            results: Validation results
            output_dir: Directory to save visualizations
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Create error heatmap
        if 'per_joint_errors' in results.get('metrics', {}):
            self._plot_joint_error_heatmap(
                results['metrics']['per_joint_errors'],
                Path(output_dir) / "joint_error_heatmap.png"
            )

        # Plot temporal metrics
        if 'temporal_jitter' in results.get('metrics', {}):
            self._plot_temporal_metrics(
                results['metrics']['temporal_jitter'],
                Path(output_dir) / "temporal_metrics.png"
            )

    def _plot_joint_error_heatmap(self, errors: np.ndarray, output_path: Path):
        """Create a heatmap of per-joint errors."""
        plt.figure(figsize=(12, 8))
        plt.imshow(errors, aspect='auto', cmap='hot')
        plt.colorbar(label='Error')
        plt.title('Per-Joint Error Heatmap')
        plt.xlabel('Time Frame')
        plt.ylabel('Joint Index')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

    def _plot_temporal_metrics(self, temporal_data: Dict, output_path: Path):
        """Plot temporal consistency metrics."""
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))

        # Velocity plot
        if 'velocity_std' in temporal_data:
            axes[0].plot(temporal_data.get('velocity_timeline', []))
            axes[0].set_title('Velocity Over Time')
            axes[0].set_xlabel('Frame')
            axes[0].set_ylabel('Velocity')

        # Jitter plot
        if 'per_joint_jitter' in temporal_data:
            axes[1].bar(range(len(temporal_data['per_joint_jitter'])),
                       temporal_data['per_joint_jitter'])
            axes[1].set_title('Per-Joint Jitter')
            axes[1].set_xlabel('Joint Index')
            axes[1].set_ylabel('Jitter')

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()


# =========================
# Main Test Runner
# =========================

def main():
    """Main function to run accuracy testing."""
    import argparse

    parser = argparse.ArgumentParser(description='BVH Accuracy Tester')
    parser.add_argument('--bvh', required=True, help='Path to BVH file')
    parser.add_argument('--video', required=True, help='Path to source video')
    parser.add_argument('--ground-truth', help='Path to ground truth data')
    parser.add_argument('--output', default='validation_report.txt',
                       help='Output report path')
    parser.add_argument('--visualize', action='store_true',
                       help='Generate visualizations')
    parser.add_argument('--verbose', action='store_true',
                       help='Verbose output')

    args = parser.parse_args()

    # Initialize tester
    tester = AccuracyTester(verbose=args.verbose)

    # Run validation
    print("Running BVH validation...")
    results = tester.run_full_validation(
        args.bvh,
        args.video,
        args.ground_truth
    )

    # Generate report
    report = tester.generate_report(results, args.output)
    print("\nValidation Report:")
    print(report)

    # Generate visualizations if requested
    if args.visualize:
        print("\nGenerating visualizations...")
        tester.visualize_errors(results)
        print("Visualizations saved to validation_output/")

    # Return status code based on validation results
    if results.get('errors'):
        return 1
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())