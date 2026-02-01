#!/usr/bin/env python3
"""
Automated BVH Tuning System

This system:
1. Generates BVH from video with various parameter settings
2. Renders comparisons against reference BVH
3. Calculates error metrics (shoulder levelness, head alignment, etc.)
4. Iteratively tunes parameters to minimize errors

Can also use AI vision (Claude/GPT) to evaluate visual quality.
"""

import numpy as np
import cv2
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import subprocess

# For BVH analysis
from visual_accuracy_tester import BVHParser, CameraView, CAMERA_VIEWS
import plotly.graph_objects as go
import plotly.io as pio


@dataclass
class SkeletonMetrics:
    """Metrics for evaluating skeleton quality."""
    shoulder_levelness: float  # 0 = perfectly level, higher = more tilted
    head_tilt: float  # Degrees of head tilt from vertical
    head_forward_angle: float  # Head forward/backward lean
    spine_straightness: float  # How straight the spine is
    left_right_symmetry: float  # 0 = symmetric, higher = asymmetric
    overall_score: float  # Combined quality score (0-100, higher = better)


class SkeletonAnalyzer:
    """Analyze skeleton quality from BVH data."""

    def __init__(self):
        self.parser = None

    def load_bvh(self, bvh_path: str):
        """Load BVH file for analysis."""
        self.parser = BVHParser()
        self.parser.parse_file(bvh_path)

    def analyze_frame(self, frame_idx: int) -> Optional[SkeletonMetrics]:
        """Analyze a single frame and return quality metrics."""
        if self.parser is None:
            return None

        positions = self.parser.get_frame_positions(frame_idx)
        if not positions:
            return None

        # Find key joints (handle both mixamorig: prefix and without)
        def find_joint(names: List[str]) -> Optional[np.ndarray]:
            for name in names:
                if name in positions:
                    return positions[name]
                # Try with mixamorig: prefix
                prefixed = f"mixamorig:{name}"
                if prefixed in positions:
                    return positions[prefixed]
            return None

        # Get key positions
        left_shoulder = find_joint(['LeftShoulder', 'LeftArm'])
        right_shoulder = find_joint(['RightShoulder', 'RightArm'])
        head = find_joint(['Head'])
        neck = find_joint(['Neck'])
        hips = find_joint(['Hips'])
        spine = find_joint(['Spine', 'Spine1', 'Spine2'])

        # Calculate metrics
        shoulder_levelness = 0.0
        if left_shoulder is not None and right_shoulder is not None:
            # Height difference between shoulders (Y axis)
            shoulder_levelness = abs(left_shoulder[1] - right_shoulder[1])

        head_tilt = 0.0
        head_forward_angle = 0.0
        if head is not None and neck is not None:
            head_vec = head - neck
            # Tilt = deviation from vertical in XY plane
            head_tilt = np.degrees(np.arctan2(abs(head_vec[0]), head_vec[1]))
            # Forward lean = deviation in YZ plane
            head_forward_angle = np.degrees(np.arctan2(head_vec[2], head_vec[1]))

        spine_straightness = 0.0
        if hips is not None and spine is not None and neck is not None:
            # Check how much spine deviates from straight line
            hip_to_neck = neck - hips
            hip_to_spine = spine - hips
            # Project spine onto hip-neck line
            hip_to_neck_norm = hip_to_neck / (np.linalg.norm(hip_to_neck) + 1e-6)
            projection = np.dot(hip_to_spine, hip_to_neck_norm)
            projected_point = hips + projection * hip_to_neck_norm
            deviation = np.linalg.norm(spine - projected_point)
            spine_straightness = deviation

        # Symmetry check
        left_right_symmetry = 0.0
        if left_shoulder is not None and right_shoulder is not None:
            # Check X positions relative to center
            center_x = (left_shoulder[0] + right_shoulder[0]) / 2
            left_dist = abs(left_shoulder[0] - center_x)
            right_dist = abs(right_shoulder[0] - center_x)
            left_right_symmetry = abs(left_dist - right_dist)

        # Calculate overall score (0-100, higher = better)
        # Penalize for each issue
        score = 100.0
        score -= min(50, shoulder_levelness * 10)  # Max 50 point penalty for shoulder tilt
        score -= min(20, head_tilt * 0.5)  # Max 20 point penalty for head tilt
        score -= min(15, abs(head_forward_angle) * 0.3)  # Max 15 for forward lean
        score -= min(10, spine_straightness * 2)  # Max 10 for spine deviation
        score -= min(5, left_right_symmetry * 5)  # Max 5 for asymmetry
        score = max(0, score)

        return SkeletonMetrics(
            shoulder_levelness=float(shoulder_levelness),
            head_tilt=float(head_tilt),
            head_forward_angle=float(head_forward_angle),
            spine_straightness=float(spine_straightness),
            left_right_symmetry=float(left_right_symmetry),
            overall_score=float(score)
        )

    def analyze_all_frames(self) -> Dict[str, float]:
        """Analyze all frames and return average metrics."""
        if self.parser is None:
            return {}

        all_metrics = []
        for i in range(self.parser.frames):
            metrics = self.analyze_frame(i)
            if metrics:
                all_metrics.append(metrics)

        if not all_metrics:
            return {}

        # Average all metrics
        avg = {
            'shoulder_levelness': np.mean([m.shoulder_levelness for m in all_metrics]),
            'head_tilt': np.mean([m.head_tilt for m in all_metrics]),
            'head_forward_angle': np.mean([m.head_forward_angle for m in all_metrics]),
            'spine_straightness': np.mean([m.spine_straightness for m in all_metrics]),
            'left_right_symmetry': np.mean([m.left_right_symmetry for m in all_metrics]),
            'overall_score': np.mean([m.overall_score for m in all_metrics]),
            'min_score': min([m.overall_score for m in all_metrics]),
            'max_score': max([m.overall_score for m in all_metrics]),
        }
        return avg


class BVHComparator:
    """Compare two BVH files and generate visual/numerical comparisons."""

    def __init__(self, output_dir: str = "tuning_output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def compare_bvh_files(self, reference_bvh: str, test_bvh: str,
                          frames: List[int] = None) -> Dict:
        """
        Compare reference BVH against test BVH.

        Returns dict with:
        - metrics: Numerical comparison metrics
        - images: Paths to comparison images
        """
        ref_analyzer = SkeletonAnalyzer()
        ref_analyzer.load_bvh(reference_bvh)

        test_analyzer = SkeletonAnalyzer()
        test_analyzer.load_bvh(test_bvh)

        if frames is None:
            frames = list(range(0, min(ref_analyzer.parser.frames,
                                       test_analyzer.parser.frames), 10))

        results = {
            'reference_metrics': ref_analyzer.analyze_all_frames(),
            'test_metrics': test_analyzer.analyze_all_frames(),
            'frame_comparisons': [],
            'images': []
        }

        # Compare specific frames
        for frame_idx in frames:
            ref_metrics = ref_analyzer.analyze_frame(frame_idx)
            test_metrics = test_analyzer.analyze_frame(frame_idx)

            if ref_metrics and test_metrics:
                comparison = {
                    'frame': frame_idx,
                    'ref_score': ref_metrics.overall_score,
                    'test_score': test_metrics.overall_score,
                    'score_diff': ref_metrics.overall_score - test_metrics.overall_score,
                    'shoulder_diff': test_metrics.shoulder_levelness - ref_metrics.shoulder_levelness,
                    'head_tilt_diff': test_metrics.head_tilt - ref_metrics.head_tilt,
                }
                results['frame_comparisons'].append(comparison)

        return results

    def generate_comparison_report(self, reference_bvh: str, test_bvh: str,
                                   output_name: str = "comparison_report") -> str:
        """Generate a detailed comparison report."""
        results = self.compare_bvh_files(reference_bvh, test_bvh)

        report_path = self.output_dir / f"{output_name}.json"
        with open(report_path, 'w') as f:
            json.dump(results, f, indent=2)

        # Print summary
        print(f"\n{'='*60}")
        print("BVH COMPARISON REPORT")
        print(f"{'='*60}")
        print(f"Reference: {reference_bvh}")
        print(f"Test: {test_bvh}")
        print(f"\nReference avg score: {results['reference_metrics'].get('overall_score', 0):.1f}")
        print(f"Test avg score: {results['test_metrics'].get('overall_score', 0):.1f}")

        if results['frame_comparisons']:
            avg_diff = np.mean([c['score_diff'] for c in results['frame_comparisons']])
            print(f"Average score difference: {avg_diff:.1f} (positive = ref better)")

        print(f"\nDetailed metrics:")
        print(f"  Shoulder levelness - Ref: {results['reference_metrics'].get('shoulder_levelness', 0):.3f}, "
              f"Test: {results['test_metrics'].get('shoulder_levelness', 0):.3f}")
        print(f"  Head tilt - Ref: {results['reference_metrics'].get('head_tilt', 0):.1f}°, "
              f"Test: {results['test_metrics'].get('head_tilt', 0):.1f}°")

        print(f"\nReport saved to: {report_path}")
        return str(report_path)


class AutomatedTuner:
    """
    Automated parameter tuning for BVH converter.

    Workflow:
    1. Run converter with current parameters
    2. Analyze output quality
    3. Adjust parameters based on errors
    4. Repeat until quality threshold met
    """

    def __init__(self, video_path: str, reference_bvh: str = None,
                 output_dir: str = "tuning_output"):
        self.video_path = video_path
        self.reference_bvh = reference_bvh
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.comparator = BVHComparator(str(self.output_dir))
        self.iteration = 0
        self.history = []

    def run_converter(self, output_bvh: str, extra_args: List[str] = None) -> bool:
        """Run the BVH converter with given arguments."""
        cmd = [
            "python", "bvh_converter.py",
            "--video", self.video_path,
            "--output", output_bvh,
            "--sample-rate", "2"
        ]

        if extra_args:
            cmd.extend(extra_args)

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            return result.returncode == 0
        except Exception as e:
            print(f"Error running converter: {e}")
            return False

    def evaluate_output(self, bvh_path: str) -> Dict:
        """Evaluate the quality of a BVH output."""
        analyzer = SkeletonAnalyzer()
        analyzer.load_bvh(bvh_path)

        metrics = analyzer.analyze_all_frames()

        # If we have a reference, compare against it
        if self.reference_bvh:
            comparison = self.comparator.compare_bvh_files(
                self.reference_bvh, bvh_path
            )
            metrics['comparison'] = comparison

        return metrics

    def run_tuning_iteration(self, params: Dict = None) -> Dict:
        """Run one iteration of the tuning process."""
        self.iteration += 1
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        output_bvh = str(self.output_dir / f"iteration_{self.iteration}_{timestamp}.bvh")

        print(f"\n{'='*60}")
        print(f"TUNING ITERATION {self.iteration}")
        print(f"{'='*60}")

        # Build extra args from params
        extra_args = []
        if params:
            if params.get('enable_ik'):
                extra_args.append('--ik')
            if params.get('enable_face'):
                extra_args.append('--face')

        # Run converter
        print(f"Running converter...")
        success = self.run_converter(output_bvh, extra_args)

        if not success:
            print("Converter failed!")
            return {'success': False, 'iteration': self.iteration}

        # Evaluate output
        print(f"Evaluating output...")
        metrics = self.evaluate_output(output_bvh)

        result = {
            'success': True,
            'iteration': self.iteration,
            'output_bvh': output_bvh,
            'params': params or {},
            'metrics': metrics,
            'timestamp': timestamp
        }

        self.history.append(result)

        # Print summary
        print(f"\nResults:")
        print(f"  Overall score: {metrics.get('overall_score', 0):.1f}")
        print(f"  Shoulder levelness: {metrics.get('shoulder_levelness', 0):.3f}")
        print(f"  Head tilt: {metrics.get('head_tilt', 0):.1f}°")

        return result

    def save_history(self):
        """Save tuning history to file."""
        history_path = self.output_dir / "tuning_history.json"
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2, default=str)
        print(f"History saved to: {history_path}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Automated BVH tuning system")
    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze a BVH file')
    analyze_parser.add_argument('--bvh', required=True, help='BVH file to analyze')

    # Compare command
    compare_parser = subparsers.add_parser('compare', help='Compare two BVH files')
    compare_parser.add_argument('--reference', required=True, help='Reference BVH')
    compare_parser.add_argument('--test', required=True, help='Test BVH')
    compare_parser.add_argument('--output-dir', default='tuning_output', help='Output directory')

    # Tune command
    tune_parser = subparsers.add_parser('tune', help='Run automated tuning')
    tune_parser.add_argument('--video', required=True, help='Source video')
    tune_parser.add_argument('--reference', help='Reference BVH for comparison')
    tune_parser.add_argument('--iterations', type=int, default=1, help='Number of iterations')
    tune_parser.add_argument('--output-dir', default='tuning_output', help='Output directory')

    args = parser.parse_args()

    if args.command == 'analyze':
        analyzer = SkeletonAnalyzer()
        analyzer.load_bvh(args.bvh)
        metrics = analyzer.analyze_all_frames()

        print(f"\nBVH Analysis: {args.bvh}")
        print(f"{'='*40}")
        for key, value in metrics.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.3f}")

    elif args.command == 'compare':
        comparator = BVHComparator(args.output_dir)
        comparator.generate_comparison_report(args.reference, args.test)

    elif args.command == 'tune':
        tuner = AutomatedTuner(args.video, args.reference, args.output_dir)

        for i in range(args.iterations):
            tuner.run_tuning_iteration()

        tuner.save_history()

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
