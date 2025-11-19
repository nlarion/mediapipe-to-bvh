"""
BVH Test Runner & Regression Testing System
Comprehensive test runner that orchestrates all validation tools and tracks regressions.
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
import subprocess
import sys
import hashlib
import cv2
from dataclasses import dataclass, asdict
import argparse
import yaml
import time

# Import our validation modules
from bvh_accuracy_tester import AccuracyTester
from bvh_overlay_visualizer import BVHOverlayVisualizer
from cross_model_validator import CrossModelValidator


@dataclass
class TestCase:
    """Definition of a test case"""
    name: str
    video_path: str
    expected_metrics: Dict[str, Any]
    thresholds: Dict[str, float]
    tags: List[str] = None
    ground_truth_path: Optional[str] = None
    description: Optional[str] = None


@dataclass
class TestResult:
    """Result from running a test case"""
    test_name: str
    timestamp: str
    passed: bool
    metrics: Dict[str, Any]
    failures: List[str]
    warnings: List[str]
    execution_time: float
    output_files: Dict[str, str]


class BVHTestRunner:
    """
    Orchestrates comprehensive testing of BVH conversion pipeline.
    """

    def __init__(self, config_path: Optional[str] = None, verbose: bool = True):
        """
        Initialize test runner.

        Args:
            config_path: Path to test configuration file
            verbose: Whether to print detailed output
        """
        self.verbose = verbose
        self.config = self._load_config(config_path)
        self.test_cases = []
        self.results_history = []

        # Initialize validators
        self.accuracy_tester = AccuracyTester(verbose=verbose)
        self.cross_validator = None  # Initialize on demand

        # Output directories
        self.output_dir = Path(self.config.get("output_dir", "test_output"))
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load test cases
        self._load_test_cases()

    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load test configuration."""
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                if config_path.endswith('.yaml') or config_path.endswith('.yml'):
                    return yaml.safe_load(f)
                else:
                    return json.load(f)

        # Default configuration
        return {
            "output_dir": "test_output",
            "enable_visual_validation": True,
            "enable_cross_model": False,
            "regression_threshold": 0.05,
            "test_cases_dir": "test_cases",
            "mediapipe_script": "mediapipe_to_bvh_complete.py",
            "save_artifacts": True,
            "parallel_tests": False
        }

    def _load_test_cases(self):
        """Load test case definitions."""
        # Load from test cases directory
        test_cases_dir = Path(self.config.get("test_cases_dir", "test_cases"))

        if test_cases_dir.exists():
            for test_file in test_cases_dir.glob("*.json"):
                with open(test_file, 'r') as f:
                    test_data = json.load(f)
                    test_case = TestCase(**test_data)
                    self.test_cases.append(test_case)

        # Add default test cases if none loaded
        if not self.test_cases:
            self._add_default_test_cases()

        if self.verbose:
            print(f"Loaded {len(self.test_cases)} test cases")

    def _add_default_test_cases(self):
        """Add default test cases for common scenarios."""
        default_cases = [
            TestCase(
                name="simple_t_pose",
                video_path="videos/t_pose.mp4",
                expected_metrics={
                    "mpjpe": 0.05,
                    "temporal_jitter": 0.01
                },
                thresholds={
                    "mpjpe": 0.1,
                    "temporal_jitter": 0.05
                },
                tags=["basic", "static"],
                description="Static T-pose validation"
            ),
            TestCase(
                name="walking_motion",
                video_path="videos/walking_00001.mp4",
                expected_metrics={
                    "mpjpe": 0.08,
                    "temporal_jitter": 0.03,
                    "limb_consistency": True
                },
                thresholds={
                    "mpjpe": 0.15,
                    "temporal_jitter": 0.1
                },
                tags=["motion", "walking"],
                description="Basic walking motion"
            ),
            TestCase(
                name="complex_dance",
                video_path="videos/dance_sequence.mp4",
                expected_metrics={
                    "mpjpe": 0.12,
                    "temporal_jitter": 0.05
                },
                thresholds={
                    "mpjpe": 0.2,
                    "temporal_jitter": 0.15
                },
                tags=["complex", "dance"],
                description="Complex dance sequence with fast movements"
            ),
            TestCase(
                name="rotation_check",
                video_path="videos/rotation_test.mp4",
                expected_metrics={
                    "global_orientation": 0,
                    "chest_rotation_error": 0
                },
                thresholds={
                    "global_orientation": 10,
                    "chest_rotation_error": 15
                },
                tags=["rotation", "orientation"],
                description="Check for 90-degree rotation errors"
            )
        ]

        # Only add test cases for videos that exist
        for case in default_cases:
            if Path(case.video_path).exists():
                self.test_cases.append(case)

    def run_mediapipe_conversion(self, video_path: str, output_path: str) -> bool:
        """
        Run MediaPipe to BVH conversion.

        Args:
            video_path: Path to input video
            output_path: Path for output BVH file

        Returns:
            True if conversion succeeded
        """
        script_path = self.config.get("mediapipe_script", "mediapipe_to_bvh_complete.py")

        if not Path(script_path).exists():
            print(f"Warning: MediaPipe script not found: {script_path}")
            return False

        cmd = [
            sys.executable,
            script_path,
            "--video", video_path,
            "--output", output_path
        ]

        try:
            if self.verbose:
                print(f"Running: {' '.join(cmd)}")

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )

            if result.returncode == 0:
                return Path(output_path).exists()
            else:
                print(f"Conversion failed: {result.stderr}")
                return False

        except subprocess.TimeoutExpired:
            print("Conversion timed out")
            return False
        except Exception as e:
            print(f"Error running conversion: {e}")
            return False

    def run_test_case(self, test_case: TestCase) -> TestResult:
        """
        Run a single test case.

        Args:
            test_case: Test case to run

        Returns:
            TestResult with metrics and pass/fail status
        """
        start_time = time.time()
        test_output_dir = self.output_dir / test_case.name
        test_output_dir.mkdir(parents=True, exist_ok=True)

        result = TestResult(
            test_name=test_case.name,
            timestamp=datetime.now().isoformat(),
            passed=True,
            metrics={},
            failures=[],
            warnings=[],
            execution_time=0,
            output_files={}
        )

        print(f"\nRunning test: {test_case.name}")
        print(f"  Description: {test_case.description}")

        # Check if video exists
        if not Path(test_case.video_path).exists():
            result.passed = False
            result.failures.append(f"Video file not found: {test_case.video_path}")
            return result

        # Run MediaPipe to BVH conversion
        bvh_output = test_output_dir / f"{test_case.name}.bvh"
        if self.verbose:
            print("  Converting video to BVH...")

        if not self.run_mediapipe_conversion(test_case.video_path, str(bvh_output)):
            result.passed = False
            result.failures.append("BVH conversion failed")
            return result

        result.output_files["bvh"] = str(bvh_output)

        # Run accuracy testing
        if self.verbose:
            print("  Running accuracy tests...")

        accuracy_results = self.accuracy_tester.run_full_validation(
            str(bvh_output),
            test_case.video_path,
            test_case.ground_truth_path
        )

        # Extract metrics
        if "metrics" in accuracy_results:
            result.metrics.update(accuracy_results["metrics"])

        # Check against thresholds
        for metric_name, threshold in test_case.thresholds.items():
            if metric_name in result.metrics:
                value = result.metrics[metric_name]

                # Handle different metric types
                if isinstance(value, bool):
                    if not value:
                        result.passed = False
                        result.failures.append(f"{metric_name} check failed")
                elif isinstance(value, (int, float)):
                    if value > threshold:
                        result.passed = False
                        result.failures.append(
                            f"{metric_name}: {value:.4f} exceeds threshold {threshold}"
                        )
                elif isinstance(value, dict) and "value" in value:
                    if value["value"] > threshold:
                        result.passed = False
                        result.failures.append(
                            f"{metric_name}: {value['value']:.4f} exceeds threshold {threshold}"
                        )

        # Visual validation
        if self.config.get("enable_visual_validation", True):
            if self.verbose:
                print("  Running visual validation...")

            overlay_output = test_output_dir / f"{test_case.name}_overlay.mp4"
            visualizer = BVHOverlayVisualizer(
                test_case.video_path,
                str(bvh_output),
                str(overlay_output)
            )

            vis_stats = visualizer.process_video(
                show_preview=False,
                save_output=self.config.get("save_artifacts", True)
            )

            if vis_stats.get("overall_mean_error", 0) > 50:  # pixels
                result.warnings.append(
                    f"High visual alignment error: {vis_stats['overall_mean_error']:.1f} pixels"
                )

            result.output_files["overlay_video"] = str(overlay_output)
            result.metrics["visual_alignment_error"] = vis_stats.get("overall_mean_error", -1)

        # Cross-model validation (optional)
        if self.config.get("enable_cross_model", False):
            if self.verbose:
                print("  Running cross-model validation...")

            if self.cross_validator is None:
                self.cross_validator = CrossModelValidator()

            cross_results = self.cross_validator.validate_video(
                test_case.video_path,
                output_dir=str(test_output_dir / "cross_validation"),
                sample_rate=30  # Sample every 30 frames
            )

            # Check for model disagreements
            if "comparisons" in cross_results:
                disagreements = [c.disagreement_score for c in cross_results["comparisons"]]
                avg_disagreement = np.mean(disagreements) if disagreements else 0

                result.metrics["cross_model_disagreement"] = avg_disagreement

                if avg_disagreement > 0.1:
                    result.warnings.append(
                        f"High cross-model disagreement: {avg_disagreement:.3f}"
                    )

        # Check for specific known issues
        self._check_known_issues(result, accuracy_results)

        # Calculate execution time
        result.execution_time = time.time() - start_time

        # Save test result
        result_file = test_output_dir / "test_result.json"
        with open(result_file, 'w') as f:
            json.dump(asdict(result), f, indent=2, default=str)

        # Generate report
        self._generate_test_report(test_case, result, test_output_dir)

        return result

    def _check_known_issues(self, result: TestResult, accuracy_results: Dict[str, Any]):
        """Check for known issues like the 90-degree rotation bug."""

        # Check for rotation errors
        if "warnings" in accuracy_results:
            for warning in accuracy_results["warnings"]:
                if "rotation" in warning.lower() or "orientation" in warning.lower():
                    result.warnings.append(f"Potential rotation issue: {warning}")

        # Check for temporal issues
        if "temporal_jitter" in result.metrics:
            jitter = result.metrics["temporal_jitter"]
            if isinstance(jitter, dict):
                max_jitter = jitter.get("max_jitter", 0)
                if max_jitter > 0.1:
                    result.warnings.append(f"High jitter detected: {max_jitter:.3f}")

    def _generate_test_report(self, test_case: TestCase, result: TestResult, output_dir: Path):
        """Generate detailed test report."""
        report = []
        report.append("=" * 60)
        report.append(f"TEST REPORT: {test_case.name}")
        report.append("=" * 60)
        report.append(f"Timestamp: {result.timestamp}")
        report.append(f"Status: {'PASSED' if result.passed else 'FAILED'}")
        report.append(f"Execution time: {result.execution_time:.2f}s")
        report.append(f"\nDescription: {test_case.description}")

        if test_case.tags:
            report.append(f"Tags: {', '.join(test_case.tags)}")

        report.append("\n" + "-" * 40)
        report.append("METRICS")
        report.append("-" * 40)

        for metric_name, value in result.metrics.items():
            expected = test_case.expected_metrics.get(metric_name, "N/A")
            threshold = test_case.thresholds.get(metric_name, "N/A")

            if isinstance(value, float):
                report.append(f"{metric_name}:")
                report.append(f"  Measured: {value:.4f}")
                report.append(f"  Expected: {expected}")
                report.append(f"  Threshold: {threshold}")
            else:
                report.append(f"{metric_name}: {value}")

        if result.failures:
            report.append("\n" + "-" * 40)
            report.append("FAILURES")
            report.append("-" * 40)
            for failure in result.failures:
                report.append(f"✗ {failure}")

        if result.warnings:
            report.append("\n" + "-" * 40)
            report.append("WARNINGS")
            report.append("-" * 40)
            for warning in result.warnings:
                report.append(f"⚠ {warning}")

        if result.output_files:
            report.append("\n" + "-" * 40)
            report.append("OUTPUT FILES")
            report.append("-" * 40)
            for file_type, file_path in result.output_files.items():
                report.append(f"{file_type}: {file_path}")

        report.append("\n" + "=" * 60)

        report_text = "\n".join(report)

        # Save report
        report_file = output_dir / "test_report.txt"
        with open(report_file, 'w') as f:
            f.write(report_text)

        if self.verbose:
            print(report_text)

    def run_all_tests(self, tags: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Run all test cases.

        Args:
            tags: Optional list of tags to filter tests

        Returns:
            Summary of all test results
        """
        print(f"\n{'=' * 60}")
        print("RUNNING BVH TEST SUITE")
        print(f"{'=' * 60}")

        # Filter test cases by tags if provided
        test_cases_to_run = self.test_cases
        if tags:
            test_cases_to_run = [
                tc for tc in self.test_cases
                if tc.tags and any(tag in tc.tags for tag in tags)
            ]

        print(f"Running {len(test_cases_to_run)} tests...")

        results = []
        passed_count = 0
        failed_count = 0

        for test_case in test_cases_to_run:
            result = self.run_test_case(test_case)
            results.append(result)

            if result.passed:
                passed_count += 1
                print(f"  ✓ {test_case.name} PASSED")
            else:
                failed_count += 1
                print(f"  ✗ {test_case.name} FAILED")

        # Generate summary
        summary = {
            "timestamp": datetime.now().isoformat(),
            "total_tests": len(test_cases_to_run),
            "passed": passed_count,
            "failed": failed_count,
            "pass_rate": (passed_count / len(test_cases_to_run) * 100) if test_cases_to_run else 0,
            "results": results
        }

        # Save summary
        summary_file = self.output_dir / "test_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)

        # Print summary
        print(f"\n{'=' * 60}")
        print("TEST SUITE SUMMARY")
        print(f"{'=' * 60}")
        print(f"Total tests: {summary['total_tests']}")
        print(f"Passed: {summary['passed']}")
        print(f"Failed: {summary['failed']}")
        print(f"Pass rate: {summary['pass_rate']:.1f}%")
        print(f"\nResults saved to: {self.output_dir}")

        return summary

    def run_regression_test(self, old_bvh: str, new_bvh: str,
                          video_path: str) -> Dict[str, Any]:
        """
        Compare two BVH files for regression.

        Args:
            old_bvh: Path to previous BVH file
            new_bvh: Path to new BVH file
            video_path: Path to source video

        Returns:
            Regression analysis results
        """
        print(f"\nRunning regression test...")
        print(f"  Old: {old_bvh}")
        print(f"  New: {new_bvh}")

        regression_results = {
            "old_bvh": old_bvh,
            "new_bvh": new_bvh,
            "video": video_path,
            "regressions": [],
            "improvements": [],
            "unchanged": []
        }

        # Run validation on both
        old_results = self.accuracy_tester.run_full_validation(old_bvh, video_path)
        new_results = self.accuracy_tester.run_full_validation(new_bvh, video_path)

        # Compare metrics
        threshold = self.config.get("regression_threshold", 0.05)

        if "metrics" in old_results and "metrics" in new_results:
            for metric_name in old_results["metrics"]:
                if metric_name in new_results["metrics"]:
                    old_value = old_results["metrics"][metric_name]
                    new_value = new_results["metrics"][metric_name]

                    # Handle different metric types
                    if isinstance(old_value, (int, float)) and isinstance(new_value, (int, float)):
                        diff = new_value - old_value
                        rel_diff = abs(diff / old_value) if old_value != 0 else abs(diff)

                        if rel_diff > threshold:
                            if diff > 0:
                                regression_results["regressions"].append({
                                    "metric": metric_name,
                                    "old": old_value,
                                    "new": new_value,
                                    "change": diff,
                                    "relative_change": rel_diff
                                })
                            else:
                                regression_results["improvements"].append({
                                    "metric": metric_name,
                                    "old": old_value,
                                    "new": new_value,
                                    "change": diff,
                                    "relative_change": rel_diff
                                })
                        else:
                            regression_results["unchanged"].append(metric_name)

        # Save results
        output_file = self.output_dir / "regression_results.json"
        with open(output_file, 'w') as f:
            json.dump(regression_results, f, indent=2)

        # Print summary
        print(f"\nRegression Test Results:")
        print(f"  Regressions: {len(regression_results['regressions'])}")
        print(f"  Improvements: {len(regression_results['improvements'])}")
        print(f"  Unchanged: {len(regression_results['unchanged'])}")

        if regression_results["regressions"]:
            print("\n  Regressions detected:")
            for reg in regression_results["regressions"]:
                print(f"    - {reg['metric']}: {reg['old']:.4f} -> {reg['new']:.4f} "
                     f"({reg['relative_change']*100:.1f}% worse)")

        if regression_results["improvements"]:
            print("\n  Improvements:")
            for imp in regression_results["improvements"]:
                print(f"    + {imp['metric']}: {imp['old']:.4f} -> {imp['new']:.4f} "
                     f"({imp['relative_change']*100:.1f}% better)")

        return regression_results

    def create_test_case(self, name: str, video_path: str,
                        auto_baseline: bool = True) -> TestCase:
        """
        Create a new test case from a video.

        Args:
            name: Test case name
            video_path: Path to video
            auto_baseline: Whether to automatically establish baseline metrics

        Returns:
            New TestCase object
        """
        print(f"Creating test case: {name}")

        # Run conversion and validation to establish baseline
        if auto_baseline:
            temp_bvh = self.output_dir / f"temp_{name}.bvh"
            if self.run_mediapipe_conversion(video_path, str(temp_bvh)):
                results = self.accuracy_tester.run_full_validation(
                    str(temp_bvh), video_path
                )

                # Extract baseline metrics
                baseline_metrics = results.get("metrics", {})

                # Set thresholds (20% tolerance)
                thresholds = {}
                for metric_name, value in baseline_metrics.items():
                    if isinstance(value, (int, float)):
                        thresholds[metric_name] = value * 1.2
                    elif isinstance(value, dict) and "value" in value:
                        thresholds[metric_name] = value["value"] * 1.2

                # Clean up temp file
                temp_bvh.unlink()
            else:
                baseline_metrics = {}
                thresholds = {}
        else:
            baseline_metrics = {}
            thresholds = {}

        # Create test case
        test_case = TestCase(
            name=name,
            video_path=video_path,
            expected_metrics=baseline_metrics,
            thresholds=thresholds,
            tags=[],
            description=f"Auto-generated test case for {Path(video_path).name}"
        )

        # Save test case
        test_case_file = Path(self.config.get("test_cases_dir", "test_cases")) / f"{name}.json"
        test_case_file.parent.mkdir(parents=True, exist_ok=True)

        with open(test_case_file, 'w') as f:
            json.dump(asdict(test_case), f, indent=2, default=str)

        print(f"Test case saved to: {test_case_file}")

        return test_case


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description='BVH Test Runner')

    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # Run tests command
    run_parser = subparsers.add_parser('run', help='Run tests')
    run_parser.add_argument('--config', help='Test configuration file')
    run_parser.add_argument('--tags', nargs='+', help='Filter tests by tags')
    run_parser.add_argument('--test', help='Run specific test by name')

    # Regression test command
    reg_parser = subparsers.add_parser('regression', help='Run regression test')
    reg_parser.add_argument('--old', required=True, help='Old BVH file')
    reg_parser.add_argument('--new', required=True, help='New BVH file')
    reg_parser.add_argument('--video', required=True, help='Source video')

    # Create test case command
    create_parser = subparsers.add_parser('create', help='Create test case')
    create_parser.add_argument('--name', required=True, help='Test case name')
    create_parser.add_argument('--video', required=True, help='Video file')
    create_parser.add_argument('--no-baseline', action='store_true',
                              help='Skip baseline metrics')

    args = parser.parse_args()

    # Initialize test runner
    runner = BVHTestRunner(config_path=args.config if hasattr(args, 'config') else None)

    if args.command == 'run':
        if args.test:
            # Run specific test
            test_case = next((tc for tc in runner.test_cases if tc.name == args.test), None)
            if test_case:
                result = runner.run_test_case(test_case)
                sys.exit(0 if result.passed else 1)
            else:
                print(f"Test case not found: {args.test}")
                sys.exit(1)
        else:
            # Run all tests
            summary = runner.run_all_tests(tags=args.tags)
            sys.exit(0 if summary["failed"] == 0 else 1)

    elif args.command == 'regression':
        results = runner.run_regression_test(args.old, args.new, args.video)
        sys.exit(0 if not results["regressions"] else 1)

    elif args.command == 'create':
        test_case = runner.create_test_case(
            args.name, args.video,
            auto_baseline=not args.no_baseline
        )
        sys.exit(0)

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()