"""
Cross-Model Validator
Compare BVH output against multiple pose estimation models to identify systematic biases.
Supports MediaPipe, OpenPose, MMPose, and other models for comprehensive validation.
"""

import numpy as np
import cv2
import json
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import mediapipe as mp
from datetime import datetime
import subprocess
import sys
import warnings

# Model availability flags
MODELS_AVAILABLE = {
    "mediapipe": True,
    "openpose": False,
    "mmpose": False,
    "movenet": False,
    "blazepose": False
}

# Try to import optional model libraries
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn("PyTorch not available. Some models will be disabled.")

try:
    # Check if OpenPose is installed
    import sys
    openpose_path = "/usr/local/python"  # Typical OpenPose Python path
    if Path(openpose_path).exists():
        sys.path.append(openpose_path)
        from openpose import pyopenpose as op
        MODELS_AVAILABLE["openpose"] = True
except ImportError:
    pass

try:
    from mmpose.apis import (inference_top_down_pose_model, init_pose_model,
                             vis_pose_result, process_mmdet_results)
    from mmdet.apis import inference_detector, init_detector
    MODELS_AVAILABLE["mmpose"] = True
except ImportError:
    pass


@dataclass
class ModelResult:
    """Container for pose estimation results from a model"""
    model_name: str
    frame_number: int
    keypoints: np.ndarray  # [N, 3] or [N, 2] array of keypoints
    confidence: np.ndarray  # Per-keypoint confidence scores
    bbox: Optional[np.ndarray] = None  # Bounding box if available
    processing_time: float = 0.0  # Time taken for inference


@dataclass
class ComparisonResult:
    """Results from comparing multiple models"""
    frame_number: int
    model_results: Dict[str, ModelResult]
    consensus_keypoints: np.ndarray  # Weighted average of all models
    disagreement_score: float  # How much models disagree
    outlier_models: List[str]  # Models that significantly differ from consensus


class CrossModelValidator:
    """
    Validate pose estimation across multiple models to ensure accuracy.
    """

    # Standard COCO keypoint format (17 keypoints)
    COCO_KEYPOINTS = [
        "nose", "left_eye", "right_eye", "left_ear", "right_ear",
        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
        "left_wrist", "right_wrist", "left_hip", "right_hip",
        "left_knee", "right_knee", "left_ankle", "right_ankle"
    ]

    # MediaPipe to COCO mapping
    MP_TO_COCO = {
        0: 0,   # nose
        2: 1,   # left_eye
        5: 2,   # right_eye
        7: 3,   # left_ear
        8: 4,   # right_ear
        11: 5,  # left_shoulder
        12: 6,  # right_shoulder
        13: 7,  # left_elbow
        14: 8,  # right_elbow
        15: 9,  # left_wrist
        16: 10, # right_wrist
        23: 11, # left_hip
        24: 12, # right_hip
        25: 13, # left_knee
        26: 14, # right_knee
        27: 15, # left_ankle
        28: 16  # right_ankle
    }

    def __init__(self, models_to_use: Optional[List[str]] = None, verbose: bool = True):
        """
        Initialize cross-model validator.

        Args:
            models_to_use: List of model names to use (defaults to all available)
            verbose: Whether to print detailed information
        """
        self.verbose = verbose

        # Determine which models to use
        if models_to_use is None:
            self.models_to_use = [name for name, available in MODELS_AVAILABLE.items() if available]
        else:
            self.models_to_use = [name for name in models_to_use if MODELS_AVAILABLE.get(name, False)]

        if self.verbose:
            print(f"Initializing Cross-Model Validator with models: {self.models_to_use}")

        # Initialize models
        self.models = {}
        self._initialize_models()

    def _initialize_models(self):
        """Initialize all selected pose estimation models."""

        # MediaPipe
        if "mediapipe" in self.models_to_use:
            self.models["mediapipe"] = {
                "pose": mp.solutions.pose.Pose(
                    static_image_mode=False,
                    model_complexity=2,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5
                )
            }
            if self.verbose:
                print("✓ MediaPipe initialized")

        # OpenPose
        if "openpose" in self.models_to_use:
            try:
                params = {
                    "model_folder": "/usr/local/openpose/models/",
                    "model_pose": "BODY_25",
                    "net_resolution": "-1x368"
                }
                self.models["openpose"] = {
                    "wrapper": op.WrapperPython(),
                    "params": params
                }
                self.models["openpose"]["wrapper"].configure(params)
                self.models["openpose"]["wrapper"].start()
                if self.verbose:
                    print("✓ OpenPose initialized")
            except Exception as e:
                print(f"Failed to initialize OpenPose: {e}")
                self.models_to_use.remove("openpose")

        # MMPose
        if "mmpose" in self.models_to_use:
            try:
                # Initialize detector and pose model
                det_config = 'mmpose/configs/detection/faster_rcnn_r50_fpn_coco.py'
                det_checkpoint = 'checkpoints/faster_rcnn_r50_fpn_1x_coco.pth'
                pose_config = 'mmpose/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/hrnet_w48_coco_256x192.py'
                pose_checkpoint = 'checkpoints/hrnet_w48_coco_256x192.pth'

                self.models["mmpose"] = {
                    "detector": init_detector(det_config, det_checkpoint, device='cuda:0'),
                    "pose_model": init_pose_model(pose_config, pose_checkpoint, device='cuda:0')
                }
                if self.verbose:
                    print("✓ MMPose initialized")
            except Exception as e:
                print(f"Failed to initialize MMPose: {e}")
                self.models_to_use.remove("mmpose")

    def process_frame_mediapipe(self, frame: np.ndarray) -> Optional[ModelResult]:
        """
        Process frame with MediaPipe.

        Args:
            frame: Input image (BGR)

        Returns:
            ModelResult or None if detection fails
        """
        import time
        start_time = time.time()

        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process
        results = self.models["mediapipe"]["pose"].process(rgb_frame)

        if results.pose_landmarks:
            # Extract keypoints
            keypoints = []
            confidences = []

            for i in range(33):  # MediaPipe has 33 landmarks
                if i in self.MP_TO_COCO:
                    landmark = results.pose_landmarks.landmark[i]
                    keypoints.append([landmark.x, landmark.y, landmark.z])
                    confidences.append(landmark.visibility)

            keypoints = np.array(keypoints)
            confidences = np.array(confidences)

            processing_time = time.time() - start_time

            return ModelResult(
                model_name="mediapipe",
                frame_number=0,
                keypoints=keypoints,
                confidence=confidences,
                processing_time=processing_time
            )

        return None

    def process_frame_openpose(self, frame: np.ndarray) -> Optional[ModelResult]:
        """
        Process frame with OpenPose.

        Args:
            frame: Input image (BGR)

        Returns:
            ModelResult or None if detection fails
        """
        if "openpose" not in self.models:
            return None

        import time
        start_time = time.time()

        # Create datum
        datum = op.Datum()
        datum.cvInputData = frame

        # Process
        self.models["openpose"]["wrapper"].emplaceAndPop([datum])

        if datum.poseKeypoints is not None and len(datum.poseKeypoints) > 0:
            # Get first person's keypoints
            keypoints = datum.poseKeypoints[0]  # Shape: [25, 3] for BODY_25

            # Convert BODY_25 to COCO format (subset of keypoints)
            coco_keypoints = []
            confidences = []

            # Mapping from BODY_25 to COCO (simplified)
            body25_to_coco = {0: 0, 1: 16, 2: 5, 3: 7, 4: 9, 5: 6, 6: 8, 7: 10,
                            8: 12, 9: 13, 10: 15, 11: 14, 12: 11, 13: 13, 14: 15}

            for coco_idx in range(17):
                if coco_idx in body25_to_coco.values():
                    # Find corresponding BODY_25 index
                    for b25_idx, c_idx in body25_to_coco.items():
                        if c_idx == coco_idx:
                            coco_keypoints.append(keypoints[b25_idx][:2])
                            confidences.append(keypoints[b25_idx][2])
                            break

            processing_time = time.time() - start_time

            return ModelResult(
                model_name="openpose",
                frame_number=0,
                keypoints=np.array(coco_keypoints),
                confidence=np.array(confidences),
                processing_time=processing_time
            )

        return None

    def process_frame_mmpose(self, frame: np.ndarray) -> Optional[ModelResult]:
        """
        Process frame with MMPose.

        Args:
            frame: Input image (BGR)

        Returns:
            ModelResult or None if detection fails
        """
        if "mmpose" not in self.models:
            return None

        import time
        start_time = time.time()

        # Detect person
        mmdet_results = inference_detector(self.models["mmpose"]["detector"], frame)
        person_results = process_mmdet_results(mmdet_results, cat_id=1)

        if len(person_results) > 0:
            # Pose estimation
            pose_results, _ = inference_top_down_pose_model(
                self.models["mmpose"]["pose_model"],
                frame,
                person_results,
                bbox_thr=0.3,
                format='xyxy'
            )

            if len(pose_results) > 0:
                # Get first person's keypoints
                keypoints = pose_results[0]['keypoints'][:, :2]
                confidences = pose_results[0]['keypoints'][:, 2]

                processing_time = time.time() - start_time

                return ModelResult(
                    model_name="mmpose",
                    frame_number=0,
                    keypoints=keypoints,
                    confidence=confidences,
                    bbox=pose_results[0].get('bbox'),
                    processing_time=processing_time
                )

        return None

    def process_frame_all_models(self, frame: np.ndarray,
                                frame_number: int = 0) -> Dict[str, ModelResult]:
        """
        Process frame with all available models.

        Args:
            frame: Input image (BGR)
            frame_number: Frame index

        Returns:
            Dictionary mapping model names to results
        """
        results = {}

        for model_name in self.models_to_use:
            if model_name == "mediapipe":
                result = self.process_frame_mediapipe(frame)
            elif model_name == "openpose":
                result = self.process_frame_openpose(frame)
            elif model_name == "mmpose":
                result = self.process_frame_mmpose(frame)
            else:
                continue

            if result:
                result.frame_number = frame_number
                results[model_name] = result

        return results

    def calculate_consensus(self, model_results: Dict[str, ModelResult]) -> Tuple[np.ndarray, float]:
        """
        Calculate consensus keypoints from multiple models.

        Args:
            model_results: Dictionary of model results

        Returns:
            Tuple of (consensus_keypoints, disagreement_score)
        """
        if not model_results:
            return np.array([]), 0.0

        # Collect all keypoints
        all_keypoints = []
        all_weights = []

        for model_name, result in model_results.items():
            # Normalize keypoints to same shape
            keypoints = result.keypoints
            if keypoints.shape[1] == 3:
                keypoints = keypoints[:, :2]  # Use only x, y for consensus

            all_keypoints.append(keypoints)
            # Use confidence as weight
            all_weights.append(result.confidence)

        # Calculate weighted average
        all_keypoints = np.array(all_keypoints)
        all_weights = np.array(all_weights)

        # Normalize weights
        weight_sum = np.sum(all_weights, axis=0)
        weight_sum[weight_sum == 0] = 1.0

        # Weighted consensus
        consensus = np.zeros_like(all_keypoints[0])
        for kpts, weights in zip(all_keypoints, all_weights):
            normalized_weights = weights / weight_sum
            consensus += kpts * normalized_weights[:, np.newaxis]

        # Calculate disagreement score (average standard deviation)
        std_devs = np.std(all_keypoints, axis=0)
        disagreement_score = np.mean(std_devs)

        return consensus, disagreement_score

    def identify_outliers(self, model_results: Dict[str, ModelResult],
                        consensus: np.ndarray,
                        threshold: float = 0.05) -> List[str]:
        """
        Identify models that significantly differ from consensus.

        Args:
            model_results: Dictionary of model results
            consensus: Consensus keypoints
            threshold: Threshold for outlier detection

        Returns:
            List of outlier model names
        """
        outliers = []

        for model_name, result in model_results.items():
            keypoints = result.keypoints
            if keypoints.shape[1] == 3:
                keypoints = keypoints[:, :2]

            # Calculate distance from consensus
            distances = np.linalg.norm(keypoints - consensus, axis=1)
            mean_distance = np.mean(distances)

            if mean_distance > threshold:
                outliers.append(model_name)

        return outliers

    def compare_models(self, frame: np.ndarray,
                      frame_number: int = 0) -> ComparisonResult:
        """
        Compare all models on a single frame.

        Args:
            frame: Input image
            frame_number: Frame index

        Returns:
            ComparisonResult with all model outputs and analysis
        """
        # Get results from all models
        model_results = self.process_frame_all_models(frame, frame_number)

        # Calculate consensus
        consensus, disagreement = self.calculate_consensus(model_results)

        # Identify outliers
        outliers = self.identify_outliers(model_results, consensus)

        return ComparisonResult(
            frame_number=frame_number,
            model_results=model_results,
            consensus_keypoints=consensus,
            disagreement_score=disagreement,
            outlier_models=outliers
        )

    def validate_video(self, video_path: str,
                      output_dir: str = "cross_validation_output",
                      sample_rate: int = 1) -> Dict[str, Any]:
        """
        Validate entire video across multiple models.

        Args:
            video_path: Path to input video
            output_dir: Directory for output files
            sample_rate: Process every Nth frame

        Returns:
            Validation statistics
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        results = {
            "video_path": video_path,
            "models_used": self.models_to_use,
            "total_frames": total_frames,
            "processed_frames": 0,
            "comparisons": [],
            "model_statistics": {model: {
                "detected_frames": 0,
                "total_confidence": 0,
                "total_time": 0
            } for model in self.models_to_use}
        }

        frame_idx = 0
        processed = 0

        print(f"Processing {total_frames} frames with {len(self.models_to_use)} models...")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % sample_rate == 0:
                # Process frame
                comparison = self.compare_models(frame, frame_idx)
                results["comparisons"].append(comparison)

                # Update statistics
                for model_name, model_result in comparison.model_results.items():
                    stats = results["model_statistics"][model_name]
                    stats["detected_frames"] += 1
                    stats["total_confidence"] += np.mean(model_result.confidence)
                    stats["total_time"] += model_result.processing_time

                processed += 1
                if processed % 10 == 0:
                    print(f"Processed {processed} frames...")

            frame_idx += 1

        cap.release()

        results["processed_frames"] = processed

        # Calculate final statistics
        for model_name, stats in results["model_statistics"].items():
            if stats["detected_frames"] > 0:
                stats["avg_confidence"] = stats["total_confidence"] / stats["detected_frames"]
                stats["avg_time"] = stats["total_time"] / stats["detected_frames"]
                stats["detection_rate"] = stats["detected_frames"] / processed

        # Save results
        self._save_validation_results(results, output_dir)

        return results

    def _save_validation_results(self, results: Dict[str, Any], output_dir: str):
        """Save validation results to files."""
        output_path = Path(output_dir)

        # Save raw results
        with open(output_path / "cross_validation_results.json", 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            serializable_results = self._make_serializable(results)
            json.dump(serializable_results, f, indent=2)

        # Generate report
        report = self._generate_validation_report(results)
        with open(output_path / "validation_report.txt", 'w') as f:
            f.write(report)

        if self.verbose:
            print(f"\nResults saved to {output_dir}")
            print(report)

    def _make_serializable(self, obj: Any) -> Any:
        """Convert numpy arrays and dataclasses to serializable format."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (ModelResult, ComparisonResult)):
            return asdict(obj)
        elif isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        return obj

    def _generate_validation_report(self, results: Dict[str, Any]) -> str:
        """Generate human-readable validation report."""
        report = []
        report.append("=" * 60)
        report.append("CROSS-MODEL VALIDATION REPORT")
        report.append("=" * 60)
        report.append(f"\nVideo: {results['video_path']}")
        report.append(f"Models used: {', '.join(results['models_used'])}")
        report.append(f"Frames processed: {results['processed_frames']} / {results['total_frames']}")

        report.append("\n" + "-" * 40)
        report.append("MODEL PERFORMANCE")
        report.append("-" * 40)

        for model_name, stats in results["model_statistics"].items():
            report.append(f"\n{model_name.upper()}:")
            report.append(f"  Detection rate: {stats.get('detection_rate', 0):.1%}")
            report.append(f"  Average confidence: {stats.get('avg_confidence', 0):.3f}")
            report.append(f"  Average processing time: {stats.get('avg_time', 0):.3f}s")

        # Analyze disagreements
        if results["comparisons"]:
            disagreements = [c.disagreement_score for c in results["comparisons"]]
            report.append("\n" + "-" * 40)
            report.append("CONSENSUS ANALYSIS")
            report.append("-" * 40)
            report.append(f"Average disagreement: {np.mean(disagreements):.4f}")
            report.append(f"Max disagreement: {np.max(disagreements):.4f}")

            # Count outliers
            outlier_counts = {}
            for comparison in results["comparisons"]:
                for outlier in comparison.outlier_models:
                    outlier_counts[outlier] = outlier_counts.get(outlier, 0) + 1

            if outlier_counts:
                report.append("\nOutlier frequency:")
                for model, count in sorted(outlier_counts.items(), key=lambda x: x[1], reverse=True):
                    percentage = (count / len(results["comparisons"])) * 100
                    report.append(f"  {model}: {count} frames ({percentage:.1f}%)")

        report.append("\n" + "=" * 60)
        return "\n".join(report)

    def visualize_comparison(self, frame: np.ndarray,
                            comparison: ComparisonResult,
                            output_path: Optional[str] = None) -> np.ndarray:
        """
        Visualize model comparison on a frame.

        Args:
            frame: Original frame
            comparison: Comparison results
            output_path: Optional path to save visualization

        Returns:
            Visualized frame
        """
        h, w = frame.shape[:2]
        num_models = len(comparison.model_results) + 1  # +1 for consensus

        # Create grid layout
        grid_width = min(3, num_models)
        grid_height = (num_models + grid_width - 1) // grid_width

        cell_width = w // grid_width
        cell_height = h // grid_height

        # Create output image
        output = np.zeros((h * grid_height, w * grid_width, 3), dtype=np.uint8)

        # Draw each model's results
        for idx, (model_name, result) in enumerate(comparison.model_results.items()):
            row = idx // grid_width
            col = idx % grid_width

            # Copy frame to grid cell
            cell = frame.copy()

            # Draw skeleton
            self._draw_skeleton(cell, result.keypoints, color=(0, 255, 0))

            # Add label
            cv2.putText(cell, model_name.upper(), (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            # Add confidence
            avg_conf = np.mean(result.confidence)
            cv2.putText(cell, f"Conf: {avg_conf:.2f}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            # Place in grid
            y_start = row * h
            x_start = col * w
            output[y_start:y_start+h, x_start:x_start+w] = cell

        # Draw consensus
        if num_models > 0:
            idx = len(comparison.model_results)
            row = idx // grid_width
            col = idx % grid_width

            cell = frame.copy()
            self._draw_skeleton(cell, comparison.consensus_keypoints, color=(255, 0, 0))
            cv2.putText(cell, "CONSENSUS", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(cell, f"Disagreement: {comparison.disagreement_score:.3f}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            y_start = row * h
            x_start = col * w
            output[y_start:y_start+h, x_start:x_start+w] = cell

        # Resize if too large
        max_width = 1920
        if output.shape[1] > max_width:
            scale = max_width / output.shape[1]
            new_height = int(output.shape[0] * scale)
            output = cv2.resize(output, (max_width, new_height))

        if output_path:
            cv2.imwrite(output_path, output)

        return output

    def _draw_skeleton(self, img: np.ndarray, keypoints: np.ndarray,
                      color: Tuple[int, int, int] = (0, 255, 0)):
        """Draw COCO skeleton on image."""
        h, w = img.shape[:2]

        # COCO skeleton connections
        connections = [
            (0, 1), (0, 2), (1, 3), (2, 4),  # Head
            (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Arms
            (5, 11), (6, 12), (11, 12),  # Torso
            (11, 13), (13, 15), (12, 14), (14, 16)  # Legs
        ]

        # Draw connections
        for connection in connections:
            if connection[0] < len(keypoints) and connection[1] < len(keypoints):
                pt1 = keypoints[connection[0]]
                pt2 = keypoints[connection[1]]

                if len(pt1) >= 2 and len(pt2) >= 2:
                    x1, y1 = int(pt1[0] * w), int(pt1[1] * h)
                    x2, y2 = int(pt2[0] * w), int(pt2[1] * h)

                    cv2.line(img, (x1, y1), (x2, y2), color, 2)

        # Draw keypoints
        for kpt in keypoints:
            if len(kpt) >= 2:
                x, y = int(kpt[0] * w), int(kpt[1] * h)
                cv2.circle(img, (x, y), 3, (0, 0, 255), -1)


def main():
    """Main function for command-line usage."""
    import argparse

    parser = argparse.ArgumentParser(description='Cross-Model Pose Validation')
    parser.add_argument('--video', required=True, help='Path to input video')
    parser.add_argument('--models', nargs='+',
                       default=['mediapipe', 'openpose', 'mmpose'],
                       help='Models to use for validation')
    parser.add_argument('--output', default='cross_validation_output',
                       help='Output directory')
    parser.add_argument('--sample-rate', type=int, default=1,
                       help='Process every Nth frame')
    parser.add_argument('--visualize', action='store_true',
                       help='Generate visualization')

    args = parser.parse_args()

    # Create validator
    validator = CrossModelValidator(models_to_use=args.models)

    # Run validation
    print("Starting cross-model validation...")
    results = validator.validate_video(
        args.video,
        output_dir=args.output,
        sample_rate=args.sample_rate
    )

    print(f"\nValidation complete!")
    print(f"Results saved to: {args.output}")

    # Generate visualizations if requested
    if args.visualize and results["comparisons"]:
        print("\nGenerating visualizations...")
        cap = cv2.VideoCapture(args.video)

        for i, comparison in enumerate(results["comparisons"][:5]):  # First 5 frames
            cap.set(cv2.CAP_PROP_POS_FRAMES, comparison.frame_number)
            ret, frame = cap.read()
            if ret:
                output_path = f"{args.output}/comparison_frame_{i}.jpg"
                validator.visualize_comparison(frame, comparison, output_path)

        cap.release()
        print(f"Visualizations saved to {args.output}")


if __name__ == "__main__":
    main()