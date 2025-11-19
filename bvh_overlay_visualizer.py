"""
BVH Overlay Visualizer
Visual validation tool that overlays BVH skeleton on original video.
Critical for detecting visual errors that numerical metrics miss.
"""

import cv2
import numpy as np
import mediapipe as mp
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any
import json
from dataclasses import dataclass
from scipy.spatial.transform import Rotation

# Try to import BVH parser
try:
    import bvh
    BVH_LIB_AVAILABLE = True
except ImportError:
    BVH_LIB_AVAILABLE = False
    print("Warning: bvh library not available. Install with: pip install bvh")


@dataclass
class BVHFrame:
    """Container for a single BVH frame data"""
    joint_positions: Dict[str, np.ndarray]  # Joint name -> 3D position
    joint_rotations: Dict[str, np.ndarray]  # Joint name -> rotation
    root_position: np.ndarray
    frame_number: int


class BVHOverlayVisualizer:
    """
    Visualize BVH skeleton overlaid on original video for visual validation.
    Helps detect orientation errors, flipped joints, and other visual issues.
    """

    # MediaPipe landmark connections for drawing
    MP_POSE_CONNECTIONS = mp.solutions.pose.POSE_CONNECTIONS

    # BVH skeleton connections (typical structure)
    BVH_CONNECTIONS = [
        # Spine
        ("Hips", "Spine"),
        ("Spine", "Spine1"),
        ("Spine1", "Spine2"),
        ("Spine2", "Neck"),
        ("Neck", "Head"),

        # Left arm
        ("Spine2", "LeftShoulder"),
        ("LeftShoulder", "LeftArm"),
        ("LeftArm", "LeftForeArm"),
        ("LeftForeArm", "LeftHand"),

        # Right arm
        ("Spine2", "RightShoulder"),
        ("RightShoulder", "RightArm"),
        ("RightArm", "RightForeArm"),
        ("RightForeArm", "RightHand"),

        # Left leg
        ("Hips", "LeftUpLeg"),
        ("LeftUpLeg", "LeftLeg"),
        ("LeftLeg", "LeftFoot"),
        ("LeftFoot", "LeftToeBase"),

        # Right leg
        ("Hips", "RightUpLeg"),
        ("RightUpLeg", "RightLeg"),
        ("RightLeg", "RightFoot"),
        ("RightFoot", "RightToeBase"),
    ]

    # Joint mapping from BVH to MediaPipe indices (simplified)
    BVH_TO_MP_MAPPING = {
        "Hips": 0,
        "LeftUpLeg": 23,
        "LeftLeg": 25,
        "LeftFoot": 27,
        "RightUpLeg": 24,
        "RightLeg": 26,
        "RightFoot": 28,
        "Spine": 0,
        "Spine1": 0,
        "Spine2": 11,
        "LeftShoulder": 11,
        "LeftArm": 13,
        "LeftForeArm": 15,
        "LeftHand": 17,
        "RightShoulder": 12,
        "RightArm": 14,
        "RightForeArm": 16,
        "RightHand": 18,
        "Neck": 0,
        "Head": 0,
    }

    def __init__(self, video_path: str, bvh_path: str, output_path: Optional[str] = None):
        """
        Initialize the overlay visualizer.

        Args:
            video_path: Path to input video
            bvh_path: Path to BVH file
            output_path: Optional path for output video
        """
        self.video_path = video_path
        self.bvh_path = bvh_path
        self.output_path = output_path or "overlay_output.mp4"

        # MediaPipe setup
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        # Load BVH data
        self.bvh_data = self.load_bvh(bvh_path)
        self.bvh_frames = []

        # Video properties
        self.cap = None
        self.video_width = 0
        self.video_height = 0
        self.video_fps = 30.0

        # Visualization settings
        self.show_mediapipe = True
        self.show_bvh = True
        self.show_comparison_metrics = True
        self.alpha_blend = 0.7

    def load_bvh(self, bvh_path: str) -> Optional[Any]:
        """
        Load and parse BVH file.

        Args:
            bvh_path: Path to BVH file

        Returns:
            Parsed BVH data or None if parsing fails
        """
        if BVH_LIB_AVAILABLE:
            try:
                with open(bvh_path) as f:
                    mocap = bvh.Bvh(f.read())
                return mocap
            except Exception as e:
                print(f"Error loading BVH: {e}")
                return None
        else:
            # Fallback: Basic BVH parser
            return self._basic_bvh_parser(bvh_path)

    def _basic_bvh_parser(self, bvh_path: str) -> Dict[str, Any]:
        """
        Basic BVH parser for when bvh library is not available.

        Args:
            bvh_path: Path to BVH file

        Returns:
            Dictionary with parsed BVH data
        """
        with open(bvh_path, 'r') as f:
            lines = f.readlines()

        # Parse structure and motion data
        hierarchy = {}
        motion_data = []
        in_motion = False

        for line in lines:
            line = line.strip()
            if line.startswith("MOTION"):
                in_motion = True
            elif in_motion and line and not line.startswith("Frames:") and not line.startswith("Frame Time:"):
                # Parse motion data
                values = list(map(float, line.split()))
                motion_data.append(values)

        return {
            "hierarchy": hierarchy,
            "motion": motion_data,
            "frames": len(motion_data)
        }

    def project_3d_to_2d(self, point_3d: np.ndarray,
                        camera_matrix: Optional[np.ndarray] = None) -> Tuple[int, int]:
        """
        Project 3D point to 2D image coordinates.

        Args:
            point_3d: 3D point [x, y, z]
            camera_matrix: Camera intrinsic matrix

        Returns:
            2D pixel coordinates (x, y)
        """
        if camera_matrix is None:
            # Simple orthographic projection with scaling
            # Adjust these parameters based on your BVH coordinate system
            scale = self.video_height / 2
            offset_x = self.video_width / 2
            offset_y = self.video_height / 2

            x = int(point_3d[0] * scale + offset_x)
            y = int(-point_3d[1] * scale + offset_y)  # Flip Y for screen coordinates
        else:
            # Perspective projection
            point_homo = np.append(point_3d, 1)
            projected = camera_matrix @ point_homo[:3]
            if projected[2] != 0:
                x = int(projected[0] / projected[2])
                y = int(projected[1] / projected[2])
            else:
                x, y = 0, 0

        # Clamp to image bounds
        x = max(0, min(x, self.video_width - 1))
        y = max(0, min(y, self.video_height - 1))

        return (x, y)

    def extract_bvh_frame(self, frame_idx: int) -> Optional[BVHFrame]:
        """
        Extract joint positions for a specific BVH frame.

        Args:
            frame_idx: Frame index

        Returns:
            BVHFrame object with joint data
        """
        if self.bvh_data is None:
            return None

        if BVH_LIB_AVAILABLE and hasattr(self.bvh_data, 'get_joint'):
            # Use bvh library
            frame_data = BVHFrame(
                joint_positions={},
                joint_rotations={},
                root_position=np.array([0, 0, 0]),
                frame_number=frame_idx
            )

            # Extract joint positions
            for joint_name in self.bvh_data.get_joints_names():
                joint_index = self.bvh_data.get_joint_index(joint_name)
                channels = self.bvh_data.get_joint_channels(joint_name)

                # Get position (usually only root has position channels)
                if 'Xposition' in channels:
                    x = self.bvh_data.frame_joint_channel(frame_idx, joint_name, 'Xposition')
                    y = self.bvh_data.frame_joint_channel(frame_idx, joint_name, 'Yposition')
                    z = self.bvh_data.frame_joint_channel(frame_idx, joint_name, 'Zposition')
                    frame_data.joint_positions[joint_name] = np.array([x, y, z])

                # Get rotation
                if 'Xrotation' in channels:
                    rx = self.bvh_data.frame_joint_channel(frame_idx, joint_name, 'Xrotation')
                    ry = self.bvh_data.frame_joint_channel(frame_idx, joint_name, 'Yrotation')
                    rz = self.bvh_data.frame_joint_channel(frame_idx, joint_name, 'Zrotation')
                    frame_data.joint_rotations[joint_name] = np.array([rx, ry, rz])

            return frame_data
        else:
            # Use basic parsed data
            if frame_idx < len(self.bvh_data.get("motion", [])):
                motion_values = self.bvh_data["motion"][frame_idx]
                # Parse motion values into joint positions
                # This would require knowledge of the BVH structure
                return None

        return None

    def draw_bvh_skeleton(self, img: np.ndarray, bvh_frame: BVHFrame,
                         color: Tuple[int, int, int] = (0, 255, 0),
                         thickness: int = 2) -> np.ndarray:
        """
        Draw BVH skeleton on image.

        Args:
            img: Input image
            bvh_frame: BVH frame data
            color: Skeleton color (BGR)
            thickness: Line thickness

        Returns:
            Image with skeleton drawn
        """
        overlay = img.copy()

        # Draw connections
        for connection in self.BVH_CONNECTIONS:
            joint1, joint2 = connection

            if joint1 in bvh_frame.joint_positions and joint2 in bvh_frame.joint_positions:
                # Project 3D points to 2D
                pt1 = self.project_3d_to_2d(bvh_frame.joint_positions[joint1])
                pt2 = self.project_3d_to_2d(bvh_frame.joint_positions[joint2])

                # Draw line
                cv2.line(overlay, pt1, pt2, color, thickness)

        # Draw joints
        for joint_name, position in bvh_frame.joint_positions.items():
            pt = self.project_3d_to_2d(position)
            cv2.circle(overlay, pt, 4, (0, 0, 255), -1)
            cv2.circle(overlay, pt, 5, color, 1)

        # Blend with original
        result = cv2.addWeighted(img, 1 - self.alpha_blend, overlay, self.alpha_blend, 0)

        return result

    def draw_mediapipe_skeleton(self, img: np.ndarray,
                               landmarks: Any,
                               color: Tuple[int, int, int] = (255, 0, 0)) -> np.ndarray:
        """
        Draw MediaPipe skeleton on image.

        Args:
            img: Input image
            landmarks: MediaPipe landmarks
            color: Skeleton color (BGR)

        Returns:
            Image with MediaPipe skeleton
        """
        overlay = img.copy()

        # Draw connections
        self.mp_drawing.draw_landmarks(
            overlay,
            landmarks,
            self.MP_POSE_CONNECTIONS,
            landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
        )

        # Blend
        result = cv2.addWeighted(img, 1 - self.alpha_blend, overlay, self.alpha_blend, 0)

        return result

    def calculate_alignment_metrics(self, bvh_frame: BVHFrame,
                                   mp_landmarks: Any) -> Dict[str, float]:
        """
        Calculate alignment metrics between BVH and MediaPipe.

        Args:
            bvh_frame: BVH frame data
            mp_landmarks: MediaPipe landmarks

        Returns:
            Dictionary with alignment metrics
        """
        metrics = {}

        # Convert MediaPipe landmarks to array
        mp_joints = []
        for landmark in mp_landmarks.landmark:
            mp_joints.append([landmark.x, landmark.y, landmark.z])
        mp_joints = np.array(mp_joints)

        # Calculate average distance between corresponding joints
        distances = []
        for bvh_joint, mp_idx in self.BVH_TO_MP_MAPPING.items():
            if bvh_joint in bvh_frame.joint_positions and mp_idx < len(mp_joints):
                bvh_pos = bvh_frame.joint_positions[bvh_joint]
                mp_pos = mp_joints[mp_idx]

                # Project both to 2D for comparison
                bvh_2d = self.project_3d_to_2d(bvh_pos)
                mp_2d = (int(mp_pos[0] * self.video_width),
                        int(mp_pos[1] * self.video_height))

                dist = np.linalg.norm(np.array(bvh_2d) - np.array(mp_2d))
                distances.append(dist)

        if distances:
            metrics["mean_pixel_error"] = np.mean(distances)
            metrics["max_pixel_error"] = np.max(distances)
            metrics["std_pixel_error"] = np.std(distances)
        else:
            metrics["mean_pixel_error"] = -1

        return metrics

    def draw_metrics_overlay(self, img: np.ndarray, metrics: Dict[str, float]) -> np.ndarray:
        """
        Draw metrics information on image.

        Args:
            img: Input image
            metrics: Metrics dictionary

        Returns:
            Image with metrics overlay
        """
        # Create semi-transparent background for text
        overlay = img.copy()
        h, w = img.shape[:2]

        # Background rectangle
        cv2.rectangle(overlay, (10, 10), (350, 120), (0, 0, 0), -1)
        img = cv2.addWeighted(img, 0.7, overlay, 0.3, 0)

        # Draw metrics text
        y_offset = 30
        for key, value in metrics.items():
            if isinstance(value, float):
                text = f"{key}: {value:.2f}"
            else:
                text = f"{key}: {value}"

            cv2.putText(img, text, (20, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                       (255, 255, 255), 1, cv2.LINE_AA)
            y_offset += 25

        return img

    def process_video(self, show_preview: bool = True,
                     save_output: bool = True) -> Dict[str, Any]:
        """
        Process video with BVH overlay.

        Args:
            show_preview: Whether to show real-time preview
            save_output: Whether to save output video

        Returns:
            Processing statistics
        """
        # Open video
        self.cap = cv2.VideoCapture(self.video_path)
        self.video_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.video_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.video_fps = self.cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Setup video writer if saving
        out = None
        if save_output:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(self.output_path, fourcc, self.video_fps,
                                 (self.video_width, self.video_height))

        # MediaPipe pose detector
        pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # Statistics
        stats = {
            "total_frames": total_frames,
            "processed_frames": 0,
            "mean_alignment_error": [],
            "warnings": []
        }

        frame_idx = 0
        print(f"Processing {total_frames} frames...")

        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break

            # Process with MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_results = pose.process(rgb_frame)

            # Get BVH frame
            bvh_frame = self.extract_bvh_frame(frame_idx)

            # Base frame for overlay
            result_frame = frame.copy()

            # Draw MediaPipe skeleton
            if self.show_mediapipe and mp_results.pose_landmarks:
                result_frame = self.draw_mediapipe_skeleton(
                    result_frame, mp_results.pose_landmarks, (255, 0, 0))

            # Draw BVH skeleton
            if self.show_bvh and bvh_frame:
                result_frame = self.draw_bvh_skeleton(
                    result_frame, bvh_frame, (0, 255, 0))

            # Calculate and show metrics
            if self.show_comparison_metrics and mp_results.pose_landmarks and bvh_frame:
                metrics = self.calculate_alignment_metrics(bvh_frame, mp_results.pose_landmarks)
                result_frame = self.draw_metrics_overlay(result_frame, metrics)
                stats["mean_alignment_error"].append(metrics.get("mean_pixel_error", 0))

            # Add frame counter
            cv2.putText(result_frame, f"Frame: {frame_idx}/{total_frames}",
                       (self.video_width - 200, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Add legend
            cv2.putText(result_frame, "MediaPipe", (self.video_width - 200, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            cv2.putText(result_frame, "BVH", (self.video_width - 200, 85),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Show preview
            if show_preview:
                cv2.imshow('BVH Overlay Validation', result_frame)
                key = cv2.waitKey(1)

                if key == ord('q'):
                    break
                elif key == ord('m'):
                    self.show_mediapipe = not self.show_mediapipe
                elif key == ord('b'):
                    self.show_bvh = not self.show_bvh
                elif key == ord('c'):
                    self.show_comparison_metrics = not self.show_comparison_metrics
                elif key == ord(' '):
                    # Pause
                    cv2.waitKey(0)

            # Save frame
            if out:
                out.write(result_frame)

            frame_idx += 1
            stats["processed_frames"] = frame_idx

            # Progress update
            if frame_idx % 30 == 0:
                progress = (frame_idx / total_frames) * 100
                print(f"Progress: {progress:.1f}%")

        # Cleanup
        self.cap.release()
        if out:
            out.release()
        cv2.destroyAllWindows()
        pose.close()

        # Calculate final statistics
        if stats["mean_alignment_error"]:
            stats["overall_mean_error"] = np.mean(stats["mean_alignment_error"])
            stats["overall_std_error"] = np.std(stats["mean_alignment_error"])
            stats["overall_max_error"] = np.max(stats["mean_alignment_error"])

        print(f"\nProcessing complete!")
        print(f"Output saved to: {self.output_path}")
        print(f"Overall alignment error: {stats.get('overall_mean_error', -1):.2f} pixels")

        return stats

    def generate_comparison_report(self, stats: Dict[str, Any],
                                  output_path: str = "overlay_report.json"):
        """
        Generate a comparison report.

        Args:
            stats: Processing statistics
            output_path: Path to save report

        Returns:
            Report dictionary
        """
        report = {
            "video_file": self.video_path,
            "bvh_file": self.bvh_path,
            "output_file": self.output_path,
            "statistics": stats,
            "configuration": {
                "show_mediapipe": self.show_mediapipe,
                "show_bvh": self.show_bvh,
                "alpha_blend": self.alpha_blend
            }
        }

        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)

        print(f"Report saved to: {output_path}")
        return report


def main():
    """Main function for command-line usage."""
    import argparse

    parser = argparse.ArgumentParser(description='BVH Overlay Visualizer')
    parser.add_argument('--video', required=True, help='Path to input video')
    parser.add_argument('--bvh', required=True, help='Path to BVH file')
    parser.add_argument('--output', help='Output video path')
    parser.add_argument('--no-preview', action='store_true',
                       help='Disable preview window')
    parser.add_argument('--no-save', action='store_true',
                       help='Don\'t save output video')
    parser.add_argument('--report', action='store_true',
                       help='Generate comparison report')

    args = parser.parse_args()

    # Create visualizer
    visualizer = BVHOverlayVisualizer(
        video_path=args.video,
        bvh_path=args.bvh,
        output_path=args.output
    )

    # Process video
    print("Starting BVH overlay visualization...")
    print("Controls (during preview):")
    print("  q - Quit")
    print("  m - Toggle MediaPipe skeleton")
    print("  b - Toggle BVH skeleton")
    print("  c - Toggle comparison metrics")
    print("  Space - Pause/Resume")
    print()

    stats = visualizer.process_video(
        show_preview=not args.no_preview,
        save_output=not args.no_save
    )

    # Generate report if requested
    if args.report:
        visualizer.generate_comparison_report(stats)


if __name__ == "__main__":
    main()