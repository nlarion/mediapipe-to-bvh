#!/usr/bin/env python3
"""
Visual Accuracy Tester for BVH Converter

Creates side-by-side visual comparisons between:
- Original video frames
- Rendered BVH skeleton (multiple camera angles)

This allows visual inspection of issues like head tracking, shoulder alignment, etc.
"""

import cv2
import numpy as np
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import json
from dataclasses import dataclass, asdict
from datetime import datetime

# For BVH rendering
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio


@dataclass
class CameraView:
    """Camera view configuration for BVH rendering."""
    name: str
    eye: Dict[str, float]  # x, y, z position of camera
    up: Dict[str, float]   # up vector
    center: Dict[str, float] = None  # look-at point

    def __post_init__(self):
        if self.center is None:
            self.center = {"x": 0, "y": 0, "z": 0}


# Predefined camera views
CAMERA_VIEWS = {
    "front": CameraView(
        name="Front",
        eye={"x": 0, "y": 0, "z": 2.5},
        up={"x": 0, "y": 1, "z": 0}
    ),
    "front_facing": CameraView(
        name="Front (Facing)",
        eye={"x": 0, "y": 0, "z": -2.5},
        up={"x": 0, "y": 1, "z": 0}
    ),
    "side_left": CameraView(
        name="Side (Left)",
        eye={"x": -2.5, "y": 0, "z": 0},
        up={"x": 0, "y": 1, "z": 0}
    ),
    "side_right": CameraView(
        name="Side (Right)",
        eye={"x": 2.5, "y": 0, "z": 0},
        up={"x": 0, "y": 1, "z": 0}
    ),
    "top_ortho": CameraView(
        name="Top (Orthographic)",
        eye={"x": 0, "y": 2.5, "z": 0},
        up={"x": 0, "y": 0, "z": -1}
    ),
    "three_quarter": CameraView(
        name="3/4 View",
        eye={"x": 1.5, "y": 1.0, "z": 2.0},
        up={"x": 0, "y": 1, "z": 0}
    ),
}


class BVHParser:
    """Parse BVH files and extract joint positions per frame."""

    def __init__(self):
        self.joints = {}
        self.root = None
        self.frames = 0
        self.frame_time = 0
        self.motion_data = []

    def parse_file(self, file_path: str):
        """Parse a BVH file."""
        with open(file_path, 'r') as f:
            content = f.read()

        if 'MOTION' in content:
            hierarchy, motion = content.split('MOTION', 1)
        else:
            hierarchy = content
            motion = ""

        self._parse_hierarchy(hierarchy)
        if motion:
            self._parse_motion(motion)

    def _parse_hierarchy(self, text: str):
        """Parse the HIERARCHY section."""
        lines = [line.strip() for line in text.splitlines() if line.strip()]

        joint_stack = []

        for line in lines:
            if 'HIERARCHY' in line:
                continue

            if 'ROOT' in line:
                name = line.split('ROOT', 1)[1].strip()
                self.joints[name] = {
                    'parent': None,
                    'children': [],
                    'offset': np.zeros(3),
                    'channels': [],
                    'end_site': False
                }
                self.root = name
                joint_stack.append(name)

            elif 'JOINT' in line:
                name = line.split('JOINT', 1)[1].strip()
                parent = joint_stack[-1] if joint_stack else None
                self.joints[name] = {
                    'parent': parent,
                    'children': [],
                    'offset': np.zeros(3),
                    'channels': [],
                    'end_site': False
                }
                if parent:
                    self.joints[parent]['children'].append(name)
                joint_stack.append(name)

            elif 'End Site' in line:
                parent = joint_stack[-1] if joint_stack else None
                name = f"{parent}_end"
                self.joints[name] = {
                    'parent': parent,
                    'children': [],
                    'offset': np.zeros(3),
                    'channels': [],
                    'end_site': True
                }
                if parent:
                    self.joints[parent]['children'].append(name)
                joint_stack.append(name)

            elif '{' in line:
                continue

            elif '}' in line:
                if joint_stack:
                    joint_stack.pop()

            elif 'OFFSET' in line:
                if joint_stack:
                    offset = [float(x) for x in line.split('OFFSET')[1].strip().split()]
                    self.joints[joint_stack[-1]]['offset'] = np.array(offset)

            elif 'CHANNELS' in line:
                if joint_stack:
                    parts = line.split()
                    num_channels = int(parts[1])
                    self.joints[joint_stack[-1]]['channels'] = parts[2:2+num_channels]

    def _parse_motion(self, text: str):
        """Parse the MOTION section."""
        lines = [line.strip() for line in text.splitlines() if line.strip()]

        for line in lines:
            if 'Frames:' in line:
                self.frames = int(line.split(':', 1)[1].strip())
            elif 'Frame Time:' in line:
                self.frame_time = float(line.split(':', 1)[1].strip())
            else:
                try:
                    values = [float(x) for x in line.split()]
                    if values:
                        self.motion_data.append(values)
                except ValueError:
                    pass

    def get_frame_positions(self, frame_idx: int) -> Dict[str, np.ndarray]:
        """Get joint positions for a specific frame using forward kinematics."""
        if frame_idx >= len(self.motion_data):
            return {}

        frame_data = self.motion_data[frame_idx]
        positions = {}

        self._compute_positions(self.root, frame_data, [0], np.zeros(3), np.eye(3), positions)
        return positions

    def _compute_positions(self, joint_name: str, frame_data: List[float],
                          channel_idx: List[int], parent_pos: np.ndarray,
                          parent_rot: np.ndarray, positions: Dict):
        """Recursively compute joint positions."""
        joint = self.joints[joint_name]

        # Start with offset rotated by parent
        offset_rotated = parent_rot @ joint['offset']
        joint_pos = parent_pos + offset_rotated

        # Apply this joint's transformations
        local_rot = np.eye(3)

        if not joint['end_site'] and joint['channels']:
            for channel in joint['channels']:
                if channel_idx[0] >= len(frame_data):
                    break

                val = frame_data[channel_idx[0]]
                channel_idx[0] += 1

                if 'position' in channel.lower():
                    # Position channels (usually only on root)
                    if 'x' in channel.lower():
                        joint_pos[0] = val + offset_rotated[0]
                    elif 'y' in channel.lower():
                        joint_pos[1] = val + offset_rotated[1]
                    elif 'z' in channel.lower():
                        joint_pos[2] = val + offset_rotated[2]

                elif 'rotation' in channel.lower():
                    angle_rad = np.radians(val)
                    c, s = np.cos(angle_rad), np.sin(angle_rad)

                    if 'x' in channel.lower():
                        rot = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
                    elif 'y' in channel.lower():
                        rot = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
                    elif 'z' in channel.lower():
                        rot = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
                    else:
                        rot = np.eye(3)

                    local_rot = local_rot @ rot

        positions[joint_name] = joint_pos
        global_rot = parent_rot @ local_rot

        # Process children
        for child_name in joint['children']:
            self._compute_positions(child_name, frame_data, channel_idx,
                                   joint_pos, global_rot, positions)

    def get_skeleton_connections(self) -> List[Tuple[str, str]]:
        """Get list of (parent, child) connections for drawing skeleton."""
        connections = []
        for name, joint in self.joints.items():
            if joint['parent']:
                connections.append((joint['parent'], name))
        return connections


class VisualAccuracyTester:
    """
    Creates visual comparisons between video frames and BVH skeleton renders.
    """

    def __init__(self, output_dir: str = "visual_accuracy_output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.bvh_parser = None
        self.video_cap = None
        self.video_fps = 30
        self.video_frame_count = 0

    def load_bvh(self, bvh_path: str):
        """Load and parse BVH file."""
        print(f"Loading BVH: {bvh_path}")
        self.bvh_parser = BVHParser()
        self.bvh_parser.parse_file(bvh_path)
        print(f"  Frames: {self.bvh_parser.frames}")
        print(f"  Frame time: {self.bvh_parser.frame_time}")
        print(f"  Joints: {len(self.bvh_parser.joints)}")

    def load_video(self, video_path: str):
        """Load video file."""
        print(f"Loading video: {video_path}")
        self.video_cap = cv2.VideoCapture(video_path)
        if not self.video_cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        self.video_fps = self.video_cap.get(cv2.CAP_PROP_FPS)
        self.video_frame_count = int(self.video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"  FPS: {self.video_fps}")
        print(f"  Total frames: {self.video_frame_count}")

    def extract_video_frame(self, frame_idx: int) -> Optional[np.ndarray]:
        """Extract a specific frame from the video."""
        self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = self.video_cap.read()
        if ret:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return None

    def render_bvh_frame(self, frame_idx: int, camera_view: CameraView,
                         width: int = 600, height: int = 600,
                         zoom: float = 1.0, show_labels: bool = False) -> Optional[np.ndarray]:
        """Render BVH skeleton for a specific frame.

        Args:
            frame_idx: Frame index to render
            camera_view: Camera view configuration
            width: Image width
            height: Image height
            zoom: Zoom factor (higher = more zoomed in, 1.0 = default)
            show_labels: Whether to show joint labels
        """
        if self.bvh_parser is None:
            return None

        positions = self.bvh_parser.get_frame_positions(frame_idx)
        if not positions:
            return None

        connections = self.bvh_parser.get_skeleton_connections()

        # Extract coordinates
        joint_names = list(positions.keys())
        xs = [positions[name][0] for name in joint_names]
        ys = [positions[name][1] for name in joint_names]
        zs = [positions[name][2] for name in joint_names]

        # Create line segments for skeleton
        lines_x, lines_y, lines_z = [], [], []
        for parent, child in connections:
            if parent in positions and child in positions:
                p1, p2 = positions[parent], positions[child]
                lines_x.extend([p1[0], p2[0], None])
                lines_y.extend([p1[1], p2[1], None])
                lines_z.extend([p1[2], p2[2], None])

        # Calculate bounds for consistent view (zoom affects range)
        all_coords = np.array([[x, y, z] for x, y, z in zip(xs, ys, zs)])
        center = np.mean(all_coords, axis=0)
        max_range = np.max(np.abs(all_coords - center)) * (1.5 / zoom)

        # Create figure
        fig = go.Figure()

        # Add joints (with optional labels)
        fig.add_trace(go.Scatter3d(
            x=xs, y=ys, z=zs,
            mode='markers+text' if show_labels else 'markers',
            marker=dict(size=8, color='blue'),
            text=joint_names if show_labels else None,
            textposition='top center',
            textfont=dict(size=10),
            name='Joints'
        ))

        # Add skeleton lines
        fig.add_trace(go.Scatter3d(
            x=lines_x, y=lines_y, z=lines_z,
            mode='lines',
            line=dict(color='red', width=6),
            name='Skeleton'
        ))

        # Configure camera and layout
        fig.update_layout(
            scene=dict(
                xaxis=dict(range=[center[0] - max_range, center[0] + max_range], title='X'),
                yaxis=dict(range=[center[1] - max_range, center[1] + max_range], title='Y'),
                zaxis=dict(range=[center[2] - max_range, center[2] + max_range], title='Z'),
                aspectmode='cube',
                camera=dict(
                    eye=camera_view.eye,
                    up=camera_view.up,
                    center=camera_view.center
                )
            ),
            margin=dict(l=0, r=0, b=0, t=30),
            title=dict(text=f"{camera_view.name} - Frame {frame_idx}", x=0.5),
            showlegend=False,
            width=width,
            height=height
        )

        # Convert to image
        try:
            img_bytes = pio.to_image(fig, format='png', width=width, height=height)
            img_array = np.frombuffer(img_bytes, dtype=np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        except Exception as e:
            print(f"Error rendering frame {frame_idx}: {e}")
            return None

    def create_comparison_image(self, video_frame: np.ndarray,
                                bvh_renders: Dict[str, np.ndarray],
                                frame_idx: int) -> np.ndarray:
        """
        Create a side-by-side comparison image.
        Layout: Video | BVH View 1 | BVH View 2 | ...
        """
        # Resize video frame to match BVH render size
        target_height = 600
        aspect = video_frame.shape[1] / video_frame.shape[0]
        target_width = int(target_height * aspect)
        video_resized = cv2.resize(video_frame, (target_width, target_height))

        # Create list of images to combine
        images = [video_resized]

        for view_name, render in bvh_renders.items():
            if render is not None:
                # Ensure same height
                if render.shape[0] != target_height:
                    aspect = render.shape[1] / render.shape[0]
                    new_width = int(target_height * aspect)
                    render = cv2.resize(render, (new_width, target_height))
                images.append(render)

        # Combine horizontally
        combined = np.hstack(images)

        # Add frame number label
        cv2.putText(combined, f"Frame: {frame_idx}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        return combined

    def run_comparison(self, video_path: str, bvh_path: str,
                      frame_interval: int = 1,
                      camera_views: List[str] = None,
                      sample_rate: int = 2,
                      max_frames: int = None,
                      zoom: float = 1.0,
                      render_size: int = 600,
                      show_labels: bool = False):
        """
        Run visual comparison between video and BVH.

        Args:
            video_path: Path to source video
            bvh_path: Path to BVH file
            frame_interval: Compare every Nth frame (1 = every frame)
            camera_views: List of camera view names to use
            sample_rate: Sample rate used during BVH conversion (to align frames)
            max_frames: Maximum number of frames to process (None = all)
            zoom: Zoom factor for skeleton render (higher = more zoomed)
            render_size: Size of skeleton render in pixels
            show_labels: Whether to show joint labels on skeleton
        """
        self.load_video(video_path)
        self.load_bvh(bvh_path)

        if camera_views is None:
            camera_views = ["front_facing", "side_left", "three_quarter"]

        # Create output subdirectory for this run
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_name = Path(video_path).stem
        run_dir = self.output_dir / f"{video_name}_{timestamp}"
        run_dir.mkdir(parents=True, exist_ok=True)

        frames_dir = run_dir / "frames"
        frames_dir.mkdir(exist_ok=True)

        # Calculate frame mapping (video frames to BVH frames)
        # BVH was created with sample_rate, so BVH frame i corresponds to video frame i * sample_rate
        bvh_frames = self.bvh_parser.frames

        print(f"\nRunning visual comparison...")
        print(f"  Video frames: {self.video_frame_count}")
        print(f"  BVH frames: {bvh_frames}")
        print(f"  Sample rate: {sample_rate}")
        print(f"  Frame interval: {frame_interval}")
        print(f"  Camera views: {camera_views}")
        print(f"  Zoom: {zoom}x, Render size: {render_size}px")
        print(f"  Output: {run_dir}")

        # Get camera view objects
        views = [CAMERA_VIEWS[v] for v in camera_views if v in CAMERA_VIEWS]

        comparison_frames = []
        frame_count = 0

        for bvh_idx in range(0, bvh_frames, frame_interval):
            if max_frames and frame_count >= max_frames:
                break

            # Map BVH frame to video frame
            video_idx = bvh_idx * sample_rate

            if video_idx >= self.video_frame_count:
                break

            print(f"  Processing BVH frame {bvh_idx} (video frame {video_idx})...")

            # Extract video frame
            video_frame = self.extract_video_frame(video_idx)
            if video_frame is None:
                print(f"    Warning: Could not extract video frame {video_idx}")
                continue

            # Render BVH from multiple views
            bvh_renders = {}
            for view in views:
                render = self.render_bvh_frame(bvh_idx, view,
                                               width=render_size, height=render_size,
                                               zoom=zoom, show_labels=show_labels)
                if render is not None:
                    bvh_renders[view.name] = render

            if not bvh_renders:
                print(f"    Warning: Could not render BVH frame {bvh_idx}")
                continue

            # Create comparison image
            comparison = self.create_comparison_image(video_frame, bvh_renders, bvh_idx)

            # Save individual frame
            frame_path = frames_dir / f"frame_{bvh_idx:04d}.png"
            cv2.imwrite(str(frame_path), cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR))

            comparison_frames.append({
                "bvh_frame": bvh_idx,
                "video_frame": video_idx,
                "image_path": str(frame_path)
            })

            frame_count += 1

        # Save metadata
        metadata = {
            "video_path": video_path,
            "bvh_path": bvh_path,
            "video_frames": self.video_frame_count,
            "bvh_frames": bvh_frames,
            "sample_rate": sample_rate,
            "frame_interval": frame_interval,
            "camera_views": camera_views,
            "comparison_frames": comparison_frames,
            "timestamp": timestamp
        }

        metadata_path = run_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"\nComparison complete!")
        print(f"  Total frames compared: {len(comparison_frames)}")
        print(f"  Output directory: {run_dir}")
        print(f"  Metadata: {metadata_path}")

        # Cleanup
        if self.video_cap:
            self.video_cap.release()

        return run_dir, comparison_frames

    def create_summary_grid(self, run_dir: Path, grid_size: Tuple[int, int] = (4, 4)):
        """Create a summary grid of key frames."""
        frames_dir = run_dir / "frames"
        frame_files = sorted(frames_dir.glob("*.png"))

        if not frame_files:
            print("No frames found")
            return

        # Select evenly spaced frames
        n_frames = grid_size[0] * grid_size[1]
        indices = np.linspace(0, len(frame_files) - 1, n_frames, dtype=int)
        selected_files = [frame_files[i] for i in indices]

        # Load and combine
        images = []
        for f in selected_files:
            img = cv2.imread(str(f))
            if img is not None:
                # Resize for grid
                img = cv2.resize(img, (400, 150))
                images.append(img)

        if not images:
            return

        # Create grid
        rows = []
        for i in range(grid_size[0]):
            row_images = images[i * grid_size[1]:(i + 1) * grid_size[1]]
            if row_images:
                rows.append(np.hstack(row_images))

        grid = np.vstack(rows)

        # Save
        grid_path = run_dir / "summary_grid.png"
        cv2.imwrite(str(grid_path), grid)
        print(f"Summary grid saved: {grid_path}")

        return grid_path


def main():
    parser = argparse.ArgumentParser(description="Visual accuracy testing for BVH conversion")
    parser.add_argument("--video", required=True, help="Path to source video")
    parser.add_argument("--bvh", required=True, help="Path to BVH file")
    parser.add_argument("--output-dir", default="visual_accuracy_output", help="Output directory")
    parser.add_argument("--frame-interval", type=int, default=1,
                       help="Compare every Nth frame (1 = every frame)")
    parser.add_argument("--sample-rate", type=int, default=2,
                       help="Sample rate used during BVH conversion")
    parser.add_argument("--max-frames", type=int, default=None,
                       help="Maximum number of frames to process")
    parser.add_argument("--views", nargs="+",
                       default=["front_facing", "side_left", "three_quarter"],
                       choices=list(CAMERA_VIEWS.keys()),
                       help="Camera views to render")
    parser.add_argument("--zoom", type=float, default=1.0,
                       help="Zoom factor for skeleton (higher = more zoomed, e.g., 1.5 or 2.0)")
    parser.add_argument("--render-size", type=int, default=600,
                       help="Size of skeleton render in pixels (default: 600)")
    parser.add_argument("--show-labels", action="store_true",
                       help="Show joint labels on skeleton")
    parser.add_argument("--create-grid", action="store_true",
                       help="Create summary grid image")

    args = parser.parse_args()

    tester = VisualAccuracyTester(args.output_dir)

    run_dir, frames = tester.run_comparison(
        video_path=args.video,
        bvh_path=args.bvh,
        frame_interval=args.frame_interval,
        camera_views=args.views,
        sample_rate=args.sample_rate,
        max_frames=args.max_frames,
        zoom=args.zoom,
        render_size=args.render_size,
        show_labels=args.show_labels
    )

    if args.create_grid:
        tester.create_summary_grid(run_dir)


if __name__ == "__main__":
    main()
