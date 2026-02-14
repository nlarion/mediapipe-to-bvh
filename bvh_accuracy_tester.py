#!/usr/bin/env python3
"""
BVH Accuracy Tester using GPT-4V vision analysis.

Compares original video frames (with MediaPipe overlay) against BVH skeleton renders
to identify pose tracking and conversion errors.
"""

import os
import sys
import json
import argparse
import base64
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from io import BytesIO
from dataclasses import dataclass, field

import numpy as np
import cv2
import mediapipe as mp
from PIL import Image

# Plotly for 3D rendering (matching bvh_viewer)
try:
    import plotly.graph_objects as go
    import plotly.io as pio
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False
    # Fallback to matplotlib
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

# Load .env file if present
try:
    from dotenv import load_dotenv
    # Look for .env in script directory
    env_path = Path(__file__).parent / '.env'
    if env_path.exists():
        load_dotenv(env_path)
except ImportError:
    # Try manual loading if dotenv not installed
    env_path = Path(__file__).parent / '.env'
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key.strip()] = value.strip().strip('"').strip("'")

# OpenAI for GPT-4V
try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

# Google Gemini
try:
    from google import genai
    from google.genai import types
    HAS_GEMINI = True
except ImportError:
    HAS_GEMINI = False


@dataclass
class FrameAnalysis:
    """Analysis result for a single frame."""
    frame_idx: int
    video_frame_idx: int
    issues: List[str] = field(default_factory=list)
    severity: str = "unknown"  # low, medium, high
    raw_response: str = ""


@dataclass
class AnalysisReport:
    """Complete analysis report for a video/BVH pair."""
    video_path: str
    bvh_path: str
    frames_analyzed: int
    issues_by_joint: Dict[str, int] = field(default_factory=dict)
    issues_by_type: Dict[str, int] = field(default_factory=dict)
    frame_analyses: List[FrameAnalysis] = field(default_factory=list)
    summary: str = ""


class BVHParser:
    """Simple BVH parser for skeleton extraction."""

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
        """Parse HIERARCHY section."""
        lines = [line.strip() for line in text.splitlines() if line.strip()]

        joint_stack = []

        for line in lines:
            if 'HIERARCHY' in line:
                continue
            elif 'ROOT' in line:
                name = line.split('ROOT', 1)[1].strip()
                joint = {'name': name, 'parent': None, 'children': [],
                        'offset': np.zeros(3), 'channels': [], 'end_site': False}
                self.root = joint
                self.joints[name] = joint
                joint_stack.append(joint)
            elif 'JOINT' in line:
                if not joint_stack:
                    continue
                name = line.split('JOINT', 1)[1].strip()
                parent = joint_stack[-1]
                joint = {'name': name, 'parent': parent['name'], 'children': [],
                        'offset': np.zeros(3), 'channels': [], 'end_site': False}
                parent['children'].append(name)
                self.joints[name] = joint
                joint_stack.append(joint)
            elif 'End Site' in line:
                if not joint_stack:
                    continue
                name = f"{joint_stack[-1]['name']}_end"
                parent = joint_stack[-1]
                joint = {'name': name, 'parent': parent['name'], 'children': [],
                        'offset': np.zeros(3), 'channels': [], 'end_site': True}
                parent['children'].append(name)
                self.joints[name] = joint
                joint_stack.append(joint)
            elif '{' in line:
                continue
            elif '}' in line:
                if joint_stack:
                    joint_stack.pop()
            elif 'OFFSET' in line:
                if joint_stack:
                    offset = [float(x) for x in line.split('OFFSET')[1].strip().split()]
                    joint_stack[-1]['offset'] = np.array(offset)
            elif 'CHANNELS' in line:
                if joint_stack:
                    parts = line.split()
                    num_channels = int(parts[1])
                    joint_stack[-1]['channels'] = parts[2:2+num_channels]

    def _parse_motion(self, text: str):
        """Parse MOTION section."""
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
        """Get joint positions for a specific frame."""
        if frame_idx >= len(self.motion_data):
            return {}

        frame_data = self.motion_data[frame_idx]
        positions = {}

        if self.root:
            self._compute_positions(self.root, frame_data, 0,
                                   np.zeros(3), np.identity(3), positions)
        return positions

    def _compute_positions(self, joint: dict, frame_data: list, channel_offset: int,
                          parent_pos: np.ndarray, parent_rot: np.ndarray,
                          positions: dict) -> int:
        """Recursively compute joint positions."""
        joint_pos = parent_pos.copy()
        local_rot = np.identity(3)

        if not joint['end_site'] and joint['channels']:
            # Handle root position
            if joint['parent'] is None:
                for i, ch in enumerate(joint['channels']):
                    if 'Xposition' in ch:
                        joint_pos[0] = frame_data[channel_offset + i]
                    elif 'Yposition' in ch:
                        joint_pos[1] = frame_data[channel_offset + i]
                    elif 'Zposition' in ch:
                        joint_pos[2] = frame_data[channel_offset + i]

            # Handle rotations
            for i, ch in enumerate(joint['channels']):
                if 'rotation' not in ch.lower():
                    continue
                angle = np.radians(frame_data[channel_offset + i])

                if 'Xrotation' in ch:
                    rot = np.array([[1, 0, 0],
                                   [0, np.cos(angle), -np.sin(angle)],
                                   [0, np.sin(angle), np.cos(angle)]])
                elif 'Yrotation' in ch:
                    rot = np.array([[np.cos(angle), 0, np.sin(angle)],
                                   [0, 1, 0],
                                   [-np.sin(angle), 0, np.cos(angle)]])
                elif 'Zrotation' in ch:
                    rot = np.array([[np.cos(angle), -np.sin(angle), 0],
                                   [np.sin(angle), np.cos(angle), 0],
                                   [0, 0, 1]])
                else:
                    rot = np.identity(3)
                local_rot = np.dot(local_rot, rot)

            channel_offset += len(joint['channels'])

        rotation = np.dot(parent_rot, local_rot)
        offset_rotated = np.dot(parent_rot, joint['offset'])
        joint_pos += offset_rotated

        positions[joint['name']] = joint_pos

        for child_name in joint['children']:
            child = self.joints[child_name]
            channel_offset = self._compute_positions(child, frame_data, channel_offset,
                                                    joint_pos, rotation, positions)
        return channel_offset


class AccuracyTester:
    """Main accuracy tester using GPT-4V or Gemini."""

    # MediaPipe pose landmark connections for drawing
    POSE_CONNECTIONS = [
        (0, 1), (1, 2), (2, 3), (3, 7),  # Left face
        (0, 4), (4, 5), (5, 6), (6, 8),  # Right face
        (9, 10),  # Mouth
        (11, 12),  # Shoulders
        (11, 13), (13, 15),  # Left arm
        (12, 14), (14, 16),  # Right arm
        (11, 23), (12, 24),  # Torso
        (23, 24),  # Hips
        (23, 25), (25, 27), (27, 29), (29, 31),  # Left leg
        (24, 26), (26, 28), (28, 30), (30, 32),  # Right leg
        (15, 17), (15, 19), (15, 21),  # Left hand
        (16, 18), (16, 20), (16, 22),  # Right hand
    ]

    def __init__(self, provider: str = "openai", api_key: Optional[str] = None, model: Optional[str] = None):
        """
        Initialize accuracy tester.

        Args:
            provider: "openai" or "gemini"
            api_key: API key (or use environment variable)
            model: Model name override (default: gpt-4o for openai, gemini-2.5-flash for gemini)
        """
        self.provider = provider.lower()
        self.model = model

        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=True,
            model_complexity=2,
            min_detection_confidence=0.5
        )

        self.openai_client = None
        self.gemini_client = None

        if self.provider == "openai":
            key = api_key or os.environ.get('OPENAI_API_KEY')
            if key and HAS_OPENAI:
                self.openai_client = OpenAI(api_key=key)
                self.model = self.model or "gpt-4o"
                print(f"Using OpenAI {self.model}")
            else:
                if not HAS_OPENAI:
                    print("OpenAI package not available. Run: pip install openai")
                elif not key:
                    print("No OPENAI_API_KEY found in environment or .env file")

        elif self.provider == "gemini":
            key = api_key or os.environ.get('GOOGLE_API_KEY')
            if key and HAS_GEMINI:
                self.gemini_client = genai.Client(api_key=key)
                self.model = self.model or "gemini-2.5-flash"
                print(f"Using Google {self.model}")
            else:
                self.gemini_client = None
                if not HAS_GEMINI:
                    print("Google AI package not available. Run: pip install google-genai")
                elif not key:
                    print("No GOOGLE_API_KEY found in environment or .env file")
        else:
            print(f"Unknown provider: {provider}. Use 'openai' or 'gemini'")

    def extract_video_frame(self, video_path: str, frame_idx: int) -> Optional[np.ndarray]:
        """Extract a specific frame from video."""
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        cap.release()
        return frame if ret else None

    def get_video_info(self, video_path: str) -> Tuple[int, float]:
        """Get video frame count and FPS."""
        cap = cv2.VideoCapture(video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        return frame_count, fps

    def draw_mediapipe_overlay(self, frame: np.ndarray) -> Tuple[np.ndarray, bool]:
        """Draw MediaPipe pose overlay on frame. Returns (frame, detection_success)."""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(frame_rgb)

        overlay = frame.copy()

        if results.pose_landmarks:
            h, w = frame.shape[:2]
            landmarks = results.pose_landmarks.landmark

            # Draw connections
            for start_idx, end_idx in self.POSE_CONNECTIONS:
                if start_idx < len(landmarks) and end_idx < len(landmarks):
                    start = landmarks[start_idx]
                    end = landmarks[end_idx]

                    if start.visibility > 0.5 and end.visibility > 0.5:
                        start_pt = (int(start.x * w), int(start.y * h))
                        end_pt = (int(end.x * w), int(end.y * h))
                        cv2.line(overlay, start_pt, end_pt, (0, 255, 0), 2)

            # Draw landmarks
            for lm in landmarks:
                if lm.visibility > 0.5:
                    pt = (int(lm.x * w), int(lm.y * h))
                    cv2.circle(overlay, pt, 4, (0, 0, 255), -1)

            return overlay, True
        return overlay, False

    def render_bvh_frame(self, bvh: BVHParser, frame_idx: int,
                         img_size: Tuple[int, int] = (640, 480)) -> np.ndarray:
        """Render BVH skeleton for a specific frame using OpenCV (fast, matches batch_convert)."""
        positions = bvh.get_frame_positions(frame_idx)

        if not positions:
            return np.ones((img_size[1], img_size[0], 3), dtype=np.uint8) * 255

        # Use fast OpenCV 2D rendering (same as batch_convert.py)
        return self._render_with_opencv(bvh, positions, frame_idx, img_size)

    def _render_with_opencv(self, bvh: BVHParser, positions: Dict[str, np.ndarray],
                            frame_idx: int, img_size: Tuple[int, int]) -> np.ndarray:
        """Render using OpenCV 2D projection (fast, matches batch_convert)."""
        # Get bounds for scaling
        pos_array = np.array(list(positions.values()))
        min_x, min_y = pos_array[:, 0].min(), pos_array[:, 1].min()
        max_x, max_y = pos_array[:, 0].max(), pos_array[:, 1].max()

        # Add padding
        pad = 20
        range_x = max(max_x - min_x, 1)
        range_y = max(max_y - min_y, 1)

        # Scale to fit image while maintaining aspect ratio
        scale = min((img_size[0] - 2*pad) / range_x, (img_size[1] - 2*pad) / range_y)

        # Center offset
        center_x = (min_x + max_x) / 2
        center_y = (min_y + max_y) / 2

        def to_pixel(pos):
            # BVH: positive X = character's left, Y up, Z forward
            # Image: X right, Y down
            # Front view: character's left (positive X) appears on viewer's right
            px = int(img_size[0]/2 + (pos[0] - center_x) * scale)  # No flip - front view
            py = int(img_size[1]/2 - (pos[1] - center_y) * scale)  # Flip Y (up to down)
            return (px, py)

        # Create white background
        img = np.ones((img_size[1], img_size[0], 3), dtype=np.uint8) * 255

        # Draw bones first (so joints appear on top)
        for joint_name, pos in positions.items():
            joint = bvh.joints[joint_name]
            if joint['parent'] and joint['parent'] in positions:
                parent_pos = positions[joint['parent']]
                pt1 = to_pixel(pos)
                pt2 = to_pixel(parent_pos)
                cv2.line(img, pt1, pt2, (0, 0, 255), 2)  # Red lines (BGR)

        # Draw joints
        for joint_name, pos in positions.items():
            pt = to_pixel(pos)
            cv2.circle(img, pt, 4, (255, 0, 0), -1)  # Blue circles (BGR)

        # Add frame label
        cv2.putText(img, f"BVH Frame {frame_idx}", (10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

        return img

    def _render_with_plotly(self, bvh: BVHParser, positions: Dict[str, np.ndarray],
                            frame_idx: int, img_size: Tuple[int, int]) -> np.ndarray:
        """Render using Plotly (same as bvh_viewer for accurate comparison)."""
        # Get all positions as array for bounds calculation
        pos_array = np.array(list(positions.values()))

        # Calculate bounds
        center = pos_array.mean(axis=0)
        max_range = np.max(np.abs(pos_array - center)) * 1.5
        if max_range < 1:
            max_range = 100

        # Extract data for plotting
        points_x, points_y, points_z = [], [], []
        lines_x, lines_y, lines_z = [], [], []

        for joint_name, pos in positions.items():
            # Keep original BVH coordinates (same as bvh_viewer)
            points_x.append(pos[0])
            points_y.append(pos[1])
            points_z.append(pos[2])

            # Add connections to parent
            joint = bvh.joints[joint_name]
            if joint['parent']:
                parent_pos = positions[joint['parent']]
                # Add line segments (separated by None)
                lines_x.extend([pos[0], parent_pos[0], None])
                lines_y.extend([pos[1], parent_pos[1], None])
                lines_z.extend([pos[2], parent_pos[2], None])

        # Create figure
        fig = go.Figure()

        # Add joints (points)
        fig.add_trace(go.Scatter3d(
            x=points_x, y=points_y, z=points_z,
            mode='markers',
            marker=dict(size=6, color='blue'),
            hoverinfo='none',
            showlegend=False
        ))

        # Add skeleton (lines)
        fig.add_trace(go.Scatter3d(
            x=lines_x, y=lines_y, z=lines_z,
            mode='lines',
            line=dict(color='red', width=4),
            hoverinfo='none',
            showlegend=False
        ))

        # Configure layout with front-facing camera (same as bvh_viewer)
        fig.update_layout(
            scene=dict(
                xaxis=dict(range=[center[0] - max_range, center[0] + max_range], title='X'),
                yaxis=dict(range=[center[1] - max_range, center[1] + max_range], title='Y'),
                zaxis=dict(range=[center[2] - max_range, center[2] + max_range], title='Z'),
                aspectmode='cube',
                camera=dict(
                    eye=dict(x=0, y=0, z=-2.5),  # Camera from front (negative Z)
                    center=dict(x=0, y=0, z=0),
                    up=dict(x=0, y=1, z=0)       # Y-axis points up
                )
            ),
            margin=dict(l=0, r=0, b=0, t=40),
            width=img_size[0],
            height=img_size[1],
            title=dict(text=f'BVH Frame {frame_idx}', x=0.5, xanchor='center')
        )

        # Render to image
        img_bytes = pio.to_image(fig, format='png', width=img_size[0], height=img_size[1])
        img_array = np.array(Image.open(BytesIO(img_bytes)))

        # Convert RGBA to RGB if needed
        if img_array.shape[2] == 4:
            img_array = img_array[:, :, :3]

        return img_array

    def _render_with_matplotlib(self, bvh: BVHParser, positions: Dict[str, np.ndarray],
                                 frame_idx: int, img_size: Tuple[int, int]) -> np.ndarray:
        """Fallback rendering using Matplotlib."""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        # Get all positions as array for bounds calculation
        pos_array = np.array(list(positions.values()))

        # Calculate bounds
        center = pos_array.mean(axis=0)
        max_range = np.max(np.abs(pos_array - center)) * 1.5
        if max_range < 1:
            max_range = 100

        # Create matplotlib figure
        fig = plt.figure(figsize=(img_size[0]/100, img_size[1]/100), dpi=100)
        ax = fig.add_subplot(111, projection='3d')

        # Draw skeleton
        for joint_name, pos in positions.items():
            joint = bvh.joints[joint_name]

            # Swap Y and Z for visualization
            plot_pos = [pos[0], pos[2], pos[1]]

            # Draw joint
            ax.scatter([plot_pos[0]], [plot_pos[1]], [plot_pos[2]], c='blue', s=20)

            # Draw bone to parent
            if joint['parent']:
                parent_pos = positions[joint['parent']]
                parent_plot = [parent_pos[0], parent_pos[2], parent_pos[1]]
                ax.plot([plot_pos[0], parent_plot[0]],
                       [plot_pos[1], parent_plot[1]],
                       [plot_pos[2], parent_plot[2]], 'r-', linewidth=2)

        # Set view
        ax.set_xlim(center[0] - max_range, center[0] + max_range)
        ax.set_ylim(center[2] - max_range, center[2] + max_range)
        ax.set_zlim(center[1] - max_range, center[1] + max_range)

        ax.set_xlabel('X')
        ax.set_ylabel('Z')
        ax.set_zlabel('Y')
        ax.set_title(f'BVH Frame {frame_idx}')

        # Render to numpy array
        fig.canvas.draw()
        img = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
        plt.close(fig)

        return img

    def create_comparison_image(self, video_frame: np.ndarray,
                                bvh_render: np.ndarray) -> np.ndarray:
        """Create side-by-side comparison image."""
        # Resize to same height
        target_height = max(video_frame.shape[0], bvh_render.shape[0])

        # Resize video frame
        scale = target_height / video_frame.shape[0]
        video_resized = cv2.resize(video_frame,
                                   (int(video_frame.shape[1] * scale), target_height))

        # Resize BVH render
        scale = target_height / bvh_render.shape[0]
        bvh_resized = cv2.resize(bvh_render,
                                 (int(bvh_render.shape[1] * scale), target_height))

        # Convert BVH from RGB to BGR for consistency
        bvh_resized = cv2.cvtColor(bvh_resized, cv2.COLOR_RGB2BGR)

        # Add labels
        cv2.putText(video_resized, "Video + MediaPipe", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(bvh_resized, "BVH Skeleton", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

        # Concatenate horizontally
        comparison = np.hstack([video_resized, bvh_resized])
        return comparison

    def image_to_base64(self, img: np.ndarray) -> str:
        """Convert numpy image to base64 string."""
        # Convert BGR to RGB if needed
        if len(img.shape) == 3 and img.shape[2] == 3:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img_rgb = img

        pil_img = Image.fromarray(img_rgb)
        buffer = BytesIO()
        pil_img.save(buffer, format='JPEG', quality=85)
        return base64.b64encode(buffer.getvalue()).decode('utf-8')

    def _get_analysis_prompt(self) -> str:
        """Get the prompt for pose analysis."""
        return """Analyze these two images showing the same animation frame:
- LEFT: Original video with green MediaPipe pose detection overlay
- RIGHT: 3D skeleton rendered from BVH file (the converted animation)

Compare the poses and identify specific differences. Focus on:
1. Joint positions that don't match (e.g., "right arm is raised in video but lowered in BVH")
2. Body orientation differences (e.g., "torso is rotated differently")
3. Limb angles that are incorrect (e.g., "elbow bend angle is wrong")

Be specific about which body parts are incorrect and how.
Rate severity: LOW (minor differences), MEDIUM (noticeable errors), HIGH (major pose mismatch)

Respond in this JSON format:
{
    "severity": "LOW|MEDIUM|HIGH",
    "issues": [
        "specific issue 1",
        "specific issue 2"
    ],
    "joints_affected": ["joint1", "joint2"]
}"""

    def _parse_analysis_response(self, raw_response: str, frame_idx: int) -> FrameAnalysis:
        """Parse the API response into FrameAnalysis."""
        try:
            # Find JSON in response
            json_start = raw_response.find('{')
            json_end = raw_response.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                json_str = raw_response[json_start:json_end]
                data = json.loads(json_str)
                return FrameAnalysis(
                    frame_idx=frame_idx,
                    video_frame_idx=frame_idx,
                    issues=data.get('issues', []),
                    severity=data.get('severity', 'unknown'),
                    raw_response=raw_response
                )
        except json.JSONDecodeError:
            pass

        # Fallback: return raw response as single issue
        return FrameAnalysis(
            frame_idx=frame_idx,
            video_frame_idx=frame_idx,
            issues=[raw_response],
            severity="unknown",
            raw_response=raw_response
        )

    def analyze_frame_with_vision(self, comparison_img: np.ndarray,
                                   frame_idx: int) -> FrameAnalysis:
        """Send comparison image to vision API for analysis."""
        if self.provider == "openai":
            return self._analyze_with_openai(comparison_img, frame_idx)
        elif self.provider == "gemini":
            return self._analyze_with_gemini(comparison_img, frame_idx)
        else:
            return FrameAnalysis(
                frame_idx=frame_idx,
                video_frame_idx=frame_idx,
                issues=["No API client configured"],
                severity="error",
                raw_response=""
            )

    def _analyze_with_openai(self, comparison_img: np.ndarray,
                              frame_idx: int) -> FrameAnalysis:
        """Analyze using OpenAI GPT-4V."""
        if not self.openai_client:
            return FrameAnalysis(
                frame_idx=frame_idx,
                video_frame_idx=frame_idx,
                issues=["OpenAI client not available"],
                severity="error",
                raw_response=""
            )

        img_b64 = self.image_to_base64(comparison_img)
        prompt = self._get_analysis_prompt()

        try:
            response = self.openai_client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{img_b64}",
                                    "detail": "high"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=500
            )
            raw_response = response.choices[0].message.content
            return self._parse_analysis_response(raw_response, frame_idx)

        except Exception as e:
            return FrameAnalysis(
                frame_idx=frame_idx,
                video_frame_idx=frame_idx,
                issues=[f"OpenAI API error: {str(e)}"],
                severity="error",
                raw_response=str(e)
            )

    def _analyze_with_gemini(self, comparison_img: np.ndarray,
                              frame_idx: int) -> FrameAnalysis:
        """Analyze using Google Gemini."""
        if not self.gemini_client:
            return FrameAnalysis(
                frame_idx=frame_idx,
                video_frame_idx=frame_idx,
                issues=["Gemini client not available"],
                severity="error",
                raw_response=""
            )

        # Convert to PIL Image for Gemini
        img_rgb = cv2.cvtColor(comparison_img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        prompt = self._get_analysis_prompt()

        try:
            response = self.gemini_client.models.generate_content(
                model=self.model,
                contents=[prompt, pil_img]
            )
            raw_response = response.text
            return self._parse_analysis_response(raw_response, frame_idx)

        except Exception as e:
            return FrameAnalysis(
                frame_idx=frame_idx,
                video_frame_idx=frame_idx,
                issues=[f"Gemini API error: {str(e)}"],
                severity="error",
                raw_response=str(e)
            )

    def analyze_video(self, video_path: str, bvh_path: str,
                      sample_rate: int = 1,
                      max_frames: Optional[int] = None,
                      output_dir: Optional[str] = None,
                      save_comparisons: bool = True) -> AnalysisReport:
        """
        Analyze a video/BVH pair for accuracy.

        Args:
            video_path: Path to source video
            bvh_path: Path to BVH file
            sample_rate: Analyze every Nth BVH frame (1 = all frames)
            max_frames: Maximum number of frames to analyze (None = all)
            output_dir: Directory to save comparison images
            save_comparisons: Whether to save comparison images
        """
        print(f"Loading video: {video_path}")
        video_frame_count, video_fps = self.get_video_info(video_path)
        print(f"  Frames: {video_frame_count}, FPS: {video_fps}")

        print(f"Loading BVH: {bvh_path}")
        bvh = BVHParser()
        bvh.parse_file(bvh_path)
        print(f"  Frames: {bvh.frames}, Frame time: {bvh.frame_time}")

        # Create output directory
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        elif save_comparisons:
            output_dir = Path(bvh_path).stem + "_analysis"
            os.makedirs(output_dir, exist_ok=True)

        # Calculate frame mapping (BVH frame to video frame)
        # BVH was created with sample_rate from video, so BVH frame 0 = video frame 0,
        # BVH frame 1 = video frame sample_rate, etc.
        bvh_sample_rate = 1  # The sample rate used during BVH creation

        report = AnalysisReport(
            video_path=video_path,
            bvh_path=bvh_path,
            frames_analyzed=0
        )

        frames_to_analyze = list(range(0, bvh.frames, sample_rate))
        if max_frames:
            frames_to_analyze = frames_to_analyze[:max_frames]

        print(f"\nAnalyzing {len(frames_to_analyze)} frames...")

        for i, bvh_frame_idx in enumerate(frames_to_analyze):
            # Map BVH frame to video frame
            video_frame_idx = bvh_frame_idx * bvh_sample_rate

            if video_frame_idx >= video_frame_count:
                print(f"  Frame {bvh_frame_idx}: Video frame {video_frame_idx} out of range")
                continue

            print(f"  [{i+1}/{len(frames_to_analyze)}] BVH frame {bvh_frame_idx} -> Video frame {video_frame_idx}")

            # Extract video frame
            video_frame = self.extract_video_frame(video_path, video_frame_idx)
            if video_frame is None:
                print(f"    Failed to extract video frame")
                continue

            # Add MediaPipe overlay
            overlay_frame, mp_success = self.draw_mediapipe_overlay(video_frame)
            if not mp_success:
                print(f"    MediaPipe detection failed")

            # Render BVH frame
            bvh_render = self.render_bvh_frame(bvh, bvh_frame_idx,
                                               (video_frame.shape[1], video_frame.shape[0]))

            # Create comparison
            comparison = self.create_comparison_image(overlay_frame, bvh_render)

            # Save comparison image
            if save_comparisons and output_dir:
                img_path = os.path.join(output_dir, f"frame_{bvh_frame_idx:04d}.jpg")
                cv2.imwrite(img_path, comparison)

            # Analyze with vision API
            if self.openai_client or self.gemini_client:
                analysis = self.analyze_frame_with_vision(comparison, bvh_frame_idx)
                analysis.video_frame_idx = video_frame_idx
                report.frame_analyses.append(analysis)

                # Update issue counts
                for issue in analysis.issues:
                    issue_lower = issue.lower()
                    # Count by body part
                    for part in ['head', 'neck', 'shoulder', 'arm', 'elbow', 'wrist', 'hand',
                                'spine', 'torso', 'hip', 'leg', 'knee', 'ankle', 'foot']:
                        if part in issue_lower:
                            report.issues_by_joint[part] = report.issues_by_joint.get(part, 0) + 1

                print(f"    Severity: {analysis.severity}")
                for issue in analysis.issues[:2]:  # Show first 2 issues
                    print(f"    - {issue[:80]}...")

            report.frames_analyzed += 1

        # Generate summary
        if report.frame_analyses:
            high_count = sum(1 for a in report.frame_analyses if a.severity == 'HIGH')
            med_count = sum(1 for a in report.frame_analyses if a.severity == 'MEDIUM')
            low_count = sum(1 for a in report.frame_analyses if a.severity == 'LOW')

            report.summary = f"""
Analysis Summary:
- Frames analyzed: {report.frames_analyzed}
- HIGH severity: {high_count}
- MEDIUM severity: {med_count}
- LOW severity: {low_count}

Most affected body parts: {sorted(report.issues_by_joint.items(), key=lambda x: -x[1])[:5]}
"""

        # Save report
        if output_dir:
            report_path = os.path.join(output_dir, "analysis_report.json")
            with open(report_path, 'w') as f:
                json.dump({
                    'video_path': report.video_path,
                    'bvh_path': report.bvh_path,
                    'frames_analyzed': report.frames_analyzed,
                    'issues_by_joint': report.issues_by_joint,
                    'summary': report.summary,
                    'frame_analyses': [
                        {
                            'frame_idx': a.frame_idx,
                            'video_frame_idx': a.video_frame_idx,
                            'severity': a.severity,
                            'issues': a.issues
                        }
                        for a in report.frame_analyses
                    ]
                }, f, indent=2)
            print(f"\nReport saved to: {report_path}")

        print(report.summary)
        return report


def main():
    parser = argparse.ArgumentParser(description="BVH Accuracy Tester using Vision AI")
    parser.add_argument("--video", required=True, help="Path to source video")
    parser.add_argument("--bvh", required=True, help="Path to BVH file")
    parser.add_argument("--provider", type=str, default="openai", choices=["openai", "gemini"],
                       help="Vision API provider: 'openai' or 'gemini' (default: openai)")
    parser.add_argument("--model", type=str, default=None,
                       help="Model name (default: gpt-4o for openai, gemini-2.5-flash for gemini)")
    parser.add_argument("--sample-rate", type=int, default=1,
                       help="Analyze every Nth frame (default: 1 = all)")
    parser.add_argument("--max-frames", type=int, default=None,
                       help="Maximum frames to analyze (default: all)")
    parser.add_argument("--output-dir", type=str, default=None,
                       help="Output directory for comparison images")
    parser.add_argument("--no-save", action="store_true",
                       help="Don't save comparison images")
    parser.add_argument("--api-key", type=str, default=None,
                       help="API key (or set OPENAI_API_KEY / GOOGLE_API_KEY env var)")

    args = parser.parse_args()

    tester = AccuracyTester(provider=args.provider, api_key=args.api_key, model=args.model)

    report = tester.analyze_video(
        video_path=args.video,
        bvh_path=args.bvh,
        sample_rate=args.sample_rate,
        max_frames=args.max_frames,
        output_dir=args.output_dir,
        save_comparisons=not args.no_save
    )

    return report


if __name__ == "__main__":
    main()
