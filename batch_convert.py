#!/usr/bin/env python3
"""
Batch convert all videos in videos/ directory to BVH format.

Features:
- Convert all .mp4 videos to BVH
- Optional: Generate side-by-side comparison videos (MediaPipe overlay + BVH skeleton)
"""

import os
import sys
import glob
import argparse
import subprocess
from pathlib import Path
from typing import Optional

import numpy as np
import cv2
import mediapipe as mp


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

    def get_frame_positions(self, frame_idx: int):
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
            if joint['parent'] is None:
                for i, ch in enumerate(joint['channels']):
                    if 'Xposition' in ch:
                        joint_pos[0] = frame_data[channel_offset + i]
                    elif 'Yposition' in ch:
                        joint_pos[1] = frame_data[channel_offset + i]
                    elif 'Zposition' in ch:
                        joint_pos[2] = frame_data[channel_offset + i]

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


class ComparisonVideoGenerator:
    """Generate side-by-side comparison videos."""

    POSE_CONNECTIONS = [
        (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
        (11, 23), (12, 24), (23, 24),
        (23, 25), (25, 27), (27, 29), (29, 31),
        (24, 26), (26, 28), (28, 30), (30, 32),
        (15, 17), (15, 19), (15, 21),
        (16, 18), (16, 20), (16, 22),
    ]

    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def draw_mediapipe_overlay(self, frame: np.ndarray) -> np.ndarray:
        """Draw MediaPipe pose overlay on frame."""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(frame_rgb)

        overlay = frame.copy()

        if results.pose_landmarks:
            h, w = frame.shape[:2]
            landmarks = results.pose_landmarks.landmark

            for start_idx, end_idx in self.POSE_CONNECTIONS:
                if start_idx < len(landmarks) and end_idx < len(landmarks):
                    start = landmarks[start_idx]
                    end = landmarks[end_idx]

                    if start.visibility > 0.5 and end.visibility > 0.5:
                        start_pt = (int(start.x * w), int(start.y * h))
                        end_pt = (int(end.x * w), int(end.y * h))
                        cv2.line(overlay, start_pt, end_pt, (0, 255, 0), 2)

            for lm in landmarks:
                if lm.visibility > 0.5:
                    pt = (int(lm.x * w), int(lm.y * h))
                    cv2.circle(overlay, pt, 4, (0, 0, 255), -1)

        return overlay

    def render_bvh_frame(self, bvh: BVHParser, frame_idx: int,
                         img_size: tuple = (640, 480)) -> np.ndarray:
        """Render BVH skeleton for a specific frame using OpenCV (very fast).

        Uses simple 2D projection from front view (X, Y in BVH space).
        """
        positions = bvh.get_frame_positions(frame_idx)

        if not positions:
            return np.ones((img_size[1], img_size[0], 3), dtype=np.uint8) * 255

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
                cv2.line(img, pt1, pt2, (0, 0, 255), 2)  # Red lines

        # Draw joints
        for joint_name, pos in positions.items():
            pt = to_pixel(pos)
            cv2.circle(img, pt, 4, (255, 0, 0), -1)  # Blue circles

        return img

    def generate_comparison_video(self, video_path: str, bvh_path: str,
                                   output_path: str, max_frames: Optional[int] = None):
        """Generate side-by-side comparison video."""
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        bvh = BVHParser()
        bvh.parse_file(bvh_path)

        # Output video is side-by-side, so double width
        out_width = width * 2
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (out_width, height))

        frames_to_process = min(total_frames, bvh.frames)
        if max_frames:
            frames_to_process = min(frames_to_process, max_frames)

        print(f"    Generating comparison video ({frames_to_process} frames)...")

        for frame_idx in range(frames_to_process):
            ret, frame = cap.read()
            if not ret:
                break

            # Left: MediaPipe overlay
            left_frame = self.draw_mediapipe_overlay(frame)
            cv2.putText(left_frame, "MediaPipe", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            # Right: BVH skeleton
            right_frame = self.render_bvh_frame(bvh, frame_idx, (width, height))
            right_frame = cv2.cvtColor(right_frame, cv2.COLOR_RGB2BGR)
            cv2.putText(right_frame, "BVH", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

            # Combine
            combined = np.hstack([left_frame, right_frame])
            out.write(combined)

            # Progress indicator every 30 frames
            if frame_idx % 30 == 0:
                print(f"      Frame {frame_idx}/{frames_to_process}", end='\r')

        cap.release()
        out.release()
        print(f"    Comparison video saved: {output_path}")
        return True

    def close(self):
        self.pose.close()


def batch_convert(generate_comparison: bool = False, comparison_only: bool = False):
    """Convert all .mp4 videos in the 'videos' directory to BVH format.

    Args:
        generate_comparison: Also generate side-by-side comparison videos
        comparison_only: Only generate comparison videos (skip BVH conversion)
    """
    bvh_dir = "bvh"
    comparison_dir = "comparison_videos"

    os.makedirs(bvh_dir, exist_ok=True)
    if generate_comparison or comparison_only:
        os.makedirs(comparison_dir, exist_ok=True)

    video_files = sorted(glob.glob("videos/*.mp4"))

    if not video_files:
        print("No .mp4 files found in 'videos' directory.")
        return

    print(f"Found {len(video_files)} videos to process.")
    if comparison_only:
        print("Mode: Comparison videos only (skipping BVH conversion)")
    elif generate_comparison:
        print("Mode: BVH conversion + comparison videos")
    print("-" * 50)

    success_count = 0
    fail_count = 0

    comparison_gen = None
    if generate_comparison or comparison_only:
        comparison_gen = ComparisonVideoGenerator()

    for video_path in video_files:
        video_name = Path(video_path).stem
        bvh_path = os.path.join(bvh_dir, f"{video_name}.bvh")
        comparison_path = os.path.join(comparison_dir, f"{video_name}_comparison.mp4")

        print(f"\nProcessing: {video_name}...")

        # BVH conversion (unless comparison_only)
        if not comparison_only:
            cmd = [
                sys.executable, "bvh_converter.py",
                "--video", video_path,
                "--output", bvh_path,
            ]

            try:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

                if result.returncode == 0:
                    outliers = ""
                    leveled = ""
                    for line in result.stdout.split('\n'):
                        if "outlier" in line.lower():
                            outliers = line.strip()
                        if "Leveled" in line:
                            leveled = line.strip()

                    if outliers:
                        print(f"  {outliers}")
                    if leveled:
                        print(f"  {leveled}")
                    print(f"  BVH: {bvh_path}")
                    success_count += 1
                else:
                    print(f"  BVH conversion failed: {video_name}")
                    if result.stderr:
                        err_lines = result.stderr.strip().split('\n')[-3:]
                        for line in err_lines:
                            print(f"    {line}")
                    fail_count += 1
                    continue  # Skip comparison if BVH failed

            except subprocess.TimeoutExpired:
                print(f"  Timeout: {video_name} (>10 min)")
                fail_count += 1
                continue
            except Exception as e:
                print(f"  Error: {e}")
                fail_count += 1
                continue

        # Generate comparison video
        if (generate_comparison or comparison_only) and comparison_gen:
            if os.path.exists(bvh_path):
                try:
                    comparison_gen.generate_comparison_video(
                        video_path, bvh_path, comparison_path
                    )
                except Exception as e:
                    print(f"    Comparison video failed: {e}")
            else:
                print(f"    Skipping comparison - BVH not found: {bvh_path}")

    if comparison_gen:
        comparison_gen.close()

    print("-" * 50)
    print(f"\nBatch processing complete!")
    if not comparison_only:
        print(f"  BVH conversions: {success_count}/{len(video_files)}")
        if fail_count > 0:
            print(f"  Failed: {fail_count}")
    if generate_comparison or comparison_only:
        print(f"  Comparison videos saved to: {comparison_dir}/")


def main():
    parser = argparse.ArgumentParser(
        description="Batch convert videos to BVH with optional comparison videos"
    )
    parser.add_argument("--compare", action="store_true",
                       help="Generate side-by-side comparison videos (MediaPipe + BVH)")
    parser.add_argument("--compare-only", action="store_true",
                       help="Only generate comparison videos (skip BVH conversion)")

    args = parser.parse_args()

    batch_convert(
        generate_comparison=args.compare or args.compare_only,
        comparison_only=args.compare_only
    )


if __name__ == "__main__":
    main()
