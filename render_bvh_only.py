#!/usr/bin/env python3
"""
Simple script to render BVH frames without video comparison.
Useful for comparing good vs bad BVH files.
"""

import numpy as np
import cv2
from pathlib import Path
import plotly.graph_objects as go
import plotly.io as pio
from visual_accuracy_tester import BVHParser, CameraView, CAMERA_VIEWS


def render_bvh_frame(parser: BVHParser, frame_idx: int, camera_view: CameraView,
                     width: int = 800, height: int = 800, zoom: float = 1.5,
                     title: str = None):
    """Render a single BVH frame."""
    positions = parser.get_frame_positions(frame_idx)
    if not positions:
        return None

    connections = parser.get_skeleton_connections()

    joint_names = list(positions.keys())
    xs = [positions[name][0] for name in joint_names]
    ys = [positions[name][1] for name in joint_names]
    zs = [positions[name][2] for name in joint_names]

    lines_x, lines_y, lines_z = [], [], []
    for parent, child in connections:
        if parent in positions and child in positions:
            p1, p2 = positions[parent], positions[child]
            lines_x.extend([p1[0], p2[0], None])
            lines_y.extend([p1[1], p2[1], None])
            lines_z.extend([p1[2], p2[2], None])

    all_coords = np.array([[x, y, z] for x, y, z in zip(xs, ys, zs)])
    center = np.mean(all_coords, axis=0)
    max_range = np.max(np.abs(all_coords - center)) * (1.5 / zoom)

    fig = go.Figure()

    fig.add_trace(go.Scatter3d(
        x=xs, y=ys, z=zs,
        mode='markers',
        marker=dict(size=8, color='blue'),
        name='Joints'
    ))

    fig.add_trace(go.Scatter3d(
        x=lines_x, y=lines_y, z=lines_z,
        mode='lines',
        line=dict(color='red', width=6),
        name='Skeleton'
    ))

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
        margin=dict(l=0, r=0, b=0, t=40),
        title=dict(text=title or f"Frame {frame_idx}", x=0.5, font=dict(size=16)),
        showlegend=False,
        width=width,
        height=height
    )

    try:
        img_bytes = pio.to_image(fig, format='png', width=width, height=height)
        img_array = np.frombuffer(img_bytes, dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    except Exception as e:
        print(f"Error rendering: {e}")
        return None


def compare_bvh_files(bvh_path1: str, bvh_path2: str, frame_idx: int,
                      output_path: str, label1: str = "BVH 1", label2: str = "BVH 2"):
    """Render two BVH files side by side for comparison."""
    parser1 = BVHParser()
    parser1.parse_file(bvh_path1)

    parser2 = BVHParser()
    parser2.parse_file(bvh_path2)

    view = CAMERA_VIEWS["front_facing"]

    img1 = render_bvh_frame(parser1, frame_idx, view, title=f"{label1} - Frame {frame_idx}")
    img2 = render_bvh_frame(parser2, frame_idx, view, title=f"{label2} - Frame {frame_idx}")

    if img1 is not None and img2 is not None:
        combined = np.hstack([img1, img2])
        cv2.imwrite(output_path, cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))
        print(f"Saved comparison to {output_path}")
        return combined
    else:
        print("Error rendering one or both BVH files")
        return None


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Compare two BVH files visually")
    parser.add_argument("--bvh1", required=True, help="First BVH file")
    parser.add_argument("--bvh2", required=True, help="Second BVH file")
    parser.add_argument("--frame", type=int, default=0, help="Frame to render")
    parser.add_argument("--output", default="bvh_comparison.png", help="Output image path")
    parser.add_argument("--label1", default="BVH 1", help="Label for first BVH")
    parser.add_argument("--label2", default="BVH 2", help="Label for second BVH")

    args = parser.parse_args()

    compare_bvh_files(args.bvh1, args.bvh2, args.frame, args.output, args.label1, args.label2)
