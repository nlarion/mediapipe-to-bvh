#!/usr/bin/env python3
"""
Analyze BVH file to check if character is moving through 3D space.
"""

import sys
import numpy as np
import argparse
from pathlib import Path

def analyze_bvh_movement(bvh_file):
    """Analyze spatial movement in a BVH file."""
    
    print(f"\n{'='*60}")
    print(f"BVH Movement Analysis: {Path(bvh_file).name}")
    print(f"{'='*60}\n")
    
    with open(bvh_file, 'r') as f:
        lines = f.readlines()
    
    # Find where motion data starts
    motion_start = -1
    num_frames = 0
    frame_time = 0
    
    for i, line in enumerate(lines):
        if line.strip() == "MOTION":
            motion_start = i
        elif line.startswith("Frames:"):
            num_frames = int(line.split()[1])
        elif line.startswith("Frame Time:"):
            frame_time = float(line.split()[2])
            motion_data_start = i + 1
            break
    
    if motion_start == -1:
        print("ERROR: No MOTION section found in BVH file")
        return
    
    print(f"📊 File Statistics:")
    print(f"  - Total frames: {num_frames}")
    print(f"  - Frame time: {frame_time:.4f}s")
    print(f"  - Total duration: {num_frames * frame_time:.2f}s")
    
    # Extract hip positions (first 3 values of each frame)
    positions = []
    for i in range(motion_data_start, min(motion_data_start + num_frames, len(lines))):
        values = lines[i].strip().split()
        if len(values) >= 3:
            x, y, z = float(values[0]), float(values[1]), float(values[2])
            positions.append([x, y, z])
    
    if not positions:
        print("ERROR: No position data found")
        return
    
    positions = np.array(positions)
    
    # Analyze movement
    print(f"\n📍 Hip Position Analysis:")
    print(f"  First frame: X={positions[0,0]:.3f}, Y={positions[0,1]:.3f}, Z={positions[0,2]:.3f}")
    print(f"  Last frame:  X={positions[-1,0]:.3f}, Y={positions[-1,1]:.3f}, Z={positions[-1,2]:.3f}")
    
    # Calculate total movement
    movement = positions[-1] - positions[0]
    print(f"\n📏 Total Movement:")
    print(f"  X-axis: {movement[0]:.3f} units {'(right)' if movement[0] > 0 else '(left)'}")
    print(f"  Y-axis: {movement[1]:.3f} units {'(up)' if movement[1] > 0 else '(down)'}")
    print(f"  Z-axis: {movement[2]:.3f} units {'(forward)' if movement[2] > 0 else '(backward)'}")
    
    # Calculate distance traveled
    total_distance = np.linalg.norm(movement)
    horizontal_distance = np.linalg.norm([movement[0], movement[2]])
    
    print(f"\n📐 Distance Metrics:")
    print(f"  Total 3D distance: {total_distance:.3f} units")
    print(f"  Horizontal distance: {horizontal_distance:.3f} units")
    
    # Check if actually moving
    is_moving = total_distance > 1.0  # More than 1 unit of movement
    
    # Calculate per-frame movement
    frame_distances = []
    for i in range(1, len(positions)):
        dist = np.linalg.norm(positions[i] - positions[i-1])
        frame_distances.append(dist)
    
    avg_speed = np.mean(frame_distances) if frame_distances else 0
    max_speed = np.max(frame_distances) if frame_distances else 0
    
    print(f"\n⚡ Speed Analysis:")
    print(f"  Average speed: {avg_speed:.4f} units/frame")
    print(f"  Maximum speed: {max_speed:.4f} units/frame")
    print(f"  Average speed: {avg_speed/frame_time:.2f} units/second")
    
    # Analyze each axis separately
    print(f"\n📈 Axis Range:")
    for axis, name in enumerate(['X', 'Y', 'Z']):
        min_val = np.min(positions[:, axis])
        max_val = np.max(positions[:, axis])
        range_val = max_val - min_val
        print(f"  {name}-axis: min={min_val:.3f}, max={max_val:.3f}, range={range_val:.3f}")
    
    # Check for drift or actual movement
    # Calculate cumulative distance vs straight-line distance
    cumulative_dist = sum(frame_distances)
    efficiency = (horizontal_distance / cumulative_dist * 100) if cumulative_dist > 0 else 0
    
    print(f"\n🎯 Movement Quality:")
    print(f"  Cumulative distance: {cumulative_dist:.3f} units")
    print(f"  Movement efficiency: {efficiency:.1f}%")
    print(f"  {'✅' if efficiency > 50 else '⚠️'} {'Direct movement' if efficiency > 50 else 'Wandering/in-place movement'}")
    
    # Final verdict
    print(f"\n{'='*60}")
    if is_moving and horizontal_distance > 10:
        print("✅ CHARACTER IS MOVING THROUGH 3D SPACE")
        print(f"   The character travels {horizontal_distance:.1f} units horizontally")
    elif is_moving:
        print("⚠️  CHARACTER HAS MINIMAL MOVEMENT")
        print(f"   Only {horizontal_distance:.1f} units of horizontal movement detected")
    else:
        print("❌ CHARACTER IS NOT MOVING THROUGH SPACE")
        print("   The character appears to be animating in place")
    print(f"{'='*60}\n")

def main():
    parser = argparse.ArgumentParser(description="Analyze BVH file for spatial movement")
    parser.add_argument("bvh_file", help="Path to BVH file")
    args = parser.parse_args()
    
    if not Path(args.bvh_file).exists():
        print(f"Error: File not found: {args.bvh_file}")
        sys.exit(1)
    
    analyze_bvh_movement(args.bvh_file)

if __name__ == "__main__":
    main()