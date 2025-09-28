#!/usr/bin/env python3
"""
MediaPipe Pose to BVH Converter
------------------------------
This script converts a video file to a BVH motion capture file using MediaPipe's pose estimation.
It extracts 3D pose landmarks from a video and maps them to a skeleton hierarchy for BVH export.

Usage: python mediapipe2bvh.py --input video.mp4 --output motion.bvh [--fps 30] [--smoothing 3]
"""

import os
import sys
import argparse
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2
import math
import time
from typing import List, Dict, Tuple

# Global scale factor for converting MediaPipe coordinates to BVH units
# This will be auto-calculated based on the subject's height in the first frame
GLOBAL_SCALE_FACTOR = 100.0

# BVH structure and export utilities
class Joint:
    def __init__(self, name, offset=(0, 0, 0), parent=None):
        self.name = name
        self.offset = offset
        self.parent = parent
        self.children = []
        self.channels = ["Zrotation", "Xrotation", "Yrotation"]
        self.positions = []
        self.rotations = []
        
    def add_child(self, child):
        self.children.append(child)
        child.parent = self
    
    def __str__(self):
        return f"Joint({self.name}, offset={self.offset}, children={len(self.children)})"

class Skeleton:
    def __init__(self, fps=30):
        self.root = None
        self.joints = {}
        self.frame_time = 1.0 / fps
        self.frames = 0
        
    def create_joint(self, name, offset=(0, 0, 0), parent=None):
        joint = Joint(name, offset, parent)
        self.joints[name] = joint
        if parent:
            self.joints[parent].add_child(joint)
        else:
            self.root = joint
        return joint
    
    def get_joint_by_name(self, name):
        return self.joints.get(name, None)
    
    def write_to_bvh(self, filename):
        """Write the skeleton and motion data to a BVH file."""
        with open(filename, 'w') as f:
            # Write hierarchy section
            f.write("HIERARCHY\n")
            self._write_joint_hierarchy(f, self.root, 0)
            
            # Write motion section
            f.write("MOTION\n")
            f.write(f"Frames: {self.frames}\n")
            f.write(f"Frame Time: {self.frame_time:.6f}\n")
            
            # Write motion data
            self._write_motion_data(f)
    
    def _write_joint_hierarchy(self, f, joint, depth):
        """Recursively write joint hierarchy to BVH file."""
        indent = "  " * depth
        if depth == 0:
            # Root joint
            f.write(f"{indent}ROOT {joint.name}\n")
        else:
            f.write(f"{indent}JOINT {joint.name}\n")
        
        f.write(f"{indent}{{\n")
        f.write(f"{indent}  OFFSET {joint.offset[0]:.6f} {joint.offset[1]:.6f} {joint.offset[2]:.6f}\n")
        
        # Write channels
        if depth == 0:
            # Root has position and rotation
            f.write(f"{indent}  CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation\n")
        else:
            # Other joints have only rotation
            f.write(f"{indent}  CHANNELS 3 Zrotation Xrotation Yrotation\n")
        
        # Process children
        for child in joint.children:
            self._write_joint_hierarchy(f, child, depth+1)
        
        # End site (for joints with no children)
        if not joint.children:
            f.write(f"{indent}  End Site\n")
            f.write(f"{indent}  {{\n")
            f.write(f"{indent}    OFFSET 0.0 0.0 0.0\n")
            f.write(f"{indent}  }}\n")
        
        f.write(f"{indent}}}\n")
    
    def _write_motion_data(self, f):
        """Write motion data for all frames."""
        for frame in range(self.frames):
            frame_data = []
            
            # Add root position and rotation
            if self.root.positions and len(self.root.positions) > frame:
                pos = self.root.positions[frame]
                frame_data.extend([pos[0], pos[1], pos[2]])
            else:
                frame_data.extend([0, 0, 0])
                
            # Add rotations for all joints in traversal order
            self._add_joint_rotations_to_frame(frame_data, self.root, frame)
            
            # Write the frame data
            f.write(" ".join(f"{val:.6f}" for val in frame_data) + "\n")
    
    def _add_joint_rotations_to_frame(self, frame_data, joint, frame):
        """Add joint rotations to frame data recursively."""
        if joint.rotations and len(joint.rotations) > frame:
            rot = joint.rotations[frame]
            frame_data.extend([rot[2], rot[0], rot[1]])  # ZXY order
        else:
            frame_data.extend([0, 0, 0])
            
        for child in joint.children:
            self._add_joint_rotations_to_frame(frame_data, child, frame)


def create_mediapipe_skeleton(fps=30) -> Skeleton:
    """Create a skeleton based on MediaPipe pose landmarks."""
    skeleton = Skeleton(fps)
    
    # Define the skeleton structure with more realistic proportions
    # Using standard T-pose proportions in BVH units (typically cm)
    # Root: Hips
    skeleton.create_joint("Hips", offset=(0, 0, 0))
    
    # Spine chain
    skeleton.create_joint("Spine", offset=(0, 10, 0), parent="Hips")
    skeleton.create_joint("Chest", offset=(0, 15, 0), parent="Spine")
    skeleton.create_joint("Neck", offset=(0, 15, 0), parent="Chest")
    skeleton.create_joint("Head", offset=(0, 10, 0), parent="Neck")
    
    # Left arm chain
    skeleton.create_joint("LeftShoulder", offset=(20, 15, 0), parent="Chest")
    skeleton.create_joint("LeftArm", offset=(15, 0, 0), parent="LeftShoulder")
    skeleton.create_joint("LeftForeArm", offset=(25, 0, 0), parent="LeftArm")
    skeleton.create_joint("LeftHand", offset=(20, 0, 0), parent="LeftForeArm")
    
    # Right arm chain
    skeleton.create_joint("RightShoulder", offset=(-20, 15, 0), parent="Chest")
    skeleton.create_joint("RightArm", offset=(-15, 0, 0), parent="RightShoulder")
    skeleton.create_joint("RightForeArm", offset=(-25, 0, 0), parent="RightArm")
    skeleton.create_joint("RightHand", offset=(-20, 0, 0), parent="RightForeArm")
    
    # Left leg chain
    skeleton.create_joint("LeftUpLeg", offset=(10, -10, 0), parent="Hips")
    skeleton.create_joint("LeftLeg", offset=(0, -45, 0), parent="LeftUpLeg")
    skeleton.create_joint("LeftFoot", offset=(0, -45, 0), parent="LeftLeg")
    skeleton.create_joint("LeftToeBase", offset=(0, 0, 10), parent="LeftFoot")
    
    # Right leg chain
    skeleton.create_joint("RightUpLeg", offset=(-10, -10, 0), parent="Hips")
    skeleton.create_joint("RightLeg", offset=(0, -45, 0), parent="RightUpLeg")
    skeleton.create_joint("RightFoot", offset=(0, -45, 0), parent="RightLeg")
    skeleton.create_joint("RightToeBase", offset=(0, 0, 10), parent="RightFoot")
    
    return skeleton


# MediaPipe landmark to BVH joint mapping
MEDIAPIPE_TO_BVH_MAPPING = {
    0: "Nose",  # Not directly mapped - used for head orientation
    11: "LeftShoulder",
    12: "RightShoulder",
    13: "LeftArm",
    14: "RightArm",
    15: "LeftForeArm",
    16: "RightForeArm",
    17: "LeftHand",
    18: "RightHand",
    23: "Hips",
    24: "RightUpLeg",
    25: "LeftUpLeg",
    26: "RightLeg",
    27: "LeftLeg",
    28: "RightFoot",
    29: "LeftFoot",
    30: "RightToeBase",
    31: "LeftToeBase",
    33: "Spine"  # Approximation
}


def calculate_rotation(vec1, vec2):
    """
    Calculate the rotation matrix that rotates vec1 to vec2.
    Returns rotation in Euler angles (X, Y, Z) in degrees.
    Uses the Gram-Schmidt process for more stable rotation calculation.
    """
    # Handle zero or near-zero length vectors
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    if norm1 < 1e-10 or norm2 < 1e-10:
        return np.array([0, 0, 0])
    
    # Normalize vectors
    vec1 = vec1 / norm1
    vec2 = vec2 / norm2
    
    # Calculate dot product (cosine of angle)
    cos_angle = np.clip(np.dot(vec1, vec2), -1.0, 1.0)
    
    # If vectors are nearly parallel (either same or opposite direction)
    if abs(cos_angle) > 0.9999:
        if cos_angle > 0:
            # Same direction, no rotation needed
            return np.array([0, 0, 0])
        else:
            # Opposite directions - need to find perpendicular axis
            # Use a different approach to find rotation axis for better stability
            if abs(vec1[0]) < abs(vec1[1]) and abs(vec1[0]) < abs(vec1[2]):
                # X is smallest component, rotate around X axis
                axis = np.array([1, 0, 0])
            elif abs(vec1[1]) < abs(vec1[2]):
                # Y is smallest component, rotate around Y axis
                axis = np.array([0, 1, 0])
            else:
                # Z is smallest component, rotate around Z axis
                axis = np.array([0, 0, 1])
                
            # Get perpendicular vector by cross product
            axis = np.cross(vec1, axis)
            axis = axis / np.linalg.norm(axis)
            
            # 180 degree rotation
            angle = np.pi
    else:
        # Calculate rotation axis using cross product
        axis = np.cross(vec1, vec2)
        axis_norm = np.linalg.norm(axis)
        
        if axis_norm < 1e-10:
            # Vectors are parallel, no need for rotation
            return np.array([0, 0, 0])
        
        axis = axis / axis_norm
        
        # Calculate rotation angle
        angle = np.arccos(cos_angle)
    
    # Convert axis-angle to rotation matrix using Rodrigues' formula
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0]
    ])
    
    R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
    
    # Convert to Euler angles (X, Y, Z) in degrees
    # We use ZXY order to match BVH standard
    sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    
    if sy > 1e-6:
        x = np.arctan2(R[2, 1], R[2, 2])
        y = np.arctan2(-R[2, 0], sy)
        z = np.arctan2(R[1, 0], R[0, 0])
    else:
        x = np.arctan2(-R[1, 2], R[1, 1])
        y = np.arctan2(-R[2, 0], sy)
        z = 0
    
    return np.array([np.degrees(x), np.degrees(y), np.degrees(z)])


def process_landmarks_to_skeleton(landmarks, skeleton: Skeleton, frame_idx: int):
    """Process MediaPipe pose landmarks for a frame and update the skeleton."""
    # First, extract 3D positions from landmarks
    positions = {}
    for idx, landmark in enumerate(landmarks.landmark):
        if idx in MEDIAPIPE_TO_BVH_MAPPING:
            # Convert from MediaPipe coordinates to BVH coordinates
            # MediaPipe: +X right, +Y down, +Z forward
            # BVH: +X right, +Y up, +Z forward (typically)
            positions[idx] = np.array([landmark.x, -landmark.y, landmark.z])
    
    # Get initial height from the first frame to use for scaling
    if frame_idx == 0 and 23 in positions and 27 in positions:
        # Calculate height as distance from hip to ankle * 1.2 (to account for feet)
        hip_to_ankle = np.linalg.norm(positions[23] - positions[27])
        # Standard human height in BVH units (typically 170-180 cm)
        target_height = 170.0  
        # Calculate scaling factor to maintain correct proportions
        global GLOBAL_SCALE_FACTOR
        GLOBAL_SCALE_FACTOR = target_height / (hip_to_ankle * 1.2)
        print(f"Auto-calculated scale factor: {GLOBAL_SCALE_FACTOR}")
    
    # Update root position (hips)
    if 23 in positions:
        hip_pos = positions[23]
        hip_joint = skeleton.get_joint_by_name("Hips")
        
        # Scale position to reasonable BVH units using the global scale factor
        # Also apply a fixed scaling that converts MediaPipe normalized coordinates to cm
        hip_pos_scaled = hip_pos * GLOBAL_SCALE_FACTOR
        
        if frame_idx >= len(hip_joint.positions):
            hip_joint.positions.append(hip_pos_scaled)
        else:
            hip_joint.positions[frame_idx] = hip_pos_scaled
    
    # Calculate and update joint rotations
    calculate_joint_rotations(positions, skeleton, frame_idx)
    
    # Update frame count if needed
    skeleton.frames = max(skeleton.frames, frame_idx + 1)


def calculate_joint_rotations(positions, skeleton: Skeleton, frame_idx: int):
    """Calculate joint rotations based on landmark positions."""
    # Define reference vectors for each joint
    reference_vectors = {
        "Hips": np.array([0, 1, 0]),  # Up vector
        "Spine": np.array([0, 1, 0]),
        "Chest": np.array([0, 1, 0]),
        "Neck": np.array([0, 1, 0]),
        "Head": np.array([0, 1, 0]),
        "LeftShoulder": np.array([1, 0, 0]),  # Out to the side
        "RightShoulder": np.array([-1, 0, 0]),
        "LeftArm": np.array([1, 0, 0]),
        "RightArm": np.array([-1, 0, 0]),
        "LeftForeArm": np.array([1, 0, 0]),
        "RightForeArm": np.array([-1, 0, 0]),
        "LeftHand": np.array([1, 0, 0]),
        "RightHand": np.array([-1, 0, 0]),
        "LeftUpLeg": np.array([0, -1, 0]),  # Down
        "RightUpLeg": np.array([0, -1, 0]),
        "LeftLeg": np.array([0, -1, 0]),
        "RightLeg": np.array([0, -1, 0]),
        "LeftFoot": np.array([0, 0, 1]),  # Forward
        "RightFoot": np.array([0, 0, 1]),
        "LeftToeBase": np.array([0, 0, 1]),
        "RightToeBase": np.array([0, 0, 1])
    }
    
    # Define joint pairs to calculate direction vectors
    joint_pairs = {
        "Hips": (23, 33),  # Hips to Spine
        "Spine": (33, 12),  # Spine to RightShoulder (approximation)
        "Chest": (33, 12),  # Same approximation
        "Neck": (33, 0),    # Spine to Nose (approximation)
        "Head": (33, 0),    # Same approximation
        "LeftShoulder": (11, 13),  # LeftShoulder to LeftArm
        "RightShoulder": (12, 14),  # RightShoulder to RightArm
        "LeftArm": (13, 15),  # LeftArm to LeftForeArm
        "RightArm": (14, 16),  # RightArm to RightForeArm
        "LeftForeArm": (15, 17),  # LeftForeArm to LeftHand
        "RightForeArm": (16, 18),  # RightForeArm to RightHand
        "LeftHand": (15, 17),  # Reuse previous vector (limitation)
        "RightHand": (16, 18),  # Reuse previous vector (limitation)
        "LeftUpLeg": (25, 27),  # LeftUpLeg to LeftLeg
        "RightUpLeg": (24, 26),  # RightUpLeg to RightLeg
        "LeftLeg": (27, 29),  # LeftLeg to LeftFoot
        "RightLeg": (26, 28),  # RightLeg to RightFoot
        "LeftFoot": (29, 31),  # LeftFoot to LeftToeBase
        "RightFoot": (28, 30),  # RightFoot to RightToeBase
        "LeftToeBase": (29, 31),  # Reuse previous vector (limitation)
        "RightToeBase": (28, 30)   # Reuse previous vector (limitation)
    }
    
    # Calculate and set rotations for each joint
    for joint_name, (idx1, idx2) in joint_pairs.items():
        if idx1 in positions and idx2 in positions:
            joint = skeleton.get_joint_by_name(joint_name)
            if not joint:
                continue
                
            # Calculate direction vector from joint positions
            direction = positions[idx2] - positions[idx1]
            if np.linalg.norm(direction) < 1e-10:
                continue
                
            # Get reference vector for this joint
            reference = reference_vectors.get(joint_name, np.array([0, 1, 0]))
            
            # Calculate rotation to align reference with direction
            rotation = calculate_rotation(reference, direction)
            
            # Store rotation
            if frame_idx >= len(joint.rotations):
                joint.rotations.append(rotation)
            else:
                joint.rotations[frame_idx] = rotation


def smooth_motion_data(skeleton: Skeleton, window_size=5):
    """Apply a simple moving average filter to smooth the motion data."""
    # Smooth root positions
    if skeleton.root and skeleton.root.positions:
        positions = np.array(skeleton.root.positions)
        smoothed_positions = []
        
        for i in range(len(positions)):
            start = max(0, i - window_size // 2)
            end = min(len(positions), i + window_size // 2 + 1)
            window = positions[start:end]
            smoothed_positions.append(np.mean(window, axis=0))
        
        skeleton.root.positions = smoothed_positions
    
    # Smooth joint rotations
    for joint_name, joint in skeleton.joints.items():
        if joint.rotations:
            rotations = np.array(joint.rotations)
            smoothed_rotations = []
            
            for i in range(len(rotations)):
                start = max(0, i - window_size // 2)
                end = min(len(rotations), i + window_size // 2 + 1)
                window = rotations[start:end]
                smoothed_rotations.append(np.mean(window, axis=0))
            
            joint.rotations = smoothed_rotations


def process_video(input_video, output_bvh, target_fps=30, smoothing_window=5, scale_factor=None):
    """Process video file and convert to BVH."""
    print(f"Processing video: {input_video}")
    print(f"Target output: {output_bvh}")
    
    # Set manual scale factor if provided
    global GLOBAL_SCALE_FACTOR
    if scale_factor is not None:
        GLOBAL_SCALE_FACTOR = scale_factor
        print(f"Using manual scale factor: {GLOBAL_SCALE_FACTOR}")
    
    # Initialize MediaPipe Pose
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=2,  # Use the most accurate model
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    # Open video file
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        print(f"Error: Could not open video file {input_video}")
        return
    
    # Get video properties
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video FPS: {video_fps}, Total frames: {total_frames}")
    print(f"Target FPS for BVH: {target_fps}")
    
    # Create skeleton for BVH
    skeleton = create_mediapipe_skeleton(fps=target_fps)
    
    # Process video frames
    frame_idx = 0
    processed_count = 0
    
    # Calculate frame sampling rate to achieve target FPS
    frame_step = max(1, round(video_fps / target_fps))
    print(f"Processing every {frame_step} frames to achieve {target_fps} FPS")
    
    start_time = time.time()
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process only every frame_step frames
        if frame_idx % frame_step == 0:
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process frame with MediaPipe
            results = pose.process(rgb_frame)
            
            if results.pose_landmarks:
                # Convert to 3D landmarks if available, otherwise use 2D with estimated depth
                landmarks = results.pose_world_landmarks or results.pose_landmarks
                
                # Process landmarks and update skeleton
                process_landmarks_to_skeleton(landmarks, skeleton, processed_count)
                processed_count += 1
            
            # Print progress
            if processed_count % 10 == 0:
                elapsed = time.time() - start_time
                fps = processed_count / elapsed if elapsed > 0 else 0
                print(f"Processed {processed_count} frames ({frame_idx}/{total_frames}, {fps:.2f} FPS)")
        
        frame_idx += 1
    
    cap.release()
    
    print(f"Finished processing {processed_count} poses from {frame_idx} frames")
    
    # Smooth motion data if requested
    if smoothing_window > 0:
        print(f"Applying motion smoothing with window size {smoothing_window}")
        smooth_motion_data(skeleton, window_size=smoothing_window)
    
    # Write BVH file
    print(f"Writing BVH file to {output_bvh}")
    skeleton.write_to_bvh(output_bvh)
    
    print(f"BVH export complete: {output_bvh}")
    print(f"Total frames in BVH: {skeleton.frames}")
    print(f"Duration: {skeleton.frames * skeleton.frame_time:.2f} seconds")

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Convert video to BVH using MediaPipe Pose')
    parser.add_argument('--video', required=True, help='Input video file')
    parser.add_argument('--output', required=True, help='Output BVH file')
    parser.add_argument('--fps', type=int, default=30, help='Target FPS for BVH output')
    parser.add_argument('--smoothing', type=int, default=3, help='Smoothing window size (0 to disable)')
    parser.add_argument('--scale', type=float, help='Manual scale factor for output (default: auto-calculated)')
    
    args = parser.parse_args()
    
    # Process video
    process_video(args.video, args.output, args.fps, args.smoothing, args.scale)


if __name__ == "__main__":
    main()