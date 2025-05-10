#!/usr/bin/env python3
"""
MediaPipe Pose to BVH Converter (Improved Version with Position Tracking)
------------------------------------------------------------------------
This script converts a video file to a BVH motion capture file using MediaPipe's pose estimation.
It extracts 3D pose landmarks from a video and maps them to a skeleton hierarchy for BVH export.
This version properly tracks global position movement across frames.

Features:
- Accurate global position tracking for walking/moving characters
- More accurate joint rotation calculations
- Improved skeleton structure and proportions
- Better handling of occlusions and missing landmarks
- Enhanced motion smoothing options
- Preview options for both video pose detection and BVH animation
- Flexible sampling rates for different frame rates

Usage: python mediapipe_to_bvh.py --video input.mp4 --output motion.bvh [options]
"""

import os
import sys
import argparse
import cv2
import numpy as np
import mediapipe as mp
import math
import time
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm

# MediaPipe setup
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Global scale factor for converting MediaPipe coordinates to BVH units
GLOBAL_SCALE_FACTOR = 100.0

# Global variable to store the accumulated movement tracking
GLOBAL_ROOT_POSITION = np.zeros(3)

@dataclass
class EmptyLandmark:
    """Simple class to substitute for MediaPipe landmarks when needed"""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    visibility: float = 0.0

class Joint:
    """Class representing a joint in the BVH skeleton"""
    def __init__(self, name, offset=(0, 0, 0), parent=None):
        self.name = name
        self.offset = np.array(offset, dtype=float)
        self.parent = parent
        self.children = []
        self.rotation_order = 'XYZ'  # Using XYZ order for better compatibility
        self.channels = ["Xrotation", "Yrotation", "Zrotation"]
        self.positions = []
        self.rotations = []
        
    def add_child(self, child):
        self.children.append(child)
        child.parent = self
    
    def __str__(self):
        return f"Joint({self.name}, offset={self.offset}, children={len(self.children)})"

class Skeleton:
    """Represents a complete skeleton with joints and animation data"""
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
        
        # Ensure non-zero offset for non-root joints
        if joint.parent and np.linalg.norm(joint.offset) < 1e-6:
            # Provide a small default offset to avoid zero-length bones
            if "Left" in joint.name:
                default_offset = np.array([-5.0, 0.0, 1.0])
            elif "Right" in joint.name:
                default_offset = np.array([5.0, 0.0, 1.0])
            else:
                default_offset = np.array([0.0, 5.0, 1.0])
            
            f.write(f"{indent}  OFFSET {default_offset[0]:.6f} {default_offset[1]:.6f} {default_offset[2]:.6f}\n")
            print(f"Warning: Zero-length bone detected for {joint.name}. Using default offset.")
        else:
            f.write(f"{indent}  OFFSET {joint.offset[0]:.6f} {joint.offset[1]:.6f} {joint.offset[2]:.6f}\n")
        
        # Write channels
        if depth == 0:
            # Root has position and rotation
            f.write(f"{indent}  CHANNELS 6 Xposition Yposition Zposition {joint.channels[0]} {joint.channels[1]} {joint.channels[2]}\n")
        else:
            # Other joints have only rotation
            f.write(f"{indent}  CHANNELS 3 {joint.channels[0]} {joint.channels[1]} {joint.channels[2]}\n")
        
        # Process children
        for child in joint.children:
            self._write_joint_hierarchy(f, child, depth+1)
        
        # End Site (for joints with no children)
        if not joint.children:
            f.write(f"{indent}  End Site\n")
            f.write(f"{indent}  {{\n")
            
            # Use anatomically appropriate end site offsets
            if joint.name == "Head":
                # Head end site should point up and slightly forward
                f.write(f"{indent}    OFFSET 0.0 5.0 2.0\n")
            elif joint.name == "LeftHand":
                # Left hand end site extends left and forward
                f.write(f"{indent}    OFFSET -3.0 0.0 2.0\n")
            elif joint.name == "RightHand":
                # Right hand end site extends right and forward
                f.write(f"{indent}    OFFSET 3.0 0.0 2.0\n")
            elif joint.name == "LeftToeBase" or joint.name == "RightToeBase":
                # Toe end sites point forward
                f.write(f"{indent}    OFFSET 0.0 -1.0 3.0\n")
            else:
                # Default end site
                f.write(f"{indent}    OFFSET 0.0 0.0 2.0\n")
            
            f.write(f"{indent}  }}\n")
        
        f.write(f"{indent}}}\n")
    
    def _write_motion_data(self, f):
        """Write motion data for all frames."""
        for frame in range(self.frames):
            frame_data = []
            
            # Add root position
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
            frame_data.extend([rot[0], rot[1], rot[2]])  # XYZ order
        else:
            frame_data.extend([0, 0, 0])
            
        for child in joint.children:
            self._add_joint_rotations_to_frame(frame_data, child, frame)


def create_mediapipe_skeleton(fps=30) -> Skeleton:
    """Create a skeleton based on MediaPipe pose landmarks with anatomically correct proportions."""
    skeleton = Skeleton(fps)
    
    # Define the skeleton structure with realistic proportions
    # Root: Hips
    skeleton.create_joint("Hips", offset=(0, 0, 0))
    
    # Spine chain
    skeleton.create_joint("Spine", offset=(0, 10, 0), parent="Hips")
    skeleton.create_joint("Chest", offset=(0, 15, 0), parent="Spine")
    skeleton.create_joint("Neck", offset=(0, 15, 0), parent="Chest")
    skeleton.create_joint("Head", offset=(0, 10, 0), parent="Neck")
    
    # Left arm chain
    skeleton.create_joint("LeftShoulder", offset=(20, 5, 0), parent="Chest")
    skeleton.create_joint("LeftArm", offset=(15, 0, 0), parent="LeftShoulder")
    skeleton.create_joint("LeftForeArm", offset=(25, 0, 0), parent="LeftArm")
    skeleton.create_joint("LeftHand", offset=(20, 0, 0), parent="LeftForeArm")
    
    # Right arm chain
    skeleton.create_joint("RightShoulder", offset=(-20, 5, 0), parent="Chest")
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


# Map MediaPipe landmarks to our BVH skeleton joints
def get_joint_mapping():
    """Map MediaPipe landmarks to BVH skeleton joints. Using more landmark points for better accuracy."""
    return {
        "Hips": [mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.RIGHT_HIP],
        
        "Spine": [mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.RIGHT_HIP, 
                 mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.RIGHT_SHOULDER],
        
        "Chest": [mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.RIGHT_SHOULDER],
        
        "Neck": [mp_pose.PoseLandmark.LEFT_EAR, mp_pose.PoseLandmark.RIGHT_EAR,
                mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.RIGHT_SHOULDER],
        
        "Head": [mp_pose.PoseLandmark.NOSE, mp_pose.PoseLandmark.LEFT_EYE, 
                mp_pose.PoseLandmark.RIGHT_EYE, mp_pose.PoseLandmark.LEFT_EAR, 
                mp_pose.PoseLandmark.RIGHT_EAR],
        
        "LeftShoulder": [mp_pose.PoseLandmark.LEFT_SHOULDER],
        
        "LeftArm": [mp_pose.PoseLandmark.LEFT_ELBOW],
        
        "LeftForeArm": [mp_pose.PoseLandmark.LEFT_WRIST],
        
        "LeftHand": [mp_pose.PoseLandmark.LEFT_PINKY, mp_pose.PoseLandmark.LEFT_INDEX, 
                    mp_pose.PoseLandmark.LEFT_THUMB],
        
        "RightShoulder": [mp_pose.PoseLandmark.RIGHT_SHOULDER],
        
        "RightArm": [mp_pose.PoseLandmark.RIGHT_ELBOW],
        
        "RightForeArm": [mp_pose.PoseLandmark.RIGHT_WRIST],
        
        "RightHand": [mp_pose.PoseLandmark.RIGHT_PINKY, mp_pose.PoseLandmark.RIGHT_INDEX, 
                     mp_pose.PoseLandmark.RIGHT_THUMB],
        
        "LeftUpLeg": [mp_pose.PoseLandmark.LEFT_HIP],
        
        "LeftLeg": [mp_pose.PoseLandmark.LEFT_KNEE],
        
        "LeftFoot": [mp_pose.PoseLandmark.LEFT_ANKLE],
        
        "LeftToeBase": [mp_pose.PoseLandmark.LEFT_FOOT_INDEX],
        
        "RightUpLeg": [mp_pose.PoseLandmark.RIGHT_HIP],
        
        "RightLeg": [mp_pose.PoseLandmark.RIGHT_KNEE],
        
        "RightFoot": [mp_pose.PoseLandmark.RIGHT_ANKLE],
        
        "RightToeBase": [mp_pose.PoseLandmark.RIGHT_FOOT_INDEX]
    }


def get_landmark_position(landmarks, idx):
    """Safely get the position of a landmark by index"""
    if landmarks and idx < len(landmarks):
        lm = landmarks[idx]
        if hasattr(lm, 'x') and hasattr(lm, 'y') and hasattr(lm, 'z'):
            if not (np.isnan(lm.x) or np.isnan(lm.y) or np.isnan(lm.z)):
                # Convert from MediaPipe coordinates to BVH coordinates
                # MediaPipe: +X right, +Y down, +Z forward
                # BVH: +X right, +Y up, +Z forward
                return np.array([lm.x, -lm.y, lm.z])
    return None


def get_joint_position(joint_name, landmarks, joint_mapping):
    """Get the average position for a joint from its mapped landmarks"""
    if joint_name not in joint_mapping:
        return None
        
    indices = joint_mapping[joint_name]
    positions = []
    
    for idx in indices:
        pos = get_landmark_position(landmarks, idx)
        if pos is not None:
            positions.append(pos)
    
    if positions:
        return np.mean(positions, axis=0)
    else:
        return None


def ensure_minimum_offset(offset_vector, min_length=0.05, default_direction=None):
    """Ensure offset vector has at least minimum length"""
    length = np.linalg.norm(offset_vector)
    
    if length < min_length:
        # If vector is too short, use default direction or normalize and scale existing vector
        if default_direction is not None:
            # Use the default direction, normalized
            direction = default_direction / np.linalg.norm(default_direction)
        elif length > 1e-10:
            # Use the existing direction, but normalize it
            direction = offset_vector / length
        else:
            # If zero vector and no default, use up direction
            direction = np.array([0.0, 1.0, 0.0])
            
        # Scale to minimum length
        return direction * min_length
    else:
        return offset_vector


def get_default_bone_offsets():
    """Define default offsets for bones when landmarks aren't reliable"""
    return {
        "Hips": np.array([0.0, 0.0, 0.0]),  # Root has no offset
        "Spine": np.array([0.0, 0.15, 0.0]),  # Up from hips
        "Chest": np.array([0.0, 0.15, 0.0]),  # Up from spine
        "Neck": np.array([0.0, 0.1, 0.0]),    # Up from chest
        "Head": np.array([0.0, 0.1, 0.0]),    # Up from neck
        
        "LeftShoulder": np.array([0.1, 0.0, 0.0]),   # Left from chest
        "LeftArm": np.array([0.15, 0.0, 0.0]),      # Left from shoulder
        "LeftForeArm": np.array([0.15, 0.0, 0.0]),  # Left from elbow
        "LeftHand": np.array([0.1, 0.0, 0.0]),      # Left from wrist
        
        "RightShoulder": np.array([-0.1, 0.0, 0.0]),   # Right from chest
        "RightArm": np.array([-0.15, 0.0, 0.0]),      # Right from shoulder
        "RightForeArm": np.array([-0.15, 0.0, 0.0]),  # Right from elbow
        "RightHand": np.array([-0.1, 0.0, 0.0]),      # Right from wrist
        
        "LeftUpLeg": np.array([0.1, -0.1, 0.0]),    # Down left from hips
        "LeftLeg": np.array([0.0, -0.25, 0.0]),      # Down from left hip
        "LeftFoot": np.array([0.0, -0.25, 0.0]),     # Down from left knee
        "LeftToeBase": np.array([0.0, 0.0, 0.1]),    # Forward from left ankle
        
        "RightUpLeg": np.array([-0.1, -0.1, 0.0]),    # Down right from hips
        "RightLeg": np.array([0.0, -0.25, 0.0]),     # Down from right hip
        "RightFoot": np.array([0.0, -0.25, 0.0]),    # Down from right knee
        "RightToeBase": np.array([0.0, 0.0, 0.1])     # Forward from right ankle
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
    # We use XYZ order to match BVH standard
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


def axis_angle_to_euler(axis_angle, order='XYZ'):
    """Convert axis-angle rotation to Euler angles"""
    angle = np.linalg.norm(axis_angle)
    
    if angle < 1e-10:
        return np.zeros(3)
    
    axis = axis_angle / angle
    
    # Convert to rotation matrix
    c = math.cos(angle)
    s = math.sin(angle)
    t = 1.0 - c
    x, y, z = axis
    
    # Compute rotation matrix
    R = np.array([
        [t*x*x + c, t*x*y - z*s, t*x*z + y*s],
        [t*x*y + z*s, t*y*y + c, t*y*z - x*s],
        [t*x*z - y*s, t*y*z + x*s, t*z*z + c]
    ])
    
    # Convert rotation matrix to Euler angles
    if order == 'XYZ':
        # Check for gimbal lock
        sy = np.sqrt(R[0,0]**2 + R[1,0]**2)
        
        if sy > 1e-6:
            x = np.arctan2(R[2,1], R[2,2])
            y = np.arctan2(-R[2,0], sy)
            z = np.arctan2(R[1,0], R[0,0])
        else:
            x = np.arctan2(-R[1,2], R[1,1])
            y = np.arctan2(-R[2,0], sy)
            z = 0
    else:
        # Default fallback for unsupported orders
        return np.zeros(3)
    
    return np.array([np.degrees(x), np.degrees(y), np.degrees(z)])


def euler_to_rotation_matrix(euler_angles, order='XYZ'):
    """Convert euler angles to rotation matrix"""
    # Unpack angles (in radians)
    if isinstance(euler_angles, (list, tuple, np.ndarray)):
        if len(euler_angles) >= 3:
            x, y, z = euler_angles[:3]
        else:
            # Handle incomplete euler angles
            missing = 3 - len(euler_angles)
            euler_angles = list(euler_angles) + [0] * missing
            x, y, z = euler_angles
    else:
        # If not a sequence, use same angle for all axes
        x = y = z = euler_angles
    
    # Calculate rotation matrices for each axis
    Rx = np.array([
        [1, 0, 0],
        [0, math.cos(x), -math.sin(x)],
        [0, math.sin(x), math.cos(x)]
    ])
    
    Ry = np.array([
        [math.cos(y), 0, math.sin(y)],
        [0, 1, 0],
        [-math.sin(y), 0, math.cos(y)]
    ])
    
    Rz = np.array([
        [math.cos(z), -math.sin(z), 0],
        [math.sin(z), math.cos(z), 0],
        [0, 0, 1]
    ])
    
    # Combine rotation matrices based on order
    if order == 'XYZ':
        R = Rz @ Ry @ Rx
    elif order == 'ZYX':
        R = Rx @ Ry @ Rz
    else:
        # Default order if unspecified
        R = Rz @ Ry @ Rx
    
    return R


def smooth_motion_data(joint_positions, joint_rotations, window_size=5):
    """
    Apply smoothing to position and rotation data
    Uses a weighted Gaussian-like window for better results
    """
    # Create a gaussian-like weighting window
    sigma = window_size / 3.0
    weights = np.array([np.exp(-(x - window_size//2)**2 / (2 * sigma**2)) 
                      for x in range(window_size)])
    weights = weights / np.sum(weights)  # Normalize weights
    
    smoothed_positions = {}
    smoothed_rotations = {}
    
    # Smooth positions
    for joint_name, positions in joint_positions.items():
        if not positions:
            smoothed_positions[joint_name] = positions
            continue
            
        positions_array = np.array(positions)
        smoothed = np.zeros_like(positions_array)
        
        for i in range(len(positions_array)):
            # Calculate window bounds
            start = max(0, i - window_size // 2)
            end = min(len(positions_array), i + window_size // 2 + 1)
            
            # Adjust weights if window is truncated
            if end - start < window_size:
                actual_weights = weights[max(0, window_size//2 - i):min(window_size, window_size//2 + (len(positions_array) - i))]
                actual_weights = actual_weights / np.sum(actual_weights)  # Renormalize
            else:
                actual_weights = weights
                
            # Apply weighted average
            window_data = positions_array[start:end]
            weighted_sum = np.zeros(3)
            
            for j in range(len(window_data)):
                weighted_sum += window_data[j] * actual_weights[j]
                
            smoothed[i] = weighted_sum
        
        smoothed_positions[joint_name] = smoothed.tolist()
    
    # Smooth rotations (same approach)
    for joint_name, rotations in joint_rotations.items():
        if not rotations:
            smoothed_rotations[joint_name] = rotations
            continue
            
        rotations_array = np.array(rotations)
        smoothed = np.zeros_like(rotations_array)
        
        for i in range(len(rotations_array)):
            start = max(0, i - window_size // 2)
            end = min(len(rotations_array), i + window_size // 2 + 1)
            
            if end - start < window_size:
                actual_weights = weights[max(0, window_size//2 - i):min(window_size, window_size//2 + (len(rotations_array) - i))]
                actual_weights = actual_weights / np.sum(actual_weights)
            else:
                actual_weights = weights
                
            window_data = rotations_array[start:end]
            weighted_sum = np.zeros(3)
            
            for j in range(len(window_data)):
                weighted_sum += window_data[j] * actual_weights[j]
                
            smoothed[i] = weighted_sum
            
        smoothed_rotations[joint_name] = smoothed.tolist()
        
    return smoothed_positions, smoothed_rotations


def calculate_bone_directions(skeleton):
    """Calculate unit direction vectors for all bones in the skeleton"""
    directions = {}
    
    def process_joint(joint):
        if joint.children:
            for child in joint.children:
                if np.linalg.norm(child.offset) > 0:
                    # Calculate unit direction vector from joint to child
                    direction = child.offset / np.linalg.norm(child.offset)
                    directions[(joint.name, child.name)] = direction
                process_joint(child)
    
    process_joint(skeleton)
    return directions


def build_skeleton_from_landmarks(frame_landmarks, joint_mapping, scale=100.0):
    """Build the skeleton structure and set offsets based on landmark positions"""
    skeleton = create_mediapipe_skeleton()
    default_offsets = get_default_bone_offsets()
    
    # Helper function to recursively process joints
    def process_joint(joint):
        # Get joint position
        joint_pos = get_joint_position(joint.name, frame_landmarks, joint_mapping)
        
        # If we have a parent, calculate offset from parent
        if joint.parent:
            parent_pos = get_joint_position(joint.parent.name, frame_landmarks, joint_mapping)
            
            if joint_pos is not None and parent_pos is not None:
                # Calculate offset from parent to this joint
                offset = joint_pos - parent_pos
                
                # For leg joints, ensure forward orientation
                if joint.name in ["LeftLeg", "RightLeg", "LeftFoot", "RightFoot"]:
                    # Add a small forward component if close to zero
                    if abs(offset[2]) < 0.02:
                        offset[2] = 0.02  # Small forward component to prevent 180° flip
                
                # Ensure minimum offset length to avoid zero-length bones
                default_dir = default_offsets.get(joint.name, np.array([0.0, 1.0, 0.0]))
                offset = ensure_minimum_offset(offset, min_length=0.05, default_direction=default_dir)
                
                joint.offset = offset * scale
            else:
                # Use default offsets if we can't determine from landmarks
                joint.offset = default_offsets.get(joint.name, np.array([0.0, 1.0, 0.0])) * scale
        else:
            # Root joint has offset from origin
            joint.offset = np.array([0.0, 0.0, 0.0])  # Root at origin
        
        # Special handling for problematic joints that often have zero length
        if joint.name == "Head":
            # Ensure head extends upward from neck
            if np.linalg.norm(joint.offset) < 5.0:
                joint.offset = np.array([0.0, 10.0, 0.0])
                
        elif joint.name == "LeftHand" or joint.name == "RightHand":
            # Ensure hands extend from forearms
            if np.linalg.norm(joint.offset) < 5.0:
                direction = -1.0 if "Left" in joint.name else 1.0
                joint.offset = np.array([direction * 8.0, 0.0, 0.0])
                
        elif joint.name == "LeftFoot" or joint.name == "RightFoot":
            # Ensure feet extend from ankles with forward orientation
            if np.linalg.norm(joint.offset) < 5.0:
                joint.offset = np.array([0.0, -8.0, 2.0])  # Down and slightly forward
            elif joint.offset[2] < 0:  # If pointing backward
                # Flip Z to point forward
                joint.offset[2] = abs(joint.offset[2])
                
        elif joint.name == "LeftToeBase" or joint.name == "RightToeBase":
            # Ensure toes extend from feet
            if np.linalg.norm(joint.offset) < 5.0:
                joint.offset = np.array([0.0, -2.0, 8.0])  # Forward from foot
        
        # Process all children
        for child in joint.children:
            process_joint(child)
    
    # Start processing from root
    process_joint(skeleton.root)
    return skeleton


def calculate_joint_rotations(landmarks, skeleton, joint_mapping, frame_idx):
    """Calculate joint rotations for each joint in the skeleton based on landmark positions"""
    # Define reference vectors for each joint in rest pose
    reference_vectors = {
        "Hips": np.array([0, 1, 0]),  # Up vector
        "Spine": np.array([0, 1, 0]),
        "Chest": np.array([0, 1, 0]),
        "Neck": np.array([0, 1, 0]),
        "Head": np.array([0, 1, 0]),
        "LeftShoulder": np.array([1, 0, 0]),  # Out to the left
        "RightShoulder": np.array([-1, 0, 0]), # Out to the right
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
        "Hips": (mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.RIGHT_HIP),  # Width axis for hips
        "Spine": (mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.LEFT_SHOULDER),  # Up from hip to shoulder
        "Chest": (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.RIGHT_SHOULDER),  # Chest width axis
        "Neck": (mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.NOSE),  # Up to head
        "Head": (mp_pose.PoseLandmark.LEFT_EAR, mp_pose.PoseLandmark.RIGHT_EAR),  # Head orientation
        "LeftShoulder": (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_ELBOW),
        "RightShoulder": (mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_ELBOW),
        "LeftArm": (mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.LEFT_WRIST),
        "RightArm": (mp_pose.PoseLandmark.RIGHT_ELBOW, mp_pose.PoseLandmark.RIGHT_WRIST),
        "LeftForeArm": (mp_pose.PoseLandmark.LEFT_WRIST, mp_pose.PoseLandmark.LEFT_INDEX),
        "RightForeArm": (mp_pose.PoseLandmark.RIGHT_WRIST, mp_pose.PoseLandmark.RIGHT_INDEX),
        "LeftHand": (mp_pose.PoseLandmark.LEFT_WRIST, mp_pose.PoseLandmark.LEFT_INDEX),
        "RightHand": (mp_pose.PoseLandmark.RIGHT_WRIST, mp_pose.PoseLandmark.RIGHT_INDEX),
        "LeftUpLeg": (mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.LEFT_KNEE),
        "RightUpLeg": (mp_pose.PoseLandmark.RIGHT_HIP, mp_pose.PoseLandmark.RIGHT_KNEE),
        "LeftLeg": (mp_pose.PoseLandmark.LEFT_KNEE, mp_pose.PoseLandmark.LEFT_ANKLE),
        "RightLeg": (mp_pose.PoseLandmark.RIGHT_KNEE, mp_pose.PoseLandmark.RIGHT_ANKLE),
        "LeftFoot": (mp_pose.PoseLandmark.LEFT_ANKLE, mp_pose.PoseLandmark.LEFT_FOOT_INDEX),
        "RightFoot": (mp_pose.PoseLandmark.RIGHT_ANKLE, mp_pose.PoseLandmark.RIGHT_FOOT_INDEX),
        "LeftToeBase": (mp_pose.PoseLandmark.LEFT_HEEL, mp_pose.PoseLandmark.LEFT_FOOT_INDEX),
        "RightToeBase": (mp_pose.PoseLandmark.RIGHT_HEEL, mp_pose.PoseLandmark.RIGHT_FOOT_INDEX)
    }

    # Additional pairs for calculating forward vectors
    forward_pairs = {
        "Hips": (mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.LEFT_KNEE),
        "Spine": (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.NOSE),
        "Chest": (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.NOSE),
        "Neck": (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.NOSE),
        "Head": (mp_pose.PoseLandmark.LEFT_EAR, mp_pose.PoseLandmark.NOSE)
    }
    
    # Calculate and set rotations for each joint
    for joint_name in skeleton.joints:
        joint = skeleton.get_joint_by_name(joint_name)
        
        if joint_name in joint_pairs:
            idx1, idx2 = joint_pairs[joint_name]
            
            # Get landmark positions for this joint pair
            pos1 = get_landmark_position(landmarks, idx1)
            pos2 = get_landmark_position(landmarks, idx2)
            
            if pos1 is not None and pos2 is not None:
                # Calculate direction vector from joint landmarks
                direction = pos2 - pos1
                
                if np.linalg.norm(direction) > 1e-10:
                    # Normalize direction
                    direction = direction / np.linalg.norm(direction)
                    
                    # Get reference vector for this joint
                    reference = reference_vectors.get(joint_name, np.array([0, 1, 0]))
                    
                    # For certain joints, we need to understand forward direction too
                    if joint_name in forward_pairs:
                        fwd_idx1, fwd_idx2 = forward_pairs[joint_name]
                        fwd_pos1 = get_landmark_position(landmarks, fwd_idx1)
                        fwd_pos2 = get_landmark_position(landmarks, fwd_idx2)
                        
                        if fwd_pos1 is not None and fwd_pos2 is not None:
                            # Use forward direction to enhance rotation calculation
                            forward = fwd_pos2 - fwd_pos1
                            if np.linalg.norm(forward) > 1e-10:
                                forward = forward / np.linalg.norm(forward)
                                
                                # Ensure perpendicular vectors to properly define a coordinate frame
                                if joint_name in ["Hips", "Spine", "Chest"]:
                                    # For torso, use cross product to get perpendicular vector
                                    side = np.cross(forward, np.array([0, 1, 0]))
                                    if np.linalg.norm(side) > 1e-10:
                                        side = side / np.linalg.norm(side)
                                        up = np.cross(side, forward)
                                        if np.linalg.norm(up) > 1e-10:
                                            direction = side  # Use side vector for better orientation
                    
                    # Apply special handling for specific joints
                    if joint_name == "Head":
                        # Make sure head is oriented correctly
                        if direction[1] < 0:  # If head is pointing down
                            direction[1] = -direction[1]  # Flip Y component
                            
                    elif joint_name in ["LeftHand", "RightHand"]:
                        # Hands should extend out from arms
                        if "Left" in joint_name and direction[0] > 0:
                            direction[0] = -abs(direction[0])  # Left hand should go left
                        elif "Right" in joint_name and direction[0] < 0:
                            direction[0] = abs(direction[0])   # Right hand should go right
                            
                    elif joint_name in ["LeftFoot", "RightFoot", "LeftToeBase", "RightToeBase"]:
                        # Feet should point forward (positive Z)
                        if direction[2] < 0 and abs(direction[2]) > 0.1:
                            direction[2] = abs(direction[2])
                    
                    # Calculate rotation using our enhanced algorithm
                    rotation = calculate_rotation(reference, direction)
                    
                    # Store rotation for this frame
                    if frame_idx >= len(joint.rotations):
                        joint.rotations.append(rotation)
                    else:
                        joint.rotations[frame_idx] = rotation
                else:
                    # Default to zero rotation if can't calculate
                    if frame_idx >= len(joint.rotations):
                        joint.rotations.append(np.zeros(3))
                    else:
                        joint.rotations[frame_idx] = np.zeros(3)
            else:
                # Use previous frame's rotation or zero if this is the first frame
                if frame_idx > 0 and len(joint.rotations) > 0:
                    prev_rotation = joint.rotations[frame_idx - 1]
                    if frame_idx >= len(joint.rotations):
                        joint.rotations.append(prev_rotation)
                    else:
                        joint.rotations[frame_idx] = prev_rotation
                else:
                    if frame_idx >= len(joint.rotations):
                        joint.rotations.append(np.zeros(3))
                    else:
                        joint.rotations[frame_idx] = np.zeros(3)


def calculate_global_position_delta(current_positions, previous_positions, frame_width, frame_height, global_scale_factor):
    """
    Calculate the change in global position between frames.
    This version aims for more consistent scaling and component contribution.
    """
    key_landmarks = [
        mp_pose.PoseLandmark.NOSE,
        mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.RIGHT_SHOULDER,
        mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.RIGHT_HIP
    ]

    if not previous_positions:
        return np.zeros(3)

    current_center_sum = np.zeros(3)
    previous_center_sum = np.zeros(3)
    num_current = 0
    num_previous = 0

    for lm_idx in key_landmarks:
        # Ensure landmark exists and is not None before using
        if lm_idx in current_positions and current_positions[lm_idx] is not None:
            current_center_sum += current_positions[lm_idx]
            num_current += 1
        if lm_idx in previous_positions and previous_positions[lm_idx] is not None:
            previous_center_sum += previous_positions[lm_idx]
            num_previous += 1

    if num_current == 0 or num_previous == 0:
        return np.zeros(3)

    current_center = current_center_sum / num_current
    previous_center = previous_center_sum / num_previous

    # 1. Base delta from body center (in world units, e.g., meters)
    delta_body_center_world = current_center - previous_center
    final_delta_world = np.copy(delta_body_center_world)

    # 2. Z-axis movement based on apparent size change (optional alternative for Z)
    # This factor determines how much size change translates to Z movement. Tune as needed.
    z_sensitivity_factor_for_size_change = 5.0
    calculated_z_from_size = False

    current_width = 0
    if (mp_pose.PoseLandmark.LEFT_SHOULDER in current_positions and current_positions[mp_pose.PoseLandmark.LEFT_SHOULDER] is not None and
        mp_pose.PoseLandmark.RIGHT_SHOULDER in current_positions and current_positions[mp_pose.PoseLandmark.RIGHT_SHOULDER] is not None):
        current_width = np.linalg.norm(
            current_positions[mp_pose.PoseLandmark.LEFT_SHOULDER] -
            current_positions[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        )

    previous_width = 0
    if (mp_pose.PoseLandmark.LEFT_SHOULDER in previous_positions and previous_positions[mp_pose.PoseLandmark.LEFT_SHOULDER] is not None and
        mp_pose.PoseLandmark.RIGHT_SHOULDER in previous_positions and previous_positions[mp_pose.PoseLandmark.RIGHT_SHOULDER] is not None):
        previous_width = np.linalg.norm(
            previous_positions[mp_pose.PoseLandmark.LEFT_SHOULDER] -
            previous_positions[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        )

    if current_width > 1e-5 and previous_width > 1e-5:
        scale_change = current_width / previous_width
        # If character appears bigger (scale_change > 1), they are closer.
        # MediaPipe world Z decreases as subject gets closer.
        # So, a positive (scale_change - 1.0) should result in negative Z movement in world coords.
        z_alternative_from_size_world = - (scale_change - 1.0) * z_sensitivity_factor_for_size_change
        final_delta_world[2] = z_alternative_from_size_world # Overwrites Z from body center
        calculated_z_from_size = True

    # 3. Walking detection from leg movements (adds to final_delta_world)
    foot_movement_world = np.zeros(3)
    # Ensure all required landmarks are present and not None
    required_foot_landmarks = [
        mp_pose.PoseLandmark.LEFT_FOOT_INDEX, mp_pose.PoseLandmark.RIGHT_FOOT_INDEX
    ]
    landmarks_available = True
    for lm_idx in required_foot_landmarks:
        if not (lm_idx in current_positions and current_positions[lm_idx] is not None and \
                lm_idx in previous_positions and previous_positions[lm_idx] is not None):
            landmarks_available = False
            break

    if landmarks_available:
        left_foot_delta_world = (current_positions[mp_pose.PoseLandmark.LEFT_FOOT_INDEX] -
                                 previous_positions[mp_pose.PoseLandmark.LEFT_FOOT_INDEX])
        right_foot_delta_world = (current_positions[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX] -
                                  previous_positions[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX])
        avg_foot_delta_world = (left_foot_delta_world + right_foot_delta_world) / 2.0

        foot_movement_threshold_world = 0.01 # e.g., 1cm if world units are meters
        if np.linalg.norm(avg_foot_delta_world) > foot_movement_threshold_world:
            walking_factor = 2.0  # Tune this amplification factor
            foot_movement_world = avg_foot_delta_world * walking_factor
            final_delta_world += foot_movement_world # Add X, Y, Z components from feet

    # 4. Convert the combined final_delta_world to BVH units
    # The problematic '* 0.1' scaling is removed here.
    # If damping is still needed, consider making it a tunable parameter.
    delta_bvh = final_delta_world * global_scale_factor

    # 5. Facing direction alignment (applied to delta_bvh)
    # Ensure shoulder landmarks are available for facing direction
    if (mp_pose.PoseLandmark.LEFT_SHOULDER in current_positions and current_positions[mp_pose.PoseLandmark.LEFT_SHOULDER] is not None and
        mp_pose.PoseLandmark.RIGHT_SHOULDER in current_positions and current_positions[mp_pose.PoseLandmark.RIGHT_SHOULDER] is not None):
        
        left_shoulder_world = current_positions[mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder_world = current_positions[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        
        shoulder_line_world = right_shoulder_world - left_shoulder_world
        
        # Assuming MediaPipe world coordinates: +X right, +Y down, +Z towards camera (closer).
        # After get_landmark_position: +X right, +Y up (inverted MP Y), +Z towards camera.
        # For BVH-like forward (perpendicular to shoulder, in XZ plane):
        # If Y is up, cross shoulder line with Y-up vector.
        world_y_axis = np.array([0, 1, 0]) # Assuming Y is 'up' in the space of shoulder_line_world
        forward_dir_world = np.cross(shoulder_line_world, world_y_axis)
        
        if np.linalg.norm(forward_dir_world) > 1e-5:
            forward_dir_world_normalized = forward_dir_world / np.linalg.norm(forward_dir_world)
            
            delta_bvh_mag = np.linalg.norm(delta_bvh)
            alignment_threshold_bvh = 0.5 # Tune: only align if movement is significant enough
            if delta_bvh_mag > alignment_threshold_bvh:
                # Project delta_bvh onto the world forward direction
                # Note: forward_dir_world_normalized is a direction, unitless in terms of GSF scaling.
                projected_forward_component_bvh = np.dot(delta_bvh, forward_dir_world_normalized) * forward_dir_world_normalized
                
                alignment_strength = 0.7  # How strongly to align (0 to 1)
                delta_bvh = delta_bvh * (1 - alignment_strength) + projected_forward_component_bvh * alignment_strength
        else: # Fallback if forward_dir cannot be determined (e.g. shoulders aligned vertically)
            pass


    # 6. Limit sudden large movements (capping applied to final delta_bvh)
    max_delta_bvh = 50.0  # Max allowed movement per frame in BVH units. Tune this.
                          # Original was 10.0, which might be too small if global_scale_factor is ~100.
    delta_bvh_mag = np.linalg.norm(delta_bvh)
    if delta_bvh_mag > max_delta_bvh:
        delta_bvh = delta_bvh * (max_delta_bvh / delta_bvh_mag)

    return delta_bvh


def process_landmarks_to_skeleton(landmarks, skeleton, frame_idx, first_frame=False, 
                                  previous_positions=None, frame_dims=None):
    """
    Process MediaPipe pose landmarks for a frame and update the skeleton.
    Now with support for tracking global position across frames.
    """
    global GLOBAL_SCALE_FACTOR, GLOBAL_ROOT_POSITION
    joint_mapping = get_joint_mapping()
    
    # Get landmark positions
    positions = {}
    for idx, landmark in enumerate(landmarks):
        pos = get_landmark_position([landmark], 0)
        if pos is not None:
            positions[idx] = pos
    
    # Get initial height from the first frame to use for scaling
    if first_frame:
        left_hip_idx = mp_pose.PoseLandmark.LEFT_HIP
        left_ankle_idx = mp_pose.PoseLandmark.LEFT_ANKLE
        
        if left_hip_idx in positions and left_ankle_idx in positions:
            # Calculate height as distance from hip to ankle * 1.2 (to account for feet)
            hip_to_ankle = np.linalg.norm(positions[left_hip_idx] - positions[left_ankle_idx])
            # Standard human height in BVH units (typically 170-180 cm)
            target_height = 170.0
            # Calculate scaling factor to maintain correct proportions
            GLOBAL_SCALE_FACTOR = target_height / (hip_to_ankle * 1.2)
            print(f"Auto-calculated scale factor: {GLOBAL_SCALE_FACTOR}")
        
        # Initialize the global position for the first frame
        GLOBAL_ROOT_POSITION = np.zeros(3)
    
    # Calculate position delta if we have previous positions
    if not first_frame and previous_positions and frame_dims:
        frame_width, frame_height = frame_dims
        position_delta = calculate_global_position_delta(positions, previous_positions, 
                                                       frame_width, frame_height, GLOBAL_SCALE_FACTOR)
        
        # Accumulate the global position
        GLOBAL_ROOT_POSITION += position_delta
        print(f"Frame {frame_idx}: Position delta: {position_delta}, Global position: {GLOBAL_ROOT_POSITION}")
    
    # Update root position (hips)
    hip_joint = skeleton.get_joint_by_name("Hips")
    hip_pos = None
    
    # Calculate hip position (average of left and right hip)
    left_hip_idx = mp_pose.PoseLandmark.LEFT_HIP
    right_hip_idx = mp_pose.PoseLandmark.RIGHT_HIP
    
    if left_hip_idx in positions and right_hip_idx in positions:
        hip_pos = (positions[left_hip_idx] + positions[right_hip_idx]) / 2
    elif left_hip_idx in positions:
        hip_pos = positions[left_hip_idx]
    elif right_hip_idx in positions:
        hip_pos = positions[right_hip_idx]
    
    if hip_pos is not None:
        # Use the accumulated global position but keep the y-coordinate from current frame
        # This allows the character to follow terrain and jump/crouch while maintaining proper
        # x and z global position tracking
        root_pos = np.copy(GLOBAL_ROOT_POSITION)
        
        # Scale the current frame's y-position (height) properly
        root_pos[1] = hip_pos[1] * GLOBAL_SCALE_FACTOR
        
        if frame_idx >= len(hip_joint.positions):
            hip_joint.positions.append(root_pos)
        else:
            hip_joint.positions[frame_idx] = root_pos
    else:
        # If no hip landmarks are detected, use previous position or default
        if frame_idx > 0 and hip_joint.positions:
            prev_pos = hip_joint.positions[frame_idx - 1]
            if frame_idx >= len(hip_joint.positions):
                hip_joint.positions.append(prev_pos)
            else:
                hip_joint.positions[frame_idx] = prev_pos
        else:
            # Default to origin if this is the first frame and no hips detected
            if frame_idx >= len(hip_joint.positions):
                hip_joint.positions.append(np.zeros(3))
            else:
                hip_joint.positions[frame_idx] = np.zeros(3)
    
    # Calculate and update joint rotations
    calculate_joint_rotations(landmarks, skeleton, joint_mapping, frame_idx)
    
    # Update frame count
    skeleton.frames = max(skeleton.frames, frame_idx + 1)
    
    # Return the current positions for the next frame to use
    return positions


def process_video(input_video, output_bvh, target_fps=30, smoothing_window=5, scale_factor=None, preview=False):
    """Process video file and convert to BVH."""
    print(f"Processing video: {input_video}")
    print(f"Target output: {output_bvh}")
    
    # Set manual scale factor if provided
    global GLOBAL_SCALE_FACTOR, GLOBAL_ROOT_POSITION
    if scale_factor is not None:
        GLOBAL_SCALE_FACTOR = scale_factor
        print(f"Using manual scale factor: {GLOBAL_SCALE_FACTOR}")
    
    # Initialize global position tracking
    GLOBAL_ROOT_POSITION = np.zeros(3)
    
    # Initialize MediaPipe Pose
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=2,  # Use the most accurate model
        enable_segmentation=False,
        smooth_landmarks=True,
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
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Video properties: {width}x{height}, {video_fps} FPS, {total_frames} frames")
    print(f"Target FPS for BVH: {target_fps}")
    
    # Create preview window if enabled
    if preview:
        cv2.namedWindow('MediaPipe Pose Preview', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('MediaPipe Pose Preview', 800, 600)
    
    # Create skeleton for BVH
    skeleton = create_mediapipe_skeleton(fps=target_fps)
    
    # Process video frames
    frame_idx = 0
    processed_count = 0
    
    # Calculate frame sampling rate to achieve target FPS
    frame_step = max(1, round(video_fps / target_fps))
    print(f"Processing every {frame_step} frames to achieve {target_fps} FPS")
    
    # Store previous frame's landmark positions for tracking movement
    previous_positions = None
    
    start_time = time.time()
    
    with tqdm(total=total_frames) as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process only every frame_step frames
            process_for_bvh = (frame_idx % frame_step == 0)
            
            # Always process for preview display if enabled
            if preview or process_for_bvh:
                # Convert BGR to RGB
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # To improve performance, mark image as not writeable
                rgb_frame.flags.writeable = False
                
                # Process frame with MediaPipe
                results = pose.process(rgb_frame)
                
                # Make image writeable again for drawing
                rgb_frame.flags.writeable = True
                
                if process_for_bvh and results.pose_world_landmarks:
                    # Process landmarks and update skeleton
                    is_first_frame = (processed_count == 0)
                    
                    # Pass frame dimensions for scaling
                    frame_dims = (width, height)
                    
                    # Process landmarks and get current positions for next frame
                    current_positions = process_landmarks_to_skeleton(
                        results.pose_world_landmarks.landmark, 
                        skeleton, 
                        processed_count, 
                        is_first_frame,
                        previous_positions,
                        frame_dims
                    )
                    
                    # Update previous positions for next frame
                    previous_positions = current_positions
                    processed_count += 1
                
                # Draw pose visualization if preview is enabled
                if preview:
                    annotated_frame = frame.copy()
                    if results.pose_landmarks:
                        mp_drawing.draw_landmarks(
                            annotated_frame,
                            results.pose_landmarks,
                            mp_pose.POSE_CONNECTIONS,
                            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                            mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)
                        )
                    
                    # Show frame number and global position
                    cv2.putText(annotated_frame, f"Frame: {frame_idx}/{total_frames}", 
                              (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(annotated_frame, f"Processed: {processed_count}", 
                              (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(annotated_frame, f"Global Pos: X={GLOBAL_ROOT_POSITION[0]:.1f} Y={GLOBAL_ROOT_POSITION[1]:.1f} Z={GLOBAL_ROOT_POSITION[2]:.1f}", 
                              (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    # Display the frame
                    cv2.imshow('MediaPipe Pose Preview', annotated_frame)
                    
                    # Break loop if 'q' is pressed
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        print("Preview stopped by user.")
                        break
            
            frame_idx += 1
            pbar.update(1)
    
    # Clean up
    cap.release()
    if preview:
        cv2.destroyAllWindows()
    
    print(f"Finished processing {processed_count} poses from {frame_idx} frames")
    
    # Smooth motion data if requested
    if smoothing_window > 0 and processed_count > 0:
        print(f"Applying motion smoothing with window size {smoothing_window}")
        
        # Prepare data for smoothing
        joint_positions = {}
        joint_rotations = {}
        
        # Extract positions and rotations for all joints
        for joint_name, joint in skeleton.joints.items():
            if joint_name == "Hips":
                joint_positions[joint_name] = joint.positions
            joint_rotations[joint_name] = joint.rotations
        
        # Apply smoothing
        smoothed_positions, smoothed_rotations = smooth_motion_data(
            joint_positions, joint_rotations, window_size=smoothing_window)
        
        # Update skeleton with smoothed data
        for joint_name, joint in skeleton.joints.items():
            if joint_name == "Hips":
                joint.positions = smoothed_positions[joint_name]
            joint.rotations = smoothed_rotations[joint_name]
    
    # Write BVH file
    print(f"Writing BVH file to {output_bvh}")
    skeleton.write_to_bvh(output_bvh)
    
    print(f"BVH export complete: {output_bvh}")
    print(f"Total frames in BVH: {skeleton.frames}")
    print(f"Duration: {skeleton.frames * skeleton.frame_time:.2f} seconds")
    
    return skeleton


def preview_bvh_animation(bvh_file, speed_factor=1.0):
    """Preview a BVH animation using Matplotlib"""
    try:
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation
        from mpl_toolkits.mplot3d import Axes3D
        import math
        
        # Parse BVH file
        print(f"Loading BVH file: {bvh_file}")
        with open(bvh_file, 'r') as f:
            content = f.read()
        
        # Split into hierarchy and motion sections
        if 'MOTION' in content:
            hierarchy_str, motion_str = content.split('MOTION', 1)
        else:
            print("Error: No MOTION section found in BVH file.")
            return
        
        # Parse joint hierarchy (simplified for preview)
        joints = {}
        joint_order = []
        parent_stack = []
        current_parent = None
        
        for line in hierarchy_str.split('\n'):
            line = line.strip()
            if not line:
                continue
                
            if 'ROOT' in line:
                joint_name = line.split('ROOT')[-1].strip()
                joints[joint_name] = {
                    'parent': None,
                    'children': [],
                    'offset': None,
                    'channels': []
                }
                current_parent = joint_name
                joint_order.append(joint_name)
                
            elif 'JOINT' in line:
                joint_name = line.split('JOINT')[-1].strip()
                joints[joint_name] = {
                    'parent': current_parent,
                    'children': [],
                    'offset': None,
                    'channels': []
                }
                
                if current_parent:
                    joints[current_parent]['children'].append(joint_name)
                    
                joint_order.append(joint_name)
                parent_stack.append(current_parent)
                current_parent = joint_name
                
            elif 'End Site' in line:
                # Skip end sites
                pass
                
            elif 'OFFSET' in line:
                if current_parent and 'End Site' not in line:
                    offset = [float(x) for x in line.split('OFFSET')[-1].strip().split()]
                    joints[current_parent]['offset'] = offset
                    
            elif 'CHANNELS' in line:
                if current_parent:
                    channels = line.split('CHANNELS')[-1].strip().split()
                    num_channels = int(channels[0])
                    channel_names = channels[1:num_channels+1]
                    joints[current_parent]['channels'] = channel_names
                    
            elif '}' in line:
                if parent_stack:
                    current_parent = parent_stack.pop()
                else:
                    current_parent = None
        
        # Parse motion data
        motion_lines = motion_str.strip().split('\n')
        
        num_frames = None
        frame_time = None
        
        # Parse frame count and frame time
        for line in motion_lines[:3]:
            line = line.strip()
            if 'Frames:' in line or 'FRAMES:' in line:
                try:
                    num_frames = int(line.split(':')[-1].strip())
                except ValueError:
                    print(f"Error parsing frame count: {line}")
                    return
            elif 'Frame Time:' in line or 'FRAME TIME:' in line:
                try:
                    frame_time = float(line.split(':')[-1].strip())
                except ValueError:
                    print(f"Error parsing frame time: {line}")
                    return
        
        if num_frames is None or frame_time is None:
            print("Error: Failed to parse frame count or frame time.")
            return
        
        # Find where the motion data starts
        data_start_idx = 0
        for i, line in enumerate(motion_lines):
            if 'Frame Time:' in line or 'FRAME TIME:' in line:
                data_start_idx = i + 1
                break
        
        # Parse motion data values
        motion_data = []
        for i in range(data_start_idx, len(motion_lines)):
            if motion_lines[i].strip():
                try:
                    frame_values = [float(x) for x in motion_lines[i].strip().split()]
                    motion_data.append(frame_values)
                except ValueError as e:
                    print(f"Error parsing motion data line {i}: {e}")
                    continue
        
        print(f"Parsed BVH file: {len(joint_order)} joints, {len(motion_data)} frames")
        
        # Create connections list for visualization
        connections = []
        for joint_name in joint_order:
            joint = joints[joint_name]
            for child_name in joint['children']:
                connections.append((joint_name, child_name))
        
        # Helper function to get positions
        def get_joint_positions(frame_data):
            positions = {}
            rotations = {}
            
            def euler_to_rotation_matrix(euler_angles):
                """Convert euler angles to rotation matrix (XYZ order)"""
                x, y, z = [math.radians(angle) for angle in euler_angles]
                
                # X rotation
                Rx = np.array([
                    [1, 0, 0],
                    [0, math.cos(x), -math.sin(x)],
                    [0, math.sin(x), math.cos(x)]
                ])
                
                # Y rotation
                Ry = np.array([
                    [math.cos(y), 0, math.sin(y)],
                    [0, 1, 0],
                    [-math.sin(y), 0, math.cos(y)]
                ])
                
                # Z rotation
                Rz = np.array([
                    [math.cos(z), -math.sin(z), 0],
                    [math.sin(z), math.cos(z), 0],
                    [0, 0, 1]
                ])
                
                # Combine rotations in XYZ order
                R = Rz @ Ry @ Rx
                return R
            
            def transform_point(point, rotation_matrix):
                """Transform a point using a rotation matrix"""
                return np.matmul(rotation_matrix, point)
            
            # Process joints in hierarchy order
            def process_joint(joint_name, parent_pos, parent_rot):
                joint = joints[joint_name]
                offset = np.array(joint['offset'] if joint['offset'] else [0, 0, 0])
                channels = joint['channels']
                
                # Root joint position comes directly from frame data
                if parent_pos is None:
                    pos_idx = 0
                    if 'Xposition' in channels:
                        pos_x = frame_data[pos_idx]
                        pos_idx += 1
                    else:
                        pos_x = 0
                        
                    if 'Yposition' in channels:
                        pos_y = frame_data[pos_idx]
                        pos_idx += 1
                    else:
                        pos_y = 0
                        
                    if 'Zposition' in channels:
                        pos_z = frame_data[pos_idx]
                        pos_idx += 1
                    else:
                        pos_z = 0
                        
                    global_pos = np.array([pos_x, pos_y, pos_z])
                    rot_idx = pos_idx
                else:
                    # Child joints: position is parent position + offset transformed by parent rotation
                    offset_transformed = transform_point(offset, parent_rot)
                    global_pos = parent_pos + offset_transformed
                    rot_idx = 0
                
                # Extract rotation values
                rot_values = [0, 0, 0]  # Default: no rotation
                rot_order = []
                
                for i, channel in enumerate(channels):
                    if 'rotation' in channel:
                        axis = channel[0]  # X, Y, or Z
                        axis_idx = {'X': 0, 'Y': 1, 'Z': 2}[axis]
                        rot_order.append(axis)
                        
                        # Calculate index in frame data
                        if parent_pos is None:
                            data_idx = rot_idx + i - (len(channels) - 3)
                        else:
                            data_idx = rot_idx + i
                            
                        if data_idx < len(frame_data):
                            rot_values[axis_idx] = frame_data[data_idx]
                
                # Calculate rotation matrix based on rotation values
                rot_matrix = euler_to_rotation_matrix(rot_values)
                
                # Combine with parent rotation
                if parent_rot is not None:
                    global_rot = np.matmul(parent_rot, rot_matrix)
                else:
                    global_rot = rot_matrix
                
                # Store position and rotation
                positions[joint_name] = global_pos
                rotations[joint_name] = global_rot
                
                # Process children recursively
                for child_name in joint['children']:
                    process_joint(child_name, global_pos, global_rot)
            
            # Start from root joint
            root_joint = joint_order[0]
            process_joint(root_joint, None, None)
            
            return positions
        
        # Create figure and 3D axes
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Find plot limits by sampling positions
        all_pos = []
        for frame_idx in range(min(10, len(motion_data))):
            positions = get_joint_positions(motion_data[frame_idx])
            all_pos.extend(list(positions.values()))
        
        all_pos = np.array(all_pos)
        if len(all_pos) > 0:
            min_vals = np.min(all_pos, axis=0)
            max_vals = np.max(all_pos, axis=0)
            
            center = (min_vals + max_vals) / 2
            max_range = max(max_vals[0] - min_vals[0], 
                            max_vals[1] - min_vals[1], 
                            max_vals[2] - min_vals[2]) / 2
            max_range = max(max_range, 50)  # Minimum range
            
            ax.set_xlim(center[0] - max_range, center[0] + max_range)
            ax.set_ylim(center[1] - max_range, center[1] + max_range)
            ax.set_zlim(center[2] - max_range, center[2] + max_range)
        else:
            # Default view if no positions found
            ax.set_xlim(-100, 100)
            ax.set_ylim(-100, 100)
            ax.set_zlim(-100, 100)
        
        # Labels and title
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('BVH Animation Preview')
        
        # Initialize display elements
        lines = [ax.plot([], [], [], 'b-')[0] for _ in connections]
        points = ax.plot([], [], [], 'ro', ms=6)[0]
        frame_text = ax.text2D(0.02, 0.98, '', transform=ax.transAxes)
        
        def update(frame):
            """Update function for animation frames"""
            # Get positions for this frame
            positions = get_joint_positions(motion_data[frame])
            
            # Update bone lines
            for i, (parent, child) in enumerate(connections):
                if parent in positions and child in positions:
                    p_pos = positions[parent]
                    c_pos = positions[child]
                    
                    lines[i].set_data([p_pos[0], c_pos[0]], [p_pos[1], c_pos[1]])
                    lines[i].set_3d_properties([p_pos[2], c_pos[2]])
                else:
                    # If joint is missing, draw empty line
                    lines[i].set_data([], [])
                    lines[i].set_3d_properties([])
            
            # Update joint points
            xs, ys, zs = [], [], []
            for joint in positions:
                pos = positions[joint]
                xs.append(pos[0])
                ys.append(pos[1])
                zs.append(pos[2])
            
            points.set_data(xs, ys)
            points.set_3d_properties(zs)
            
            # Update frame text
            frame_text.set_text(f'Frame: {frame}/{num_frames}')
            
            return lines + [points, frame_text]
        
        # Create animation
        interval = frame_time * 1000 / speed_factor  # Convert to milliseconds
        anim = FuncAnimation(
            fig, update, frames=num_frames, interval=interval, blit=True
        )
        
        plt.tight_layout()
        plt.show()
        
    except ImportError:
        print("Error: This feature requires matplotlib. Please install it with:")
        print("  pip install matplotlib")
    except Exception as e:
        print(f"Error previewing BVH animation: {e}")
        import traceback
        traceback.print_exc()


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Convert video to BVH using MediaPipe Pose')
    parser.add_argument('--video', required=True, help='Input video file')
    parser.add_argument('--output', required=True, help='Output BVH file')
    parser.add_argument('--fps', type=int, default=30, help='Target FPS for BVH output')
    parser.add_argument('--smoothing', type=int, default=5, help='Smoothing window size (0 to disable)')
    parser.add_argument('--scale', type=float, help='Manual scale factor for output (default: auto-calculated)')
    parser.add_argument('--preview', action='store_true', help='Show preview of pose detection during processing')
    parser.add_argument('--preview-bvh', action='store_true', help='Preview the BVH animation after processing')
    parser.add_argument('--speed', type=float, default=1.0, help='Playback speed for BVH preview')
    
    args = parser.parse_args()
    
    # Process video
    start_time = time.time()
    skeleton = process_video(args.video, args.output, args.fps, args.smoothing, args.scale, args.preview)
    elapsed = time.time() - start_time
    
    print(f"Processing completed in {elapsed:.2f} seconds")
    
    # Preview BVH if requested
    if args.preview_bvh:
        preview_bvh_animation(args.output, args.speed)


if __name__ == "__main__":
    main()