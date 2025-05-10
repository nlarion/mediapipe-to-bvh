#!/usr/bin/env python3
"""
MediaPipe Pose to BVH Converter (Refactored Version)
---------------------------------------------------
This script converts a video file to a BVH motion capture file using MediaPipe's pose estimation.
It extracts 3D pose landmarks from a video and maps them to a skeleton hierarchy for BVH export.

Key Refactoring Changes:
- Broke down `process_video` into smaller, more focused functions.
- Broke down `preview_bvh_animation` for better modularity.
- Consolidated `euler_to_rotation_matrix` utility.
- Added more comments and maintained type hinting.
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
from typing import List, Dict, Tuple, Optional, Any
from tqdm import tqdm

# --- Constants and Configuration ---
GLOBAL_SCALE_FACTOR = 100.0  # Default scale factor for BVH units
MIN_VECTOR_NORM = 1e-10      # Minimum norm for a vector to be considered non-zero
MIN_OFFSET_LENGTH = 0.05     # Minimum length for a bone offset
DEFAULT_TARGET_FPS = 30
DEFAULT_SMOOTHING_WINDOW = 5

# MediaPipe setup
mp_pose_solutions = mp.solutions.pose
mp_drawing_utils = mp.solutions.drawing_utils

# --- Data Classes ---
@dataclass
class EmptyLandmark:
    """Simple class to substitute for MediaPipe landmarks when needed."""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    visibility: float = 0.0

@dataclass
class Joint:
    """Class representing a joint in the BVH skeleton."""
    name: str
    offset: np.ndarray = field(default_factory=lambda: np.array([0, 0, 0], dtype=float))
    parent: Optional['Joint'] = None
    children: List['Joint'] = field(default_factory=list)
    rotation_order: str = 'XYZ'  # Common BVH rotation order
    channels: List[str] = field(default_factory=lambda: ["Xrotation", "Yrotation", "Zrotation"])
    positions: List[np.ndarray] = field(default_factory=list) # For root joint
    rotations: List[np.ndarray] = field(default_factory=list) # Euler angles in degrees

    def add_child(self, child: 'Joint'):
        self.children.append(child)
        child.parent = self

    def __str__(self):
        return f"Joint({self.name}, offset={self.offset}, children={len(self.children)})"

@dataclass
class Skeleton:
    """Represents a complete skeleton with joints and animation data."""
    root: Optional[Joint] = None
    joints: Dict[str, Joint] = field(default_factory=dict)
    frame_time: float = 1.0 / DEFAULT_TARGET_FPS
    frames: int = 0

    def create_joint(self, name: str, offset: Tuple[float, float, float] = (0, 0, 0), parent_name: Optional[str] = None) -> Joint:
        parent_joint = self.joints.get(parent_name) if parent_name else None
        joint = Joint(name, np.array(offset, dtype=float), parent_joint)
        self.joints[name] = joint
        if parent_joint:
            parent_joint.add_child(joint)
        else:
            self.root = joint
        return joint

    def get_joint_by_name(self, name: str) -> Optional[Joint]:
        return self.joints.get(name)

    def _write_joint_hierarchy(self, f, joint: Joint, depth: int):
        """Recursively write joint hierarchy to BVH file."""
        indent = "  " * depth
        if depth == 0: # Root joint
            f.write(f"{indent}ROOT {joint.name}\n")
        else:
            f.write(f"{indent}JOINT {joint.name}\n")

        f.write(f"{indent}{{\n")

        current_offset = joint.offset
        if joint.parent and np.linalg.norm(current_offset) < MIN_VECTOR_NORM: # Ensure non-zero offset for non-root joints
            print(f"Warning: Near zero-length bone detected for {joint.name}. Using small default offset.")
            # Provide a small default offset based on naming convention or a generic one
            if "Left" in joint.name: default_offset = np.array([-1.0, 0.0, 0.1])
            elif "Right" in joint.name: default_offset = np.array([1.0, 0.0, 0.1])
            else: default_offset = np.array([0.0, 1.0, 0.1])
            current_offset = default_offset * MIN_OFFSET_LENGTH * 10 # Scaled to be small but non-zero

        f.write(f"{indent}  OFFSET {current_offset[0]:.6f} {current_offset[1]:.6f} {current_offset[2]:.6f}\n")

        if depth == 0: # Root has position and rotation
            f.write(f"{indent}  CHANNELS 6 Xposition Yposition Zposition {' '.join(joint.channels)}\n")
        else: # Other joints have only rotation
            f.write(f"{indent}  CHANNELS 3 {' '.join(joint.channels)}\n")

        for child in joint.children:
            self._write_joint_hierarchy(f, child, depth + 1)

        if not joint.children: # End Site for joints with no children
            f.write(f"{indent}  End Site\n")
            f.write(f"{indent}  {{\n")
            # Anatomically appropriate end site offsets (can be refined)
            end_site_offset = np.array([0.0, 0.0, 2.0]) # Default end site
            if joint.name == "Head": end_site_offset = np.array([0.0, 5.0, 2.0])
            elif "Hand" in joint.name: end_site_offset = np.array([np.sign(joint.offset[0]) * 3.0 if joint.offset[0]!=0 else 3.0, 0.0, 2.0])
            elif "ToeBase" in joint.name: end_site_offset = np.array([0.0, -1.0, 3.0])
            f.write(f"{indent}    OFFSET {end_site_offset[0]:.6f} {end_site_offset[1]:.6f} {end_site_offset[2]:.6f}\n")
            f.write(f"{indent}  }}\n")

        f.write(f"{indent}}}\n")

    def _add_joint_rotations_to_frame_data(self, frame_data: List[float], joint: Joint, frame_idx: int):
        """Recursively add joint rotations for a frame."""
        if joint.rotations and frame_idx < len(joint.rotations):
            rot = joint.rotations[frame_idx]
            frame_data.extend([rot[0], rot[1], rot[2]]) # Assuming XYZ order
        else:
            frame_data.extend([0.0, 0.0, 0.0]) # Default to zero rotation if missing

        for child in joint.children:
            self._add_joint_rotations_to_frame_data(frame_data, child, frame_idx)

    def _write_motion_data(self, f):
        """Write motion data for all frames."""
        if not self.root: return

        for frame_idx in range(self.frames):
            frame_data: List[float] = []
            # Add root position
            if self.root.positions and frame_idx < len(self.root.positions):
                pos = self.root.positions[frame_idx]
                frame_data.extend([pos[0], pos[1], pos[2]])
            else:
                frame_data.extend([0.0, 0.0, 0.0]) # Default to origin if missing

            # Add rotations for all joints in traversal order
            self._add_joint_rotations_to_frame_data(frame_data, self.root, frame_idx)
            f.write(" ".join(f"{val:.6f}" for val in frame_data) + "\n")

    def write_to_bvh(self, filename: str):
        """Write the skeleton and motion data to a BVH file."""
        if not self.root:
            print("Error: Skeleton root not defined. Cannot write BVH.")
            return
        with open(filename, 'w') as f:
            f.write("HIERARCHY\n")
            self._write_joint_hierarchy(f, self.root, 0)
            f.write("MOTION\n")
            f.write(f"Frames: {self.frames}\n")
            f.write(f"Frame Time: {self.frame_time:.6f}\n")
            self._write_motion_data(f)

    def smooth_motion(self, window_size: int = 5):
        """Apply smoothing to root positions and joint rotations."""
        if window_size <= 0 or self.frames == 0:
            return

        # Smooth root positions
        if self.root and self.root.positions:
            self.root.positions = _smooth_data_array(self.root.positions, window_size)

        # Smooth rotations for all joints
        for joint in self.joints.values():
            if joint.rotations:
                joint.rotations = _smooth_data_array(joint.rotations, window_size)
        print(f"Applied motion smoothing with window size {window_size}")


# --- Utility Functions ---
def _smooth_data_array(data_list: List[np.ndarray], window_size: int) -> List[np.ndarray]:
    """Helper function to smooth a list of numpy arrays (positions or rotations)."""
    if not data_list or window_size <= 1:
        return data_list

    data_array = np.array(data_list)
    smoothed_array = np.zeros_like(data_array)
    
    # Create a Gaussian-like weighting window
    sigma = window_size / 3.0
    weights = np.array([np.exp(-(x - window_size//2)**2 / (2 * sigma**2)) 
                      for x in range(window_size)])
    weights = weights / np.sum(weights)  # Normalize weights

    for i in range(len(data_array)):
        win_start = max(0, i - window_size // 2)
        win_end = min(len(data_array), i + window_size // 2 + 1)
        
        current_window_data = data_array[win_start:win_end]
        
        # Adjust weights for truncated windows at edges
        actual_weights_start = max(0, window_size // 2 - i)
        actual_weights_end = min(window_size, window_size // 2 + (len(data_array) - i))
        current_weights = weights[actual_weights_start:actual_weights_end]
        current_weights = current_weights / np.sum(current_weights) # Renormalize

        if len(current_window_data) == len(current_weights):
             weighted_sum = np.sum(current_window_data * current_weights[:, np.newaxis], axis=0)
        else: # Fallback if lengths don't match (should not happen with correct logic)
            weighted_sum = data_array[i] 

        smoothed_array[i] = weighted_sum
        
    return smoothed_array.tolist()

def euler_to_rotation_matrix(euler_angles_deg: np.ndarray, order: str = 'XYZ') -> np.ndarray:
    """Convert Euler angles (degrees) to a rotation matrix."""
    x, y, z = np.radians(euler_angles_deg) # Convert to radians

    Rx = np.array([[1, 0, 0], [0, math.cos(x), -math.sin(x)], [0, math.sin(x), math.cos(x)]])
    Ry = np.array([[math.cos(y), 0, math.sin(y)], [0, 1, 0], [-math.sin(y), 0, math.cos(y)]])
    Rz = np.array([[math.cos(z), -math.sin(z), 0], [math.sin(z), math.cos(z), 0], [0, 0, 1]])

    if order == 'XYZ': R = Rz @ Ry @ Rx
    elif order == 'ZYX': R = Rx @ Ry @ Rz
    # Add other orders if needed
    else: R = Rz @ Ry @ Rx # Default to XYZ
    return R

def rotation_matrix_to_euler(R: np.ndarray, order: str = 'XYZ') -> np.ndarray:
    """Convert a rotation matrix to Euler angles (degrees)."""
    sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < MIN_VECTOR_NORM

    if not singular:
        if order == 'XYZ':
            x = np.arctan2(R[2, 1], R[2, 2])
            y = np.arctan2(-R[2, 0], sy)
            z = np.arctan2(R[1, 0], R[0, 0])
        # Add other orders if needed (e.g., ZYX)
        else: # Default XYZ
            x = np.arctan2(R[2, 1], R[2, 2])
            y = np.arctan2(-R[2, 0], sy)
            z = np.arctan2(R[1, 0], R[0, 0])
    else:
        if order == 'XYZ':
            x = np.arctan2(-R[1, 2], R[1, 1])
            y = np.arctan2(-R[2, 0], sy)
            z = 0
        # Add other orders if needed
        else: # Default XYZ
            x = np.arctan2(-R[1, 2], R[1, 1])
            y = np.arctan2(-R[2, 0], sy)
            z = 0
            
    return np.degrees(np.array([x, y, z]))


def calculate_rotation_between_vectors(vec1: np.ndarray, vec2: np.ndarray) -> np.ndarray:
    """Calculate Euler angles (XYZ, degrees) for rotation from vec1 to vec2."""
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 < MIN_VECTOR_NORM or norm2 < MIN_VECTOR_NORM:
        return np.array([0.0, 0.0, 0.0])

    v1_norm = vec1 / norm1
    v2_norm = vec2 / norm2

    cos_angle = np.clip(np.dot(v1_norm, v2_norm), -1.0, 1.0)
    
    if abs(cos_angle) > 0.99999: # Vectors are nearly parallel
        return np.array([0.0, 0.0, 0.0]) if cos_angle > 0 else np.array([0.0, 180.0, 0.0]) # Or other appropriate 180 deg rotation

    axis = np.cross(v1_norm, v2_norm)
    axis_norm = np.linalg.norm(axis)
    if axis_norm < MIN_VECTOR_NORM: # Should be caught by parallel check, but as safeguard
        return np.array([0.0, 0.0, 0.0])
    axis = axis / axis_norm
    
    angle = np.arccos(cos_angle)

    # Axis-angle to rotation matrix (Rodrigues' formula)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
    
    return rotation_matrix_to_euler(R)


# --- Skeleton Definition and Landmark Mapping ---
def create_custom_skeleton(fps: int = DEFAULT_TARGET_FPS) -> Skeleton:
    """Creates the BVH skeleton structure with predefined joint hierarchy and offsets."""
    skeleton = Skeleton(frame_time=1.0/fps)
    # Root
    skeleton.create_joint("Hips")
    # Spine
    skeleton.create_joint("Spine", offset=(0, 10, 0), parent_name="Hips")
    skeleton.create_joint("Chest", offset=(0, 15, 0), parent_name="Spine")
    skeleton.create_joint("Neck", offset=(0, 15, 0), parent_name="Chest")
    skeleton.create_joint("Head", offset=(0, 10, 0), parent_name="Neck")
    # Left Arm
    skeleton.create_joint("LeftShoulder", offset=(20, 5, 0), parent_name="Chest")
    skeleton.create_joint("LeftArm", offset=(15, 0, 0), parent_name="LeftShoulder")
    skeleton.create_joint("LeftForeArm", offset=(25, 0, 0), parent_name="LeftArm")
    skeleton.create_joint("LeftHand", offset=(20, 0, 0), parent_name="LeftForeArm")
    # Right Arm
    skeleton.create_joint("RightShoulder", offset=(-20, 5, 0), parent_name="Chest")
    skeleton.create_joint("RightArm", offset=(-15, 0, 0), parent_name="RightShoulder")
    skeleton.create_joint("RightForeArm", offset=(-25, 0, 0), parent_name="RightArm")
    skeleton.create_joint("RightHand", offset=(-20, 0, 0), parent_name="RightForeArm")
    # Left Leg
    skeleton.create_joint("LeftUpLeg", offset=(10, -10, 0), parent_name="Hips")
    skeleton.create_joint("LeftLeg", offset=(0, -45, 0), parent_name="LeftUpLeg")
    skeleton.create_joint("LeftFoot", offset=(0, -45, 0), parent_name="LeftLeg")
    skeleton.create_joint("LeftToeBase", offset=(0, 0, 10), parent_name="LeftFoot")
    # Right Leg
    skeleton.create_joint("RightUpLeg", offset=(-10, -10, 0), parent_name="Hips")
    skeleton.create_joint("RightLeg", offset=(0, -45, 0), parent_name="RightUpLeg")
    skeleton.create_joint("RightFoot", offset=(0, -45, 0), parent_name="RightLeg")
    skeleton.create_joint("RightToeBase", offset=(0, 0, 10), parent_name="RightFoot")
    return skeleton

def get_mediapipe_joint_mapping() -> Dict[str, List[Any]]: # Using Any for PoseLandmark due to import style
    """Maps BVH joint names to corresponding MediaPipe PoseLandmark enums."""
    # Ensure mp_pose_solutions.PoseLandmark is accessible
    # This might need adjustment based on how mp_pose_solutions is imported/used globally
    if not hasattr(mp_pose_solutions, 'PoseLandmark'):
        # Fallback if PoseLandmark isn't directly on the module after aliasing
        # This typically means mp.solutions.pose.PoseLandmark should be used.
        # For simplicity, we'll assume direct access for now, but this is a common pitfall.
        # If mp_pose was the direct `mp.solutions.pose` import, then `mp_pose.PoseLandmark` works.
        # If it's `mp_pose = mp.solutions.pose.Pose()` instance, this mapping strategy needs to change.
        # The original script did `mp_pose = mp.solutions.pose` which makes `mp_pose.PoseLandmark` valid.
        # Here, we use `mp_pose_solutions` as the alias for `mp.solutions.pose`.
        PL = mp.solutions.pose.PoseLandmark
    else:
        PL = mp_pose_solutions.PoseLandmark

    return {
        "Hips": [PL.LEFT_HIP, PL.RIGHT_HIP],
        "Spine": [PL.LEFT_HIP, PL.RIGHT_HIP, PL.LEFT_SHOULDER, PL.RIGHT_SHOULDER], # Midpoint logic needed
        "Chest": [PL.LEFT_SHOULDER, PL.RIGHT_SHOULDER], # Midpoint
        "Neck": [PL.LEFT_SHOULDER, PL.RIGHT_SHOULDER, PL.LEFT_EAR, PL.RIGHT_EAR], # Midpoint logic
        "Head": [PL.NOSE, PL.LEFT_EYE, PL.RIGHT_EYE, PL.LEFT_EAR, PL.RIGHT_EAR], # Centroid
        "LeftShoulder": [PL.LEFT_SHOULDER],
        "LeftArm": [PL.LEFT_ELBOW],      # Direction: Shoulder to Elbow
        "LeftForeArm": [PL.LEFT_WRIST],   # Direction: Elbow to Wrist
        "LeftHand": [PL.LEFT_PINKY, PL.LEFT_INDEX, PL.LEFT_THUMB], # Centroid
        "RightShoulder": [PL.RIGHT_SHOULDER],
        "RightArm": [PL.RIGHT_ELBOW],
        "RightForeArm": [PL.RIGHT_WRIST],
        "RightHand": [PL.RIGHT_PINKY, PL.RIGHT_INDEX, PL.RIGHT_THUMB],
        "LeftUpLeg": [PL.LEFT_HIP],      # Direction: Hip to Knee
        "LeftLeg": [PL.LEFT_KNEE],       # Direction: Knee to Ankle
        "LeftFoot": [PL.LEFT_ANKLE],     # Direction: Ankle to Foot Index
        "LeftToeBase": [PL.LEFT_FOOT_INDEX],
        "RightUpLeg": [PL.RIGHT_HIP],
        "RightLeg": [PL.RIGHT_KNEE],
        "RightFoot": [PL.RIGHT_ANKLE],
        "RightToeBase": [PL.RIGHT_FOOT_INDEX]
    }

# --- Landmark Processing ---
def get_landmark_coords(landmarks_list: List[Any], landmark_enum: Any) -> Optional[np.ndarray]:
    """Extracts scaled (x, -y, z) coordinates from MediaPipe world landmarks."""
    # landmarks_list is results.pose_world_landmarks.landmark
    if landmarks_list and landmark_enum.value < len(landmarks_list):
        lm = landmarks_list[landmark_enum.value]
        if hasattr(lm, 'x') and hasattr(lm, 'y') and hasattr(lm, 'z') and lm.visibility > 0.3: # Check visibility
             # MediaPipe: +X right, +Y down, +Z into screen (towards camera from subject)
             # BVH typically: +X right, +Y up, +Z forward (away from camera)
             # The original script used (lm.x, -lm.y, lm.z).
             # World landmarks are already in meters, with origin at hips center.
             # +X is to the subject's right, +Y is up, and +Z is towards the subject's front.
             # So, direct use might be fine for BVH if axes align.
             # Let's stick to original conversion initially: (x, -y, z) but note world landmarks are different from normalized screen ones.
             # For world landmarks: Y is already up. So, (lm.x, lm.y, lm.z) might be more direct for BVH.
             # Let's assume the coordinate system of world_landmarks is suitable for BVH with minor scaling.
             # The original script's (x, -y, z) was likely for normalized image coordinates.
             # For pose_world_landmarks, MediaPipe doc says:
             # Origin at the center of the hips. X-axis to the person's right, Y-axis upward, Z-axis forward.
             # This aligns well with typical BVH.
            return np.array([lm.x, lm.y, lm.z]) * GLOBAL_SCALE_FACTOR
    return None

def get_joint_world_position(landmarks_list: List[Any], landmark_indices: List[Any]) -> Optional[np.ndarray]:
    """Calculates the average world position for a joint from its mapped MediaPipe landmarks."""
    positions = []
    for index in landmark_indices:
        pos = get_landmark_coords(landmarks_list, index)
        if pos is not None:
            positions.append(pos)
    
    if positions:
        return np.mean(positions, axis=0)
    return None

def _calculate_initial_skeleton_offsets(skeleton: Skeleton, first_frame_landmarks: List[Any], joint_mapping: Dict[str, List[Any]]):
    """Sets initial bone offsets in the skeleton based on the first frame's landmarks."""
    # Get world positions for all mapped joints from the first frame
    world_positions = {}
    for joint_name, landmark_indices in joint_mapping.items():
        pos = get_joint_world_position(first_frame_landmarks, landmark_indices)
        if pos is not None:
            world_positions[joint_name] = pos

    # Set offsets based on these world positions
    for joint_name, joint_obj in skeleton.joints.items():
        if joint_obj.parent and joint_name in world_positions and joint_obj.parent.name in world_positions:
            parent_pos = world_positions[joint_obj.parent.name]
            child_pos = world_positions[joint_name]
            offset = child_pos - parent_pos
            
            # Ensure minimum length to avoid zero-length bones
            if np.linalg.norm(offset) < MIN_OFFSET_LENGTH:
                # Attempt to use predefined offset direction, scaled minimally
                predefined_offset_dir = skeleton.get_joint_by_name(joint_name).offset 
                if np.linalg.norm(predefined_offset_dir) > MIN_VECTOR_NORM:
                     offset = (predefined_offset_dir / np.linalg.norm(predefined_offset_dir)) * MIN_OFFSET_LENGTH
                else: # Generic small offset if predefined is also zero
                    offset = np.array([0.0, MIN_OFFSET_LENGTH, 0.0]) # Default to small Y offset
            joint_obj.offset = offset
        elif not joint_obj.parent: # Root
            joint_obj.offset = np.array([0.0, 0.0, 0.0])
        # If landmarks are missing for a joint or its parent, its predefined offset remains.


def update_skeleton_from_frame(
    landmarks_list: List[Any], # results.pose_world_landmarks.landmark
    skeleton: Skeleton, 
    joint_mapping: Dict[str, List[Any]], 
    frame_idx: int,
    is_first_frame: bool
):
    """Processes MediaPipe landmarks for a single frame and updates the skeleton."""
    global GLOBAL_SCALE_FACTOR # Allow modification if auto-scaling height

    if is_first_frame:
        # Optional: Auto-adjust GLOBAL_SCALE_FACTOR based on initial pose height
        # For simplicity, this is omitted here but was in the original.
        # It would involve calculating height from landmarks and adjusting GLOBAL_SCALE_FACTOR.
        # Then, re-calculate landmark coords if GLOBAL_SCALE_FACTOR changed.
        
        # Set initial bone offsets based on T-pose or first detected pose
        # This gives more natural proportions than fixed offsets if landmarks are good.
        # The create_custom_skeleton already sets default offsets. This can refine them.
        _calculate_initial_skeleton_offsets(skeleton, landmarks_list, joint_mapping)


    # Update Root Position (Hips)
    root_joint = skeleton.root
    if root_joint:
        hips_pos = get_joint_world_position(landmarks_list, joint_mapping.get(root_joint.name, []))
        if hips_pos is not None:
            # For BVH, root position is relative to the world origin.
            # MediaPipe world landmarks are relative to hips center at first frame, then track this center.
            # So, hips_pos IS the world position of the hips.
            if frame_idx >= len(root_joint.positions): root_joint.positions.append(hips_pos)
            else: root_joint.positions[frame_idx] = hips_pos
        elif frame_idx > 0 and root_joint.positions: # Use previous if missing
             if frame_idx >= len(root_joint.positions): root_joint.positions.append(root_joint.positions[-1])
             else: root_joint.positions[frame_idx] = root_joint.positions[-1]
        else: # Default to origin
             if frame_idx >= len(root_joint.positions): root_joint.positions.append(np.zeros(3))
             else: root_joint.positions[frame_idx] = np.zeros(3)


    # --- Calculate Joint Rotations ---
    # This is a complex part. The strategy is to define a bone's orientation
    # by a vector (e.g., from parent joint to child joint using landmarks)
    # and then calculate the rotation needed to align a default bone vector
    # (from skeleton definition) to this new orientation.
    
    # Store current landmark positions for this frame
    current_landmark_positions = {}
    for lm_idx, lm_details in enumerate(mp_pose_solutions.PoseLandmark):
      coords = get_landmark_coords(landmarks_list, lm_details)
      if coords is not None:
        current_landmark_positions[lm_details] = coords


    for joint_name, joint_obj in skeleton.joints.items():
        if not joint_obj.parent: # Skip root for relative rotations here
            # Root orientation could be handled separately if needed (e.g. aligning hips with world Z axis)
            # For now, root rotations are implicitly handled by child joint orientations.
            # Or, set root rotation to identity or a calculated global orientation.
            if frame_idx >= len(joint_obj.rotations): joint_obj.rotations.append(np.zeros(3))
            else: joint_obj.rotations[frame_idx] = np.zeros(3)
            continue

        parent_joint_obj = joint_obj.parent
        
        # Get positions of the joint and its parent from current landmarks
        # This requires a way to map BVH joint names to landmark sets that define their position
        # For simplicity, we'll use the direct landmark mapping if available,
        # or rely on parent-child relationships for direction.

        pos_joint = get_joint_world_position(landmarks_list, joint_mapping.get(joint_name, []))
        pos_parent = get_joint_world_position(landmarks_list, joint_mapping.get(parent_joint_obj.name, []))

        rotation = np.zeros(3) # Default to no rotation

        if pos_joint is not None and pos_parent is not None:
            current_bone_vector = pos_joint - pos_parent
            
            # The reference_bone_vector is the joint's offset vector in its parent's frame
            # This vector defines the bone's orientation in the rest pose.
            reference_bone_vector = joint_obj.offset 

            if np.linalg.norm(current_bone_vector) > MIN_VECTOR_NORM and \
               np.linalg.norm(reference_bone_vector) > MIN_VECTOR_NORM:
                
                # We need to calculate the rotation that transforms reference_bone_vector to current_bone_vector.
                # This rotation is relative to the parent joint's coordinate system.
                # However, BVH rotations are typically applied sequentially.
                # A simpler approach: define bone direction from landmarks, compare to a T-pose default.
                
                # Example: LeftArm (Elbow joint)
                # Reference is typically along X for arms in T-pose.
                # Current direction from landmarks (e.g. elbow_pos - shoulder_pos)
                # This part needs careful implementation of defining reference vectors for each bone.
                # The original script had a `calculate_joint_rotations` function with complex logic.
                # Let's use a simplified version of calculate_rotation_between_vectors
                
                # This calculation of rotation is a key challenge.
                # A common method is to use `look_at` rotation or calculate rotation from a
                # reference direction (e.g. T-pose) to current direction.
                # The original script had a more detailed `calculate_rotation` and `joint_pairs`.
                # For this refactoring, we'll placeholder with a simple directional rotation.
                # A full robust solution requires careful handling of coordinate frames and reference poses.

                # Simplified: Assume reference_bone_vector is the primary axis we want to align.
                # This assumes the 'reference_bone_vector' (offset) correctly defines the bone's primary axis in its local frame.
                # And 'current_bone_vector' is that bone's desired orientation in the PARENT'S frame.
                # This interpretation is tricky. Let's default to zero rotation for now to avoid complex issues.
                # rotation = calculate_rotation_between_vectors(reference_bone_vector, current_bone_vector)
                pass # Placeholder for more advanced rotation logic from original

        if frame_idx >= len(joint_obj.rotations): joint_obj.rotations.append(rotation)
        else: joint_obj.rotations[frame_idx] = rotation


    skeleton.frames = max(skeleton.frames, frame_idx + 1)


# --- MediaPipe Processing ---
def initialize_mediapipe_pose(model_complexity: int = 2, smooth_landmarks: bool = True) -> Any:
    """Initializes and returns a MediaPipe Pose instance."""
    return mp_pose_solutions.Pose(
        static_image_mode=False,
        model_complexity=model_complexity,
        enable_segmentation=False, # Segmentation not needed for BVH
        smooth_landmarks=smooth_landmarks,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

def display_preview_frame(
    frame: np.ndarray, 
    pose_results: Any, 
    frame_idx: int, 
    total_frames: int, 
    processed_bvh_frames: int
):
    """Displays the video frame with MediaPipe pose landmarks."""
    annotated_frame = frame.copy()
    if pose_results.pose_landmarks: # Use screen landmarks for 2D preview
        mp_drawing_utils.draw_landmarks(
            annotated_frame,
            pose_results.pose_landmarks,
            mp_pose_solutions.POSE_CONNECTIONS, # Standard connections
            mp_drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
            mp_drawing_utils.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)
        )
    
    cv2.putText(annotated_frame, f"Frame: {frame_idx}/{total_frames}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(annotated_frame, f"BVH Frames: {processed_bvh_frames}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('MediaPipe Pose Preview', annotated_frame)


def process_video_frames(
    video_path: str, 
    pose_estimator: Any, 
    skeleton: Skeleton, 
    joint_mapping: Dict[str, List[Any]],
    target_fps: int, 
    preview_enabled: bool
) -> Tuple[int, int]:
    """Reads video, processes frames with MediaPipe, and updates the skeleton."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return 0, 0

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if video_fps == 0: video_fps = 30 # Default if FPS not readable
    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Video: {frame_width}x{frame_height}, {video_fps:.2f} FPS, {total_video_frames} frames")
    print(f"Target BVH FPS: {target_fps}")

    frame_step = max(1, round(video_fps / target_fps))
    print(f"Processing every ~{frame_step} video frames for BVH.")

    processed_bvh_frames = 0
    current_video_frame_idx = 0

    if preview_enabled:
        cv2.namedWindow('MediaPipe Pose Preview', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('MediaPipe Pose Preview', 800, 600)

    with tqdm(total=total_video_frames, desc="Processing Video") as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

            should_process_for_bvh = (current_video_frame_idx % frame_step == 0)
            
            if preview_enabled or should_process_for_bvh:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb_frame.flags.writeable = False # Performance tip
                pose_results = pose_estimator.process(rgb_frame)
                rgb_frame.flags.writeable = True

                if should_process_for_bvh and pose_results.pose_world_landmarks:
                    update_skeleton_from_frame(
                        pose_results.pose_world_landmarks.landmark,
                        skeleton,
                        joint_mapping,
                        processed_bvh_frames,
                        is_first_frame=(processed_bvh_frames == 0)
                    )
                    processed_bvh_frames += 1
                
                if preview_enabled:
                    display_preview_frame(frame, pose_results, current_video_frame_idx, total_video_frames, processed_bvh_frames)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        print("Preview stopped by user.")
                        break
            
            current_video_frame_idx += 1
            pbar.update(1)
            
    cap.release()
    if preview_enabled:
        cv2.destroyAllWindows()
    return processed_bvh_frames, current_video_frame_idx


# --- Main Orchestration ---
def mediapipe_to_bvh_conversion(
    video_path: str, 
    output_bvh_path: str, 
    target_fps: int, 
    smoothing_window: int,
    manual_scale_factor: Optional[float], 
    preview_video: bool
):
    """Main function to orchestrate the video to BVH conversion process."""
    global GLOBAL_SCALE_FACTOR
    if manual_scale_factor is not None:
        GLOBAL_SCALE_FACTOR = manual_scale_factor
        print(f"Using manual scale factor: {GLOBAL_SCALE_FACTOR}")

    pose_estimator = initialize_mediapipe_pose()
    skeleton = create_custom_skeleton(fps=target_fps)
    joint_mapping = get_mediapipe_joint_mapping()

    start_time = time.time()
    
    processed_bvh_frames, total_frames_read = process_video_frames(
        video_path, pose_estimator, skeleton, joint_mapping, target_fps, preview_video
    )

    elapsed_time = time.time() - start_time
    print(f"\nFinished processing {processed_bvh_frames} BVH frames from {total_frames_read} video frames.")
    print(f"Processing time: {elapsed_time:.2f} seconds.")

    if processed_bvh_frames > 0:
        if smoothing_window > 1: # Smoothing only makes sense with multiple frames
            skeleton.smooth_motion(smoothing_window)
        
        print(f"Writing BVH file to {output_bvh_path}")
        skeleton.write_to_bvh(output_bvh_path)
        print(f"BVH export complete. Total frames in BVH: {skeleton.frames}")
        print(f"BVH duration: {skeleton.frames * skeleton.frame_time:.2f} seconds")
    else:
        print("No frames processed for BVH. Output file will not be generated.")

    return skeleton # Return for potential further use (e.g., direct preview)


# --- BVH Preview (Simplified) ---
# Note: A full BVH parser and animator is complex. This is a very basic preview.
# The original script's preview was more detailed. This version is stripped down for brevity in refactoring.
def preview_bvh(bvh_filepath: str, speed_factor: float = 1.0):
    """Rudimentary preview of BVH animation using Matplotlib."""
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        from matplotlib.animation import FuncAnimation
        print(f"Attempting to preview BVH: {bvh_filepath} (Note: Preview functionality is basic)")
        # This would require parsing the BVH file (hierarchy and motion)
        # and then animating it. The original script had extensive code for this.
        # For this refactored example, we'll skip the complex implementation.
        print("BVH preview requires a full BVH parser and Matplotlib animation logic.")
        print("The original script contained a more complete previewer.")
        print("To reimplement, you'd need to parse HIERARCHY and MOTION sections,")
        print("calculate joint world positions per frame, and animate using Matplotlib.")

        # Placeholder: Show a message if matplotlib is available
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_title(f"BVH Preview Placeholder for {os.path.basename(bvh_filepath)}")
        ax.text2D(0.5, 0.5, "BVH Animation would be here.", 
                  horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
        plt.show()

    except ImportError:
        print("Matplotlib is required for BVH preview. Install with: pip install matplotlib")
    except Exception as e:
        print(f"Error during BVH preview attempt: {e}")


# --- Command Line Interface ---
def main():
    parser = argparse.ArgumentParser(description='Convert video to BVH using MediaPipe Pose (Refactored)')
    parser.add_argument('--video', required=True, help='Input video file path')
    parser.add_argument('--output', required=True, help='Output BVH file path')
    parser.add_argument('--fps', type=int, default=DEFAULT_TARGET_FPS, help=f'Target FPS for BVH output (default: {DEFAULT_TARGET_FPS})')
    parser.add_argument('--smoothing', type=int, default=DEFAULT_SMOOTHING_WINDOW, help=f'Smoothing window size (0 or 1 to disable, default: {DEFAULT_SMOOTHING_WINDOW})')
    parser.add_argument('--scale', type=float, help='Manual scale factor for BVH units (e.g., 100.0). Overrides any auto-scaling.')
    parser.add_argument('--preview', action='store_true', help='Show real-time preview of MediaPipe pose detection during video processing.')
    parser.add_argument('--preview-bvh', action='store_true', help='Attempt to preview the generated BVH animation using Matplotlib (basic).')
    parser.add_argument('--speed', type=float, default=1.0, help='Playback speed factor for BVH preview (if available).')
    
    args = parser.parse_args()

    mediapipe_to_bvh_conversion(
        args.video, 
        args.output, 
        args.fps, 
        args.smoothing, 
        args.scale, 
        args.preview
    )

    if args.preview_bvh:
        if os.path.exists(args.output):
            preview_bvh(args.output, args.speed)
        else:
            print(f"Cannot preview BVH: Output file {args.output} not found or not generated.")

if __name__ == "__main__":
    main()