#!/usr/bin/env python3
"""
MediaPipe Pose to BVH Converter (Combined Version)
---------------------------------------------------
This script converts a video file to a BVH motion capture file using MediaPipe's pose estimation.
It aims to combine accurate global position tracking with refined limb pose estimation.

Strengths from mediapipe_to_bvh_redux.py:
- Global position tracking using calculate_global_position_delta.

Strengths from bvh9.py:
- Skeleton structure and joint mapping.
- Limb pose estimation and rotation calculation.
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
# This might be adjusted or made configurable
GLOBAL_SCALE_FACTOR = 100.0

# Global variable to store the accumulated movement tracking
GLOBAL_ROOT_POSITION = np.zeros(3)
PREVIOUS_FRAME_WORLD_LANDMARKS = None # For global position delta calculation

@dataclass
class EmptyLandmark:
    """Simple class to substitute for MediaPipe landmarks when needed"""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    visibility: float = 0.0

class Joint:
    """Class representing a joint in the BVH skeleton (adapted from bvh9.py)"""
    def __init__(self, name, parent=None):
        self.name = name
        self.parent = parent
        self.children = []
        self.offset = np.zeros(3) # Will be calculated
        # Channels will be set based on root or child
        self.rotation_order = 'XYZ'  # Common rotation order
        self.positions = []  # For root joint
        self.rotations = []  # For all joints (stores Euler angles)

    def add_child(self, child):
        self.children.append(child)
        child.parent = self # Ensure parent is set

    def __str__(self):
        return f"Joint({self.name}, children={len(self.children)})"

def create_skeleton_bvh9():
    """Create a skeleton structure based on bvh9.py."""
    hips = Joint("Hips")

    spine = Joint("Spine", hips)
    hips.add_child(spine)

    chest = Joint("Chest", spine)
    spine.add_child(chest)

    neck = Joint("Neck", chest)
    chest.add_child(neck)

    head = Joint("Head", neck)
    neck.add_child(head)

    left_shoulder = Joint("LeftShoulder", chest)
    chest.add_child(left_shoulder)
    left_arm = Joint("LeftArm", left_shoulder)
    left_shoulder.add_child(left_arm)
    left_forearm = Joint("LeftForeArm", left_arm)
    left_arm.add_child(left_forearm)
    left_hand = Joint("LeftHand", left_forearm)
    left_forearm.add_child(left_hand)

    right_shoulder = Joint("RightShoulder", chest)
    chest.add_child(right_shoulder)
    right_arm = Joint("RightArm", right_shoulder)
    right_shoulder.add_child(right_arm)
    right_forearm = Joint("RightForeArm", right_arm)
    right_arm.add_child(right_forearm)
    right_hand = Joint("RightHand", right_forearm)
    right_forearm.add_child(right_hand)

    left_up_leg = Joint("LeftUpLeg", hips)
    hips.add_child(left_up_leg)
    left_leg = Joint("LeftLeg", left_up_leg)
    left_up_leg.add_child(left_leg)
    left_foot = Joint("LeftFoot", left_leg)
    left_leg.add_child(left_foot)
    left_toe = Joint("LeftToeBase", left_foot) # bvh9 calls it LeftToeBase
    left_foot.add_child(left_toe)

    right_up_leg = Joint("RightUpLeg", hips)
    hips.add_child(right_up_leg)
    right_leg = Joint("RightLeg", right_up_leg)
    right_up_leg.add_child(right_leg)
    right_foot = Joint("RightFoot", right_leg)
    right_leg.add_child(right_foot)
    right_toe = Joint("RightToeBase", right_foot) # bvh9 calls it RightToeBase
    right_foot.add_child(right_toe)

    return hips

def get_joint_mapping_bvh9():
    """Joint mapping from bvh9.py."""
    return {
        "Hips": [mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.RIGHT_HIP],
        "Spine": [mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.RIGHT_HIP,
                  mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.RIGHT_SHOULDER],
        "Chest": [mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.RIGHT_SHOULDER],
        "Neck": [mp_pose.PoseLandmark.LEFT_EAR, mp_pose.PoseLandmark.RIGHT_EAR], # bvh9 uses EARs for Neck
        "Head": [mp_pose.PoseLandmark.NOSE], # bvh9 uses NOSE for Head
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

# Helper function to get landmark 3D position from MediaPipe results
def get_landmark_world_position(world_landmarks, idx):
    if world_landmarks and idx < len(world_landmarks):
        lm = world_landmarks[idx]
        if hasattr(lm, 'x') and hasattr(lm, 'y') and hasattr(lm, 'z'):
            # MediaPipe world coordinates: +X right, +Y down, +Z away from camera (further = larger Z)
            # Convert to BVH like: +X right, +Y up, +Z forward (towards camera)
            # This conversion might need adjustment based on desired BVH orientation.
            # For now, let's use the redux version's conversion which seemed to work with its global positioning
            return np.array([lm.x, -lm.y, lm.z]) # redux: Y negated, Z as is (MP z is depth)
    return None

# Adapted from bvh9.py
def get_joint_world_position_from_landmarks(joint_name, world_landmarks, joint_mapping):
    if joint_name not in joint_mapping:
        return None
    indices = joint_mapping[joint_name]
    positions = []
    for idx in indices:
        pos = get_landmark_world_position(world_landmarks, idx)
        if pos is not None:
            positions.append(pos)
    if positions:
        return np.mean(positions, axis=0)
    return None

# Adapted from bvh9.py (ensure_minimum_offset and get_default_bone_offsets)
def ensure_minimum_offset(offset_vector, min_length=0.05, default_direction=None):
    length = np.linalg.norm(offset_vector)
    if length < min_length:
        if default_direction is not None and np.linalg.norm(default_direction) > 1e-9:
            direction = default_direction / np.linalg.norm(default_direction)
        elif length > 1e-10:
            direction = offset_vector / length
        else:
            direction = np.array([0.0, 1.0, 0.0]) # Default up
        return direction * min_length
    return offset_vector

def get_default_bone_offsets_bvh9():
    return {
        "Hips": np.array([0.0, 0.0, 0.0]),
        "Spine": np.array([0.0, 0.15, 0.0]), "Chest": np.array([0.0, 0.15, 0.0]),
        "Neck": np.array([0.0, 0.1, 0.0]), "Head": np.array([0.0, 0.1, 0.0]),
        "LeftShoulder": np.array([-0.1, 0.0, 0.0]), "LeftArm": np.array([-0.15, 0.0, 0.0]),
        "LeftForeArm": np.array([-0.15, 0.0, 0.0]), "LeftHand": np.array([-0.1, 0.0, 0.0]),
        "RightShoulder": np.array([0.1, 0.0, 0.0]), "RightArm": np.array([0.15, 0.0, 0.0]),
        "RightForeArm": np.array([0.15, 0.0, 0.0]), "RightHand": np.array([0.1, 0.0, 0.0]),
        "LeftUpLeg": np.array([-0.1, -0.1, 0.0]), "LeftLeg": np.array([0.0, -0.25, 0.0]),
        "LeftFoot": np.array([0.0, -0.25, 0.0]), "LeftToeBase": np.array([0.0, 0.0, 0.1]),
        "RightUpLeg": np.array([0.1, -0.1, 0.0]), "RightLeg": np.array([0.0, -0.25, 0.0]),
        "RightFoot": np.array([0.0, -0.25, 0.0]), "RightToeBase": np.array([0.0, 0.0, 0.1])
    }

def build_skeleton_offsets_from_landmarks_bvh9(skeleton_root, frame_world_landmarks, joint_mapping, scale):
    """Builds skeleton offsets based on bvh9.py logic."""
    default_offsets = get_default_bone_offsets_bvh9()
    
    # Store all joints for easier lookup
    all_joints_dict = {}
    q = [skeleton_root]
    while q:
        curr = q.pop(0)
        all_joints_dict[curr.name] = curr
        for child in curr.children:
            q.append(child)

    def process_joint_offset(joint):
        joint_pos = get_joint_world_position_from_landmarks(joint.name, frame_world_landmarks, joint_mapping)

        if joint.parent:
            parent_pos = get_joint_world_position_from_landmarks(joint.parent.name, frame_world_landmarks, joint_mapping)
            if joint_pos is not None and parent_pos is not None:
                offset = joint_pos - parent_pos
                default_dir = default_offsets.get(joint.name, np.array([0.0, 0.1, 0.0])) # Default Y up
                offset = ensure_minimum_offset(offset, default_direction=default_dir)
                joint.offset = offset * scale
            else:
                joint.offset = default_offsets.get(joint.name, np.array([0.0, 0.1, 0.0])) * scale
        else: # Root joint
            joint.offset = np.array([0.0, 0.0, 0.0])

        # Specific adjustments from bvh9
        if joint.name == "Head" and np.linalg.norm(joint.offset) < 5.0: joint.offset = np.array([0.0, 10.0, 0.0])
        elif joint.name == "LeftHand" and np.linalg.norm(joint.offset) < 5.0: joint.offset = np.array([-8.0, 0.0, 0.0])
        elif joint.name == "RightHand" and np.linalg.norm(joint.offset) < 5.0: joint.offset = np.array([8.0, 0.0, 0.0])
        elif joint.name in ["LeftFoot", "RightFoot"]:
            if np.linalg.norm(joint.offset) < 5.0: joint.offset = np.array([0.0, -8.0, 2.0])
            elif joint.offset[2] < 0: joint.offset[2] = abs(joint.offset[2])
        elif joint.name in ["LeftToeBase", "RightToeBase"] and np.linalg.norm(joint.offset) < 5.0:
            joint.offset = np.array([0.0, -2.0, 8.0])


        for child in joint.children:
            process_joint_offset(child)

    process_joint_offset(skeleton_root)


# Rotation calculation from bvh9.py
def axis_angle_to_euler_bvh9(axis_angle, order='XYZ'):
    angle = np.linalg.norm(axis_angle)
    if angle < 1e-10: return np.zeros(3)
    axis = axis_angle / angle
    c, s, t = math.cos(angle), math.sin(angle), 1.0 - math.cos(angle)
    x, y, z = axis
    
    # Rotation matrix components
    r00,r10,r11,r12,r20,r21,r22 = 0,0,0,0,0,0,0 # silence linter
    if order == 'XYZ': # Assuming XYZ Euler order
        r00 = c + x*x*t; r10 = y*x*t + z*s; r11 = c + y*y*t;
        r12 = y*z*t - x*s; r20 = z*x*t - y*s; r21 = z*y*t + x*s; r22 = c + z*z*t;

        sy = math.sqrt(r00*r00 + r10*r10)
        if sy > 1e-6:
            ex = math.atan2(r21, r22)
            ey = math.atan2(-r20, sy)
            ez = math.atan2(r10, r00)
        else:
            ex = math.atan2(-r12, r11)
            ey = math.atan2(-r20, sy)
            ez = 0
        return np.array([ex, ey, ez]) * (180.0 / math.pi) # Degrees
    else: # Fallback for other orders (not implemented in bvh9 example)
        print(f"Warning: Euler order {order} not fully implemented in axis_angle_to_euler_bvh9")
        return np.zeros(3)


def calculate_frame_rotations_bvh9(world_landmarks, joint_mapping, skeleton_root, all_joints_list):
    """Calculates rotations for a single frame using bvh9 logic."""
    # Initialize rotations for all joints in this frame
    current_frame_joint_rotations = {joint.name: np.zeros(3) for joint in all_joints_list}

    # Store world positions of joints for this frame
    current_frame_joint_positions = {
        name: get_joint_world_position_from_landmarks(name, world_landmarks, joint_mapping)
        for name in current_frame_joint_rotations.keys()
    }

    def get_bone_direction(parent_name, child_name):
        parent_pos = current_frame_joint_positions.get(parent_name)
        child_pos = current_frame_joint_positions.get(child_name)
        if parent_pos is not None and child_pos is not None:
            direction = child_pos - parent_pos
            if np.linalg.norm(direction) > 1e-10:
                return direction / np.linalg.norm(direction)
        return None

    # Recursive function to calculate rotations
    def calculate_joint_rotation_recursive(joint):
        for child in joint.children:
            # Use the pre-calculated offset of the child as its rest direction relative to parent
            if np.linalg.norm(child.offset) > 1e-9:
                rest_direction_normalized = child.offset / np.linalg.norm(child.offset)
            else: # Should not happen if offsets are built correctly
                rest_direction_normalized = get_default_bone_offsets_bvh9().get(child.name, np.array([0,1,0]))
                if np.linalg.norm(rest_direction_normalized) < 1e-9: rest_direction_normalized = np.array([0,1,0])
                rest_direction_normalized /= np.linalg.norm(rest_direction_normalized)


            current_bone_dir = get_bone_direction(joint.name, child.name)

            if current_bone_dir is not None:
                # bvh9 specific adjustments to current_bone_dir before calculating rotation
                if child.name in ["Head", "LeftHand", "RightHand", "LeftFoot", "RightFoot", "LeftToeBase", "RightToeBase"]:
                    if current_bone_dir[2] < 0 and abs(current_bone_dir[2]) > 0.1: current_bone_dir[2] = abs(current_bone_dir[2])
                if child.name == "LeftHand" and current_bone_dir[0] > 0: current_bone_dir[0] = -abs(current_bone_dir[0])
                elif child.name == "RightHand" and current_bone_dir[0] < 0: current_bone_dir[0] = abs(current_bone_dir[0])

                # Calculate rotation from rest_direction to current_bone_dir
                # This represents the local rotation of the child joint
                cross_prod = np.cross(rest_direction_normalized, current_bone_dir)
                dot_prod = np.dot(rest_direction_normalized, current_bone_dir)

                if np.linalg.norm(cross_prod) > 1e-10: # If not parallel
                    angle = math.acos(np.clip(dot_prod, -1.0, 1.0))
                    axis_angle = (cross_prod / np.linalg.norm(cross_prod)) * angle
                    euler_angles = axis_angle_to_euler_bvh9(axis_angle, order=child.rotation_order)
                    current_frame_joint_rotations[child.name] = euler_angles # Store local rotation
                # else: already zero (parallel or anti-parallel, but for anti-parallel, bvh9 doesn't show specific handling here)
            
            calculate_joint_rotation_recursive(child) # Recurse for children of child

    # Start recursion from the children of the Hips (as Hips itself has no parent to determine its rotation from this perspective)
    # The Hips/Root rotation is typically its orientation in world space, which can be derived differently or set to an initial state.
    # For simplicity here, we will calculate local rotations for all children.
    # Hips rotation will be stored as [0,0,0] unless specifically set by global orientation logic (not detailed in bvh9 for this part)
    for child_of_hips in skeleton_root.children:
        calculate_joint_rotation_recursive(child_of_hips)
        
    return current_frame_joint_rotations


# Global position calculation from mediapipe_to_bvh_redux.py
def calculate_global_position_delta(current_world_landmarks, previous_world_landmarks_dict, global_scale_factor):
    """
    Calculate the change in global position between frames.
    Uses world landmarks.
    """
    if previous_world_landmarks_dict is None:
        return np.zeros(3)

    key_landmarks_indices = [
        mp_pose.PoseLandmark.NOSE.value,
        mp_pose.PoseLandmark.LEFT_SHOULDER.value, mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
        mp_pose.PoseLandmark.LEFT_HIP.value, mp_pose.PoseLandmark.RIGHT_HIP.value
    ]

    current_center_sum = np.zeros(3)
    previous_center_sum = np.zeros(3)
    num_current, num_previous = 0, 0

    for lm_idx in key_landmarks_indices:
        curr_pos = get_landmark_world_position(current_world_landmarks, lm_idx)
        prev_pos = previous_world_landmarks_dict.get(lm_idx) # Already converted

        if curr_pos is not None:
            current_center_sum += curr_pos
            num_current += 1
        if prev_pos is not None:
            previous_center_sum += prev_pos
            num_previous += 1
    
    if num_current == 0 or num_previous == 0: return np.zeros(3)

    current_center = current_center_sum / num_current
    previous_center = previous_center_sum / num_previous
    
    delta_body_center_world = current_center - previous_center # In MediaPipe world units
    
    # This is a simplified version from mediapipe_to_bvh_redux.
    # The original had more complex logic for Z based on size, walking detection.
    # For this merge, we'll use the basic delta and scale it.
    # A more sophisticated merge would integrate those features too.
    delta_bvh = delta_body_center_world * global_scale_factor

    # Limit large movements (capping)
    max_delta_bvh = 50.0 # Tune this
    delta_bvh_mag = np.linalg.norm(delta_bvh)
    if delta_bvh_mag > max_delta_bvh:
        delta_bvh = delta_bvh * (max_delta_bvh / delta_bvh_mag)
        
    return delta_bvh


def process_landmarks_for_frame(
    world_landmarks, # From results.pose_world_landmarks.landmark
    skeleton_root, # The root of your Joint tree
    joint_mapping,
    all_joints_list, # List of all Joint objects
    frame_idx,
    is_first_bvh_frame,
    current_fps):
    """
    Processes landmarks for a single frame to get root position and joint rotations.
    """
    global GLOBAL_ROOT_POSITION, PREVIOUS_FRAME_WORLD_LANDMARKS, GLOBAL_SCALE_FACTOR

    # 1. Calculate Global Position Delta and update GLOBAL_ROOT_POSITION
    if not is_first_bvh_frame and PREVIOUS_FRAME_WORLD_LANDMARKS is not None:
        delta_pos = calculate_global_position_delta(
            world_landmarks,
            PREVIOUS_FRAME_WORLD_LANDMARKS,
            GLOBAL_SCALE_FACTOR
        )
        GLOBAL_ROOT_POSITION += delta_pos
    
    # Store current landmarks for next frame's delta calculation
    # These should be in the same coordinate system as used by calculate_global_position_delta
    PREVIOUS_FRAME_WORLD_LANDMARKS = {
        idx: get_landmark_world_position(world_landmarks, idx)
        for idx in range(len(world_landmarks))
    }
    
    current_root_position = np.copy(GLOBAL_ROOT_POSITION)

    # 2. Calculate Joint Rotations for the current frame using bvh9's logic
    # This function should return a dictionary: {joint_name: [rx, ry, rz]}
    frame_joint_rotations = calculate_frame_rotations_bvh9(
        world_landmarks,
        joint_mapping,
        skeleton_root,
        all_joints_list
    )
    
    # Store data for BVH
    skeleton_root.positions.append(current_root_position) # Only root has positions
    for joint_obj in all_joints_list:
        joint_obj.rotations.append(frame_joint_rotations.get(joint_obj.name, np.zeros(3)))
        
    return current_root_position, frame_joint_rotations


def write_bvh_file(skeleton_root, num_bvh_frames, frame_time_bvh, output_file, all_joints_list):
    """Write the BVH file with hierarchy and motion data."""
    print(f"Writing BVH file to {output_file}...")
    with open(output_file, 'w') as f:
        f.write("HIERARCHY\n")
        
        # Write joint hierarchy (adapted from bvh9.py)
        # This recursive function needs the file handle, joint, and indent level
        def write_joint_hierarchy_recursive(file_handle, joint, indent_level):
            indent = "  " * indent_level
            node_type = "ROOT" if joint.parent is None else "JOINT"
            file_handle.write(f"{indent}{node_type} {joint.name}\n")
            file_handle.write(f"{indent}{{\n")
            
            # Offset from parent
            offset_str = " ".join([f"{val:.6f}" for val in joint.offset])
            file_handle.write(f"{indent}  OFFSET {offset_str}\n")

            # Channels
            if joint.parent is None: # Root
                channels = f"CHANNELS 6 Xposition Yposition Zposition {joint.rotation_order[0]}rotation {joint.rotation_order[1]}rotation {joint.rotation_order[2]}rotation\n"
            else: # Child
                channels = f"CHANNELS 3 {joint.rotation_order[0]}rotation {joint.rotation_order[1]}rotation {joint.rotation_order[2]}rotation\n"
            file_handle.write(f"{indent}  {channels}")

            for child in joint.children:
                write_joint_hierarchy_recursive(file_handle, child, indent_level + 1)

            if not joint.children: # End Site for leaf joints
                file_handle.write(f"{indent}  End Site\n")
                file_handle.write(f"{indent}  {{\n")
                # bvh9.py uses more anatomically appropriate end site offsets.
                # For simplicity, using a small default. This can be improved.
                end_site_offset = "0.0 5.0 0.0" # Default end site offset
                if "Hand" in joint.name: end_site_offset = "3.0 0.0 0.0" if "Right" in joint.name else "-3.0 0.0 0.0"
                elif "Toe" in joint.name: end_site_offset = "0.0 0.0 3.0"

                file_handle.write(f"{indent}    OFFSET {end_site_offset}\n")
                file_handle.write(f"{indent}  }}\n")
            
            file_handle.write(f"{indent}}}\n")

        write_joint_hierarchy_recursive(f, skeleton_root, 0)

        # Write Motion Data
        f.write("MOTION\n")
        f.write(f"Frames: {num_bvh_frames}\n")
        f.write(f"Frame Time: {frame_time_bvh:.8f}\n")

        for i in range(num_bvh_frames):
            line_data = []
            # Root position
            line_data.extend(skeleton_root.positions[i])
            
            # Rotations for all joints in the order they appear in all_joints_list (should be HIERARCHY order)
            def append_rotations_recursive(joint_for_motion):
                line_data.extend(joint_for_motion.rotations[i])
                for child in joint_for_motion.children:
                    append_rotations_recursive(child)
            
            append_rotations_recursive(skeleton_root) # Start with root for rotation order

            f.write(" ".join([f"{val:.6f}" for val in line_data]) + "\n")

    print(f"BVH file created: {output_file}")


def get_all_joints_in_hierarchy_order(root_joint):
    """ Traverses the skeleton hierarchy (depth-first) to get joints in BVH order."""
    ordered_joints = []
    def traverse(joint):
        ordered_joints.append(joint)
        for child in joint.children:
            traverse(child)
    traverse(root_joint)
    return ordered_joints

def process_video(input_video, output_bvh, target_bvh_fps=30, preview=False):
    global GLOBAL_ROOT_POSITION, PREVIOUS_FRAME_WORLD_LANDMARKS, GLOBAL_SCALE_FACTOR
    GLOBAL_ROOT_POSITION = np.zeros(3) # Reset global position
    PREVIOUS_FRAME_WORLD_LANDMARKS = None


    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        print(f"Error: Could not open video {input_video}")
        return

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Video: {frame_width}x{frame_height} @ {video_fps:.2f} FPS, {total_frames} frames.")
    print(f"Target BVH FPS: {target_bvh_fps}")

    # Determine which video frames to process for BVH
    frame_skip = max(1, round(video_fps / target_bvh_fps))
    frame_time_bvh = 1.0 / target_bvh_fps # Actual frame time for BVH
    print(f"Processing every {frame_skip} video frames for BVH. BVH frame time: {frame_time_bvh:.4f}s.")

    # --- Skeleton and Mappings Setup ---
    skeleton_root = create_skeleton_bvh9()
    joint_mapping = get_joint_mapping_bvh9()
    all_joints_list = get_all_joints_in_hierarchy_order(skeleton_root) # Get joints in DFS order for BVH writing

    # Build initial skeleton offsets from a reference frame (e.g., first good detection)
    # This is important for bvh9's rotation calculation method.
    ref_landmarks_found = False
    temp_pose_for_ref = mp_pose.Pose(static_image_mode=True, model_complexity=1) # temp for one frame
    
    for _ in range(min(int(video_fps), total_frames)): # Try first few frames
        ret_ref, ref_frame_img = cap.read()
        if not ret_ref: break
        ref_results = temp_pose_for_ref.process(cv2.cvtColor(ref_frame_img, cv2.COLOR_BGR2RGB))
        if ref_results.pose_world_landmarks:
            print("Reference frame found for initial skeleton offsets.")
            build_skeleton_offsets_from_landmarks_bvh9(
                skeleton_root,
                ref_results.pose_world_landmarks.landmark,
                joint_mapping,
                GLOBAL_SCALE_FACTOR # Use the global scale factor
            )
            ref_landmarks_found = True
            break
    temp_pose_for_ref.close()
    if not ref_landmarks_found:
        print("Warning: Could not find a good reference frame for initial skeleton. Using default offsets scaled.")
        # Fallback: Apply default offsets if no reference frame worked
        default_offsets_map = get_default_bone_offsets_bvh9()
        for joint_obj in all_joints_list:
            joint_obj.offset = default_offsets_map.get(joint_obj.name, np.array([0,1,0])) * GLOBAL_SCALE_FACTOR
            if joint_obj.name == "Hips": joint_obj.offset = np.zeros(3)


    cap.set(cv2.CAP_PROP_POS_FRAMES, 0) # Reset video to beginning for processing

    # --- MediaPipe Pose for Video Processing ---
    pose_detector = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1, # 0, 1, or 2. Higher is more accurate but slower.
        smooth_landmarks=True,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    bvh_frame_count = 0
    video_frame_idx = 0
    
    # Lists to store per-frame data before writing to BVH
    # Data is already being appended to joint.positions and joint.rotations

    with tqdm(total=total_frames, desc="Processing Video") as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if video_frame_idx % frame_skip == 0:
                # Process this frame for BVH
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb_frame.flags.writeable = False
                results = pose_detector.process(rgb_frame)
                rgb_frame.flags.writeable = True

                if results.pose_world_landmarks:
                    process_landmarks_for_frame(
                        results.pose_world_landmarks.landmark,
                        skeleton_root,
                        joint_mapping,
                        all_joints_list,
                        bvh_frame_count, # current bvh frame index
                        bvh_frame_count == 0, # is_first_bvh_frame
                        target_bvh_fps
                    )
                    bvh_frame_count +=1
                else:
                    # Handle no detection: duplicate last frame's data or use zeros
                    # For simplicity, let's assume if process_landmarks_for_frame doesn't get landmarks,
                    # it should handle appending default (e.g. zero) rotations.
                    # And global position would just not update from this frame.
                    # A robust solution might involve interpolation or holding last pose.
                    print(f"Warning: No pose detected in video frame {video_frame_idx}. BVH frame {bvh_frame_count} may be affected.")
                    # Ensure lists get populated even if no landmarks
                    if bvh_frame_count > 0 : # if not first frame
                        skeleton_root.positions.append(skeleton_root.positions[-1]) # duplicate last position
                        for joint_obj in all_joints_list:
                            joint_obj.rotations.append(joint_obj.rotations[-1]) # duplicate last rotation
                    else: # if first frame and no detection
                        skeleton_root.positions.append(np.zeros(3))
                        for joint_obj in all_joints_list:
                            joint_obj.rotations.append(np.zeros(3))
                    bvh_frame_count += 1


                if preview and results.pose_landmarks: # screen landmarks for preview
                    annotated_frame = frame.copy()
                    mp_drawing.draw_landmarks(
                        annotated_frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                        mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2)
                    )
                    cv2.putText(annotated_frame, f"BVH Frame: {bvh_frame_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.imshow('MediaPipe Pose - Combined', annotated_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
            
            video_frame_idx += 1
            pbar.update(1)

    print(f"Finished processing. Generated {bvh_frame_count} frames for BVH.")
    
    # Clean up
    cap.release()
    pose_detector.close()
    if preview:
        cv2.destroyAllWindows()

    if bvh_frame_count > 0:
        write_bvh_file(skeleton_root, bvh_frame_count, frame_time_bvh, output_bvh, all_joints_list)
    else:
        print("No frames were processed for BVH. Output file not written.")


def main():
    parser = argparse.ArgumentParser(description='Convert video to BVH using MediaPipe Pose (Combined Logic)')
    parser.add_argument('--video', required=True, help='Input video file')
    parser.add_argument('--output', required=True, help='Output BVH file')
    parser.add_argument('--fps', type=int, default=30, help='Target FPS for BVH output')
    parser.add_argument('--scale', type=float, default=100.0, help='Global scale factor for BVH units')
    parser.add_argument('--preview', action='store_true', help='Show preview of pose detection')
    
    args = parser.parse_args()

    global GLOBAL_SCALE_FACTOR
    GLOBAL_SCALE_FACTOR = args.scale
    
    process_video(args.video, args.output, target_bvh_fps=args.fps, preview=args.preview)

if __name__ == "__main__":
    main()