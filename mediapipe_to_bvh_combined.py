#!/usr/bin/env python3
"""
MediaPipe Pose to BVH Converter (Combined Version)
-------------------------------------------------
This script converts a video file to a BVH motion capture file using MediaPipe's pose estimation.
It combines the accurate limb pose tracking from bvh9.py with the global position tracking 
from mediapipe_to_bvh_redux.py, providing a more complete motion capture solution.

Features:
- Accurate global position tracking for walking/moving characters
- High quality joint rotation calculations for better limb poses
- Improved skeleton structure and proportions
- Better handling of occlusions and missing landmarks
- Preview options for both video pose detection and BVH animation
- Flexible sampling rates for different frame rates

Usage: python mediapipe_to_bvh_combined.py --video input.mp4 --output motion.bvh [options]
"""

import cv2
import mediapipe as mp
import numpy as np
import os
import argparse
import math
import time
from dataclasses import dataclass
from tqdm import tqdm

# MediaPipe setup
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Global variables for position tracking
GLOBAL_SCALE_FACTOR = 100.0
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
    def __init__(self, name, parent=None):
        self.name = name
        self.parent = parent
        self.children = []
        self.offset = np.zeros(3)
        self.channels = []
        self.rotation_order = 'XYZ'  # Using XYZ order for better Blender compatibility
        self.positions = []
        self.rotations = []
        
    def add_child(self, child):
        self.children.append(child)
        child.parent = self

def create_skeleton():
    """Create a skeleton structure that matches MediaPipe's pose landmarks."""
    # Root joint
    hips = Joint("Hips")
    
    # Spine
    spine = Joint("Spine", hips)
    hips.add_child(spine)
    
    chest = Joint("Chest", spine)
    spine.add_child(chest)
    
    neck = Joint("Neck", chest)
    chest.add_child(neck)
    
    head = Joint("Head", neck)
    neck.add_child(head)
    
    # Left arm
    left_shoulder = Joint("LeftShoulder", chest)
    chest.add_child(left_shoulder)
    
    left_arm = Joint("LeftArm", left_shoulder)
    left_shoulder.add_child(left_arm)
    
    left_forearm = Joint("LeftForeArm", left_arm)
    left_arm.add_child(left_forearm)
    
    left_hand = Joint("LeftHand", left_forearm)
    left_forearm.add_child(left_hand)
    
    # Right arm
    right_shoulder = Joint("RightShoulder", chest)
    chest.add_child(right_shoulder)
    
    right_arm = Joint("RightArm", right_shoulder)
    right_shoulder.add_child(right_arm)
    
    right_forearm = Joint("RightForeArm", right_arm)
    right_arm.add_child(right_forearm)
    
    right_hand = Joint("RightHand", right_forearm)
    right_forearm.add_child(right_hand)
    
    # Left leg
    left_up_leg = Joint("LeftUpLeg", hips)
    hips.add_child(left_up_leg)
    
    left_leg = Joint("LeftLeg", left_up_leg)
    left_up_leg.add_child(left_leg)
    
    left_foot = Joint("LeftFoot", left_leg)
    left_leg.add_child(left_foot)
    
    left_toe = Joint("LeftToeBase", left_foot)
    left_foot.add_child(left_toe)
    
    # Right leg
    right_up_leg = Joint("RightUpLeg", hips)
    hips.add_child(right_up_leg)
    
    right_leg = Joint("RightLeg", right_up_leg)
    right_up_leg.add_child(right_leg)
    
    right_foot = Joint("RightFoot", right_leg)
    right_leg.add_child(right_foot)
    
    right_toe = Joint("RightToeBase", right_foot)
    right_foot.add_child(right_toe)
    
    return hips

def get_joint_mapping():
    """Map MediaPipe landmarks to BVH skeleton joints."""
    return {
        "Hips": [mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.RIGHT_HIP],
        
        "Spine": [mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.RIGHT_HIP, 
                 mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.RIGHT_SHOULDER],
        
        "Chest": [mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.RIGHT_SHOULDER],
        
        "Neck": [mp_pose.PoseLandmark.LEFT_EAR, mp_pose.PoseLandmark.RIGHT_EAR],
        
        "Head": [mp_pose.PoseLandmark.NOSE],
        
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

def build_skeleton_from_landmarks(frame_landmarks, joint_mapping, scale=100.0):
    """Build the skeleton structure and set offsets based on landmark positions"""
    root = create_skeleton()
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
            if np.linalg.norm(joint.offset) < 5.0:  # If less than 5 units (scaled)
                joint.offset = np.array([0.0, 10.0, 0.0])  # Default head height
                
        elif joint.name == "LeftHand" or joint.name == "RightHand":
            # Ensure hands extend from forearms
            if np.linalg.norm(joint.offset) < 5.0:
                direction = 1.0 if "Right" in joint.name else -1.0
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
    process_joint(root)
    return root

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

def euler_to_rotation_matrix(euler_angles):
    """Convert euler angles to rotation matrix (XYZ order)"""
    # Unpack angles
    x, y, z = euler_angles
    
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
    
    # Combine the matrices in the correct order for XYZ extrinsic rotation
    R = Rz @ Ry @ Rx
    return R

def transform_point(point, rotation_matrix):
    """Transform a point using a rotation matrix"""
    return np.matmul(rotation_matrix, point)

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
    
    # Compute only the rotation matrix elements we actually need for XYZ order
    r00 = c + x*x*t
    r10 = y*x*t + z*s
    r11 = c + y*y*t
    r12 = y*z*t - x*s
    r20 = z*x*t - y*s
    r21 = z*y*t + x*s
    r22 = c + z*z*t
    
    # Convert rotation matrix to Euler angles based on order
    if order == 'XYZ':
        # Check for gimbal lock (singularity)
        sy = math.sqrt(r00*r00 + r10*r10)
        
        if sy > 1e-6:
            # Normal case - not at singularity
            x = math.atan2(r21, r22)
            y = math.atan2(-r20, sy)
            z = math.atan2(r10, r00)
        else:
            # At singularity (gimbal lock)
            x = math.atan2(-r12, r11)
            y = math.atan2(-r20, sy)
            z = 0
    elif order == 'ZYX':
        # We'd use different elements for ZYX order
        # (not implemented as we're using XYZ in this script)
        return np.zeros(3)
    else:
        # Default fallback for unsupported orders
        return np.zeros(3)
    
    return np.array([x, y, z]) * (180.0 / math.pi)  # Convert to degrees

def calculate_frame_rotations(frame_landmarks, joint_mapping, skeleton, default_rotations):
    """Calculate rotations for all joints based on the current frame landmarks"""
    rotations = default_rotations.copy()  # Start with default rotations
    
    # Helper function to calculate bone direction
    def get_bone_direction(joint_name, child_name):
        parent_pos = get_joint_position(joint_name, frame_landmarks, joint_mapping)
        child_pos = get_joint_position(child_name, frame_landmarks, joint_mapping)
        
        if parent_pos is not None and child_pos is not None:
            direction = child_pos - parent_pos
            if np.linalg.norm(direction) > 1e-10:
                return direction / np.linalg.norm(direction)
        
        return None
    
    # Process joints
    def process_joint(joint):
        if joint.children:
            for child in joint.children:
                # Try to calculate rotation from rest pose to current pose
                direction = get_bone_direction(joint.name, child.name)
                
                if direction is not None:
                    # Calculate rest pose direction (normalized offset)
                    rest_direction = child.offset / np.linalg.norm(child.offset) if np.linalg.norm(child.offset) > 0 else None
                    
                    if rest_direction is not None:
                        # Apply corrections for specific joints that need orientation fixes
                        if joint.name in ["Head", "LeftHand", "RightHand", "LeftFoot", "RightFoot", "LeftToeBase", "RightToeBase"]:
                            # For problematic joints, ensure the forward direction is positive
                            if direction[2] < 0 and abs(direction[2]) > 0.1:
                                # If significantly pointing backward, fix the direction
                                direction[2] = abs(direction[2])
                                
                            # For hands, ensure they extend out from arms
                            if joint.name == "LeftHand" and direction[0] > 0:
                                direction[0] = -abs(direction[0])  # Ensure left hand goes left
                            elif joint.name == "RightHand" and direction[0] < 0:
                                direction[0] = abs(direction[0])   # Ensure right hand goes right
                        
                        # Calculate rotation from rest to current
                        cross = np.cross(rest_direction, direction)
                        dot = np.dot(rest_direction, direction)
                        
                        if np.linalg.norm(cross) > 1e-10:
                            angle = math.acos(np.clip(dot, -1.0, 1.0))
                            axis_angle = (cross / np.linalg.norm(cross)) * angle
                            euler_angles = axis_angle_to_euler(axis_angle)
                            
                            # For some joints, we may need to flip rotation axes
                            if joint.name in ["Head", "LeftToeBase", "RightToeBase"]:
                                # If rotation seems wrong, try flipping axes
                                if abs(euler_angles[1]) > 90:  # Y rotation seems excessive
                                    euler_angles[1] = -euler_angles[1]
                                if abs(euler_angles[0]) > 90:  # X rotation seems excessive
                                    euler_angles[0] = -euler_angles[0]
                            
                            rotations[joint.name] = euler_angles
                
                process_joint(child)
    
    process_joint(skeleton)
    return rotations

def process_motion(frames_landmarks, skeleton, joint_mapping):
    """Process all frames to calculate rotations for animation"""
    num_frames = len(frames_landmarks)
    print(f"Calculating joint rotations for {num_frames} frames...")
    
    # Calculate default rotations (all zeros)
    default_rotations = {joint.name: np.zeros(3) for joint in get_all_joints(skeleton)}
    
    # Calculate all frame rotations
    all_rotations = []
    for frame_idx in tqdm(range(num_frames), desc="Processing frames"):
        frame_landmarks = frames_landmarks[frame_idx]
        frame_rotations = calculate_frame_rotations(frame_landmarks, joint_mapping, skeleton, default_rotations)
        all_rotations.append(frame_rotations)
    
    return all_rotations

def get_all_joints(skeleton):
    """Get a list of all joints in the skeleton"""
    joints = []
    
    def collect_joints(joint):
        joints.append(joint)
        for child in joint.children:
            collect_joints(child)
    
    collect_joints(skeleton)
    return joints

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

def write_bvh_file(skeleton, frame_rotations, frame_time, output_file, global_positions):
    """Write the BVH file with motion data"""
    print(f"Writing BVH file to {output_file}...")
    try:
        with open(output_file, 'w') as f:
            # Write header
            f.write("HIERARCHY\n")
            
            # Write joint hierarchy recursively
            write_joint_hierarchy(f, skeleton, 0)
            
            # Write motion data
            num_frames = len(frame_rotations)
            f.write("MOTION\n")
            f.write(f"Frames: {num_frames}\n")
            f.write(f"Frame Time: {frame_time:.6f}\n")
            
            # For each frame, write position (for root only) and rotation data for all joints
            for frame_idx in tqdm(range(num_frames), desc="Writing animation data"):
                frame_rotations_data = frame_rotations[frame_idx]
                frame_data = []
                
                # Root position from hip positions
                if frame_idx < len(global_positions):
                    frame_data.extend(global_positions[frame_idx])
                else:
                    frame_data.extend([0.0, 0.0, 0.0])  # Default if not available
                
                # Write rotations for all joints in depth-first order
                write_joint_rotations(skeleton, frame_rotations_data, frame_data)
                
                f.write(" ".join([f"{val:.6f}" for val in frame_data]) + "\n")
                
        print(f"BVH file created successfully: {output_file}")
    except Exception as e:
        print(f"Error writing BVH file: {e}")

def write_joint_hierarchy(f, joint, indent_level):
    """Write the joint hierarchy recursively to the BVH file"""
    indent = "  " * indent_level
    
    if joint.parent is None:
        # Root joint
        f.write(f"{indent}ROOT {joint.name}\n")
    else:
        # Child joint
        f.write(f"{indent}JOINT {joint.name}\n")
    
    f.write(f"{indent}" + "{\n")
    
    # Write offset - verify it's not zero for non-root joints
    if joint.parent is not None and np.linalg.norm(joint.offset) < 1e-10:
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
    if joint.parent is None:
        # Root has 6 channels: position and rotation
        f.write(f"{indent}  CHANNELS 6 Xposition Yposition Zposition {joint.rotation_order[0]}rotation {joint.rotation_order[1]}rotation {joint.rotation_order[2]}rotation\n")
    else:
        # Other joints have 3 channels: rotation only
        f.write(f"{indent}  CHANNELS 3 {joint.rotation_order[0]}rotation {joint.rotation_order[1]}rotation {joint.rotation_order[2]}rotation\n")
    
    # Process children
    for child in joint.children:
        write_joint_hierarchy(f, child, indent_level + 1)
    
    # If no children, write end site
    if not joint.children:
        f.write(f"{indent}  End Site\n")
        f.write(f"{indent}  " + "{\n")
        
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
            f.write(f"{indent}    OFFSET 0.0 0.0 5.0\n")
        
        f.write(f"{indent}  " + "}\n")
    
    f.write(f"{indent}" + "}\n")

def write_joint_rotations(joint, frame_rotations, frame_data):
    """Write rotations for a joint and its children recursively"""
    # Add this joint's rotation
    if joint.name in frame_rotations:
        frame_data.extend(frame_rotations[joint.name])
    else:
        # Default to zero rotation if not found
        frame_data.extend([0.0, 0.0, 0.0])
    
    # Process all children in order
    for child in joint.children:
        write_joint_rotations(child, frame_rotations, frame_data)

def process_video(video_path, output_bvh, confidence_threshold=0.5, sample_rate=2, preview=False):
    """Process video and create BVH file"""
    print(f"Opening video file: {video_path}")
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Video properties: {width}x{height}, {fps} FPS, {frame_count} frames")
    print(f"Sampling every {sample_rate} frames, resulting in approximately {frame_count//sample_rate} animation frames")
    
    # Calculate frame time based on original FPS and sampling rate
    frame_time = 1.0 / (fps / sample_rate)
    
    # Set up preview window if enabled
    if preview:
        cv2.namedWindow('MediaPipe Pose Preview', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('MediaPipe Pose Preview', 800, 600)
    
    # Create pose detector with static_image_mode=False for video
    print("Initializing MediaPipe Pose detector...")
    with mp_pose.Pose(
        static_image_mode=False,          # Video mode
        model_complexity=2,               # Highest accuracy
        smooth_landmarks=True,            # Enable temporal smoothing
        enable_segmentation=False,        # No need for segmentation
        smooth_segmentation=False,
        min_detection_confidence=0.5,     # Initial detection confidence
        min_tracking_confidence=0.5       # Tracking confidence between frames
    ) as pose:
        
        # Process frames
        all_landmarks = []
        frame_idx = 0
        sampled_frames = 0
        
        print(f"Processing video frames (sampling every {sample_rate} frames)...")
        
        with tqdm(total=frame_count) as pbar:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process only every sample_rate frames for BVH
                process_for_bvh = (frame_idx % sample_rate == 0)
                
                # For preview, process every frame to make it smooth
                if preview or process_for_bvh:
                    # Convert BGR to RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # To improve performance, mark the image as not writeable
                    frame_rgb.flags.writeable = False
                    
                    # Process the frame
                    results = pose.process(frame_rgb)
                    
                    # Make image writeable again for drawing
                    frame_rgb.flags.writeable = True
                    
                    if process_for_bvh:
                        if results.pose_world_landmarks:
                            # Store landmarks for BVH
                            all_landmarks.append(results.pose_world_landmarks.landmark)
                            sampled_frames += 1
                        else:
                            # If no landmarks detected, use empty landmarks
                            empty_landmarks = [EmptyLandmark() for _ in range(33)]  # MediaPipe has 33 pose landmarks
                            all_landmarks.append(empty_landmarks)
                            print(f"Warning: No pose detected in frame {frame_idx}. Using empty landmarks.")
                    
                    # Draw pose landmarks on the frame for preview
                    if preview:
                        # Convert back to BGR for OpenCV
                        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                        
                        # Draw pose landmarks
                        if results.pose_landmarks:
                            mp_drawing.draw_landmarks(
                                frame_bgr,
                                results.pose_landmarks,
                                mp_pose.POSE_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=1)
                            )
                            
                            # Add frame info
                            cv2.putText(frame_bgr, f"Frame: {frame_idx}", (10, 30), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                            
                            # Add instruction for quitting
                            cv2.putText(frame_bgr, "Press 'q' to quit preview", (10, height - 10),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        
                        # Show the frame
                        cv2.imshow('MediaPipe Pose Preview', frame_bgr)
                        
                        # Break loop if 'q' is pressed
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            print("Preview stopped early by user.")
                            break
                
                frame_idx += 1
                pbar.update(1)
        
        # Close preview window if it was open
        if preview:
            cv2.destroyAllWindows()
            
        cap.release()
        
        if not all_landmarks:
            print("Error: No frames with detected poses found in the video.")
            return
        
        print(f"Video processing complete. Collected {len(all_landmarks)} frames of pose data.")
        
        # Create joint mapping
        joint_mapping = get_joint_mapping()
        
        # Find a good reference frame for the skeleton structure
        print("Finding a good reference frame for skeletal structure...")
        ref_frame_idx = 0
        best_detection_score = 0
        
        for i in range(min(len(all_landmarks), 30)):  # Check first 30 frames at most
            # Count how many key landmarks are detected
            landmarks = all_landmarks[i]
            detection_score = 0
            
            for idx in [mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.RIGHT_SHOULDER, 
                       mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.RIGHT_ELBOW,
                       mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.RIGHT_HIP,
                       mp_pose.PoseLandmark.LEFT_KNEE, mp_pose.PoseLandmark.RIGHT_KNEE]:
                pos = get_landmark_position(landmarks, idx)
                if pos is not None:
                    detection_score += 1
            
            if detection_score > best_detection_score:
                best_detection_score = detection_score
                ref_frame_idx = i
                
                # If all key points detected, break early
                if detection_score == 8:
                    break
        
        print(f"Using frame {ref_frame_idx} for skeletal structure (detection score: {best_detection_score}/8)")
        
        # Build the skeleton from the reference frame
        skeleton = build_skeleton_from_landmarks(all_landmarks[ref_frame_idx], joint_mapping)
        
        # Process global positions using the global position tracking method
        print("Calculating global position across frames...")
        global_positions = []
        previous_positions = None
        
        # Initialize global position
        global GLOBAL_ROOT_POSITION
        GLOBAL_ROOT_POSITION = np.zeros(3)
        
        for frame_idx, landmarks in enumerate(tqdm(all_landmarks, desc="Tracking global position")):
            # Extract positions for this frame
            positions = {}
            for idx, lm in enumerate(landmarks):
                pos = get_landmark_position([lm], 0)
                if pos is not None:
                    positions[idx] = pos
            
            # Calculate position delta if we have previous positions
            if frame_idx > 0 and previous_positions:
                position_delta = calculate_global_position_delta(
                    positions, previous_positions, width, height, GLOBAL_SCALE_FACTOR)
                
                # Accumulate the global position
                GLOBAL_ROOT_POSITION += position_delta
            
            # Store the global position for this frame
            global_positions.append(np.copy(GLOBAL_ROOT_POSITION))
            
            # Update previous positions for next frame
            previous_positions = positions
        
        # Process all frames to calculate limb rotations 
        frame_rotations = process_motion(all_landmarks, skeleton, joint_mapping)
        
        # Write BVH file with both accurate limb rotations and global position
        write_bvh_file(skeleton, frame_rotations, frame_time, output_bvh, global_positions)
        
        return skeleton, frame_rotations, global_positions

def preview_bvh_animation(bvh_file, speed_factor=1.0):
    """Preview a BVH animation using Matplotlib"""
    try:
        # Import required libraries
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation
        from mpl_toolkits.mplot3d import Axes3D
        
        # Parse the BVH file
        print(f"Loading BVH file for preview: {bvh_file}")
        with open(bvh_file, 'r') as f:
            content = f.read()
        
        # Split into hierarchy and motion sections
        if 'MOTION' in content:
            hierarchy_str, motion_str = content.split('MOTION', 1)
        else:
            print("Error: No MOTION section found in BVH file.")
            return
        
        # Parse joint hierarchy
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
        
        # Create a list of connections (parent-child pairs)
        connections = []
        for joint_name in joint_order:
            joint = joints[joint_name]
            for child_name in joint['children']:
                connections.append((joint_name, child_name))
        
        # Create figure and 3D axes
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Find the limits for the plot
        all_positions = []
        for frame_idx in range(min(10, len(motion_data))):  # Sample a few frames to estimate bounds
            positions = get_joint_positions(joints, joint_order, motion_data[frame_idx])
            for pos in positions.values():
                all_positions.append(pos)
        
        # Convert to numpy array for efficient processing
        all_positions = np.array(all_positions)
        
        # Handle case where all_positions is empty
        if len(all_positions) == 0:
            print("Warning: Could not determine any joint positions. Using default plot limits.")
            min_vals = np.array([-50, -50, -50])
            max_vals = np.array([50, 50, 50])
        else:
            min_vals = np.min(all_positions, axis=0)
            max_vals = np.max(all_positions, axis=0)
        
        center = (min_vals + max_vals) / 2
        
        # Ensure aspect ratio is equal
        max_range = max(max_vals[0] - min_vals[0], 
                        max_vals[1] - min_vals[1], 
                        max_vals[2] - min_vals[2]) / 2
        max_range = max(max_range, 50)  # Ensure a minimum viewing range
        
        ax.set_xlim(center[0] - max_range, center[0] + max_range)
        ax.set_ylim(center[1] - max_range, center[1] + max_range)
        ax.set_zlim(center[2] - max_range, center[2] + max_range)
        
        # Labels and title
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('BVH Animation Preview')
        
        # Create lines and points for the skeleton
        lines = [ax.plot([], [], [], 'b-')[0] for _ in connections]
        points = ax.plot([], [], [], 'ro', ms=6)[0]
        
        frame_text = ax.text2D(0.02, 0.98, '', transform=ax.transAxes)
        
        # Update function for animation
        def update(frame):
            # Get global positions for this frame
            frame_data = motion_data[frame]
            positions = get_joint_positions(joints, joint_order, frame_data)
            
            # Update lines (bones)
            for i, (parent, child) in enumerate(connections):
                if parent in positions and child in positions:
                    parent_pos = positions[parent]
                    child_pos = positions[child]
                    
                    xs = [parent_pos[0], child_pos[0]]
                    ys = [parent_pos[1], child_pos[1]]
                    zs = [parent_pos[2], child_pos[2]]
                    
                    lines[i].set_data(xs, ys)
                    lines[i].set_3d_properties(zs)
                else:
                    # If a joint is missing, just draw a tiny invisible line
                    lines[i].set_data([0, 0], [0, 0])
                    lines[i].set_3d_properties([0, 0])
            
            # Update points (joints)
            xs, ys, zs = [], [], []
            for joint in joint_order:
                if joint in positions:
                    pos = positions[joint]
                    xs.append(pos[0])
                    ys.append(pos[1])
                    zs.append(pos[2])
            
            points.set_data(xs, ys)
            points.set_3d_properties(zs)
            
            # Update frame text
            frame_text.set_text(f'Frame: {frame}')
            
            return lines + [points, frame_text]
        
        # Create animation
        num_frames = len(motion_data)
        print(f"Creating animation with {num_frames} frames...")
        
        interval = frame_time * 1000 / speed_factor  # Convert to milliseconds
        anim = FuncAnimation(
            fig, update, frames=num_frames, interval=interval, blit=True
        )
        
        # Display controls info
        print("Animation controls:")
        print("  - Left/Right arrows: Go backward/forward")
        print("  - Home/End: Go to first/last frame")
        print("  - Space: Play/Pause")
        print("  - Close the window to exit")
        
        plt.show()
        
    except ImportError:
        print("Error: This feature requires matplotlib. Please install it with:")
        print("  pip install matplotlib")
    except Exception as e:
        print(f"Error previewing BVH animation: {e}")
        import traceback
        traceback.print_exc()  # Print the full error traceback for debugging

def get_joint_positions(joints, joint_order, frame_data):
    """Calculate the global positions of all joints for a frame"""
    positions = {}
    rotations = {}
    
    # Process joints in order (depth-first)
    def process_joint(joint_name, parent_pos, parent_rot):
        # Get joint info
        joint = joints[joint_name]
        offset = joint['offset'] if joint['offset'] else [0, 0, 0]
        channels = joint['channels']
        
        # Start with current offset and parent position
        if parent_pos is None:
            # Root joint - position directly from frame data
            pos_index = 0
            if 'Xposition' in channels:
                pos_x = frame_data[pos_index]
                pos_index += 1
            else:
                pos_x = 0
                
            if 'Yposition' in channels:
                pos_y = frame_data[pos_index]
                pos_index += 1
            else:
                pos_y = 0
                
            if 'Zposition' in channels:
                pos_z = frame_data[pos_index]
                pos_index += 1
            else:
                pos_z = 0
                
            global_pos = np.array([pos_x, pos_y, pos_z])
            rot_start_index = pos_index
            
        else:
            # Child joint - position is parent position + offset transformed by parent rotation
            offset_transformed = transform_point(np.array(offset), parent_rot)
            global_pos = parent_pos + offset_transformed
            rot_start_index = 0
            
        # Extract rotation channels - simplified for now
        channel_indices = {'X': None, 'Y': None, 'Z': None}
        rot_values = {'X': 0, 'Y': 0, 'Z': 0}
        
        for i, channel in enumerate(channels):
            if 'rotation' in channel:
                axis = channel[0]  # X, Y, or Z
                channel_indices[axis] = rot_start_index + i - (3 if parent_pos is None else 0)
        
        for axis, index in channel_indices.items():
            if index is not None and index < len(frame_data):
                rot_values[axis] = frame_data[index] * (math.pi / 180.0)  # Convert to radians
                
        # Calculate rotation matrix
        rot_matrix = euler_to_rotation_matrix([rot_values['X'], rot_values['Y'], rot_values['Z']])
        
        # Combine with parent rotation
        if parent_rot is not None:
            global_rot = np.matmul(parent_rot, rot_matrix)
        else:
            global_rot = rot_matrix
            
        # Store position and rotation
        positions[joint_name] = global_pos
        rotations[joint_name] = global_rot
        
        # Process children
        for child_name in joint['children']:
            process_joint(child_name, global_pos, global_rot)
    
    # Start with the root joint
    root_joint = joint_order[0]
    process_joint(root_joint, None, None)
    
    return positions

def main():
    parser = argparse.ArgumentParser(description="Convert video to BVH using MediaPipe")
    parser.add_argument("--video", required=True, help="Path to input video file")
    parser.add_argument("--output", required=True, help="Path to output BVH file")
    parser.add_argument("--confidence", type=float, default=0.5, help="Confidence threshold for pose detection")
    parser.add_argument("--sample-rate", type=int, default=2, help="Process every Nth frame")
    parser.add_argument("--preview", action="store_true", help="Show preview window with pose detection visualization")
    parser.add_argument("--preview-bvh", action="store_true", help="Preview the final BVH animation after processing")
    parser.add_argument("--speed", type=float, default=1.0, help="Speed factor for BVH preview (default: 1.0)")
    
    args = parser.parse_args()
    
    print("Starting MediaPipe to BVH conversion (Combined Version)...")
    start_time = time.time()
    
    process_video(args.video, args.output, args.confidence, args.sample_rate, args.preview)
    
    end_time = time.time()
    print(f"Conversion completed in {end_time - start_time:.2f} seconds")
    
    # Preview BVH animation if requested
    if args.preview_bvh:
        preview_bvh_animation(args.output, args.speed)

if __name__ == "__main__":
    main()