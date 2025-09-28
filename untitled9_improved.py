#!/usr/bin/env python3
"""
Improved MediaPipe to BVH Converter with Ground Contact Detection
Based on untitled9.py with added foot planting and ground contact features
"""

import cv2
import mediapipe as mp
import numpy as np
import argparse
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# ============= GROUND CONTACT DETECTION =============

class GroundContactDetector:
    """Detects and manages foot ground contacts to prevent sliding"""
    
    def __init__(self, ground_threshold=5.0, velocity_threshold=2.0, lock_threshold=3.0):
        """
        ground_threshold: Y distance from ground to consider contact (cm)
        velocity_threshold: Max Y velocity to consider stationary (cm/frame)
        lock_threshold: Max horizontal movement allowed when locked (cm)
        """
        self.ground_threshold = ground_threshold
        self.velocity_threshold = velocity_threshold
        self.lock_threshold = lock_threshold
        
        # Track foot states
        self.left_foot_locked = False
        self.right_foot_locked = False
        self.left_foot_lock_pos = None
        self.right_foot_lock_pos = None
        
        # Track previous positions for velocity calculation
        self.prev_left_pos = None
        self.prev_right_pos = None
        
        # Ground plane estimation
        self.ground_y = None
        self.ground_history = []
        
    def update_ground_plane(self, left_foot_y, right_foot_y):
        """Estimate ground plane from foot positions"""
        min_y = min(left_foot_y, right_foot_y)
        self.ground_history.append(min_y)
        
        # Keep only recent history
        if len(self.ground_history) > 30:
            self.ground_history.pop(0)
        
        # Use median of recent minimum Y values as ground estimate
        if len(self.ground_history) >= 5:
            self.ground_y = np.median(self.ground_history)
        else:
            self.ground_y = min_y
    
    def detect_contact(self, foot_pos, prev_pos, foot_name):
        """Detect if foot is in contact with ground"""
        if self.ground_y is None:
            return False
        
        # Check if foot is near ground
        near_ground = (foot_pos[1] - self.ground_y) < self.ground_threshold
        
        # Calculate velocity if we have previous position
        if prev_pos is not None:
            y_velocity = abs(foot_pos[1] - prev_pos[1])
            stationary = y_velocity < self.velocity_threshold
        else:
            stationary = True
        
        return near_ground and stationary
    
    def process_feet(self, left_foot_pos, right_foot_pos):
        """Process both feet and apply ground contact constraints"""
        # Update ground plane
        self.update_ground_plane(left_foot_pos[1], right_foot_pos[1])
        
        # Process left foot
        left_corrected = self._process_single_foot(
            left_foot_pos, self.prev_left_pos, 'left',
            self.left_foot_locked, self.left_foot_lock_pos
        )
        
        # Process right foot
        right_corrected = self._process_single_foot(
            right_foot_pos, self.prev_right_pos, 'right',
            self.right_foot_locked, self.right_foot_lock_pos
        )
        
        # Update previous positions
        self.prev_left_pos = left_corrected
        self.prev_right_pos = right_corrected
        
        return left_corrected, right_corrected
    
    def _process_single_foot(self, foot_pos, prev_pos, foot_name, is_locked, lock_pos):
        """Process a single foot with ground contact logic"""
        corrected_pos = foot_pos.copy()
        
        # Check for ground contact
        in_contact = self.detect_contact(foot_pos, prev_pos, foot_name)
        
        if in_contact:
            if not is_locked:
                # Start locking the foot
                if foot_name == 'left':
                    self.left_foot_locked = True
                    self.left_foot_lock_pos = foot_pos.copy()
                else:
                    self.right_foot_locked = True
                    self.right_foot_lock_pos = foot_pos.copy()
                lock_pos = foot_pos.copy()
            
            # Apply lock - keep horizontal position, allow small vertical movement
            if lock_pos is not None:
                corrected_pos[0] = lock_pos[0]  # Lock X
                corrected_pos[2] = lock_pos[2]  # Lock Z
                # Allow small Y movement but keep near ground
                corrected_pos[1] = max(foot_pos[1], self.ground_y)
        else:
            # Release lock if foot is moving away from ground
            if foot_name == 'left':
                self.left_foot_locked = False
                self.left_foot_lock_pos = None
            else:
                self.right_foot_locked = False
                self.right_foot_lock_pos = None
        
        return corrected_pos

# ============= SMOOTHING FUNCTIONS =============

def apply_temporal_smoothing(positions, window_size=3, preserve_dynamics=True):
    """Apply temporal smoothing while preserving motion dynamics"""
    if len(positions) < window_size:
        return positions
    
    smoothed = []
    half_window = window_size // 2
    
    for i in range(len(positions)):
        # Calculate window bounds
        start = max(0, i - half_window)
        end = min(len(positions), i + half_window + 1)
        
        # Get window of positions
        window = positions[start:end]
        
        if preserve_dynamics and len(window) > 2:
            # Use weighted average that preserves motion dynamics
            weights = np.exp(-0.5 * np.arange(len(window))**2)
            weights = weights / weights.sum()
            smoothed_pos = np.average(window, axis=0, weights=weights)
        else:
            # Simple moving average
            smoothed_pos = np.mean(window, axis=0)
        
        smoothed.append(smoothed_pos)
    
    return smoothed

# ============= ORIGINAL FUNCTIONS WITH MODIFICATIONS =============

def parse_bvh(file_path):
    """Parse a BVH file and extract joint hierarchy and motion data"""
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    # Find where MOTION section starts
    motion_start = 0
    for i, line in enumerate(lines):
        if line.strip() == "MOTION":
            motion_start = i
            break
    
    # Parse hierarchy
    hierarchy_lines = lines[:motion_start]
    
    # Parse joint structure
    joints = {}
    joint_stack = []
    current_joint = None
    joint_order = []  # Keep track of joint order for motion data
    
    for line in hierarchy_lines:
        line = line.strip()
        
        if line.startswith("ROOT") or line.startswith("JOINT"):
            # Extract joint name
            parts = line.split()
            joint_name = parts[1] if len(parts) > 1 else "unnamed"
            
            # Create joint entry
            parent = joint_stack[-1] if joint_stack else None
            joints[joint_name] = {
                'parent': parent,
                'offset': [0, 0, 0],
                'channels': [],
                'children': []
            }
            
            # Update parent's children list
            if parent:
                joints[parent]['children'].append(joint_name)
            
            # Update stack
            joint_stack.append(joint_name)
            current_joint = joint_name
            joint_order.append(joint_name)
            
        elif line.startswith("OFFSET"):
            # Parse offset
            parts = line.split()
            offset = [float(parts[1]), float(parts[2]), float(parts[3])]
            if current_joint:
                joints[current_joint]['offset'] = offset
                
        elif line.startswith("CHANNELS"):
            # Parse channels
            parts = line.split()
            num_channels = int(parts[1])
            channels = parts[2:2+num_channels]
            if current_joint:
                joints[current_joint]['channels'] = channels
                
        elif line == "}":
            # End of joint definition
            if joint_stack:
                joint_stack.pop()
                current_joint = joint_stack[-1] if joint_stack else None
    
    # Parse motion data
    motion_lines = lines[motion_start+1:]
    frames = 0
    frame_time = 0.0
    motion_data = []
    
    for line in motion_lines:
        line = line.strip()
        if line.startswith("Frames:"):
            frames = int(line.split(":")[1])
        elif line.startswith("Frame Time:"):
            frame_time = float(line.split(":")[1])
        elif line and not line.startswith("Frame"):
            # This is motion data
            values = [float(x) for x in line.split()]
            if values:
                motion_data.append(values)
    
    return {
        'joints': joints,
        'joint_order': joint_order,
        'frames': frames,
        'frame_time': frame_time,
        'motion_data': motion_data
    }

def get_joint_positions(joints, joint_order, frame_data):
    """Calculate world positions of all joints for a given frame"""
    positions = {}
    rotations = {}
    
    # Process joints in order, accumulating transformations
    def process_joint(joint_name, parent_pos, parent_rot):
        joint = joints[joint_name]
        
        # Get channels for this joint
        channel_data = []
        for channel in joint['channels']:
            if frame_data:
                channel_data.append(frame_data.pop(0))
            else:
                channel_data.append(0)
        
        # Apply transformations
        pos = parent_pos.copy()
        rot = parent_rot.copy()
        
        # Process channels
        for i, channel in enumerate(joint['channels']):
            if 'position' in channel:
                if 'X' in channel:
                    pos[0] += channel_data[i]
                elif 'Y' in channel:
                    pos[1] += channel_data[i]
                elif 'Z' in channel:
                    pos[2] += channel_data[i]
            elif 'rotation' in channel:
                # For simplicity, accumulate rotations
                if 'X' in channel:
                    rot[0] += channel_data[i]
                elif 'Y' in channel:
                    rot[1] += channel_data[i]
                elif 'Z' in channel:
                    rot[2] += channel_data[i]
        
        # Apply offset
        offset = np.array(joint['offset'])
        
        # Apply rotation to offset
        rotated_offset = transform_point(offset, euler_to_rotation_matrix(np.radians(rot)))
        final_pos = pos + rotated_offset
        
        positions[joint_name] = final_pos
        rotations[joint_name] = rot
        
        # Process children
        for child in joint['children']:
            process_joint(child, final_pos, rot)
    
    # Find root joint
    root = None
    for joint_name, joint in joints.items():
        if joint['parent'] is None:
            root = joint_name
            break
    
    if root:
        frame_data_copy = frame_data.copy()
        process_joint(root, np.array([0, 0, 0]), np.array([0, 0, 0]))
    
    return positions

def euler_to_rotation_matrix(euler_angles):
    """Convert Euler angles (in radians) to rotation matrix"""
    x, y, z = euler_angles
    
    # Rotation around X axis
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(x), -np.sin(x)],
        [0, np.sin(x), np.cos(x)]
    ])
    
    # Rotation around Y axis
    Ry = np.array([
        [np.cos(y), 0, np.sin(y)],
        [0, 1, 0],
        [-np.sin(y), 0, np.cos(y)]
    ])
    
    # Rotation around Z axis
    Rz = np.array([
        [np.cos(z), -np.sin(z), 0],
        [np.sin(z), np.cos(z), 0],
        [0, 0, 1]
    ])
    
    # Combined rotation (order: ZYX)
    R = Rz @ Ry @ Rx
    return R

def transform_point(point, rotation_matrix):
    """Transform a 3D point using a rotation matrix"""
    return rotation_matrix @ point

def preview_bvh_animation(bvh_file, speed_factor=1.0):
    """Preview BVH animation in 3D"""
    print(f"Previewing BVH animation: {bvh_file}")
    
    # Parse BVH file
    bvh_data = parse_bvh(bvh_file)
    
    # Set up the figure and 3D axis
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Define connections between joints (simplified skeleton)
    connections = [
        ('Hips', 'Spine'),
        ('Spine', 'Chest'),
        ('Chest', 'Neck'),
        ('Neck', 'Head'),
        ('Chest', 'LeftShoulder'),
        ('LeftShoulder', 'LeftArm'),
        ('LeftArm', 'LeftForeArm'),
        ('LeftForeArm', 'LeftHand'),
        ('Chest', 'RightShoulder'),
        ('RightShoulder', 'RightArm'),
        ('RightArm', 'RightForeArm'),
        ('RightForeArm', 'RightHand'),
        ('Hips', 'LeftUpLeg'),
        ('LeftUpLeg', 'LeftLeg'),
        ('LeftLeg', 'LeftFoot'),
        ('Hips', 'RightUpLeg'),
        ('RightUpLeg', 'RightLeg'),
        ('RightLeg', 'RightFoot')
    ]
    
    # Check if motion data exists
    if not bvh_data['motion_data']:
        print("No motion data found in BVH file")
        ax.set_title("No motion data available")
        plt.show()
        return
    
    # Analyze the data to set appropriate limits
    print("Analyzing motion data...")
    all_positions = []
    sample_frames = range(0, len(bvh_data['motion_data']), max(1, len(bvh_data['motion_data']) // 10))
    
    for frame_idx in sample_frames:
        frame_data = bvh_data['motion_data'][frame_idx]
        positions = get_joint_positions(bvh_data['joints'], bvh_data['joint_order'], frame_data.copy())
        if positions:
            all_positions.extend(positions.values())
    
    if all_positions:
        all_positions = np.array(all_positions)
        
        # Calculate ranges with some padding
        padding = 20
        x_range = [all_positions[:, 0].min() - padding, all_positions[:, 0].max() + padding]
        y_range = [all_positions[:, 1].min() - padding, all_positions[:, 1].max() + padding]
        z_range = [all_positions[:, 2].min() - padding, all_positions[:, 2].max() + padding]
        
        # Make sure we have a reasonable view
        max_range = max(x_range[1] - x_range[0], y_range[1] - y_range[0], z_range[1] - z_range[0])
        if max_range < 100:
            center_x = (x_range[0] + x_range[1]) / 2
            center_y = (y_range[0] + y_range[1]) / 2
            center_z = (z_range[0] + z_range[1]) / 2
            x_range = [center_x - 100, center_x + 100]
            y_range = [center_y - 100, center_y + 100]
            z_range = [center_z - 100, center_z + 100]
    else:
        print("Warning: Could not determine joint positions. Using default plot limits.")
        x_range = [-100, 100]
        y_range = [0, 200]
        z_range = [-100, 100]
    
    print(f"Plot ranges - X: {x_range}, Y: {y_range}, Z: {z_range}")
    
    # Animation function
    def update(frame):
        ax.clear()
        
        # Get frame data
        frame_idx = int(frame * speed_factor) % len(bvh_data['motion_data'])
        frame_data = bvh_data['motion_data'][frame_idx]
        
        # Calculate joint positions
        positions = get_joint_positions(bvh_data['joints'], bvh_data['joint_order'], frame_data.copy())
        
        # Plot joints
        if positions:
            coords = np.array(list(positions.values()))
            ax.scatter(coords[:, 0], coords[:, 2], coords[:, 1], c='red', s=30, alpha=0.6)
            
            # Plot connections
            for parent, child in connections:
                if parent in positions and child in positions:
                    p1 = positions[parent]
                    p2 = positions[child]
                    ax.plot([p1[0], p2[0]], [p1[2], p2[2]], [p1[1], p2[1]], 'b-', linewidth=2)
        
        # Set labels and limits
        ax.set_xlabel('X')
        ax.set_ylabel('Z')
        ax.set_zlabel('Y')
        ax.set_xlim(x_range)
        ax.set_ylim(z_range)
        ax.set_zlim(y_range)
        ax.set_title(f'BVH Animation - Frame {frame_idx}/{len(bvh_data["motion_data"])}')
        
        # Set view angle for better visualization
        ax.view_init(elev=10, azim=45)
        
        return ax,
    
    # Create animation
    frames = len(bvh_data['motion_data'])
    interval = bvh_data['frame_time'] * 1000 / speed_factor  # Convert to milliseconds
    
    anim = FuncAnimation(fig, update, frames=frames, interval=interval, blit=False, repeat=True)
    
    print(f"Animation ready:")
    print(f"  - Total frames: {frames}")
    print(f"  - Frame time: {bvh_data['frame_time']:.3f} seconds")
    print(f"  - Speed factor: {speed_factor}x")
    print("  - Close the window to exit")
    
    plt.show()

from dataclasses import dataclass

# Placeholder for Landmark when missing
@dataclass
class EmptyLandmark:
    """Simple class to substitute for MediaPipe landmarks when needed"""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    visibility: float = 0.0

class Joint:
    """Represents a joint in the BVH skeleton"""
    def __init__(self, name, parent=None):
        self.name = name
        self.parent = parent
        self.children = []
        self.offset = np.array([0.0, 0.0, 0.0])
        
        # For root joint (Hips), store position for each frame
        self.positions = []
        # For all joints, store rotation for each frame
        self.rotations = []
    
    def add_child(self, child):
        self.children.append(child)

def create_skeleton():
    """Create the BVH skeleton hierarchy matching MediaPipe landmarks"""
    # Create root joint (Hips)
    hips = Joint("Hips")
    
    # Spine chain
    spine = Joint("Spine", hips)
    chest = Joint("Chest", spine)
    neck = Joint("Neck", chest)
    head = Joint("Head", neck)
    
    # Left arm chain
    left_shoulder = Joint("LeftShoulder", chest)
    left_arm = Joint("LeftArm", left_shoulder)
    left_forearm = Joint("LeftForeArm", left_arm)
    left_hand = Joint("LeftHand", left_forearm)
    
    # Right arm chain
    right_shoulder = Joint("RightShoulder", chest)
    right_arm = Joint("RightArm", right_shoulder)
    right_forearm = Joint("RightForeArm", right_arm)
    right_hand = Joint("RightHand", right_forearm)
    
    # Left leg chain
    left_upleg = Joint("LeftUpLeg", hips)
    left_leg = Joint("LeftLeg", left_upleg)
    left_foot = Joint("LeftFoot", left_leg)
    
    # Right leg chain
    right_upleg = Joint("RightUpLeg", hips)
    right_leg = Joint("RightLeg", right_upleg)
    right_foot = Joint("RightFoot", right_leg)
    
    # Build hierarchy
    hips.add_child(spine)
    hips.add_child(left_upleg)
    hips.add_child(right_upleg)
    
    spine.add_child(chest)
    
    chest.add_child(neck)
    chest.add_child(left_shoulder)
    chest.add_child(right_shoulder)
    
    neck.add_child(head)
    
    left_shoulder.add_child(left_arm)
    left_arm.add_child(left_forearm)
    left_forearm.add_child(left_hand)
    
    right_shoulder.add_child(right_arm)
    right_arm.add_child(right_forearm)
    right_forearm.add_child(right_hand)
    
    left_upleg.add_child(left_leg)
    left_leg.add_child(left_foot)
    
    right_upleg.add_child(right_leg)
    right_leg.add_child(right_foot)
    
    return hips

def get_joint_mapping():
    """Map MediaPipe landmarks to BVH joints"""
    return {
        # Using average of hip landmarks for pelvis/hip center
        'Hips': [mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.RIGHT_HIP],
        
        # Spine approximation (between hips and shoulders)
        'Spine': [mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.RIGHT_HIP,
                  mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.RIGHT_SHOULDER],
        
        # Chest (shoulder center)
        'Chest': [mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.RIGHT_SHOULDER],
        
        # Neck (between shoulders and ears/nose)
        'Neck': [mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.RIGHT_SHOULDER,
                 mp_pose.PoseLandmark.LEFT_EAR, mp_pose.PoseLandmark.RIGHT_EAR],
        
        # Head
        'Head': [mp_pose.PoseLandmark.NOSE, mp_pose.PoseLandmark.LEFT_EAR, mp_pose.PoseLandmark.RIGHT_EAR],
        
        # Arms
        'LeftShoulder': [mp_pose.PoseLandmark.LEFT_SHOULDER],
        'LeftArm': [mp_pose.PoseLandmark.LEFT_ELBOW],
        'LeftForeArm': [mp_pose.PoseLandmark.LEFT_WRIST],
        'LeftHand': [mp_pose.PoseLandmark.LEFT_WRIST, mp_pose.PoseLandmark.LEFT_PINKY, 
                     mp_pose.PoseLandmark.LEFT_INDEX],
        
        'RightShoulder': [mp_pose.PoseLandmark.RIGHT_SHOULDER],
        'RightArm': [mp_pose.PoseLandmark.RIGHT_ELBOW],
        'RightForeArm': [mp_pose.PoseLandmark.RIGHT_WRIST],
        'RightHand': [mp_pose.PoseLandmark.RIGHT_WRIST, mp_pose.PoseLandmark.RIGHT_PINKY,
                      mp_pose.PoseLandmark.RIGHT_INDEX],
        
        # Legs
        'LeftUpLeg': [mp_pose.PoseLandmark.LEFT_HIP],
        'LeftLeg': [mp_pose.PoseLandmark.LEFT_KNEE],
        'LeftFoot': [mp_pose.PoseLandmark.LEFT_ANKLE, mp_pose.PoseLandmark.LEFT_HEEL, 
                     mp_pose.PoseLandmark.LEFT_FOOT_INDEX],
        
        'RightUpLeg': [mp_pose.PoseLandmark.RIGHT_HIP],
        'RightLeg': [mp_pose.PoseLandmark.RIGHT_KNEE],
        'RightFoot': [mp_pose.PoseLandmark.RIGHT_ANKLE, mp_pose.PoseLandmark.RIGHT_HEEL,
                      mp_pose.PoseLandmark.RIGHT_FOOT_INDEX],
    }

def get_landmark_position(landmarks, idx):
    """Safely get landmark position"""
    if idx < len(landmarks):
        lm = landmarks[idx]
        if hasattr(lm, 'visibility') and lm.visibility > 0.5:
            return np.array([lm.x, lm.y, lm.z])
    return None

def get_joint_position(joint_name, landmarks, joint_mapping):
    """Get the position of a joint from MediaPipe landmarks"""
    landmark_indices = joint_mapping.get(joint_name, [])
    positions = []
    
    for idx in landmark_indices:
        pos = get_landmark_position(landmarks, idx)
        if pos is not None:
            positions.append(pos)
    
    if positions:
        # Average multiple landmark positions
        return np.mean(positions, axis=0)
    return None

def ensure_minimum_offset(offset_vector, min_length=0.05, default_direction=None):
    """Ensure offset vector has minimum length to avoid zero-length bones"""
    length = np.linalg.norm(offset_vector)
    
    if length < min_length:
        # If vector is too short, use default direction or normalize and scale existing vector
        if default_direction is not None:
            # Use the default direction, normalized
            direction = default_direction / np.linalg.norm(default_direction)
        elif length > 0:
            # Scale up the existing vector
            direction = offset_vector / length
        else:
            # If zero vector and no default, use up direction
            direction = np.array([0, 1, 0])
        
        return direction * min_length
    
    return offset_vector

def get_default_bone_offsets():
    """Define default offsets for bones when landmarks aren't reliable"""
    return {
        'Spine': np.array([0, 10, 0]),      # Upward from hips
        'Chest': np.array([0, 15, 0]),      # Upward from spine
        'Neck': np.array([0, 5, 0]),        # Upward from chest
        'Head': np.array([0, 10, 0]),       # Upward from neck
        'LeftShoulder': np.array([-8, 2, 0]),   # Left and slightly up from chest
        'RightShoulder': np.array([8, 2, 0]),   # Right and slightly up from chest
        'LeftArm': np.array([-15, -5, 0]),      # Left arm down
        'RightArm': np.array([15, -5, 0]),      # Right arm down
        'LeftForeArm': np.array([-12, -8, 0]),  # Continue left arm
        'RightForeArm': np.array([12, -8, 0]),  # Continue right arm
        'LeftHand': np.array([-5, -5, 0]),      # Hand
        'RightHand': np.array([5, -5, 0]),      # Hand
        'LeftUpLeg': np.array([-5, -10, 0]),    # Left leg down
        'RightUpLeg': np.array([5, -10, 0]),    # Right leg down
        'LeftLeg': np.array([0, -20, 0]),       # Shin down
        'RightLeg': np.array([0, -20, 0]),      # Shin down
        'LeftFoot': np.array([0, -5, 5]),       # Foot forward
        'RightFoot': np.array([0, -5, 5]),      # Foot forward
    }

def build_skeleton_from_landmarks(frame_landmarks, joint_mapping, scale=100.0):
    """Build skeleton structure from a reference frame of landmarks"""
    skeleton = create_skeleton()
    default_offsets = get_default_bone_offsets()
    
    # Process joints to calculate offsets based on landmark positions
    def process_joint(joint):
        joint_pos = get_joint_position(joint.name, frame_landmarks, joint_mapping)
        
        if joint.parent:
            parent_pos = get_joint_position(joint.parent.name, frame_landmarks, joint_mapping)
            
            if joint_pos is not None and parent_pos is not None:
                # Calculate offset from parent to this joint (in centimeters)
                offset = (joint_pos - parent_pos) * scale
                
                # MediaPipe Y is down, BVH Y is up
                offset[1] = -offset[1]
                
                # Use default direction for minimum offset
                default_direction = default_offsets.get(joint.name, None)
                joint.offset = ensure_minimum_offset(offset, min_length=1.0, default_direction=default_direction)
            else:
                # Use default offset
                joint.offset = default_offsets.get(joint.name, np.array([0, 5, 0]))
        
        # Process children
        for child in joint.children:
            process_joint(child)
    
    process_joint(skeleton)
    return skeleton

def calculate_rotation_between_vectors(v1, v2):
    """Calculate rotation that transforms v1 to v2"""
    # Normalize vectors
    v1 = v1 / (np.linalg.norm(v1) + 1e-10)
    v2 = v2 / (np.linalg.norm(v2) + 1e-10)
    
    # Calculate axis of rotation (cross product)
    axis = np.cross(v1, v2)
    axis_length = np.linalg.norm(axis)
    
    # If vectors are parallel, no rotation needed
    if axis_length < 1e-10:
        return np.zeros(3)
    
    axis = axis / axis_length
    
    # Calculate angle
    angle = np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0))
    
    # Convert to Euler angles (simplified - proper conversion would use rotation matrices)
    # This is a basic approximation
    euler = axis * angle
    return np.degrees(euler)

def calculate_joint_rotation(joint_name, current_pos, parent_pos, child_pos=None, 
                           rest_offset=None):
    """Calculate rotation for a joint based on current positions"""
    if rest_offset is None or np.linalg.norm(rest_offset) < 0.01:
        return np.zeros(3)
    
    if current_pos is None or parent_pos is None:
        return np.zeros(3)
    
    # Current vector from parent to joint
    current_vector = current_pos - parent_pos
    
    # MediaPipe Y is down, BVH Y is up
    current_vector[1] = -current_vector[1]
    
    # Rest pose vector (the offset)
    rest_vector = rest_offset
    
    # Calculate rotation needed to go from rest to current
    rotation = calculate_rotation_between_vectors(rest_vector, current_vector)
    
    # Apply some constraints for more natural motion
    # Limit extreme rotations
    rotation = np.clip(rotation, -180, 180)
    
    return rotation

def calculate_frame_rotations(frame_landmarks, joint_mapping, skeleton, default_rotations):
    """Calculate rotations for all joints in a single frame"""
    frame_rotations = {}
    
    def process_joint(joint):
        joint_pos = get_joint_position(joint.name, frame_landmarks, joint_mapping)
        
        if joint.parent:
            parent_pos = get_joint_position(joint.parent.name, frame_landmarks, joint_mapping)
            
            # Get child position for better rotation calculation
            child_pos = None
            if joint.children:
                # Use first child for orientation
                child_pos = get_joint_position(joint.children[0].name, frame_landmarks, joint_mapping)
            
            if joint_pos is not None and parent_pos is not None:
                # Scale positions
                joint_pos = joint_pos * 100  # Convert to centimeters
                parent_pos = parent_pos * 100
                if child_pos is not None:
                    child_pos = child_pos * 100
                
                rotation = calculate_joint_rotation(
                    joint.name, joint_pos, parent_pos, child_pos, joint.offset
                )
                frame_rotations[joint.name] = rotation
            else:
                frame_rotations[joint.name] = default_rotations.get(joint.name, np.zeros(3))
        else:
            # Root joint - no rotation, just position
            frame_rotations[joint.name] = np.zeros(3)
        
        # Process children
        for child in joint.children:
            process_joint(child)
    
    process_joint(skeleton)
    return frame_rotations

def process_motion(frames_landmarks, skeleton, joint_mapping):
    """Process all frames to calculate rotations for animation with ground contact"""
    num_frames = len(frames_landmarks)
    print(f"Calculating joint rotations for {num_frames} frames...")
    
    # Initialize ground contact detector
    ground_detector = GroundContactDetector()
    
    # Calculate default rotations (all zeros)
    default_rotations = {joint.name: np.zeros(3) for joint in get_all_joints(skeleton)}
    
    # First pass: collect all foot positions for smoothing
    left_foot_positions = []
    right_foot_positions = []
    
    for frame_idx in range(num_frames):
        frame_landmarks = frames_landmarks[frame_idx]
        left_foot_pos = get_joint_position('LeftFoot', frame_landmarks, joint_mapping)
        right_foot_pos = get_joint_position('RightFoot', frame_landmarks, joint_mapping)
        
        if left_foot_pos is not None:
            left_foot_positions.append(left_foot_pos * 100)  # Convert to cm
        else:
            left_foot_positions.append(np.array([0, 0, 0]))
            
        if right_foot_pos is not None:
            right_foot_positions.append(right_foot_pos * 100)  # Convert to cm
        else:
            right_foot_positions.append(np.array([0, 0, 0]))
    
    # Apply temporal smoothing to foot positions
    left_foot_positions = apply_temporal_smoothing(left_foot_positions, window_size=3)
    right_foot_positions = apply_temporal_smoothing(right_foot_positions, window_size=3)
    
    # Second pass: process with ground contact detection
    all_rotations = []
    corrected_foot_positions = []
    
    for frame_idx in tqdm(range(num_frames), desc="Processing frames with ground contact"):
        frame_landmarks = frames_landmarks[frame_idx]
        
        # Get smoothed foot positions
        left_foot = left_foot_positions[frame_idx]
        right_foot = right_foot_positions[frame_idx]
        
        # Apply ground contact detection
        left_corrected, right_corrected = ground_detector.process_feet(left_foot, right_foot)
        corrected_foot_positions.append((left_corrected, right_corrected))
        
        # Create modified landmarks with corrected foot positions
        modified_landmarks = list(frame_landmarks)
        
        # Update foot landmark positions (scale back to normalized units)
        if frame_idx < len(frames_landmarks):
            # Update left foot landmarks
            for idx in [mp_pose.PoseLandmark.LEFT_ANKLE, mp_pose.PoseLandmark.LEFT_HEEL, 
                       mp_pose.PoseLandmark.LEFT_FOOT_INDEX]:
                if idx < len(modified_landmarks):
                    lm = modified_landmarks[idx]
                    if hasattr(lm, 'x'):
                        # Apply correction (convert back from cm)
                        lm.x = left_corrected[0] / 100
                        lm.y = -left_corrected[1] / 100  # Flip Y back
                        lm.z = left_corrected[2] / 100
            
            # Update right foot landmarks
            for idx in [mp_pose.PoseLandmark.RIGHT_ANKLE, mp_pose.PoseLandmark.RIGHT_HEEL,
                       mp_pose.PoseLandmark.RIGHT_FOOT_INDEX]:
                if idx < len(modified_landmarks):
                    lm = modified_landmarks[idx]
                    if hasattr(lm, 'x'):
                        # Apply correction (convert back from cm)
                        lm.x = right_corrected[0] / 100
                        lm.y = -right_corrected[1] / 100  # Flip Y back
                        lm.z = right_corrected[2] / 100
        
        # Calculate frame rotations with corrected positions
        frame_rotations = calculate_frame_rotations(modified_landmarks, joint_mapping, skeleton, default_rotations)
        all_rotations.append(frame_rotations)
    
    # Print ground contact statistics
    print(f"Ground contacts - Left foot: {ground_detector.left_foot_locked}, Right foot: {ground_detector.right_foot_locked}")
    
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

def write_bvh_file(skeleton, frame_rotations, frame_time, output_file):
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
            
            # Calculate hip positions for each frame
            hip_positions = calculate_hip_positions(frame_rotations)
            
            # For each frame, write position (for root only) and rotation data for all joints
            for frame_idx in tqdm(range(num_frames), desc="Writing animation data"):
                frame_data = []
                
                # Write root position and rotation
                hip_pos = hip_positions[frame_idx]
                frame_data.extend([hip_pos[0], hip_pos[1], hip_pos[2]])  # Position
                
                # Write rotations for all joints in order
                write_joint_rotations(skeleton, frame_rotations[frame_idx], frame_data)
                
                # Write the frame data as a single line
                f.write(" ".join([f"{val:.6f}" for val in frame_data]) + "\n")
        
        print(f"BVH file created successfully: {output_file}")
        return True
        
    except Exception as e:
        print(f"Error writing BVH file: {e}")
        return False

def write_joint_hierarchy(f, joint, level):
    """Recursively write joint hierarchy"""
    indent = "  " * level
    
    if level == 0:
        f.write(f"{indent}ROOT {joint.name}\n")
    else:
        f.write(f"{indent}JOINT {joint.name}\n")
    
    f.write(f"{indent}{{\n")
    f.write(f"{indent}  OFFSET {joint.offset[0]:.6f} {joint.offset[1]:.6f} {joint.offset[2]:.6f}\n")
    
    # Channels
    if level == 0:
        # Root has position and rotation
        f.write(f"{indent}  CHANNELS 6 Xposition Yposition Zposition Zrotation Yrotation Xrotation\n")
    else:
        # Other joints only have rotation
        f.write(f"{indent}  CHANNELS 3 Zrotation Yrotation Xrotation\n")
    
    # Write children
    for child in joint.children:
        write_joint_hierarchy(f, child, level + 1)
    
    # Add end site for leaf joints
    if not joint.children:
        f.write(f"{indent}  End Site\n")
        f.write(f"{indent}  {{\n")
        # Use a small offset for end site
        end_offset = joint.offset * 0.3 if np.linalg.norm(joint.offset) > 0 else np.array([0, -5, 0])
        f.write(f"{indent}    OFFSET {end_offset[0]:.6f} {end_offset[1]:.6f} {end_offset[2]:.6f}\n")
        f.write(f"{indent}  }}\n")
    
    f.write(f"{indent}}}\n")

def calculate_hip_positions(frame_rotations):
    """Calculate hip positions for each frame (simplified - just returns origin)"""
    # For now, keep hips at origin. Could be enhanced to track actual hip movement
    return [np.array([0.0, 60.0, 0.0]) for _ in frame_rotations]

def write_joint_rotations(joint, frame_rotations, frame_data):
    """Write rotation data for a joint and its children"""
    if joint.name in frame_rotations:
        rotation = frame_rotations[joint.name]
        # Write in ZYX order
        frame_data.extend([rotation[2], rotation[1], rotation[0]])
    else:
        frame_data.extend([0.0, 0.0, 0.0])
    
    # Process children
    for child in joint.children:
        write_joint_rotations(child, frame_rotations, frame_data)

def process_video(video_path, output_bvh, confidence_threshold=0.5, sample_rate=1, preview=False):
    """Process video and create BVH file with improved ground contact"""
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
        model_complexity=1,               # Balanced accuracy and speed
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
                            cv2.putText(frame_bgr, f"Sampled: {sampled_frames}", (10, 60), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        
                        # Show the frame
                        cv2.imshow('MediaPipe Pose Preview', frame_bgr)
                        
                        # Check for exit
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            print("Preview cancelled by user")
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
        
        # Process all frames to calculate rotations with ground contact
        frame_rotations = process_motion(all_landmarks, skeleton, joint_mapping)
        
        # Write BVH file
        write_bvh_file(skeleton, frame_rotations, frame_time, output_bvh)

def main():
    parser = argparse.ArgumentParser(description="Convert video to BVH using MediaPipe with ground contact detection")
    parser.add_argument("--video", required=True, help="Path to input video file")
    parser.add_argument("--output", required=True, help="Path to output BVH file")
    parser.add_argument("--confidence", type=float, default=0.5, help="Confidence threshold for pose detection")
    parser.add_argument("--sample-rate", type=int, default=2, help="Process every Nth frame")
    parser.add_argument("--preview", action="store_true", help="Show preview window with pose detection visualization")
    parser.add_argument("--preview-bvh", action="store_true", help="Preview the final BVH animation after processing")
    parser.add_argument("--speed", type=float, default=1.0, help="Speed factor for BVH preview (default: 1.0)")
    
    args = parser.parse_args()
    
    print("Starting MediaPipe to BVH conversion with improved ground contact...")
    start_time = time.time()
    
    process_video(args.video, args.output, args.confidence, args.sample_rate, args.preview)
    
    end_time = time.time()
    print(f"Conversion completed in {end_time - start_time:.2f} seconds")
    
    # Preview BVH animation if requested
    if args.preview_bvh:
        preview_bvh_animation(args.output, args.speed)

if __name__ == "__main__":
    main()