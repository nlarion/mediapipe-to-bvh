#!/usr/bin/env python3
"""
BVH Visualization and Validation Tool

This script helps visualize and compare the MediaPipe skeleton detection
with the output BVH file to identify and correct discrepancies.
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2
import mediapipe as mp
from mediapipe_to_bvh import MediaPipeToBVH
import re
import math
from matplotlib.animation import FuncAnimation

class BVHParser:
    def __init__(self, bvh_file_path):
        self.file_path = bvh_file_path
        self.joint_hierarchy = []
        self.joint_offsets = {}
        self.joint_channels = {}
        self.motion_data = []
        self.frame_time = 0
        self.num_frames = 0
        self.parse_bvh()
    
    def parse_bvh(self):
        with open(self.file_path, 'r') as f:
            content = f.read()
        
        # Split into hierarchy and motion sections
        hierarchy_section, motion_section = content.split('MOTION')
        
        # Parse hierarchy
        self._parse_hierarchy(hierarchy_section)
        
        # Parse motion
        self._parse_motion(motion_section)
    
    def _parse_hierarchy(self, hierarchy_text):
        # This is a simplified parser for demonstration
        lines = hierarchy_text.split('\n')
        
        # Stack for keeping track of the joint hierarchy
        stack = []
        current_joint = None
        
        for line in lines:
            line = line.strip()
            if line.startswith('ROOT') or line.startswith('JOINT'):
                joint_name = line.split()[1]
                parent = stack[-1] if stack else None
                self.joint_hierarchy.append((joint_name, parent))
                current_joint = joint_name
                stack.append(joint_name)
            elif line.startswith('OFFSET'):
                if current_joint:
                    offset = [float(val) for val in line.split()[1:4]]
                    self.joint_offsets[current_joint] = offset
            elif line.startswith('CHANNELS'):
                if current_joint:
                    parts = line.split()
                    num_channels = int(parts[1])
                    channel_types = parts[2:2+num_channels]
                    self.joint_channels[current_joint] = channel_types
            elif line.startswith('}'):
                if stack:
                    stack.pop()
                    if stack:
                        current_joint = stack[-1]
                    else:
                        current_joint = None
    
    def _parse_motion(self, motion_text):
        lines = motion_text.strip().split('\n')
        
        # Parse frame count and frame time
        for line in lines:
            if line.startswith('Frames:'):
                self.num_frames = int(line.split()[1])
            elif line.startswith('Frame Time:'):
                self.frame_time = float(line.split()[1])
        
        # Parse motion data
        motion_lines = [line for line in lines if not line.startswith('Frames:') and 
                        not line.startswith('Frame Time:') and line.strip()]
        
        for line in motion_lines:
            values = [float(val) for val in line.split()]
            self.motion_data.append(values)
    
    def calculate_joint_positions(self, frame_idx=0):
        """
        Calculate the 3D positions of all joints at a given frame.
        This is more accurate than get_skeleton_at_frame for visualization.
        """
        if frame_idx >= self.num_frames:
            raise ValueError(f"Frame index {frame_idx} out of range (0-{self.num_frames-1})")
        
        # Get motion data for this frame
        motion_data = self.motion_data[frame_idx]
        
        # Find the root joint
        root_joint = None
        for joint, parent in self.joint_hierarchy:
            if parent is None:
                root_joint = joint
                break
        
        if root_joint is None:
            raise ValueError("No root joint found in hierarchy")
        
        # Track channel index in motion data
        channel_idx = 0
        
        # Store joint transformations (position and rotation)
        joint_positions = {}
        joint_rotations = {}
        
        # Recursive function to calculate joint positions
        def process_joint(joint_name, parent_pos, parent_rot_matrix):
            nonlocal channel_idx
            
            # Get joint offset from parent
            offset = self.joint_offsets.get(joint_name, [0, 0, 0])
            
            # Get joint channels
            channels = self.joint_channels.get(joint_name, [])
            
            # Initialize position and rotation matrix
            position = list(parent_pos)
            local_rotation_matrix = np.identity(3)
            
            # Apply channels in order
            for channel in channels:
                value = motion_data[channel_idx]
                channel_idx += 1
                
                if channel.endswith('position'):
                    axis = channel[0].lower()
                    if axis == 'x':
                        position[0] = value
                    elif axis == 'y':
                        position[1] = value
                    elif axis == 'z':
                        position[2] = value
                elif channel.endswith('rotation'):
                    # Convert from degrees to radians
                    angle_rad = math.radians(value)
                    
                    # Create rotation matrix for this channel
                    channel_rot = np.identity(3)
                    axis = channel[0].lower()
                    
                    if axis == 'x':
                        # Rotation around X axis
                        channel_rot = np.array([
                            [1, 0, 0],
                            [0, math.cos(angle_rad), -math.sin(angle_rad)],
                            [0, math.sin(angle_rad), math.cos(angle_rad)]
                        ])
                    elif axis == 'y':
                        # Rotation around Y axis
                        channel_rot = np.array([
                            [math.cos(angle_rad), 0, math.sin(angle_rad)],
                            [0, 1, 0],
                            [-math.sin(angle_rad), 0, math.cos(angle_rad)]
                        ])
                    elif axis == 'z':
                        # Rotation around Z axis
                        channel_rot = np.array([
                            [math.cos(angle_rad), -math.sin(angle_rad), 0],
                            [math.sin(angle_rad), math.cos(angle_rad), 0],
                            [0, 0, 1]
                        ])
                    
                    # Accumulate rotation
                    local_rotation_matrix = np.dot(local_rotation_matrix, channel_rot)
            
            # If this is not the root, apply parent rotation to offset
            if parent_pos is not None:
                # Rotate offset by parent's accumulated rotation
                rotated_offset = np.dot(parent_rot_matrix, offset)
                
                # Add to parent position
                position = [
                    parent_pos[0] + rotated_offset[0],
                    parent_pos[1] + rotated_offset[1],
                    parent_pos[2] + rotated_offset[2]
                ]
            
            # Combine parent rotation with local rotation
            global_rotation_matrix = np.dot(parent_rot_matrix, local_rotation_matrix)
            
            # Store results
            joint_positions[joint_name] = position
            joint_rotations[joint_name] = global_rotation_matrix
            
            # Process child joints
            for child_joint, parent in self.joint_hierarchy:
                if parent == joint_name:
                    process_joint(child_joint, position, global_rotation_matrix)
        
        # Start with root joint (no parent position/rotation)
        process_joint(root_joint, None, np.identity(3))
        
        return joint_positions
    
    def get_skeleton_at_frame(self, frame_idx=0):
        """
        Simplified method for getting joint positions.
        For more accurate visualization, use calculate_joint_positions.
        """
        return self.calculate_joint_positions(frame_idx)


class SkeletonVisualizer:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Define skeleton connections for visualization
        self.connections = [
            # Torso
            ('Hips', 'Spine'),
            ('Spine', 'Spine1'),
            ('Spine1', 'Spine2'),
            ('Spine2', 'Neck'),
            ('Neck', 'Head'),
            # Left arm
            ('Spine2', 'LeftShoulder'),
            ('LeftShoulder', 'LeftArm'),
            ('LeftArm', 'LeftForeArm'),
            ('LeftForeArm', 'LeftHand'),
            # Right arm
            ('Spine2', 'RightShoulder'),
            ('RightShoulder', 'RightArm'),
            ('RightArm', 'RightForeArm'),
            ('RightForeArm', 'RightHand'),
            # Left leg
            ('Hips', 'LeftUpLeg'),
            ('LeftUpLeg', 'LeftLeg'),
            ('LeftLeg', 'LeftFoot'),
            ('LeftFoot', 'LeftToeBase'),
            # Right leg
            ('Hips', 'RightUpLeg'),
            ('RightUpLeg', 'RightLeg'),
            ('RightLeg', 'RightFoot'),
            ('RightFoot', 'RightToeBase')
        ]
    
    def process_video_frame(self, video_path, frame_idx):
        """Process a single frame from a video using MediaPipe."""
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        success, image = cap.read()
        cap.release()
        
        if not success:
            return None, None
        
        with self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=2) as pose:
            
            # Convert image to RGB and process
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = pose.process(image_rgb)
            
            return image_rgb, results.pose_landmarks
    
    def visualize_comparison(self, video_path, bvh_path, frame_idx=0):
        """
        Create a side-by-side visualization of MediaPipe skeleton and BVH skeleton.
        """
        # Process video frame with MediaPipe
        image, mp_landmarks = self.process_video_frame(video_path, frame_idx)
        if image is None or mp_landmarks is None:
            print(f"Failed to process frame {frame_idx} from video")
            return
        
        # Extract MediaPipe landmarks
        mp_converter = MediaPipeToBVH()
        landmark_list = [mp_landmarks.landmark[i] for i in range(len(mp_landmarks.landmark))]
        joint_data = mp_converter.process_landmarks(landmark_list)
        
        # Parse BVH file
        bvh_parser = BVHParser(bvh_path)
        bvh_positions = bvh_parser.get_skeleton_at_frame(min(frame_idx, bvh_parser.num_frames-1))
        
        # Create figure with 3 subplots (video frame, MP skeleton, BVH skeleton)
        fig = plt.figure(figsize=(18, 6))
        
        # Plot video frame with MediaPipe landmarks
        ax_img = fig.add_subplot(131)
        ax_img.imshow(image)
        ax_img.set_title('Video Frame with MediaPipe Detection')
        ax_img.axis('off')
        
        # Plot MediaPipe skeleton
        ax_mp = fig.add_subplot(132, projection='3d')
        self._plot_mp_skeleton(ax_mp, joint_data)
        ax_mp.set_title('MediaPipe Skeleton')
        
        # Plot BVH skeleton
        ax_bvh = fig.add_subplot(133, projection='3d')
        self._plot_bvh_skeleton(ax_bvh, bvh_positions)
        ax_bvh.set_title('BVH Skeleton')
        
        # Standardize view for both 3D plots
        for ax in [ax_mp, ax_bvh]:
            ax.set_xlim([-150, 150])
            ax.set_ylim([-150, 150])
            ax.set_zlim([-50, 250])
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.view_init(elev=10, azim=-80)
        
        plt.tight_layout()
        plt.show()
    
    def _plot_mp_skeleton(self, ax, joint_data):
        """Plot MediaPipe skeleton in 3D."""
        # Create dictionary of joint positions
        joints = {}
        for joint in joint_data:
            joints[joint['name']] = joint['position']
        
        # Plot joints as points
        for joint_name, position in joints.items():
            ax.scatter(position[0], position[1], position[2], color='blue', s=30)
            ax.text(position[0], position[1], position[2], joint_name, fontsize=8)
        
        # Plot connections
        for start, end in self.connections:
            if start in joints and end in joints:
                ax.plot([joints[start][0], joints[end][0]],
                        [joints[start][1], joints[end][1]],
                        [joints[start][2], joints[end][2]], color='blue')
    
    def _plot_bvh_skeleton(self, ax, joint_positions):
        """Plot BVH skeleton in 3D."""
        # Plot joints as points
        for joint_name, position in joint_positions.items():
            ax.scatter(position[0], position[1], position[2], color='red', s=30)
            ax.text(position[0], position[1], position[2], joint_name, fontsize=8)
        
        # Plot connections
        for start, end in self.connections:
            if start in joint_positions and end in joint_positions:
                ax.plot([joint_positions[start][0], joint_positions[end][0]],
                        [joint_positions[start][1], joint_positions[end][1]],
                        [joint_positions[start][2], joint_positions[end][2]], color='red')
    
    def create_animation(self, video_path, bvh_path, start_frame=0, num_frames=30, output_path=None):
        """
        Create an animation showing both skeletons moving together.
        """
        # Parse BVH file
        bvh_parser = BVHParser(bvh_path)
        
        # Cap num_frames to available frames in BVH
        max_frames = min(bvh_parser.num_frames - start_frame, num_frames)
        
        # Initialize figure
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Initialize empty scatter and line objects for both skeletons
        mp_points = ax.scatter([], [], [], color='blue', s=30)
        bvh_points = ax.scatter([], [], [], color='red', s=30)
        
        mp_lines = []
        bvh_lines = []
        
        for _ in self.connections:
            mp_line, = ax.plot([], [], [], color='blue', lw=2)
            bvh_line, = ax.plot([], [], [], color='red', lw=2)
            mp_lines.append(mp_line)
            bvh_lines.append(bvh_line)
        
        # Set plot limits and labels
        ax.set_xlim([-150, 150])
        ax.set_ylim([-150, 150])
        ax.set_zlim([-50, 250])
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('MediaPipe (Blue) vs BVH (Red) Skeleton Animation')
        
        def init():
            mp_points._offsets3d = ([], [], [])
            bvh_points._offsets3d = ([], [], [])
            
            for line in mp_lines + bvh_lines:
                line.set_data([], [])
                line.set_3d_properties([])
            
            return [mp_points, bvh_points] + mp_lines + bvh_lines
        
        def update(frame):
            frame_idx = start_frame + frame
            
            # Process video frame with MediaPipe
            _, mp_landmarks = self.process_video_frame(video_path, frame_idx)
            
            # Process MediaPipe landmarks
            mp_converter = MediaPipeToBVH()
            if mp_landmarks:
                landmark_list = [mp_landmarks.landmark[i] for i in range(len(mp_landmarks.landmark))]
                mp_joint_data = mp_converter.process_landmarks(landmark_list)
                
                # Create dictionary of joint positions for MediaPipe
                mp_joints = {}
                for joint in mp_joint_data:
                    mp_joints[joint['name']] = joint['position']
                
                # Get BVH joint positions
                bvh_joints = bvh_parser.get_skeleton_at_frame(min(frame_idx, bvh_parser.num_frames-1))
                
                # Update points
                mp_xs, mp_ys, mp_zs = [], [], []
                bvh_xs, bvh_ys, bvh_zs = [], [], []
                
                for joint_name in mp_joints:
                    pos = mp_joints[joint_name]
                    mp_xs.append(pos[0])
                    mp_ys.append(pos[1])
                    mp_zs.append(pos[2])
                
                for joint_name in bvh_joints:
                    pos = bvh_joints[joint_name]
                    bvh_xs.append(pos[0])
                    bvh_ys.append(pos[1])
                    bvh_zs.append(pos[2])
                
                mp_points._offsets3d = (mp_xs, mp_ys, mp_zs)
                bvh_points._offsets3d = (bvh_xs, bvh_ys, bvh_zs)
                
                # Update lines
                for i, (start, end) in enumerate(self.connections):
                    if start in mp_joints and end in mp_joints:
                        mp_lines[i].set_data(
                            [mp_joints[start][0], mp_joints[end][0]],
                            [mp_joints[start][1], mp_joints[end][1]]
                        )
                        mp_lines[i].set_3d_properties(
                            [mp_joints[start][2], mp_joints[end][2]]
                        )
                    
                    if start in bvh_joints and end in bvh_joints:
                        bvh_lines[i].set_data(
                            [bvh_joints[start][0], bvh_joints[end][0]],
                            [bvh_joints[start][1], bvh_joints[end][1]]
                        )
                        bvh_lines[i].set_3d_properties(
                            [bvh_joints[start][2], bvh_joints[end][2]]
                        )
            
            return [mp_points, bvh_points] + mp_lines + bvh_lines
        
        anim = FuncAnimation(fig, update, frames=max_frames, init_func=init, blit=True, interval=50)
        
        if output_path:
            # Save animation
            anim.save(output_path, writer='pillow', fps=10)
            print(f"Animation saved to {output_path}")
        else:
            # Display animation
            plt.show()
        
        return anim


def main():
    parser = argparse.ArgumentParser(description='Visualize and compare MediaPipe skeleton and BVH output')
    parser.add_argument('video', help='Input video file path')
    parser.add_argument('bvh', help='BVH file path')
    parser.add_argument('--frame', type=int, default=0, help='Frame to visualize (default: 0)')
    parser.add_argument('--animate', action='store_true', help='Create animation instead of static comparison')
    parser.add_argument('--frames', type=int, default=30, help='Number of frames to animate (default: 30)')
    parser.add_argument('--output', help='Output file for animation (optional)')
    
    args = parser.parse_args()
    
    visualizer = SkeletonVisualizer()
    
    if args.animate:
        visualizer.create_animation(args.video, args.bvh, args.frame, args.frames, args.output)
    else:
        visualizer.visualize_comparison(args.video, args.bvh, args.frame)


if __name__ == "__main__":
    main()