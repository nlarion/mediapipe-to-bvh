"""
Main BVH converter module that orchestrates the conversion pipeline.
Combines MediaPipe extraction, skeleton mapping, and BVH file generation.
"""

import numpy as np
import argparse
import time
from pathlib import Path
from typing import List, Dict, Optional
from tqdm import tqdm
import mediapipe as mp

from mediapipe_extractor import MediaPipeExtractor, PoseFrame
from skeleton_mapper import SkeletonMapper, BVHJoint
from math_utils import calculate_rotation_from_directions, smooth_rotations, smooth_positions
from config import BVH_CONFIG, PROCESSING_CONFIG, SMOOTHING_CONFIG

mp_pose = mp.solutions.pose


class BVHConverter:
    """Converts MediaPipe pose data to BVH format."""
    
    def __init__(self):
        self.skeleton_mapper = SkeletonMapper()
        self.frame_time = 1.0 / BVH_CONFIG['fps']
        self.rotation_order = BVH_CONFIG['rotation_order']
        self.scale = PROCESSING_CONFIG['scale_factor']
        
    def convert(self, pose_frames: List[PoseFrame], output_path: str) -> bool:
        """Convert pose frames to BVH file.
        
        Args:
            pose_frames: List of extracted pose frames
            output_path: Path for output BVH file
            
        Returns:
            True if successful, False otherwise
        """
        if not pose_frames:
            print("Error: No pose frames to convert")
            return False
        
        # Store pose_frames for hip position calculation
        self.pose_frames = pose_frames
        
        # Find reference frame for skeleton setup
        extractor = MediaPipeExtractor(use_holistic=True)
        ref_idx = extractor.find_reference_frame(pose_frames)
        
        if not pose_frames[ref_idx].is_valid():
            print("Error: No valid reference frame found")
            return False
        
        # Calculate bone offsets from reference frame
        print("Setting up skeleton from reference frame...")
        ref_landmarks = pose_frames[ref_idx].world_landmarks
        self.skeleton_mapper.calculate_bone_offsets(ref_landmarks, self.scale)
        
        # Process all frames to calculate rotations
        print("Calculating joint rotations...")
        all_rotations = self._process_motion(pose_frames)
        
        # Apply smoothing if enabled
        if SMOOTHING_CONFIG['enable_temporal_smoothing']:
            print("Applying temporal smoothing...")
            all_rotations = self._smooth_motion(all_rotations)
        
        # Calculate hip positions from actual landmark data
        print("Calculating hip movement through 3D space...")
        hip_positions = self._calculate_hip_positions(pose_frames)
        
        # Write BVH file
        print(f"Writing BVH file to {output_path}...")
        success = self._write_bvh(all_rotations, hip_positions, output_path)
        
        if success:
            print(f"BVH file created successfully: {output_path}")
        else:
            print("Error writing BVH file")
        
        return success
    
    def _process_motion(self, pose_frames: List[PoseFrame]) -> List[Dict[str, np.ndarray]]:
        """Process all frames to calculate joint rotations.
        
        Args:
            pose_frames: List of pose frames
            
        Returns:
            List of rotation dictionaries for each frame
        """
        all_rotations = []
        
        for frame in tqdm(pose_frames, desc="Processing frames"):
            if frame.is_valid():
                frame_rotations = self._calculate_frame_rotations(
                    frame.world_landmarks,
                    left_hand_landmarks=frame.left_hand_landmarks,
                    right_hand_landmarks=frame.right_hand_landmarks
                )
            else:
                # Use zero rotations for invalid frames
                frame_rotations = self._get_zero_rotations()

            all_rotations.append(frame_rotations)
        
        return all_rotations
    
    def _calculate_frame_rotations(self, landmarks,
                                   left_hand_landmarks=None,
                                   right_hand_landmarks=None) -> Dict[str, np.ndarray]:
        """Calculate rotations for all joints in a single frame.

        Uses the untitled9.py approach with axis-angle to Euler conversion.
        Now includes hand landmark data for better ForeArm rotation calculation.

        Args:
            landmarks: MediaPipe world landmarks (pose)
            left_hand_landmarks: Optional left hand landmarks (21 points)
            right_hand_landmarks: Optional right hand landmarks (21 points)

        Returns:
            Dictionary mapping joint names to rotation angles
        """
        # Start with all zero rotations like untitled9
        rotations = {joint.name: np.zeros(3) for joint in self.skeleton_mapper.get_all_joints()}
        skeleton = self.skeleton_mapper.skeleton
        
        def get_bone_direction(joint_name: str, child_name: str) -> Optional[np.ndarray]:
            """Calculate normalized direction from parent to child."""
            parent_pos = self.skeleton_mapper.get_joint_position(
                joint_name, landmarks, self.scale
            )
            child_pos = self.skeleton_mapper.get_joint_position(
                child_name, landmarks, self.scale
            )
            
            if parent_pos is not None and child_pos is not None:
                direction = child_pos - parent_pos
                if np.linalg.norm(direction) > 1e-10:
                    return direction / np.linalg.norm(direction)
            
            return None
        
        def process_joint(joint: BVHJoint):
            """Recursively process joint hierarchy."""
            # Initialize with zero rotation
            rotations[joint.name] = np.zeros(3)
            
            # Special handling for specific joints
            if joint.name == "Head":
                # Head is a leaf but needs rotation based on orientation
                # For Head, we'll calculate a simple tilt/nod based on face direction
                # Using a minimal rotation for now to avoid zero
                rotations[joint.name] = np.array([5.0, 0.0, 0.0])  # Small default rotation
            
            elif joint.name in ["Chest", "Neck"]:
                # For Chest and Neck, calculate based on orientation of shoulders/ears
                if joint.name == "Chest":
                    # Chest orientation from shoulders
                    left_shoulder = self.skeleton_mapper.get_joint_position("LeftShoulder", landmarks, self.scale)
                    right_shoulder = self.skeleton_mapper.get_joint_position("RightShoulder", landmarks, self.scale)
                    chest_pos = self.skeleton_mapper.get_joint_position("Chest", landmarks, self.scale)

                    if left_shoulder is not None and right_shoulder is not None and chest_pos is not None:
                        # Calculate chest forward direction
                        shoulder_axis = right_shoulder - left_shoulder
                        shoulder_center = (left_shoulder + right_shoulder) / 2
                        up_direction = shoulder_center - chest_pos

                        if np.linalg.norm(shoulder_axis) > 1e-10 and np.linalg.norm(up_direction) > 1e-10:
                            # Forward is cross product of shoulder axis and up
                            forward = np.cross(shoulder_axis, up_direction)
                            if np.linalg.norm(forward) > 1e-10:
                                forward = forward / np.linalg.norm(forward)
                                # Rest forward is Z axis
                                rest_forward = np.array([0, 0, 1])

                                euler_angles = calculate_rotation_from_directions(
                                    rest_forward, forward, order='XYZ'
                                )
                                rotations[joint.name] = euler_angles
                
                elif joint.name == "Neck" and joint.children:
                    # Neck uses child direction but with special handling
                    child = joint.children[0]  # Head
                    direction = get_bone_direction(joint.name, child.name)
                    
                    if direction is not None and np.linalg.norm(child.offset) > 0:
                        rest_direction = child.offset / np.linalg.norm(child.offset)
                        euler_angles = calculate_rotation_from_directions(
                            rest_direction, direction, order='XYZ'
                        )
                        rotations[joint.name] = euler_angles
            
            elif joint.name in ["LeftShoulder", "RightShoulder"]:
                # Shoulders need special handling as they're connection points
                shoulder_pos = self.skeleton_mapper.get_joint_position(joint.name, landmarks, self.scale)

                if joint.children and shoulder_pos is not None:
                    child = joint.children[0]  # Arm
                    arm_pos = self.skeleton_mapper.get_joint_position(child.name, landmarks, self.scale)

                    if arm_pos is not None:
                        direction = arm_pos - shoulder_pos
                        if np.linalg.norm(direction) > 1e-10 and np.linalg.norm(child.offset) > 0:
                            direction = direction / np.linalg.norm(direction)
                            rest_direction = child.offset / np.linalg.norm(child.offset)

                            euler_angles = calculate_rotation_from_directions(
                                rest_direction, direction, order='XYZ'
                            )
                            rotations[joint.name] = euler_angles

            elif joint.name in ["LeftArm", "RightArm"]:
                # Arms (elbow joints) - calculate elbow rotation
                # The elbow joint controls the angle between upper arm and forearm
                parent = joint.parent
                if parent and joint.children:
                    # Get shoulder, elbow, and wrist positions
                    shoulder_pos = self.skeleton_mapper.get_joint_position(parent.name, landmarks, self.scale)
                    elbow_pos = self.skeleton_mapper.get_joint_position(joint.name, landmarks, self.scale)
                    child = joint.children[0]  # ForeArm
                    wrist_pos = self.skeleton_mapper.get_joint_position(child.name, landmarks, self.scale)

                    if shoulder_pos is not None and elbow_pos is not None and wrist_pos is not None:
                        # Calculate upper arm and forearm vectors
                        upper_arm = elbow_pos - shoulder_pos
                        forearm = wrist_pos - elbow_pos

                        if np.linalg.norm(forearm) > 1e-10 and np.linalg.norm(child.offset) > 0:
                            forearm_direction = forearm / np.linalg.norm(forearm)
                            rest_direction = child.offset / np.linalg.norm(child.offset)

                            # Calculate rotation for the elbow joint
                            euler_angles = calculate_rotation_from_directions(
                                rest_direction, forearm_direction, order='XYZ'
                            )

                            # Elbow is primarily a hinge joint (flexion/extension)
                            # It has very limited rotation in other axes
                            euler_angles[1] *= 0.2  # Minimal Y rotation
                            euler_angles[2] *= 0.2  # Minimal Z rotation

                            rotations[joint.name] = euler_angles

            elif joint.name in ["LeftForeArm", "RightForeArm"]:
                # ForeArm joints - now we can use hand landmarks for proper orientation!
                use_hand_landmarks = False
                hand_orientation = None

                # Get the wrist world position from pose landmarks
                wrist_world_pos = None
                if joint.name == "LeftForeArm":
                    wrist_world_pos = self.skeleton_mapper.get_joint_position("LeftForeArm", landmarks, self.scale)
                else:  # RightForeArm
                    wrist_world_pos = self.skeleton_mapper.get_joint_position("RightForeArm", landmarks, self.scale)

                # Check if we have hand landmarks
                if joint.name == "LeftForeArm" and left_hand_landmarks:
                    hand_orientation = self._calculate_hand_orientation(
                        left_hand_landmarks, "left", wrist_world_pos
                    )
                    use_hand_landmarks = hand_orientation is not None
                elif joint.name == "RightForeArm" and right_hand_landmarks:
                    hand_orientation = self._calculate_hand_orientation(
                        right_hand_landmarks, "right", wrist_world_pos
                    )
                    use_hand_landmarks = hand_orientation is not None

                if use_hand_landmarks and joint.children:
                    # Use hand orientation to calculate proper ForeArm rotation
                    child = joint.children[0]  # Hand
                    if np.linalg.norm(child.offset) > 0:
                        rest_direction = child.offset / np.linalg.norm(child.offset)

                        # Calculate rotation from rest to current hand orientation
                        euler_angles = calculate_rotation_from_directions(
                            rest_direction, hand_orientation, order='XYZ'
                        )

                        # Apply some constraints for natural wrist movement
                        euler_angles[0] *= 0.9  # Allow most flexion/extension
                        euler_angles[1] *= 0.6  # Moderate radial/ulnar deviation
                        euler_angles[2] *= 0.7  # Some pronation/supination

                        rotations[joint.name] = euler_angles
                else:
                    # Fallback to minimal rotation if no hand data
                    rotations[joint.name] = np.zeros(3)

            elif joint.children:
                # Standard joint with children
                child = joint.children[0]
                
                # Get current direction
                direction = get_bone_direction(joint.name, child.name)
                
                if direction is not None:
                    # Calculate rest pose direction (normalized offset)
                    if np.linalg.norm(child.offset) > 0:
                        rest_direction = child.offset / np.linalg.norm(child.offset)
                        
                        # Apply corrections for specific joints (from untitled9)
                        if joint.name in ["LeftHand", "RightHand", "LeftFoot", "RightFoot"]:
                            # Ensure forward direction is positive
                            if direction[2] < 0 and abs(direction[2]) > 0.1:
                                direction[2] = abs(direction[2])
                            
                            # For hands, ensure proper lateral direction
                            if joint.name == "LeftHand" and direction[0] > 0:
                                direction[0] = -abs(direction[0])
                            elif joint.name == "RightHand" and direction[0] < 0:
                                direction[0] = abs(direction[0])
                        
                        # Calculate rotation using untitled9's approach
                        euler_angles = calculate_rotation_from_directions(
                            rest_direction, direction, order='XYZ'
                        )
                        
                        rotations[joint.name] = euler_angles
            
            # Process children
            for child in joint.children:
                process_joint(child)
        
        # Start from root
        process_joint(skeleton)
        
        return rotations
    
    def _calculate_hand_orientation(self, hand_landmarks, hand_side: str,
                                   wrist_world_pos: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        """Calculate hand orientation from 21 hand landmarks.

        Transforms 2D hand landmarks to 3D world space using wrist anchor,
        then calculates orientation from the hand plane.

        Args:
            hand_landmarks: MediaPipe hand landmarks (21 points, in image space)
            hand_side: "left" or "right" to handle coordinate system
            wrist_world_pos: 3D world position of wrist from pose landmarks

        Returns:
            Normalized direction vector representing hand orientation, or None if calculation fails
        """
        if not hand_landmarks or len(hand_landmarks) < 21:
            return None

        try:
            # Hand landmarks are in normalized image coordinates (0-1 range)
            # We need to transform them to world space

            # First, get the 2D hand points
            wrist_2d = np.array([hand_landmarks[0].x, hand_landmarks[0].y])
            index_mcp_2d = np.array([hand_landmarks[5].x, hand_landmarks[5].y])
            pinky_mcp_2d = np.array([hand_landmarks[17].x, hand_landmarks[17].y])
            middle_mcp_2d = np.array([hand_landmarks[9].x, hand_landmarks[9].y])

            # If we have the wrist world position, use it for depth reference
            if wrist_world_pos is not None:
                # Estimate depth for other hand points based on typical hand proportions
                # Hand is roughly planar, so we'll use a simple depth model
                base_depth = wrist_world_pos[2]  # Z coordinate of wrist

                # Convert to 3D with estimated depth
                # Scale factor based on typical hand size (about 0.2 units in world space)
                hand_scale = 0.2

                # Transform 2D points to 3D relative to wrist
                wrist = wrist_world_pos.copy()

                # Calculate relative positions in hand plane
                index_mcp = wrist + np.array([
                    (index_mcp_2d[0] - wrist_2d[0]) * hand_scale,
                    (index_mcp_2d[1] - wrist_2d[1]) * hand_scale,
                    0.02  # Slight depth variation
                ])

                pinky_mcp = wrist + np.array([
                    (pinky_mcp_2d[0] - wrist_2d[0]) * hand_scale,
                    (pinky_mcp_2d[1] - wrist_2d[1]) * hand_scale,
                    0.01
                ])

                middle_mcp = wrist + np.array([
                    (middle_mcp_2d[0] - wrist_2d[0]) * hand_scale,
                    (middle_mcp_2d[1] - wrist_2d[1]) * hand_scale,
                    0.015
                ])
            else:
                # Fallback: use normalized coordinates as-is (less accurate)
                wrist = np.array([hand_landmarks[0].x, hand_landmarks[0].y, 0.0])
                index_mcp = np.array([hand_landmarks[5].x, hand_landmarks[5].y, 0.02])
                pinky_mcp = np.array([hand_landmarks[17].x, hand_landmarks[17].y, 0.01])
                middle_mcp = np.array([hand_landmarks[9].x, hand_landmarks[9].y, 0.015])

            # Calculate hand plane vectors
            v1 = index_mcp - wrist  # Vector from wrist to index MCP
            v2 = pinky_mcp - wrist  # Vector from wrist to pinky MCP

            # Calculate hand normal (perpendicular to palm)
            normal = np.cross(v1, v2)

            # Flip normal for left hand to maintain consistency
            if hand_side == "left":
                normal = -normal

            # Normalize
            if np.linalg.norm(normal) > 1e-10:
                normal = normal / np.linalg.norm(normal)

                # The hand orientation is a combination of the normal and forward direction
                forward = middle_mcp - wrist

                if np.linalg.norm(forward) > 1e-10:
                    forward = forward / np.linalg.norm(forward)

                    # Combine normal and forward for hand orientation
                    hand_direction = (forward + normal * 0.3)
                    hand_direction = hand_direction / np.linalg.norm(hand_direction)

                    return hand_direction

        except (IndexError, AttributeError) as e:
            # Handle any issues with landmark access
            return None

        return None

    def _get_zero_rotations(self) -> Dict[str, np.ndarray]:
        """Get zero rotations for all joints.

        Returns:
            Dictionary with zero rotations
        """
        rotations = {}
        for joint in self.skeleton_mapper.get_all_joints():
            rotations[joint.name] = np.zeros(3)
        return rotations
    
    def _smooth_motion(self, all_rotations: List[Dict[str, np.ndarray]]) -> List[Dict[str, np.ndarray]]:
        """Apply temporal smoothing to motion data.
        
        Args:
            all_rotations: List of rotation dictionaries
            
        Returns:
            Smoothed rotations
        """
        if not all_rotations:
            return all_rotations
        
        # Get joint names
        joint_names = list(all_rotations[0].keys())
        
        # Smooth each joint's rotations independently
        smoothed_rotations = [{} for _ in range(len(all_rotations))]
        
        for joint_name in joint_names:
            # Collect rotations for this joint across all frames
            joint_rotations = np.array([
                frame_rots[joint_name] for frame_rots in all_rotations
            ])
            
            # Apply smoothing
            smoothed = smooth_rotations(
                joint_rotations,
                window_size=SMOOTHING_CONFIG['temporal_window_size'],
                preserve_dynamics=SMOOTHING_CONFIG['preserve_dynamics']
            )
            
            # Store smoothed rotations
            for i, rotation in enumerate(smoothed):
                smoothed_rotations[i][joint_name] = rotation
        
        return smoothed_rotations
    
    def _calculate_hip_positions(self, pose_frames: List[PoseFrame]) -> List[np.ndarray]:
        """Calculate actual hip positions from MediaPipe landmarks.
        
        This tracks the character's movement through 3D space, not just poses.
        
        Args:
            pose_frames: List of pose frames with world landmarks
            
        Returns:
            List of hip positions for each frame
        """
        positions = []
        
        for frame in pose_frames:
            if frame.world_landmarks:
                # Get left and right hip landmarks
                left_hip = frame.world_landmarks[mp_pose.PoseLandmark.LEFT_HIP]
                right_hip = frame.world_landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
                
                # Calculate hip center (average of left and right)
                # Use movement_scale for X and Z to amplify movement through space
                # Keep Y with normal scale to maintain proper height
                hip_center = np.array([
                    (left_hip.x + right_hip.x) / 2 * PROCESSING_CONFIG['movement_scale'],
                    -(left_hip.y + right_hip.y) / 2 * self.scale,  # MediaPipe Y is down, BVH Y is up
                    (left_hip.z + right_hip.z) / 2 * PROCESSING_CONFIG['movement_scale']
                ])
                
                positions.append(hip_center)
            else:
                # Use previous position or default if no landmarks
                if positions:
                    positions.append(positions[-1])
                else:
                    positions.append(np.array([0.0, BVH_CONFIG['root_height'], 0.0]))
        
        # Apply temporal smoothing to reduce jitter (NEW)
        if SMOOTHING_CONFIG['enable_temporal_smoothing'] and len(positions) > 3:
            positions_array = np.array(positions)
            smoothed_positions = smooth_positions(
                positions_array,
                window_size=SMOOTHING_CONFIG['temporal_window_size'],
                preserve_dynamics=SMOOTHING_CONFIG['preserve_dynamics'],
                preserve_y_axis=True  # Less smoothing on Y to maintain ground contact
            )
            positions = [pos for pos in smoothed_positions]
        
        # Make positions relative to first frame (delta positions)
        # This prevents the character from appearing at arbitrary world coordinates
        if positions:
            origin = positions[0].copy()
            for i in range(len(positions)):
                positions[i] = positions[i] - origin
                
                # Filter small Y movements as noise (helps with foot sliding)
                if abs(positions[i][1]) < 1.0:  # Less than 1cm movement
                    positions[i][1] = 0.0
                
                # Add back the default hip height to Y
                positions[i][1] += BVH_CONFIG['root_height']
        
        return positions
    
    def _write_bvh(self, all_rotations: List[Dict[str, np.ndarray]], 
                   hip_positions: List[np.ndarray], output_path: str) -> bool:
        """Write BVH file with motion data.
        
        Args:
            all_rotations: List of rotation dictionaries
            hip_positions: List of hip positions for each frame
            output_path: Path for output file
            
        Returns:
            True if successful
        """
        try:
            with open(output_path, 'w') as f:
                # Write hierarchy
                f.write("HIERARCHY\n")
                self._write_hierarchy(f, self.skeleton_mapper.skeleton, 0)
                
                # Write motion data
                num_frames = len(all_rotations)
                f.write("MOTION\n")
                f.write(f"Frames: {num_frames}\n")
                f.write(f"Frame Time: {self.frame_time:.6f}\n")
                
                # Write each frame
                for frame_idx in tqdm(range(num_frames), desc="Writing frames"):
                    frame_data = []
                    
                    # Root position from tracked hip movement
                    hip_pos = hip_positions[frame_idx]
                    frame_data.extend([hip_pos[0], hip_pos[1], hip_pos[2]])
                    
                    # Write rotations for all joints
                    frame_rotations = all_rotations[frame_idx]
                    self._write_joint_rotations(
                        self.skeleton_mapper.skeleton,
                        frame_rotations,
                        frame_data
                    )
                    
                    # Write frame line
                    f.write(" ".join([f"{val:.6f}" for val in frame_data]) + "\n")
            
            return True
            
        except Exception as e:
            print(f"Error writing BVH file: {e}")
            return False
    
    def _write_hierarchy(self, f, joint: BVHJoint, level: int):
        """Recursively write joint hierarchy.
        
        Args:
            f: File handle
            joint: Current joint
            level: Indentation level
        """
        indent = "  " * level
        
        if level == 0:
            f.write(f"{indent}ROOT {joint.name}\n")
        else:
            f.write(f"{indent}JOINT {joint.name}\n")
        
        f.write(f"{indent}{{\n")
        f.write(f"{indent}  OFFSET {joint.offset[0]:.6f} {joint.offset[1]:.6f} {joint.offset[2]:.6f}\n")
        
        # Write channels
        if joint.channels:
            channels_str = " ".join(joint.channels)
            f.write(f"{indent}  CHANNELS {len(joint.channels)} {channels_str}\n")
        
        # Write children
        for child in joint.children:
            self._write_hierarchy(f, child, level + 1)
        
        # Add end site for leaf joints
        if not joint.children:
            f.write(f"{indent}  End Site\n")
            f.write(f"{indent}  {{\n")
            end_offset = joint.offset * 0.3 if np.linalg.norm(joint.offset) > 0 else np.array([0, -5, 0])
            f.write(f"{indent}    OFFSET {end_offset[0]:.6f} {end_offset[1]:.6f} {end_offset[2]:.6f}\n")
            f.write(f"{indent}  }}\n")
        
        f.write(f"{indent}}}\n")
    
    def _write_joint_rotations(self, joint: BVHJoint, frame_rotations: Dict, frame_data: List):
        """Write rotation data for a joint and its children.
        
        Args:
            joint: Current joint
            frame_rotations: Dictionary of rotations for this frame
            frame_data: List to append rotation values to
        """
        if joint.name in frame_rotations:
            rotation = frame_rotations[joint.name]
            # Write in XYZ order as specified in channels
            if 'Xrotation' in joint.channels:
                frame_data.append(rotation[0])
            if 'Yrotation' in joint.channels:
                frame_data.append(rotation[1])
            if 'Zrotation' in joint.channels:
                frame_data.append(rotation[2])
        
        # Process children
        for child in joint.children:
            self._write_joint_rotations(child, frame_rotations, frame_data)


def main():
    """Main entry point for the converter."""
    parser = argparse.ArgumentParser(description="Convert video to BVH using MediaPipe")
    parser.add_argument("--video", required=True, help="Path to input video file")
    parser.add_argument("--output", required=True, help="Path to output BVH file")
    parser.add_argument("--preview", action="store_true", help="Show pose detection preview")
    parser.add_argument("--sample-rate", type=int, default=2,
                       help="Process every Nth frame (default: 2)")
    parser.add_argument("--ik", action="store_true",
                       help="Enable IK foot locking to reduce sliding")

    args = parser.parse_args()
    
    # Update config with command line arguments
    PROCESSING_CONFIG['sample_rate'] = args.sample_rate
    
    print("=" * 60)
    print("MediaPipe to BVH Converter")
    print("Based on lessons learned from experiments")
    print("=" * 60)
    
    start_time = time.time()
    
    # Extract poses from video
    with MediaPipeExtractor(use_holistic=True) as extractor:
        extractor.sample_rate = args.sample_rate
        print("Using MediaPipe Holistic model for improved hand tracking...")
        pose_frames = extractor.extract_from_video(args.video, preview=args.preview)
        
        if not pose_frames:
            print("Error: No poses extracted from video")
            return
        
        # Interpolate missing frames if needed
        pose_frames = extractor.interpolate_missing_frames(pose_frames)
    
    # Convert to BVH
    if args.ik:
        # Use IK-enabled converter
        from bvh_converter_with_ik import BVHConverterWithIK
        print("Using IK foot locking to reduce sliding...")
        converter = BVHConverterWithIK(enable_ik=True)
    else:
        converter = BVHConverter()
    success = converter.convert(pose_frames, args.output)
    
    elapsed_time = time.time() - start_time
    
    if success:
        print(f"\nConversion completed in {elapsed_time:.2f} seconds")
        print(f"Output saved to: {args.output}")
    else:
        print("\nConversion failed")


if __name__ == "__main__":
    main()