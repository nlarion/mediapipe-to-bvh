import cv2
import numpy as np
import torch
import os
from pathlib import Path
import logging
from typing import List, Dict, Tuple, Optional
import math
import argparse

# ViTPose implementation (using easy_ViTPose for simplicity)
from easy_ViTPose import VitInference

class MP4ToBVHConverter:
    """
    Complete pipeline for converting MP4 videos to BVH motion capture files
    using ViTPose for pose estimation and direct BVH generation.
    """
    
    def __init__(self, vitpose_model_path: str, yolo_model_path: str, 
                 model_size: str = 'base', device: str = 'cuda'):
        """
        Initialize the converter with ViTPose models
        
        Args:
            vitpose_model_path: Path to ViTPose checkpoint (.pth file)
            yolo_model_path: Path to YOLO detection model
            model_size: ViTPose model size ('s', 'base', 'l', 'h')
            device: Computing device ('cuda' or 'cpu')
        """
        self.device = device
        self.model_size = model_size
        self.setup_logging()
        
        # Initialize ViTPose model
        self.pose_model = VitInference(
            vitpose_model_path,
            yolo_model_path,
            model_name=model_size,
            yolo_size=320,
            is_video=True,  # Enable temporal tracking
            device=device
        )
        
        # BVH joint hierarchy for COCO 17-keypoint format
        self.joint_hierarchy = self._create_bvh_hierarchy()
        self.joint_names = self._get_joint_names()
        
    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def _create_bvh_hierarchy(self) -> Dict:
        """
        Define BVH joint hierarchy mapping from COCO keypoints
        COCO 17 keypoints: nose, eyes, ears, shoulders, elbows, wrists, 
                          hips, knees, ankles
        """
        return {
            'Hips': {
                'parent': None,
                'offset': [0.0, 0.0, 0.0],
                'channels': ['Xposition', 'Yposition', 'Zposition', 
                            'Zrotation', 'Xrotation', 'Yrotation'],
                'children': ['Spine', 'LeftUpLeg', 'RightUpLeg']
            },
            'Spine': {
                'parent': 'Hips',
                'offset': [0.0, 6.0, 0.0],
                'channels': ['Zrotation', 'Xrotation', 'Yrotation'],
                'children': ['Chest']
            },
            'Chest': {
                'parent': 'Spine',
                'offset': [0.0, 8.0, 0.0],
                'channels': ['Zrotation', 'Xrotation', 'Yrotation'],
                'children': ['Neck', 'LeftShoulder', 'RightShoulder']
            },
            'Neck': {
                'parent': 'Chest',
                'offset': [0.0, 4.0, 0.0],
                'channels': ['Zrotation', 'Xrotation', 'Yrotation'],
                'children': ['Head']
            },
            'Head': {
                'parent': 'Neck',
                'offset': [0.0, 3.0, 0.0],
                'channels': ['Zrotation', 'Xrotation', 'Yrotation'],
                'children': []
            },
            # Left arm chain
            'LeftShoulder': {
                'parent': 'Chest',
                'offset': [-4.0, 2.0, 0.0],
                'channels': ['Zrotation', 'Xrotation', 'Yrotation'],
                'children': ['LeftArm']
            },
            'LeftArm': {
                'parent': 'LeftShoulder',
                'offset': [-8.0, 0.0, 0.0],
                'channels': ['Zrotation', 'Xrotation', 'Yrotation'],
                'children': ['LeftForeArm']
            },
            'LeftForeArm': {
                'parent': 'LeftArm',
                'offset': [-8.0, 0.0, 0.0],
                'channels': ['Zrotation', 'Xrotation', 'Yrotation'],
                'children': ['LeftHand']
            },
            'LeftHand': {
                'parent': 'LeftForeArm',
                'offset': [-6.0, 0.0, 0.0],
                'channels': ['Zrotation', 'Xrotation', 'Yrotation'],
                'children': []
            },
            # Right arm chain (mirrored)
            'RightShoulder': {
                'parent': 'Chest',
                'offset': [4.0, 2.0, 0.0],
                'channels': ['Zrotation', 'Xrotation', 'Yrotation'],
                'children': ['RightArm']
            },
            'RightArm': {
                'parent': 'RightShoulder',
                'offset': [8.0, 0.0, 0.0],
                'channels': ['Zrotation', 'Xrotation', 'Yrotation'],
                'children': ['RightForeArm']
            },
            'RightForeArm': {
                'parent': 'RightArm',
                'offset': [8.0, 0.0, 0.0],
                'channels': ['Zrotation', 'Xrotation', 'Yrotation'],
                'children': ['RightHand']
            },
            'RightHand': {
                'parent': 'RightForeArm',
                'offset': [6.0, 0.0, 0.0],
                'channels': ['Zrotation', 'Xrotation', 'Yrotation'],
                'children': []
            },
            # Left leg chain
            'LeftUpLeg': {
                'parent': 'Hips',
                'offset': [-2.0, -2.0, 0.0],
                'channels': ['Zrotation', 'Xrotation', 'Yrotation'],
                'children': ['LeftLeg']
            },
            'LeftLeg': {
                'parent': 'LeftUpLeg',
                'offset': [0.0, -18.0, 0.0],
                'channels': ['Zrotation', 'Xrotation', 'Yrotation'],
                'children': ['LeftFoot']
            },
            'LeftFoot': {
                'parent': 'LeftLeg',
                'offset': [0.0, -18.0, 0.0],
                'channels': ['Zrotation', 'Xrotation', 'Yrotation'],
                'children': []
            },
            # Right leg chain (mirrored)
            'RightUpLeg': {
                'parent': 'Hips',
                'offset': [2.0, -2.0, 0.0],
                'channels': ['Zrotation', 'Xrotation', 'Yrotation'],
                'children': ['RightLeg']
            },
            'RightLeg': {
                'parent': 'RightUpLeg',
                'offset': [0.0, -18.0, 0.0],
                'channels': ['Zrotation', 'Xrotation', 'Yrotation'],
                'children': ['RightFoot']
            },
            'RightFoot': {
                'parent': 'RightLeg',
                'offset': [0.0, -18.0, 0.0],
                'channels': ['Zrotation', 'Xrotation', 'Yrotation'],
                'children': []
            }
        }
    
    def _get_joint_names(self) -> List[str]:
        """Get ordered list of joint names for BVH output"""
        return ['Hips', 'Spine', 'Chest', 'Neck', 'Head',
                'LeftShoulder', 'LeftArm', 'LeftForeArm', 'LeftHand',
                'RightShoulder', 'RightArm', 'RightForeArm', 'RightHand',
                'LeftUpLeg', 'LeftLeg', 'LeftFoot',
                'RightUpLeg', 'RightLeg', 'RightFoot']
    
    def extract_frames_from_video(self, video_path: str, 
                                 max_frames: Optional[int] = None) -> List[np.ndarray]:
        """
        Extract frames from MP4 video with robust error handling
        
        Args:
            video_path: Path to input MP4 file
            max_frames: Maximum number of frames to extract (None for all)
            
        Returns:
            List of RGB frame arrays
        """
        frames = []
        
        # Initialize video capture with FFMPEG backend for MP4 compatibility 
        cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
        
        if not cap.isOpened():
            raise IOError(f"Cannot open video file: {video_path}")
        
        # Get video properties 
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        self.logger.info(f"Video: {total_frames} frames at {fps:.2f} FPS")
        
        frame_count = 0
        
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                
                if not ret:
                    break
                
                # Validate frame
                if frame is None or frame.size == 0:
                    self.logger.warning(f"Empty frame at position {frame_count}")
                    continue
                
                # Convert BGR to RGB for pose estimation
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(rgb_frame)
                
                frame_count += 1
                
                # Check max frames limit
                if max_frames and frame_count >= max_frames:
                    break
                    
                # Progress logging
                if frame_count % 100 == 0:
                    progress = (frame_count / total_frames) * 100
                    self.logger.info(f"Extracted {frame_count} frames ({progress:.1f}%)")
        
        except Exception as e:
            self.logger.error(f"Error extracting frames: {str(e)}")
        finally:
            cap.release()
        
        self.logger.info(f"Successfully extracted {len(frames)} frames")
        return frames
    
    def estimate_poses_from_frames(self, frames: List[np.ndarray]) -> List[Dict]:
        """
        Extract pose estimations from video frames using ViTPose
        
        Args:
            frames: List of RGB frame arrays
            
        Returns:
            List of pose estimation dictionaries per frame
        """
        pose_data = []
        
        self.logger.info(f"Processing {len(frames)} frames for pose estimation")
        
        for i, frame in enumerate(frames):
            try:
                # ViTPose inference - returns dict with person IDs and keypoints
                results = self.pose_model.inference(frame)
                
                # Process results for each detected person
                frame_poses = {}
                
                if results:
                    for person_id, keypoints in results.items():
                        # Keypoints format: [[y, x, confidence], ...]
                        processed_keypoints = []
                        
                        for keypoint in keypoints:
                            y, x, conf = keypoint
                            processed_keypoints.append([x, y, conf])  # Convert to [x, y, conf]
                        
                        frame_poses[person_id] = {
                            'keypoints': np.array(processed_keypoints),
                            'confidence': np.mean([kp[2] for kp in processed_keypoints])
                        }
                
                pose_data.append(frame_poses)
                
                # Progress logging
                if (i + 1) % 50 == 0:
                    self.logger.info(f"Processed {i + 1}/{len(frames)} frames")
                
            except Exception as e:
                self.logger.error(f"Error processing frame {i}: {str(e)}")
                pose_data.append({})  # Empty pose data for failed frames
        
        # Reset pose model for next video
        self.pose_model.reset()
        
        self.logger.info(f"Pose estimation complete for {len(pose_data)} frames")
        return pose_data
    
    def calculate_joint_angles(self, keypoints: np.ndarray) -> Dict[str, List[float]]:
        """
        Convert 2D pose keypoints to joint angles for BVH format
        Using simplified approach based on keypoint vectors
        
        Args:
            keypoints: Array of shape (17, 3) with [x, y, confidence]
            
        Returns:
            Dictionary of joint angles for each joint
        """
        # COCO keypoint indices
        NOSE = 0; L_EYE = 1; R_EYE = 2; L_EAR = 3; R_EAR = 4
        L_SHOULDER = 5; R_SHOULDER = 6; L_ELBOW = 7; R_ELBOW = 8
        L_WRIST = 9; R_WRIST = 10; L_HIP = 11; R_HIP = 12
        L_KNEE = 13; R_KNEE = 14; L_ANKLE = 15; R_ANKLE = 16
        
        joint_angles = {}
        
        # Calculate hip/pelvis position (root joint with translation)
        hip_center = (keypoints[L_HIP][:2] + keypoints[R_HIP][:2]) / 2
        
        # Convert 2D positions to approximate 3D with simple depth estimation
        def calculate_rotation_angles(parent_pos, child_pos, reference_vector=[0, 1]):
            """Calculate rotation angles from parent to child position"""
            vector = child_pos - parent_pos
            
            # Calculate rotation angles (simplified 2D to 3D mapping)
            angle_z = math.atan2(vector[0], vector[1])  # Z rotation (yaw)
            angle_x = 0.0  # X rotation (pitch) - simplified
            angle_y = 0.0  # Y rotation (roll) - simplified
            
            return [
                math.degrees(angle_z),
                math.degrees(angle_x), 
                math.degrees(angle_y)
            ]
        
        # Root joint (Hips) - include translation
        joint_angles['Hips'] = [
            hip_center[0],  # X position
            hip_center[1],  # Y position  
            0.0,            # Z position (depth)
            0.0, 0.0, 0.0   # Rotations (simplified)
        ]
        
        # Spine and torso
        shoulder_center = (keypoints[L_SHOULDER][:2] + keypoints[R_SHOULDER][:2]) / 2
        joint_angles['Spine'] = calculate_rotation_angles(hip_center, shoulder_center)
        joint_angles['Chest'] = [0.0, 0.0, 0.0]  # Simplified
        
        # Head and neck
        head_pos = keypoints[NOSE][:2]
        joint_angles['Neck'] = calculate_rotation_angles(shoulder_center, head_pos)
        joint_angles['Head'] = [0.0, 0.0, 0.0]  # Simplified
        
        # Left arm chain
        joint_angles['LeftShoulder'] = [0.0, 0.0, 0.0]  # Simplified
        joint_angles['LeftArm'] = calculate_rotation_angles(
            keypoints[L_SHOULDER][:2], keypoints[L_ELBOW][:2]
        )
        joint_angles['LeftForeArm'] = calculate_rotation_angles(
            keypoints[L_ELBOW][:2], keypoints[L_WRIST][:2]
        )
        joint_angles['LeftHand'] = [0.0, 0.0, 0.0]  # End effector
        
        # Right arm chain (mirrored)
        joint_angles['RightShoulder'] = [0.0, 0.0, 0.0]  # Simplified
        joint_angles['RightArm'] = calculate_rotation_angles(
            keypoints[R_SHOULDER][:2], keypoints[R_ELBOW][:2]
        )
        joint_angles['RightForeArm'] = calculate_rotation_angles(
            keypoints[R_ELBOW][:2], keypoints[R_WRIST][:2]
        )
        joint_angles['RightHand'] = [0.0, 0.0, 0.0]  # End effector
        
        # Left leg chain
        joint_angles['LeftUpLeg'] = calculate_rotation_angles(
            keypoints[L_HIP][:2], keypoints[L_KNEE][:2]
        )
        joint_angles['LeftLeg'] = calculate_rotation_angles(
            keypoints[L_KNEE][:2], keypoints[L_ANKLE][:2]
        )
        joint_angles['LeftFoot'] = [0.0, 0.0, 0.0]  # End effector
        
        # Right leg chain
        joint_angles['RightUpLeg'] = calculate_rotation_angles(
            keypoints[R_HIP][:2], keypoints[R_KNEE][:2]
        )
        joint_angles['RightLeg'] = calculate_rotation_angles(
            keypoints[R_KNEE][:2], keypoints[R_ANKLE][:2]
        )
        joint_angles['RightFoot'] = [0.0, 0.0, 0.0]  # End effector
        
        return joint_angles
    
    def smooth_motion_data(self, motion_frames: List[Dict], window_size: int = 5) -> List[Dict]:
        """
        Apply temporal smoothing to reduce jitter in motion data
        
        Args:
            motion_frames: List of joint angle dictionaries per frame
            window_size: Size of smoothing window
            
        Returns:
            Smoothed motion data
        """
        if len(motion_frames) < window_size:
            return motion_frames
        
        smoothed_frames = []
        
        for i in range(len(motion_frames)):
            smoothed_frame = {}
            
            # Define smoothing window
            start_idx = max(0, i - window_size // 2)
            end_idx = min(len(motion_frames), i + window_size // 2 + 1)
            
            # Average joint angles within window
            for joint_name in self.joint_names:
                joint_data = []
                
                for j in range(start_idx, end_idx):
                    if joint_name in motion_frames[j]:
                        joint_data.append(motion_frames[j][joint_name])
                
                if joint_data:
                    # Average the angles
                    smoothed_frame[joint_name] = np.mean(joint_data, axis=0).tolist()
                else:
                    # Use current frame data if no surrounding data
                    smoothed_frame[joint_name] = motion_frames[i].get(joint_name, [0.0] * 6)
            
            smoothed_frames.append(smoothed_frame)
        
        return smoothed_frames
    
    def write_bvh_file(self, motion_data: List[Dict], output_path: str, fps: float = 30.0):
        """
        Write motion data to BVH file compatible with Blender
        
        Args:
            motion_data: List of joint angle dictionaries per frame
            output_path: Output BVH file path
            fps: Frame rate for the animation
        """
        with open(output_path, 'w') as f:
            # Write HIERARCHY section 
            f.write("HIERARCHY\n")
            self._write_joint_hierarchy(f, 'Hips', 0)
            
            # Write MOTION section 
            f.write("MOTION\n")
            f.write(f"Frames: {len(motion_data)}\n")
            f.write(f"Frame Time: {1.0/fps:.6f}\n")
            
            # Write motion data
            for frame_data in motion_data:
                frame_values = []
                
                for joint_name in self.joint_names:
                    if joint_name in frame_data:
                        values = frame_data[joint_name]
                        frame_values.extend(values)
                    else:
                        # Default values for missing joints
                        if joint_name == 'Hips':
                            frame_values.extend([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
                        else:
                            frame_values.extend([0.0, 0.0, 0.0])
                
                # Write frame data
                f.write(" ".join(f"{val:.6f}" for val in frame_values) + "\n")
        
        self.logger.info(f"BVH file written: {output_path}")
    
    def _write_joint_hierarchy(self, f, joint_name: str, indent_level: int):
        """Recursively write joint hierarchy to BVH file"""
        indent = "  " * indent_level
        joint_info = self.joint_hierarchy[joint_name]
        
        # Write joint header
        if joint_info['parent'] is None:
            f.write(f"{indent}ROOT {joint_name}\n")
        else:
            f.write(f"{indent}JOINT {joint_name}\n")
        
        f.write(f"{indent}{{\n")
        
        # Write offset 
        offset = joint_info['offset']
        f.write(f"{indent}  OFFSET {offset[0]:.6f} {offset[1]:.6f} {offset[2]:.6f}\n")
        
        # Write channels 
        channels = joint_info['channels']
        f.write(f"{indent}  CHANNELS {len(channels)} {' '.join(channels)}\n")
        
        # Write child joints recursively
        for child_name in joint_info['children']:
            self._write_joint_hierarchy(f, child_name, indent_level + 1)
        
        # End site for leaf joints
        if not joint_info['children']:
            f.write(f"{indent}  End Site\n")
            f.write(f"{indent}  {{\n")
            f.write(f"{indent}    OFFSET 0.0 0.0 0.0\n")
            f.write(f"{indent}  }}\n")
        
        f.write(f"{indent}}}\n")
    
    def convert_video_to_bvh(self, video_path: str, output_path: str, 
                           fps: float = 30.0, max_frames: Optional[int] = None) -> bool:
        """
        Complete pipeline: MP4 video -> BVH motion capture file
        
        Args:
            video_path: Path to input MP4 file
            output_path: Path for output BVH file
            fps: Target frame rate for BVH animation
            max_frames: Maximum frames to process (None for all)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.logger.info(f"Starting conversion: {video_path} -> {output_path}")
            
            # Step 1: Extract frames from video
            frames = self.extract_frames_from_video(video_path, max_frames)
            if not frames:
                self.logger.error("No frames extracted from video")
                return False
            
            # Step 2: Estimate poses from frames
            pose_data = self.estimate_poses_from_frames(frames)
            
            # Step 3: Convert poses to joint angles
            motion_data = []
            
            for i, frame_poses in enumerate(pose_data):
                if frame_poses:
                    # Use first detected person
                    person_id = list(frame_poses.keys())[0]
                    keypoints = frame_poses[person_id]['keypoints']
                    
                    # Calculate joint angles
                    joint_angles = self.calculate_joint_angles(keypoints)
                    motion_data.append(joint_angles)
                else:
                    # No pose detected - use neutral pose
                    neutral_pose = {joint: [0.0] * len(self.joint_hierarchy[joint]['channels']) 
                                  for joint in self.joint_names}
                    motion_data.append(neutral_pose)
            
            # Step 4: Apply temporal smoothing
            smoothed_motion = self.smooth_motion_data(motion_data)
            
            # Step 5: Write BVH file
            self.write_bvh_file(smoothed_motion, output_path, fps)
            
            self.logger.info("Conversion completed successfully!")
            return True
            
        except Exception as e:
            self.logger.error(f"Conversion failed: {str(e)}")
            return False

# Usage example and installation requirements
def main():
    """Example usage of the MP4 to BVH converter"""
    # Model paths (download required)
    parser = argparse.ArgumentParser(description='Convert video to BVH using MediaPipe Pose')
    parser.add_argument('--video', required=True, help='Input video file')
    parser.add_argument('--output', required=True, help='Output BVH file')
    vitpose_model = "model/vitpose-s-coco_25.pth"
    yolo_model = "model/yolov8l.pt"
    args = parser.parse_args()
    
    # Initialize converter
    converter = MP4ToBVHConverter(
        vitpose_model_path=vitpose_model,
        yolo_model_path=yolo_model,
        model_size='base',
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # Convert video to BVH
    success = converter.convert_video_to_bvh(
        # video_path="input_video.mp4",
        # output_path="output_motion.bvh",
        video_path=args.video,
        output_path=args.output,
        fps=30.0,
        max_frames=300  # Limit for testing
    )
    
    if success:
        print("✅ Conversion successful! Import output_motion.bvh into Blender.")
    else:
        print("❌ Conversion failed. Check logs for details.")

if __name__ == "__main__":
    main()