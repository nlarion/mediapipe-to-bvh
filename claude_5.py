import argparse
import cv2
import numpy as np
import torch
import os
from pathlib import Path
import logging
from typing import List, Dict, Tuple, Optional
import math
from easy_ViTPose import VitInference

class FixedMP4ToBVHConverter:
    """
    Fixed pipeline for converting MP4 videos to BVH with proper coordinate handling
    """
    
    def __init__(self, vitpose_model_path: str, yolo_model_path: str, 
                 model_size: str = 'b', device: str = 'cuda'):
        self.device = device
        self.model_size = model_size
        self.setup_logging()
        
        # Initialize ViTPose model
        self.pose_model = VitInference(
            vitpose_model_path,
            yolo_model_path,
            model_name=model_size,
            yolo_size=320,
            is_video=True,
            device=device
        )
        
        # Setup joint hierarchy
        self.joint_hierarchy = self._create_bvh_hierarchy()
        self.joint_names = self._get_joint_names()
        
        # Reference pose for normalization (T-pose)
        self.reference_height = 180.0  # cm
        
    def setup_logging(self):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def _create_bvh_hierarchy(self) -> Dict:
        """BVH joint hierarchy with proper offsets"""
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
                'offset': [0.0, 10.0, 0.0],
                'channels': ['Zrotation', 'Xrotation', 'Yrotation'],
                'children': ['Chest']
            },
            'Chest': {
                'parent': 'Spine', 
                'offset': [0.0, 15.0, 0.0],
                'channels': ['Zrotation', 'Xrotation', 'Yrotation'],
                'children': ['Neck', 'LeftShoulder', 'RightShoulder']
            },
            'Neck': {
                'parent': 'Chest',
                'offset': [0.0, 10.0, 0.0],
                'channels': ['Zrotation', 'Xrotation', 'Yrotation'],
                'children': ['Head']
            },
            'Head': {
                'parent': 'Neck',
                'offset': [0.0, 10.0, 0.0],
                'channels': ['Zrotation', 'Xrotation', 'Yrotation'],
                'children': []
            },
            'LeftShoulder': {
                'parent': 'Chest',
                'offset': [-5.0, 5.0, 0.0],
                'channels': ['Zrotation', 'Xrotation', 'Yrotation'],
                'children': ['LeftArm']
            },
            'LeftArm': {
                'parent': 'LeftShoulder',
                'offset': [-15.0, 0.0, 0.0],
                'channels': ['Zrotation', 'Xrotation', 'Yrotation'],
                'children': ['LeftForeArm']
            },
            'LeftForeArm': {
                'parent': 'LeftArm',
                'offset': [-12.0, 0.0, 0.0],
                'channels': ['Zrotation', 'Xrotation', 'Yrotation'],
                'children': ['LeftHand']
            },
            'LeftHand': {
                'parent': 'LeftForeArm',
                'offset': [-8.0, 0.0, 0.0],
                'channels': ['Zrotation', 'Xrotation', 'Yrotation'],
                'children': []
            },
            'RightShoulder': {
                'parent': 'Chest',
                'offset': [5.0, 5.0, 0.0],
                'channels': ['Zrotation', 'Xrotation', 'Yrotation'],
                'children': ['RightArm']
            },
            'RightArm': {
                'parent': 'RightShoulder',
                'offset': [15.0, 0.0, 0.0],
                'channels': ['Zrotation', 'Xrotation', 'Yrotation'],
                'children': ['RightForeArm']
            },
            'RightForeArm': {
                'parent': 'RightArm',
                'offset': [12.0, 0.0, 0.0],
                'channels': ['Zrotation', 'Xrotation', 'Yrotation'],
                'children': ['RightHand']
            },
            'RightHand': {
                'parent': 'RightForeArm',
                'offset': [8.0, 0.0, 0.0],
                'channels': ['Zrotation', 'Xrotation', 'Yrotation'],
                'children': []
            },
            'LeftUpLeg': {
                'parent': 'Hips',
                'offset': [-8.0, -5.0, 0.0],
                'channels': ['Zrotation', 'Xrotation', 'Yrotation'],
                'children': ['LeftLeg']
            },
            'LeftLeg': {
                'parent': 'LeftUpLeg',
                'offset': [0.0, -40.0, 0.0],
                'channels': ['Zrotation', 'Xrotation', 'Yrotation'],
                'children': ['LeftFoot']
            },
            'LeftFoot': {
                'parent': 'LeftLeg',
                'offset': [0.0, -40.0, 0.0],
                'channels': ['Zrotation', 'Xrotation', 'Yrotation'],
                'children': []
            },
            'RightUpLeg': {
                'parent': 'Hips',
                'offset': [8.0, -5.0, 0.0],
                'channels': ['Zrotation', 'Xrotation', 'Yrotation'],
                'children': ['RightLeg']
            },
            'RightLeg': {
                'parent': 'RightUpLeg',
                'offset': [0.0, -40.0, 0.0],
                'channels': ['Zrotation', 'Xrotation', 'Yrotation'],
                'children': ['RightFoot']
            },
            'RightFoot': {
                'parent': 'RightLeg',
                'offset': [0.0, -40.0, 0.0],
                'channels': ['Zrotation', 'Xrotation', 'Yrotation'],
                'children': []
            }
        }
    
    def _get_joint_names(self) -> List[str]:
        return ['Hips', 'Spine', 'Chest', 'Neck', 'Head',
                'LeftShoulder', 'LeftArm', 'LeftForeArm', 'LeftHand',
                'RightShoulder', 'RightArm', 'RightForeArm', 'RightHand',
                'LeftUpLeg', 'LeftLeg', 'LeftFoot',
                'RightUpLeg', 'RightLeg', 'RightFoot']
    
    def extract_frames_from_video(self, video_path: str, 
                                 max_frames: Optional[int] = None) -> Tuple[List[np.ndarray], float]:
        """Extract frames and return with FPS"""
        frames = []
        cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
        
        if not cap.isOpened():
            raise IOError(f"Cannot open video file: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        self.video_width = width
        self.video_height = height
        
        self.logger.info(f"Video: {total_frames} frames at {fps:.2f} FPS, {width}x{height}")
        
        frame_count = 0
        
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                
                if not ret:
                    break
                
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(rgb_frame)
                
                frame_count += 1
                
                if max_frames and frame_count >= max_frames:
                    break
                
        finally:
            cap.release()
        
        self.logger.info(f"Extracted {len(frames)} frames")
        return frames, fps
    
    def estimate_poses_from_frames(self, frames: List[np.ndarray]) -> List[Dict]:
        """Extract poses with proper normalization"""
        pose_data = []
        
        for i, frame in enumerate(frames):
            try:
                results = self.pose_model.inference(frame)
                
                frame_poses = {}
                
                if results:
                    for person_id, keypoints in results.items():
                        # Convert [y, x, conf] to [x, y, conf]
                        processed_keypoints = []
                        
                        for keypoint in keypoints:
                            y, x, conf = keypoint
                            processed_keypoints.append([x, y, conf])
                        
                        frame_poses[person_id] = {
                            'keypoints': np.array(processed_keypoints),
                            'confidence': np.mean([kp[2] for kp in processed_keypoints])
                        }
                
                pose_data.append(frame_poses)
                
            except Exception as e:
                self.logger.error(f"Error in frame {i}: {str(e)}")
                pose_data.append({})
        
        self.pose_model.reset()
        return pose_data
    
    def normalize_keypoints(self, keypoints: np.ndarray) -> np.ndarray:
        """Normalize keypoints to consistent scale and center"""
        # COCO keypoint indices
        L_HIP = 11
        R_HIP = 12
        L_ANKLE = 15
        R_ANKLE = 16
        NOSE = 0
        
        # Calculate hip center
        hip_center = (keypoints[L_HIP][:2] + keypoints[R_HIP][:2]) / 2
        
        # Estimate height from pose
        if keypoints[L_ANKLE][2] > 0.5 and keypoints[R_ANKLE][2] > 0.5:
            # Use ankle to nose distance as height estimate
            ankle_center = (keypoints[L_ANKLE][:2] + keypoints[R_ANKLE][:2]) / 2
            pose_height = np.linalg.norm(keypoints[NOSE][:2] - ankle_center)
        else:
            # Fallback: use hip to nose distance * 2.5
            pose_height = np.linalg.norm(keypoints[NOSE][:2] - hip_center) * 2.5
        
        # Avoid division by zero
        if pose_height < 10:
            pose_height = self.video_height * 0.6
        
        # Scale factor to normalize to reference height
        scale = self.reference_height / pose_height
        
        # Center and scale keypoints
        normalized = keypoints.copy()
        for i in range(len(keypoints)):
            # Center on hip
            normalized[i][0] = (keypoints[i][0] - hip_center[0]) * scale
            normalized[i][1] = (keypoints[i][1] - hip_center[1]) * scale
            
            # Flip Y axis (video Y is down, BVH Y is up)
            normalized[i][1] = -normalized[i][1]
        
        return normalized
    
    def calculate_joint_angles_fixed(self, keypoints: np.ndarray) -> Dict[str, List[float]]:
        """Calculate joint angles with proper coordinate handling"""
        # Normalize keypoints first
        norm_kp = self.normalize_keypoints(keypoints)
        
        # COCO keypoint indices
        NOSE = 0
        L_SHOULDER = 5; R_SHOULDER = 6
        L_ELBOW = 7; R_ELBOW = 8
        L_WRIST = 9; R_WRIST = 10
        L_HIP = 11; R_HIP = 12
        L_KNEE = 13; R_KNEE = 14
        L_ANKLE = 15; R_ANKLE = 16
        
        joint_angles = {}
        
        # Hip center (root position)
        hip_center = (norm_kp[L_HIP][:2] + norm_kp[R_HIP][:2]) / 2
        
        # Root joint with world position
        joint_angles['Hips'] = [
            hip_center[0],    # X position
            hip_center[1],    # Y position (adjusted for ground level)
            0.0,              # Z position
            0.0, 0.0, 0.0     # Rotations
        ]
        
        # Helper function for safe angle calculation
        def calculate_angle_2d(parent_pos, child_pos):
            """Calculate angle between two 2D points"""
            vector = child_pos - parent_pos
            # Avoid division by zero
            if np.linalg.norm(vector) < 0.001:
                return [0.0, 0.0, 0.0]
            
            # Calculate angles
            angle_y = math.atan2(vector[0], vector[1])  # Rotation around Y (yaw)
            
            return [0.0, 0.0, math.degrees(angle_y)]  # [X, Y, Z] rotations
        
        # Spine and torso
        shoulder_center = (norm_kp[L_SHOULDER][:2] + norm_kp[R_SHOULDER][:2]) / 2
        spine_vec = shoulder_center - hip_center
        spine_angle = math.atan2(spine_vec[0], spine_vec[1])
        joint_angles['Spine'] = [0.0, 0.0, math.degrees(spine_angle)]
        
        # Chest (minimal rotation)
        joint_angles['Chest'] = [0.0, 0.0, 0.0]
        
        # Neck and head
        if norm_kp[NOSE][2] > 0.3:  # If nose is visible
            neck_vec = norm_kp[NOSE][:2] - shoulder_center
            neck_angle = math.atan2(neck_vec[0], neck_vec[1])
            joint_angles['Neck'] = [0.0, 0.0, math.degrees(neck_angle) - math.degrees(spine_angle)]
        else:
            joint_angles['Neck'] = [0.0, 0.0, 0.0]
        
        joint_angles['Head'] = [0.0, 0.0, 0.0]
        
        # Arms
        # Left arm
        joint_angles['LeftShoulder'] = [0.0, 0.0, 0.0]
        if norm_kp[L_SHOULDER][2] > 0.3 and norm_kp[L_ELBOW][2] > 0.3:
            arm_vec = norm_kp[L_ELBOW][:2] - norm_kp[L_SHOULDER][:2]
            arm_angle = math.atan2(arm_vec[1], -arm_vec[0])  # Adjust for left side
            joint_angles['LeftArm'] = [0.0, 0.0, math.degrees(arm_angle)]
            
            if norm_kp[L_WRIST][2] > 0.3:
                forearm_vec = norm_kp[L_WRIST][:2] - norm_kp[L_ELBOW][:2]
                forearm_angle = math.atan2(forearm_vec[1], -forearm_vec[0])
                elbow_bend = math.degrees(forearm_angle - arm_angle)
                joint_angles['LeftForeArm'] = [0.0, elbow_bend, 0.0]
            else:
                joint_angles['LeftForeArm'] = [0.0, 0.0, 0.0]
        else:
            joint_angles['LeftArm'] = [0.0, 0.0, 0.0]
            joint_angles['LeftForeArm'] = [0.0, 0.0, 0.0]
        
        joint_angles['LeftHand'] = [0.0, 0.0, 0.0]
        
        # Right arm
        joint_angles['RightShoulder'] = [0.0, 0.0, 0.0]
        if norm_kp[R_SHOULDER][2] > 0.3 and norm_kp[R_ELBOW][2] > 0.3:
            arm_vec = norm_kp[R_ELBOW][:2] - norm_kp[R_SHOULDER][:2]
            arm_angle = math.atan2(arm_vec[1], arm_vec[0])
            joint_angles['RightArm'] = [0.0, 0.0, math.degrees(arm_angle)]
            
            if norm_kp[R_WRIST][2] > 0.3:
                forearm_vec = norm_kp[R_WRIST][:2] - norm_kp[R_ELBOW][:2]
                forearm_angle = math.atan2(forearm_vec[1], forearm_vec[0])
                elbow_bend = math.degrees(forearm_angle - arm_angle)
                joint_angles['RightForeArm'] = [0.0, elbow_bend, 0.0]
            else:
                joint_angles['RightForeArm'] = [0.0, 0.0, 0.0]
        else:
            joint_angles['RightArm'] = [0.0, 0.0, 0.0]
            joint_angles['RightForeArm'] = [0.0, 0.0, 0.0]
        
        joint_angles['RightHand'] = [0.0, 0.0, 0.0]
        
        # Legs
        # Left leg
        if norm_kp[L_HIP][2] > 0.3 and norm_kp[L_KNEE][2] > 0.3:
            thigh_vec = norm_kp[L_KNEE][:2] - norm_kp[L_HIP][:2]
            thigh_angle = math.atan2(thigh_vec[0], -thigh_vec[1])  # Y is up
            joint_angles['LeftUpLeg'] = [math.degrees(thigh_angle), 0.0, 0.0]
            
            if norm_kp[L_ANKLE][2] > 0.3:
                shin_vec = norm_kp[L_ANKLE][:2] - norm_kp[L_KNEE][:2]
                shin_angle = math.atan2(shin_vec[0], -shin_vec[1])
                knee_bend = math.degrees(shin_angle - thigh_angle)
                joint_angles['LeftLeg'] = [knee_bend, 0.0, 0.0]
            else:
                joint_angles['LeftLeg'] = [0.0, 0.0, 0.0]
        else:
            joint_angles['LeftUpLeg'] = [0.0, 0.0, 0.0]
            joint_angles['LeftLeg'] = [0.0, 0.0, 0.0]
        
        joint_angles['LeftFoot'] = [0.0, 0.0, 0.0]
        
        # Right leg
        if norm_kp[R_HIP][2] > 0.3 and norm_kp[R_KNEE][2] > 0.3:
            thigh_vec = norm_kp[R_KNEE][:2] - norm_kp[R_HIP][:2]
            thigh_angle = math.atan2(thigh_vec[0], -thigh_vec[1])
            joint_angles['RightUpLeg'] = [math.degrees(thigh_angle), 0.0, 0.0]
            
            if norm_kp[R_ANKLE][2] > 0.3:
                shin_vec = norm_kp[R_ANKLE][:2] - norm_kp[R_KNEE][:2]
                shin_angle = math.atan2(shin_vec[0], -shin_vec[1])
                knee_bend = math.degrees(shin_angle - thigh_angle)
                joint_angles['RightLeg'] = [knee_bend, 0.0, 0.0]
            else:
                joint_angles['RightLeg'] = [0.0, 0.0, 0.0]
        else:
            joint_angles['RightUpLeg'] = [0.0, 0.0, 0.0]
            joint_angles['RightLeg'] = [0.0, 0.0, 0.0]
        
        joint_angles['RightFoot'] = [0.0, 0.0, 0.0]
        
        return joint_angles
    
    def smooth_motion_data(self, motion_frames: List[Dict], window_size: int = 5) -> List[Dict]:
        """Apply temporal smoothing"""
        if len(motion_frames) < window_size:
            return motion_frames
        
        smoothed_frames = []
        
        for i in range(len(motion_frames)):
            smoothed_frame = {}
            
            start_idx = max(0, i - window_size // 2)
            end_idx = min(len(motion_frames), i + window_size // 2 + 1)
            
            for joint_name in self.joint_names:
                joint_data = []
                
                for j in range(start_idx, end_idx):
                    if joint_name in motion_frames[j]:
                        joint_data.append(motion_frames[j][joint_name])
                
                if joint_data:
                    smoothed_frame[joint_name] = np.mean(joint_data, axis=0).tolist()
                else:
                    # Default values based on joint type
                    if joint_name == 'Hips':
                        smoothed_frame[joint_name] = [0.0, 90.0, 0.0, 0.0, 0.0, 0.0]  # Standing height
                    else:
                        smoothed_frame[joint_name] = [0.0, 0.0, 0.0]
            
            smoothed_frames.append(smoothed_frame)
        
        return smoothed_frames
    
    def write_bvh_file(self, motion_data: List[Dict], output_path: str, fps: float = 30.0):
        """Write BVH file"""
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
                        # Default values
                        if joint_name == 'Hips':
                            frame_values.extend([0.0, 90.0, 0.0, 0.0, 0.0, 0.0])
                        else:
                            frame_values.extend([0.0, 0.0, 0.0])
                
                f.write(" ".join(f"{val:.6f}" for val in frame_values) + "\n")
        
        self.logger.info(f"BVH file written: {output_path}")
    
    def _write_joint_hierarchy(self, f, joint_name: str, indent_level: int):
        """Write joint hierarchy"""
        indent = "  " * indent_level
        joint_info = self.joint_hierarchy[joint_name]
        
        if joint_info['parent'] is None:
            f.write(f"{indent}ROOT {joint_name}\n")
        else:
            f.write(f"{indent}JOINT {joint_name}\n")
        
        f.write(f"{indent}{{\n")
        
        offset = joint_info['offset']
        f.write(f"{indent}  OFFSET {offset[0]:.6f} {offset[1]:.6f} {offset[2]:.6f}\n")
        
        channels = joint_info['channels']
        f.write(f"{indent}  CHANNELS {len(channels)} {' '.join(channels)}\n")
        
        for child_name in joint_info['children']:
            self._write_joint_hierarchy(f, child_name, indent_level + 1)
        
        if not joint_info['children']:
            f.write(f"{indent}  End Site\n")
            f.write(f"{indent}  {{\n")
            f.write(f"{indent}    OFFSET 0.0 0.0 0.0\n")
            f.write(f"{indent}  }}\n")
        
        f.write(f"{indent}}}\n")
    
    def convert_video_to_bvh(self, video_path: str, output_path: str, 
                           max_frames: Optional[int] = None) -> bool:
        """Complete pipeline with fixes"""
        try:
            self.logger.info(f"Starting conversion: {video_path} -> {output_path}")
            
            # Extract frames
            frames, fps = self.extract_frames_from_video(video_path, max_frames)
            if not frames:
                self.logger.error("No frames extracted")
                return False
            
            # Estimate poses
            pose_data = self.estimate_poses_from_frames(frames)
            
            # Convert to joint angles
            motion_data = []
            
            for i, frame_poses in enumerate(pose_data):
                if frame_poses:
                    # Use first detected person
                    person_id = list(frame_poses.keys())[0]
                    keypoints = frame_poses[person_id]['keypoints']
                    
                    # Use the fixed angle calculation
                    joint_angles = self.calculate_joint_angles_fixed(keypoints)
                    motion_data.append(joint_angles)
                else:
                    # Default standing pose
                    default_pose = {}
                    for joint in self.joint_names:
                        if joint == 'Hips':
                            default_pose[joint] = [0.0, 90.0, 0.0, 0.0, 0.0, 0.0]
                        else:
                            default_pose[joint] = [0.0, 0.0, 0.0]
                    motion_data.append(default_pose)
            
            # Smooth motion
            smoothed_motion = self.smooth_motion_data(motion_data)
            
            # Write BVH
            self.write_bvh_file(smoothed_motion, output_path, fps)
            
            self.logger.info("Conversion completed successfully!")
            return True
            
        except Exception as e:
            self.logger.error(f"Conversion failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

def main():
    """Example usage"""
    # Model paths
    parser = argparse.ArgumentParser(description='Convert video to BVH using MediaPipe Pose')
    parser.add_argument('--video', required=True, help='Input video file')
    parser.add_argument('--output', required=True, help='Output BVH file')

    vitpose_model = "model/vitpose-s-coco_25.pth"
    yolo_model = "model/yolov8l.pt"
    args = parser.parse_args()
    
    # Initialize converter
    converter = MathematicallyCorrectMP4ToBVH(
        vitpose_model_path=vitpose_model,
        yolo_model_path=yolo_model,
        model_size='b',
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # Convert video
    success = converter.convert_video_to_bvh(
        video_path=args.video,
        output_path=args.output,
        max_frames=300
    )
    
    if success:
        print("✅ Conversion successful!")
        print("\nTo import in Blender:")
        print("1. File > Import > Motion Capture (.bvh)")
        print("2. Select 'output_motion_fixed.bvh'")
        print("3. In import settings:")
        print("   - Forward: -Z Forward")
        print("   - Up: Y Up")
        print("   - Scale: 0.01 (if skeleton appears too large)")
    else:
        print("❌ Conversion failed.")

if __name__ == "__main__":
    main()