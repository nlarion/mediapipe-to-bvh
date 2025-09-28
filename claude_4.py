import cv2
import numpy as np
import argparse
import torch
import os
from pathlib import Path
import logging
from typing import List, Dict, Tuple, Optional
import math

# ViTPose implementation (using easy_ViTPose for simplicity)
from easy_ViTPose import VitInference

class MP4ToBVHConverter:
    """
    Fixed pipeline for converting MP4 videos to BVH motion capture files
    with proper coordinate transformation and joint calculations
    """
    
    def __init__(self, vitpose_model_path: str, yolo_model_path: str, 
                 model_size: str = 'b', device: str = 'cuda'):
        """
        Initialize the converter with ViTPose models
        """
        self.device = device
        
        # Map common model names to easy_ViTPose convention
        model_name_map = {
            'small': 's', 'base': 'b', 'large': 'l', 'huge': 'h',
            's': 's', 'b': 'b', 'l': 'l', 'h': 'h'
        }
        
        if model_size in model_name_map:
            self.model_size = model_name_map[model_size]
        else:
            raise ValueError(f"Invalid model_size: {model_size}")
        
        self.setup_logging()
        
        # Initialize ViTPose model
        self.pose_model = VitInference(
            vitpose_model_path,
            yolo_model_path,
            model_name=self.model_size,
            yolo_size=320,
            is_video=True,
            device=device
        )
        
        # Store video dimensions for coordinate normalization
        self.video_width = None
        self.video_height = None
        
        # BVH joint hierarchy for COCO 17-keypoint format
        self.joint_hierarchy = self._create_bvh_hierarchy()
        self.joint_names = self._get_joint_names()
        
    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def _create_bvh_hierarchy(self) -> Dict:
        """
        Define BVH joint hierarchy with proper offsets in meters
        """
        # Average human proportions in centimeters (converted to BVH units)
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
                'offset': [0.0, 15.0, 0.0],  # ~15cm up from hips
                'channels': ['Zrotation', 'Xrotation', 'Yrotation'],
                'children': ['Chest']
            },
            'Chest': {
                'parent': 'Spine',
                'offset': [0.0, 20.0, 0.0],  # ~20cm up from spine base
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
                'offset': [0.0, 15.0, 0.0],
                'channels': ['Zrotation', 'Xrotation', 'Yrotation'],
                'children': []
            },
            # Left arm chain
            'LeftShoulder': {
                'parent': 'Chest',
                'offset': [-15.0, 5.0, 0.0],
                'channels': ['Zrotation', 'Xrotation', 'Yrotation'],
                'children': ['LeftArm']
            },
            'LeftArm': {
                'parent': 'LeftShoulder',
                'offset': [-15.0, -5.0, 0.0],  # Upper arm ~30cm
                'channels': ['Zrotation', 'Xrotation', 'Yrotation'],
                'children': ['LeftForeArm']
            },
            'LeftForeArm': {
                'parent': 'LeftArm',
                'offset': [-25.0, 0.0, 0.0],  # Forearm ~25cm
                'channels': ['Zrotation', 'Xrotation', 'Yrotation'],
                'children': ['LeftHand']
            },
            'LeftHand': {
                'parent': 'LeftForeArm',
                'offset': [-15.0, 0.0, 0.0],
                'channels': ['Zrotation', 'Xrotation', 'Yrotation'],
                'children': []
            },
            # Right arm chain (mirrored)
            'RightShoulder': {
                'parent': 'Chest',
                'offset': [15.0, 5.0, 0.0],
                'channels': ['Zrotation', 'Xrotation', 'Yrotation'],
                'children': ['RightArm']
            },
            'RightArm': {
                'parent': 'RightShoulder',
                'offset': [15.0, -5.0, 0.0],
                'channels': ['Zrotation', 'Xrotation', 'Yrotation'],
                'children': ['RightForeArm']
            },
            'RightForeArm': {
                'parent': 'RightArm',
                'offset': [25.0, 0.0, 0.0],
                'channels': ['Zrotation', 'Xrotation', 'Yrotation'],
                'children': ['RightHand']
            },
            'RightHand': {
                'parent': 'RightForeArm',
                'offset': [15.0, 0.0, 0.0],
                'channels': ['Zrotation', 'Xrotation', 'Yrotation'],
                'children': []
            },
            # Left leg chain
            'LeftUpLeg': {
                'parent': 'Hips',
                'offset': [-10.0, -5.0, 0.0],
                'channels': ['Zrotation', 'Xrotation', 'Yrotation'],
                'children': ['LeftLeg']
            },
            'LeftLeg': {
                'parent': 'LeftUpLeg',
                'offset': [0.0, -40.0, 0.0],  # Upper leg ~40cm
                'channels': ['Zrotation', 'Xrotation', 'Yrotation'],
                'children': ['LeftFoot']
            },
            'LeftFoot': {
                'parent': 'LeftLeg',
                'offset': [0.0, -40.0, 0.0],  # Lower leg ~40cm
                'channels': ['Zrotation', 'Xrotation', 'Yrotation'],
                'children': []
            },
            # Right leg chain (mirrored)
            'RightUpLeg': {
                'parent': 'Hips',
                'offset': [10.0, -5.0, 0.0],
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
        """Get ordered list of joint names for BVH output"""
        return ['Hips', 'Spine', 'Chest', 'Neck', 'Head',
                'LeftShoulder', 'LeftArm', 'LeftForeArm', 'LeftHand',
                'RightShoulder', 'RightArm', 'RightForeArm', 'RightHand',
                'LeftUpLeg', 'LeftLeg', 'LeftFoot',
                'RightUpLeg', 'RightLeg', 'RightFoot']
    
    def extract_frames_from_video(self, video_path: str, 
                                 max_frames: Optional[int] = None) -> List[np.ndarray]:
        """Extract frames from MP4 video and store video dimensions"""
        frames = []
        
        cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
        
        if not cap.isOpened():
            raise IOError(f"Cannot open video file: {video_path}")
        
        # Store video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        self.video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        self.logger.info(f"Video: {total_frames} frames at {fps:.2f} FPS")
        self.logger.info(f"Dimensions: {self.video_width}x{self.video_height}")
        
        frame_count = 0
        
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                
                if not ret:
                    break
                
                if frame is None or frame.size == 0:
                    self.logger.warning(f"Empty frame at position {frame_count}")
                    continue
                
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(rgb_frame)
                
                frame_count += 1
                
                if max_frames and frame_count >= max_frames:
                    break
                    
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
        """Extract pose estimations from video frames using ViTPose"""
        pose_data = []
        
        self.logger.info(f"Processing {len(frames)} frames for pose estimation")
        
        for i, frame in enumerate(frames):
            try:
                results = self.pose_model.inference(frame)
                
                frame_poses = {}
                
                if results:
                    for person_id, keypoints in results.items():
                        # Convert to [x, y, conf] format
                        processed_keypoints = []
                        
                        for keypoint in keypoints:
                            y, x, conf = keypoint
                            processed_keypoints.append([x, y, conf])
                        
                        frame_poses[person_id] = {
                            'keypoints': np.array(processed_keypoints),
                            'confidence': np.mean([kp[2] for kp in processed_keypoints])
                        }
                
                pose_data.append(frame_poses)
                
                if (i + 1) % 50 == 0:
                    self.logger.info(f"Processed {i + 1}/{len(frames)} frames")
                
            except Exception as e:
                self.logger.error(f"Error processing frame {i}: {str(e)}")
                pose_data.append({})
        
        self.pose_model.reset()
        
        self.logger.info(f"Pose estimation complete for {len(pose_data)} frames")
        return pose_data
    
    def normalize_keypoints(self, keypoints: np.ndarray) -> np.ndarray:
        """
        Normalize keypoints from pixel coordinates to centered coordinates
        with proper scaling for BVH
        """
        # COCO keypoint indices
        L_HIP = 11
        R_HIP = 12
        
        # Calculate hip center in pixel coordinates
        hip_center = (keypoints[L_HIP][:2] + keypoints[R_HIP][:2]) / 2
        
        # Normalize coordinates:
        # - Center at hip position
        # - Scale to approximate human proportions
        # - Convert Y axis (image Y is down, BVH Y is up)
        normalized = keypoints.copy()
        
        # Center at hips
        normalized[:, 0] -= hip_center[0]  # X
        normalized[:, 1] -= hip_center[1]  # Y
        
        # Scale from pixels to approximate meters
        # Assume average human height ~170cm, typical in frame ~400 pixels
        scale_factor = 170.0 / 400.0  # cm per pixel
        
        normalized[:, 0] *= scale_factor
        normalized[:, 1] *= -scale_factor  # Flip Y axis
        
        return normalized
    
    def calculate_joint_angles(self, keypoints: np.ndarray) -> Dict[str, List[float]]:
        """
        Convert normalized 2D pose keypoints to joint angles for BVH format
        """
        # First normalize the keypoints
        norm_kp = self.normalize_keypoints(keypoints)
        
        # COCO keypoint indices
        NOSE = 0; L_EYE = 1; R_EYE = 2; L_EAR = 3; R_EAR = 4
        L_SHOULDER = 5; R_SHOULDER = 6; L_ELBOW = 7; R_ELBOW = 8
        L_WRIST = 9; R_WRIST = 10; L_HIP = 11; R_HIP = 12
        L_KNEE = 13; R_KNEE = 14; L_ANKLE = 15; R_ANKLE = 16
        
        joint_angles = {}
        
        # Helper function to calculate angle between two vectors
        def vector_angle(v1, v2):
            """Calculate angle between two 2D vectors"""
            dot = np.dot(v1, v2)
            det = v1[0] * v2[1] - v1[1] * v2[0]
            angle = math.atan2(det, dot)
            return math.degrees(angle)
        
        # Helper function to calculate joint angle
        def calculate_joint_rotation(parent_pos, child_pos):
            """Calculate rotation from parent to child joint"""
            # Vector from parent to child
            v = child_pos - parent_pos
            
            # Calculate angles
            # For 2D to 3D, we estimate rotations
            length = np.linalg.norm(v)
            if length < 0.001:  # Avoid division by zero
                return [0.0, 0.0, 0.0]
            
            # Primary rotation around Z (in 2D plane)
            angle_z = math.degrees(math.atan2(v[0], -v[1]))
            
            # Estimate X and Y rotations (depth)
            # Use confidence and position to estimate depth
            angle_x = 0.0  # Pitch
            angle_y = 0.0  # Roll
            
            return [angle_z, angle_x, angle_y]
        
        # Root joint (Hips) - position in world space
        hip_center = (norm_kp[L_HIP][:2] + norm_kp[R_HIP][:2]) / 2
        joint_angles['Hips'] = [
            0.0,  # X position (centered)
            100.0,  # Y position (height above ground)
            0.0,   # Z position (depth)
            0.0, 0.0, 0.0  # Initial rotation
        ]
        
        # Spine chain
        shoulder_center = (norm_kp[L_SHOULDER][:2] + norm_kp[R_SHOULDER][:2]) / 2
        spine_rotation = calculate_joint_rotation(hip_center, shoulder_center)
        joint_angles['Spine'] = spine_rotation
        joint_angles['Chest'] = [0.0, 0.0, 0.0]  # Neutral
        
        # Head chain
        head_rotation = calculate_joint_rotation(shoulder_center, norm_kp[NOSE][:2])
        joint_angles['Neck'] = [r * 0.5 for r in head_rotation]  # Reduce neck rotation
        joint_angles['Head'] = [r * 0.5 for r in head_rotation]  # Rest goes to head
        
        # Left arm
        joint_angles['LeftShoulder'] = [0.0, 0.0, -15.0]  # Slight outward rotation
        l_arm_rot = calculate_joint_rotation(norm_kp[L_SHOULDER][:2], norm_kp[L_ELBOW][:2])
        joint_angles['LeftArm'] = l_arm_rot
        l_forearm_rot = calculate_joint_rotation(norm_kp[L_ELBOW][:2], norm_kp[L_WRIST][:2])
        joint_angles['LeftForeArm'] = l_forearm_rot
        joint_angles['LeftHand'] = [0.0, 0.0, 0.0]
        
        # Right arm
        joint_angles['RightShoulder'] = [0.0, 0.0, 15.0]  # Slight outward rotation
        r_arm_rot = calculate_joint_rotation(norm_kp[R_SHOULDER][:2], norm_kp[R_ELBOW][:2])
        joint_angles['RightArm'] = r_arm_rot
        r_forearm_rot = calculate_joint_rotation(norm_kp[R_ELBOW][:2], norm_kp[R_WRIST][:2])
        joint_angles['RightForeArm'] = r_forearm_rot
        joint_angles['RightHand'] = [0.0, 0.0, 0.0]
        
        # Left leg
        l_upleg_rot = calculate_joint_rotation(norm_kp[L_HIP][:2], norm_kp[L_KNEE][:2])
        joint_angles['LeftUpLeg'] = l_upleg_rot
        l_leg_rot = calculate_joint_rotation(norm_kp[L_KNEE][:2], norm_kp[L_ANKLE][:2])
        joint_angles['LeftLeg'] = l_leg_rot
        joint_angles['LeftFoot'] = [0.0, 0.0, 0.0]
        
        # Right leg
        r_upleg_rot = calculate_joint_rotation(norm_kp[R_HIP][:2], norm_kp[R_KNEE][:2])
        joint_angles['RightUpLeg'] = r_upleg_rot
        r_leg_rot = calculate_joint_rotation(norm_kp[R_KNEE][:2], norm_kp[R_ANKLE][:2])
        joint_angles['RightLeg'] = r_leg_rot
        joint_angles['RightFoot'] = [0.0, 0.0, 0.0]
        
        return joint_angles
    
    def smooth_motion_data(self, motion_frames: List[Dict], window_size: int = 5) -> List[Dict]:
        """Apply temporal smoothing to reduce jitter in motion data"""
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
                    # Provide default values based on joint type
                    if joint_name == 'Hips':
                        smoothed_frame[joint_name] = [0.0, 100.0, 0.0, 0.0, 0.0, 0.0]
                    else:
                        smoothed_frame[joint_name] = [0.0, 0.0, 0.0]
            
            smoothed_frames.append(smoothed_frame)
        
        return smoothed_frames
    
    def write_bvh_file(self, motion_data: List[Dict], output_path: str, fps: float = 30.0):
        """Write motion data to BVH file compatible with Blender"""
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
                            frame_values.extend([0.0, 100.0, 0.0, 0.0, 0.0, 0.0])
                        else:
                            frame_values.extend([0.0, 0.0, 0.0])
                
                # Write frame data
                f.write(" ".join(f"{val:.6f}" for val in frame_values) + "\n")
        
        self.logger.info(f"BVH file written: {output_path}")
    
    def _write_joint_hierarchy(self, f, joint_name: str, indent_level: int):
        """Recursively write joint hierarchy to BVH file"""
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
            f.write(f"{indent}    OFFSET 0.0 -5.0 0.0\n")  # Small end site offset
            f.write(f"{indent}  }}\n")
        
        f.write(f"{indent}}}\n")
    
    def convert_video_to_bvh(self, video_path: str, output_path: str, 
                           fps: float = 30.0, max_frames: Optional[int] = None) -> bool:
        """Complete pipeline: MP4 video -> BVH motion capture file"""
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
                    
                    # Calculate joint angles with proper normalization
                    joint_angles = self.calculate_joint_angles(keypoints)
                    motion_data.append(joint_angles)
                else:
                    # No pose detected - use default standing pose
                    default_pose = {
                        'Hips': [0.0, 100.0, 0.0, 0.0, 0.0, 0.0],
                        'Spine': [0.0, 0.0, 0.0],
                        'Chest': [0.0, 0.0, 0.0],
                        'Neck': [0.0, 0.0, 0.0],
                        'Head': [0.0, 0.0, 0.0],
                        'LeftShoulder': [0.0, 0.0, -15.0],
                        'LeftArm': [0.0, 0.0, -45.0],
                        'LeftForeArm': [0.0, 0.0, 0.0],
                        'LeftHand': [0.0, 0.0, 0.0],
                        'RightShoulder': [0.0, 0.0, 15.0],
                        'RightArm': [0.0, 0.0, 45.0],
                        'RightForeArm': [0.0, 0.0, 0.0],
                        'RightHand': [0.0, 0.0, 0.0],
                        'LeftUpLeg': [0.0, 0.0, 0.0],
                        'LeftLeg': [0.0, 0.0, 0.0],
                        'LeftFoot': [0.0, 0.0, 0.0],
                        'RightUpLeg': [0.0, 0.0, 0.0],
                        'RightLeg': [0.0, 0.0, 0.0],
                        'RightFoot': [0.0, 0.0, 0.0]
                    }
                    motion_data.append(default_pose)
            
            # Step 4: Apply temporal smoothing
            smoothed_motion = self.smooth_motion_data(motion_data)
            
            # Step 5: Write BVH file
            self.write_bvh_file(smoothed_motion, output_path, fps)
            
            self.logger.info("Conversion completed successfully!")
            return True
            
        except Exception as e:
            self.logger.error(f"Conversion failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

# Usage example
def main():
    """Example usage of the MP4 to BVH converter"""
    
    # Model paths (download required)
    # vitpose_model = "./models/vitpose_base_coco_256x192.pth"
    # yolo_model = "./models/yolov8s.pt"
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
        model_size='s',  # 's'=small, 'b'=base, 'l'=large, 'h'=huge
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # Convert video to BVH
    success = converter.convert_video_to_bvh(
        video_path=args.video,
        output_path=args.output,
        fps=30.0,
        max_frames=300  # Limit for testing
    )
    
    if success:
        print("✅ Conversion successful! Import output_motion.bvh into Blender.")
        print("\nBlender Import Tips:")
        print("1. File > Import > Motion Capture (.bvh)")
        print("2. In import settings, try:")
        print("   - Scale: 0.01 (if skeleton is too large)")
        print("   - Forward: -Z Forward")
        print("   - Up: Y Up")
    else:
        print("❌ Conversion failed. Check logs for details.")

if __name__ == "__main__":
    main()