import cv2
import numpy as np
import torch
import os
from pathlib import Path
import logging
from typing import List, Dict, Tuple, Optional
import math
import json
import argparse

# Fix 1: Modify easy_ViTPose import to avoid torch_tensorrt
import sys
import importlib.util

# Temporarily disable torch_tensorrt import
original_import = __builtins__.__import__

def custom_import(name, *args, **kwargs):
    if name == "torch_tensorrt":
        # Return a dummy module to avoid the import error
        from types import ModuleType
        return ModuleType("torch_tensorrt")
    return original_import(name, *args, **kwargs)

__builtins__.__import__ = custom_import

try:
    from easy_ViTPose import VitInference
except ImportError:
    print("easy_ViTPose not found. Using alternative implementation...")
    VitInference = None

# Restore original import
__builtins__.__import__ = original_import

# Alternative: Use MMPose directly if easy_ViTPose fails
try:
    from mmpose.apis import init_model, inference_topdown
    from mmdet.apis import init_detector, inference_detector
    MMPOSE_AVAILABLE = True
except ImportError:
    MMPOSE_AVAILABLE = False
    print("MMPose not available. Will use MediaPipe as fallback.")

# Fallback: Use MediaPipe for pose estimation
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    print("MediaPipe not available.")

class MP4ToBVHConverter:
    """
    Complete pipeline for converting MP4 videos to BVH motion capture files
    with multiple pose estimation backend support.
    """
    
    def __init__(self, backend='auto', device='cuda'):
        """
        Initialize the converter with flexible backend support
        
        Args:
            backend: 'vitpose', 'mmpose', 'mediapipe', or 'auto'
            device: Computing device ('cuda' or 'cpu')
        """
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.setup_logging()
        
        # Auto-select backend based on availability
        if backend == 'auto':
            if VitInference is not None:
                backend = 'vitpose'
            elif MMPOSE_AVAILABLE:
                backend = 'mmpose'
            elif MEDIAPIPE_AVAILABLE:
                backend = 'mediapipe'
            else:
                raise RuntimeError("No pose estimation backend available!")
        
        self.backend = backend
        self.logger.info(f"Using {backend} backend for pose estimation")
        
        # Initialize the selected backend
        if backend == 'vitpose':
            self._init_vitpose()
        elif backend == 'mmpose':
            self._init_mmpose()
        elif backend == 'mediapipe':
            self._init_mediapipe()
        
        # BVH joint hierarchy
        self.joint_hierarchy = self._create_bvh_hierarchy()
        self.joint_names = self._get_joint_names()
    
    def _init_vitpose(self):
        """Initialize ViTPose with error handling"""
        try:
            # Download models if not present
            vitpose_model = self._download_vitpose_model()
            yolo_model = self._download_yolo_model()
            
            self.pose_model = VitInference(
                vitpose_model,
                yolo_model,
                model_name='s',  # Use small model for better compatibility
                yolo_size=320,
                is_video=True,
                device=self.device
            )
        except Exception as e:
            self.logger.error(f"Failed to initialize ViTPose: {e}")
            raise
    
    def _init_mmpose(self):
        """Initialize MMPose backend"""
        # Configuration for MMPose
        det_config = 'configs/faster_rcnn_r50_fpn_coco.py'
        det_checkpoint = 'https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
        
        pose_config = 'configs/td-hm_hrnet-w48_8xb32-210e_coco-256x192.py'
        pose_checkpoint = 'https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_coco_256x192-b9e0b3ab_20200708.pth'
        
        self.detector = init_detector(det_config, det_checkpoint, device=self.device)
        self.pose_model = init_model(pose_config, pose_checkpoint, device=self.device)
    
    def _init_mediapipe(self):
        """Initialize MediaPipe as fallback"""
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose_model = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
    
    def _download_vitpose_model(self):
        """Download ViTPose model if not present"""
        model_dir = Path("models")
        model_dir.mkdir(exist_ok=True)
        
        model_path = model_dir / "vitpose_small_coco.pth"
        
        if not model_path.exists():
            self.logger.info("Downloading ViTPose model...")
            # Use a smaller, more compatible model
            url = "https://github.com/ViTAE-Transformer/ViTPose/releases/download/v0.0.1/vitpose_small.pth"
            import urllib.request
            urllib.request.urlretrieve(url, model_path)
        
        return str(model_path)
    
    def _download_yolo_model(self):
        """Download YOLO model if not present"""
        model_dir = Path("models")
        model_dir.mkdir(exist_ok=True)
        
        model_path = model_dir / "yolov8s.pt"
        
        if not model_path.exists():
            self.logger.info("Downloading YOLO model...")
            # Download YOLOv8s
            from ultralytics import YOLO
            model = YOLO('yolov8s.pt')
            model.save(model_path)
        
        return str(model_path)
    
    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def _create_bvh_hierarchy(self) -> Dict:
        """Define BVH joint hierarchy"""
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
        """Get ordered list of joint names"""
        return ['Hips', 'Spine', 'Chest', 'Neck', 'Head',
                'LeftShoulder', 'LeftArm', 'LeftForeArm', 'LeftHand',
                'RightShoulder', 'RightArm', 'RightForeArm', 'RightHand',
                'LeftUpLeg', 'LeftLeg', 'LeftFoot',
                'RightUpLeg', 'RightLeg', 'RightFoot']
    
    def extract_frames_from_video(self, video_path: str, 
                                 max_frames: Optional[int] = None) -> List[np.ndarray]:
        """Extract frames from MP4 video"""
        frames = []
        
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise IOError(f"Cannot open video file: {video_path}")
        
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        self.logger.info(f"Video: {total_frames} frames at {self.fps:.2f} FPS")
        
        frame_count = 0
        
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                
                if not ret:
                    break
                
                if frame is None or frame.size == 0:
                    continue
                
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(rgb_frame)
                
                frame_count += 1
                
                if max_frames and frame_count >= max_frames:
                    break
                    
                if frame_count % 100 == 0:
                    progress = (frame_count / total_frames) * 100
                    self.logger.info(f"Extracted {frame_count} frames ({progress:.1f}%)")
        
        finally:
            cap.release()
        
        self.logger.info(f"Successfully extracted {len(frames)} frames")
        return frames
    
    def estimate_poses_from_frames(self, frames: List[np.ndarray]) -> List[Dict]:
        """Extract pose estimations using selected backend"""
        if self.backend == 'vitpose':
            return self._estimate_poses_vitpose(frames)
        elif self.backend == 'mmpose':
            return self._estimate_poses_mmpose(frames)
        elif self.backend == 'mediapipe':
            return self._estimate_poses_mediapipe(frames)
    
    def _estimate_poses_mediapipe(self, frames: List[np.ndarray]) -> List[Dict]:
        """Use MediaPipe for pose estimation"""
        pose_data = []
        
        for i, frame in enumerate(frames):
            try:
                # Process frame with MediaPipe
                results = self.pose_model.process(frame)
                
                frame_poses = {}
                
                if results.pose_landmarks:
                    # Convert MediaPipe landmarks to COCO format
                    keypoints = self._mediapipe_to_coco(results.pose_landmarks)
                    
                    frame_poses[0] = {
                        'keypoints': keypoints,
                        'confidence': np.mean(keypoints[:, 2])
                    }
                
                pose_data.append(frame_poses)
                
                if (i + 1) % 50 == 0:
                    self.logger.info(f"Processed {i + 1}/{len(frames)} frames")
                
            except Exception as e:
                self.logger.error(f"Error processing frame {i}: {str(e)}")
                pose_data.append({})
        
        return pose_data
    
    def _mediapipe_to_coco(self, landmarks):
        """Convert MediaPipe 33 landmarks to COCO 17 keypoints"""
        # Mapping from MediaPipe to COCO indices
        mp_to_coco = {
            0: 0,    # nose
            2: 1,    # left_eye
            5: 2,    # right_eye
            7: 3,    # left_ear
            8: 4,    # right_ear
            11: 5,   # left_shoulder
            12: 6,   # right_shoulder
            13: 7,   # left_elbow
            14: 8,   # right_elbow
            15: 9,   # left_wrist
            16: 10,  # right_wrist
            23: 11,  # left_hip
            24: 12,  # right_hip
            25: 13,  # left_knee
            26: 14,  # right_knee
            27: 15,  # left_ankle
            28: 16   # right_ankle
        }
        
        keypoints = np.zeros((17, 3))
        
        for coco_idx, mp_idx in mp_to_coco.items():
            landmark = landmarks.landmark[mp_idx]
            keypoints[coco_idx] = [landmark.x, landmark.y, landmark.visibility]
        
        return keypoints
    
    def _estimate_poses_vitpose(self, frames: List[np.ndarray]) -> List[Dict]:
        """Original ViTPose estimation method"""
        pose_data = []
        
        for i, frame in enumerate(frames):
            try:
                results = self.pose_model.inference(frame)
                
                frame_poses = {}
                
                if results:
                    for person_id, keypoints in results.items():
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
        
        if hasattr(self.pose_model, 'reset'):
            self.pose_model.reset()
        
        return pose_data
    
    def _estimate_poses_mmpose(self, frames: List[np.ndarray]) -> List[Dict]:
        """MMPose estimation method"""
        pose_data = []
        
        for i, frame in enumerate(frames):
            try:
                # Detect persons
                det_results = inference_detector(self.detector, frame)
                
                # Get person bounding boxes
                bboxes = det_results[0]  # person class
                
                # Run pose estimation
                pose_results = inference_topdown(self.pose_model, frame, bboxes)
                
                frame_poses = {}
                
                for idx, result in enumerate(pose_results):
                    keypoints = result['keypoints']
                    frame_poses[idx] = {
                        'keypoints': keypoints,
                        'confidence': np.mean(keypoints[:, 2])
                    }
                
                pose_data.append(frame_poses)
                
                if (i + 1) % 50 == 0:
                    self.logger.info(f"Processed {i + 1}/{len(frames)} frames")
                
            except Exception as e:
                self.logger.error(f"Error processing frame {i}: {str(e)}")
                pose_data.append({})
        
        return pose_data
    
    def calculate_joint_angles(self, keypoints: np.ndarray) -> Dict[str, List[float]]:
        """Convert 2D pose keypoints to joint angles"""
        # COCO keypoint indices
        NOSE = 0; L_EYE = 1; R_EYE = 2; L_EAR = 3; R_EAR = 4
        L_SHOULDER = 5; R_SHOULDER = 6; L_ELBOW = 7; R_ELBOW = 8
        L_WRIST = 9; R_WRIST = 10; L_HIP = 11; R_HIP = 12
        L_KNEE = 13; R_KNEE = 14; L_ANKLE = 15; R_ANKLE = 16
        
        joint_angles = {}
        
        # Normalize keypoints to [-1, 1] range for better angle calculation
        if keypoints[:, :2].max() > 1.0:
            # Assume pixel coordinates, normalize
            height, width = 1080, 1920  # Default assumption
            keypoints[:, 0] = (keypoints[:, 0] / width) * 2 - 1
            keypoints[:, 1] = (keypoints[:, 1] / height) * 2 - 1
        
        # Calculate hip center
        hip_center = (keypoints[L_HIP][:2] + keypoints[R_HIP][:2]) / 2
        
        def calculate_rotation_angles(parent_pos, child_pos):
            """Calculate rotation angles from parent to child position"""
            vector = child_pos - parent_pos
            
            # Prevent division by zero
            if np.linalg.norm(vector) < 1e-6:
                return [0.0, 0.0, 0.0]
            
            # Calculate angles with proper bounds checking
            angle_z = np.arctan2(vector[0], -vector[1])  # Y-up coordinate system
            angle_x = 0.0  # Simplified for 2D
            angle_y = 0.0  # Simplified for 2D
            
            return [
                np.degrees(angle_z),
                np.degrees(angle_x), 
                np.degrees(angle_y)
            ]
        
        # Root joint (Hips) - scale positions for reasonable BVH units
        joint_angles['Hips'] = [
            hip_center[0] * 50,  # X position (scaled)
            hip_center[1] * 50,  # Y position (scaled) 
            0.0,                 # Z position
            0.0, 0.0, 0.0        # Rotations
        ]
        
        # Spine and torso
        shoulder_center = (keypoints[L_SHOULDER][:2] + keypoints[R_SHOULDER][:2]) / 2
        joint_angles['Spine'] = calculate_rotation_angles(hip_center, shoulder_center)
        joint_angles['Chest'] = [0.0, 0.0, 0.0]
        
        # Head and neck
        head_pos = keypoints[NOSE][:2]
        joint_angles['Neck'] = calculate_rotation_angles(shoulder_center, head_pos)
        joint_angles['Head'] = [0.0, 0.0, 0.0]
        
        # Left arm
        joint_angles['LeftShoulder'] = [0.0, 0.0, 0.0]
        joint_angles['LeftArm'] = calculate_rotation_angles(
            keypoints[L_SHOULDER][:2], keypoints[L_ELBOW][:2]
        )
        joint_angles['LeftForeArm'] = calculate_rotation_angles(
            keypoints[L_ELBOW][:2], keypoints[L_WRIST][:2]
        )
        joint_angles['LeftHand'] = [0.0, 0.0, 0.0]
        
        # Right arm
        joint_angles['RightShoulder'] = [0.0, 0.0, 0.0]
        joint_angles['RightArm'] = calculate_rotation_angles(
            keypoints[R_SHOULDER][:2], keypoints[R_ELBOW][:2]
        )
        joint_angles['RightForeArm'] = calculate_rotation_angles(
            keypoints[R_ELBOW][:2], keypoints[R_WRIST][:2]
        )
        joint_angles['RightHand'] = [0.0, 0.0, 0.0]
        
        # Left leg
        joint_angles['LeftUpLeg'] = calculate_rotation_angles(
            keypoints[L_HIP][:2], keypoints[L_KNEE][:2]
        )
        joint_angles['LeftLeg'] = calculate_rotation_angles(
            keypoints[L_KNEE][:2], keypoints[L_ANKLE][:2]
        )
        joint_angles['LeftFoot'] = [0.0, 0.0, 0.0]
        
        # Right leg
        joint_angles['RightUpLeg'] = calculate_rotation_angles(
            keypoints[R_HIP][:2], keypoints[R_KNEE][:2]
        )
        joint_angles['RightLeg'] = calculate_rotation_angles(
            keypoints[R_KNEE][:2], keypoints[R_ANKLE][:2]
        )
        joint_angles['RightFoot'] = [0.0, 0.0, 0.0]
        
        return joint_angles
    
    def smooth_motion_data(self, motion_frames: List[Dict], window_size: int = 5) -> List[Dict]:
        """Apply temporal smoothing to reduce jitter"""
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
                    smoothed_frame[joint_name] = motion_frames[i].get(
                        joint_name, 
                        [0.0] * (6 if joint_name == 'Hips' else 3)
                    )
            
            smoothed_frames.append(smoothed_frame)
        
        return smoothed_frames
    
    def write_bvh_file(self, motion_data: List[Dict], output_path: str, fps: float = 30.0):
        """Write motion data to BVH file"""
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
                        if joint_name == 'Hips':
                            frame_values.extend([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
                        else:
                            frame_values.extend([0.0, 0.0, 0.0])
                
                f.write(" ".join(f"{val:.6f}" for val in frame_values) + "\n")
        
        self.logger.info(f"BVH file written: {output_path}")
    
    def _write_joint_hierarchy(self, f, joint_name: str, indent_level: int):
        """Recursively write joint hierarchy"""
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
                           fps: float = 30.0, max_frames: Optional[int] = None) -> bool:
        """Complete pipeline: MP4 video -> BVH motion capture file"""
        try:
            self.logger.info(f"Starting conversion: {video_path} -> {output_path}")
            
            # Step 1: Extract frames
            frames = self.extract_frames_from_video(video_path, max_frames)
            if not frames:
                self.logger.error("No frames extracted from video")
                return False
            
            # Step 2: Estimate poses
            pose_data = self.estimate_poses_from_frames(frames)
            
            # Step 3: Convert to joint angles
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
            
            # Step 4: Apply smoothing
            smoothed_motion = self.smooth_motion_data(motion_data)
            
            # Step 5: Write BVH
            self.write_bvh_file(smoothed_motion, output_path, fps)
            
            self.logger.info("Conversion completed successfully!")
            return True
            
        except Exception as e:
            self.logger.error(f"Conversion failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

# Main execution
def main():
    """Example usage with automatic backend selection"""
    # Create converter with automatic backend selection
    converter = MP4ToBVHConverter(
        backend='auto',  # Will try vitpose -> mmpose -> mediapipe
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    parser = argparse.ArgumentParser(description='Convert video to BVH using MediaPipe Pose')
    parser.add_argument('--video', required=True, help='Input video file')
    parser.add_argument('--output', required=True, help='Output BVH file')
    args = parser.parse_args()

    # Convert video to BVH
    success = converter.convert_video_to_bvh(
        video_path=args.video,
        output_path=args.output,
        fps=30.0,
        max_frames=300  # Limit for testing
    )
    
    if success:
        print("✅ Conversion successful! Import output_motion.bvh into Blender.")
        print("\nBlender Import Settings:")
        print("- File > Import > Motion Capture (.bvh)")
        print("- Scale: 0.01 (if the character appears too large)")
        print("- Forward: -Z Forward")
        print("- Up: Y Up")
    else:
        print("❌ Conversion failed. Check logs for details.")

if __name__ == "__main__":
    main()