"""
MediaPipe pose extraction module.
Handles video processing and landmark detection.
"""

import cv2
import mediapipe as mp
import numpy as np
from typing import List, Optional, Tuple
from dataclasses import dataclass
from tqdm import tqdm

from config import MEDIAPIPE_CONFIG, PROCESSING_CONFIG, QUALITY_THRESHOLDS

mp_pose = mp.solutions.pose
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands


@dataclass
class PoseFrame:
    """Container for pose data from a single frame."""
    frame_index: int
    timestamp: float
    landmarks: Optional[List]  # MediaPipe landmarks
    world_landmarks: Optional[List]  # World space landmarks
    left_hand_landmarks: Optional[List]  # Left hand landmarks (21 points)
    right_hand_landmarks: Optional[List]  # Right hand landmarks (21 points)
    detection_confidence: float

    def is_valid(self):
        """Check if frame has valid pose data."""
        return self.landmarks is not None and self.detection_confidence > 0.5


class MediaPipeExtractor:
    """Extracts pose and hand landmarks from video using MediaPipe Holistic."""

    def __init__(self, config=None, use_holistic=True):
        """Initialize the extractor with configuration.

        Args:
            config: MediaPipe configuration
            use_holistic: Whether to use Holistic model (True) or just Pose (False)
        """
        self.config = config or MEDIAPIPE_CONFIG
        self.use_holistic = use_holistic
        self.model = None
        self.sample_rate = PROCESSING_CONFIG['sample_rate']

    def __enter__(self):
        """Context manager entry."""
        if self.use_holistic:
            # Use Holistic model for pose + hands + face
            self.model = mp_holistic.Holistic(
                model_complexity=self.config.get('model_complexity', 1),
                min_detection_confidence=self.config.get('min_detection_confidence', 0.5),
                min_tracking_confidence=self.config.get('min_tracking_confidence', 0.5),
                enable_segmentation=False,
                smooth_segmentation=False,
                refine_face_landmarks=True
            )
        else:
            # Fallback to Pose-only model
            self.model = mp_pose.Pose(**self.config)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if self.model:
            self.model.close()
    
    def extract_from_video(self, video_path: str, preview: bool = False) -> List[PoseFrame]:
        """Extract pose landmarks from video file.
        
        Args:
            video_path: Path to input video file
            preview: Whether to show preview window
            
        Returns:
            List of PoseFrame objects
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"Video properties: {width}x{height}, {fps:.1f} FPS, {frame_count} frames")
        print(f"Sampling every {self.sample_rate} frames")
        
        # Setup preview if requested
        if preview:
            cv2.namedWindow('Pose Detection', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Pose Detection', 800, 600)
        
        # Process frames
        pose_frames = []
        frame_idx = 0
        
        with tqdm(total=frame_count, desc="Extracting poses") as pbar:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Sample frames according to sample rate
                if frame_idx % self.sample_rate == 0:
                    timestamp = frame_idx / fps
                    
                    # Process frame
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame_rgb.flags.writeable = False
                    results = self.model.process(frame_rgb)
                    frame_rgb.flags.writeable = True

                    # Create PoseFrame with pose and hand data
                    confidence = self._calculate_detection_confidence(results)

                    if self.use_holistic:
                        # Extract from Holistic results
                        pose_frame = PoseFrame(
                            frame_index=frame_idx,
                            timestamp=timestamp,
                            landmarks=results.pose_landmarks.landmark if results.pose_landmarks else None,
                            world_landmarks=results.pose_world_landmarks.landmark if results.pose_world_landmarks else None,
                            left_hand_landmarks=results.left_hand_landmarks.landmark if results.left_hand_landmarks else None,
                            right_hand_landmarks=results.right_hand_landmarks.landmark if results.right_hand_landmarks else None,
                            detection_confidence=confidence
                        )
                    else:
                        # Extract from Pose-only results
                        pose_frame = PoseFrame(
                            frame_index=frame_idx,
                            timestamp=timestamp,
                            landmarks=results.pose_landmarks.landmark if results.pose_landmarks else None,
                            world_landmarks=results.pose_world_landmarks.landmark if results.pose_world_landmarks else None,
                            left_hand_landmarks=None,
                            right_hand_landmarks=None,
                            detection_confidence=confidence
                        )
                    pose_frames.append(pose_frame)
                    
                    # Preview if requested
                    if preview and results.pose_landmarks:
                        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                        mp_drawing.draw_landmarks(
                            frame_bgr,
                            results.pose_landmarks,
                            mp_pose.POSE_CONNECTIONS,
                            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                            mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=1)
                        )
                        
                        # Add info text
                        cv2.putText(frame_bgr, f"Frame: {frame_idx}", (10, 30),
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        cv2.putText(frame_bgr, f"Confidence: {confidence:.2f}", (10, 60),
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        
                        cv2.imshow('Pose Detection', frame_bgr)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            print("Preview cancelled by user")
                            break
                
                frame_idx += 1
                pbar.update(1)
        
        # Cleanup
        cap.release()
        if preview:
            cv2.destroyAllWindows()
        
        print(f"Extracted {len(pose_frames)} pose frames")
        return pose_frames
    
    def _calculate_detection_confidence(self, results) -> float:
        """Calculate overall detection confidence from key landmarks.

        Args:
            results: MediaPipe detection results (Pose or Holistic)

        Returns:
            Confidence score between 0 and 1
        """
        # Check for pose landmarks first
        if hasattr(results, 'pose_world_landmarks'):
            if not results.pose_world_landmarks:
                return 0.0
        elif hasattr(results, 'pose_landmarks'):
            if not results.pose_landmarks:
                return 0.0
        else:
            return 0.0
        
        # Check key landmarks for visibility
        key_landmarks = [
            mp_pose.PoseLandmark.LEFT_SHOULDER,
            mp_pose.PoseLandmark.RIGHT_SHOULDER,
            mp_pose.PoseLandmark.LEFT_ELBOW,
            mp_pose.PoseLandmark.RIGHT_ELBOW,
            mp_pose.PoseLandmark.LEFT_HIP,
            mp_pose.PoseLandmark.RIGHT_HIP,
            mp_pose.PoseLandmark.LEFT_KNEE,
            mp_pose.PoseLandmark.RIGHT_KNEE
        ]
        
        visible_count = 0
        for idx in key_landmarks:
            landmark = results.pose_world_landmarks.landmark[idx]
            if landmark.visibility > 0.5:
                visible_count += 1
        
        return visible_count / len(key_landmarks)
    
    def find_reference_frame(self, pose_frames: List[PoseFrame], max_frames: int = 30) -> int:
        """Find the best frame to use as reference for skeleton setup.
        
        Args:
            pose_frames: List of extracted pose frames
            max_frames: Maximum number of frames to check
            
        Returns:
            Index of the best reference frame
        """
        best_idx = 0
        best_score = 0.0
        
        for i in range(min(len(pose_frames), max_frames)):
            if pose_frames[i].is_valid():
                score = pose_frames[i].detection_confidence
                if score > best_score:
                    best_score = score
                    best_idx = i
                    
                    # If we found a perfect frame, stop searching
                    if score >= 1.0:
                        break
        
        print(f"Using frame {best_idx} as reference (confidence: {best_score:.2f})")
        return best_idx
    
    def interpolate_missing_frames(self, pose_frames: List[PoseFrame]) -> List[PoseFrame]:
        """Interpolate missing or low-confidence frames.
        
        Args:
            pose_frames: List of pose frames with possible gaps
            
        Returns:
            List with interpolated frames
        """
        # For now, simple implementation - just mark invalid frames
        # Could be enhanced with actual interpolation
        interpolated = []
        
        for i, frame in enumerate(pose_frames):
            if frame.is_valid():
                interpolated.append(frame)
            else:
                # For invalid frames, try to use previous valid frame
                if i > 0 and interpolated[-1].is_valid():
                    # Copy previous frame's landmarks
                    frame.landmarks = interpolated[-1].landmarks
                    frame.world_landmarks = interpolated[-1].world_landmarks
                    frame.detection_confidence = 0.3  # Mark as interpolated
                interpolated.append(frame)
        
        return interpolated