#!/usr/bin/env python3
"""
Robust MediaPipe to BVH converter with comprehensive error handling
"""

import cv2
import mediapipe as mp
import numpy as np
import argparse
import os
import sys
from datetime import datetime

class RobustBVHConverter:
    def __init__(self, default_fps=30.0):
        self.default_fps = default_fps
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
    def process_video(self, video_path, output_path=None, show_preview=False):
        """Process video with comprehensive error handling"""
        
        # Validate input
        if not os.path.exists(video_path):
            print(f"Error: Video file '{video_path}' not found")
            return False
        
        # Generate output path if not provided
        if output_path is None:
            base_name = os.path.splitext(os.path.basename(video_path))[0]
            output_path = f"{base_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.bvh"
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video file '{video_path}'")
            print("This might be due to:")
            print("  - Unsupported video codec")
            print("  - Corrupted video file")
            print("  - Missing codec libraries")
            return False
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Handle invalid FPS
        if fps <= 0 or fps > 1000:  # Sanity check
            print(f"Warning: Invalid FPS value ({fps}), using default {self.default_fps} FPS")
            fps = self.default_fps
        
        print(f"Video Info:")
        print(f"  File: {video_path}")
        print(f"  Resolution: {width}x{height}")
        print(f"  FPS: {fps}")
        print(f"  Total frames: {total_frames}")
        print(f"  Duration: {total_frames/fps:.2f} seconds")
        
        # Process frames
        frames = []
        frame_count = 0
        failed_frames = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # Convert and process
                try:
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = self.pose.process(rgb_frame)
                    
                    if results.pose_landmarks:
                        # Extract BVH data
                        frame_data = self.extract_frame_data(results.pose_landmarks)
                        frames.append(frame_data)
                        
                        # Show preview if requested
                        if show_preview:
                            self.mp_drawing.draw_landmarks(
                                frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
                            cv2.imshow('MediaPipe Pose', frame)
                            if cv2.waitKey(1) & 0xFF == ord('q'):
                                print("\nPreview stopped by user")
                                break
                    else:
                        failed_frames += 1
                        # Use last valid frame or neutral pose
                        if frames:
                            frames.append(frames[-1])  # Repeat last frame
                        else:
                            frames.append(self.get_neutral_pose())
                    
                except Exception as e:
                    print(f"Warning: Error processing frame {frame_count}: {e}")
                    failed_frames += 1
                    if frames:
                        frames.append(frames[-1])
                    else:
                        frames.append(self.get_neutral_pose())
                
                # Progress update
                if frame_count % 30 == 0 or frame_count == total_frames:
                    progress = (frame_count / total_frames * 100) if total_frames > 0 else 0
                    print(f"Progress: {frame_count}/{total_frames} frames ({progress:.1f}%)")
                    
        except KeyboardInterrupt:
            print("\nProcessing interrupted by user")
        except Exception as e:
            print(f"Error during processing: {e}")
        finally:
            cap.release()
            if show_preview:
                cv2.destroyAllWindows()
            self.pose.close()
        
        # Check results
        if not frames:
            print("Error: No valid frames were processed")
            return False
        
        print(f"\nProcessing complete:")
        print(f"  Processed frames: {frame_count}")
        print(f"  Valid frames: {len(frames)}")
        print(f"  Failed frames: {failed_frames}")
        
        # Write BVH
        try:
            self.write_bvh(output_path, frames, fps)
            print(f"\nBVH file saved: {output_path}")
            return True
        except Exception as e:
            print(f"Error writing BVH file: {e}")
            return False
    
    def extract_frame_data(self, landmarks):
        """Extract BVH frame data from MediaPipe landmarks"""
        lm = landmarks.landmark
        
        # Calculate key positions
        left_hip = np.array([lm[23].x, lm[23].y, lm[23].z])
        right_hip = np.array([lm[24].x, lm[24].y, lm[24].z])
        left_shoulder = np.array([lm[11].x, lm[11].y, lm[11].z])
        right_shoulder = np.array([lm[12].x, lm[12].y, lm[12].z])
        
        hip_center = (left_hip + right_hip) / 2
        shoulder_center = (left_shoulder + right_shoulder) / 2
        
        # Convert to BVH coordinate system
        hip_pos = [
            (hip_center[0] - 0.5) * 100,
            (0.5 - hip_center[1]) * 100,
            hip_center[2] * 50
        ]
        
        # Calculate rotations (simplified for robustness)
        frame_data = []
        
        # Hip position and rotation
        frame_data.extend(hip_pos)
        frame_data.extend([0, 0, 0])  # Hip rotation
        
        # Spine chain (neutral for now)
        for _ in range(4):  # Spine, Spine1, Neck, Head
            frame_data.extend([0, 0, 0])
        
        # Arms
        for side in [11, 12]:  # Left and right shoulders
            shoulder = np.array([lm[side].x, lm[side].y, lm[side].z])
            elbow = np.array([lm[side+2].x, lm[side+2].y, lm[side+2].z])
            wrist = np.array([lm[side+4].x, lm[side+4].y, lm[side+4].z])
            
            # Simple angle calculations
            shoulder_angle = self.calculate_simple_angle(shoulder_center, shoulder, elbow)
            elbow_angle = self.calculate_simple_angle(shoulder, elbow, wrist)
            
            frame_data.extend([0, shoulder_angle, 0])  # Shoulder
            frame_data.extend([0, elbow_angle, 0])     # Elbow
            frame_data.extend([0, 0, 0])               # Forearm
            frame_data.extend([0, 0, 0])               # Hand
        
        # Legs
        for side in [23, 24]:  # Left and right hips
            hip = np.array([lm[side].x, lm[side].y, lm[side].z])
            knee = np.array([lm[side+2].x, lm[side+2].y, lm[side+2].z])
            ankle = np.array([lm[side+4].x, lm[side+4].y, lm[side+4].z])
            
            # Calculate angles
            hip_angle_x = np.degrees(np.arctan2(knee[1] - hip[1], 
                                               np.sqrt((knee[0]-hip[0])**2 + (knee[2]-hip[2])**2)))
            hip_angle_y = np.degrees(np.arctan2(knee[0] - hip[0], knee[2] - hip[2]))
            knee_angle = self.calculate_simple_angle(hip, knee, ankle)
            
            frame_data.extend([0, -hip_angle_x, hip_angle_y])  # Hip
            frame_data.extend([-knee_angle, 0, 0])             # Knee
            frame_data.extend([0, 0, 0])                       # Foot
        
        return frame_data
    
    def calculate_simple_angle(self, p1, p2, p3):
        """Calculate angle at p2 between p1-p2 and p2-p3"""
        v1 = p1 - p2
        v2 = p3 - p2
        
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        angle = np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))
        
        # Return complementary angle for more natural joint representation
        return 180 - angle
    
    def get_neutral_pose(self):
        """Return neutral T-pose frame data"""
        return [
            0, 0, 0,  # Hip position
            0, 0, 0,  # Hip rotation
            0, 0, 0,  # Spine
            0, 0, 0,  # Spine1
            0, 0, 0,  # Neck
            0, 0, 0,  # Head
            0, -15, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  # Left arm
            0, 15, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,   # Right arm
            0, 0, 0, 0, 0, 0, 0, 0, 0,             # Left leg
            0, 0, 0, 0, 0, 0, 0, 0, 0              # Right leg
        ]
    
    def write_bvh(self, filename, frames, fps):
        """Write BVH file"""
        with open(filename, 'w') as f:
            # Standard BVH header
            f.write("""HIERARCHY
ROOT Hips
{
  OFFSET 0.000000 0.000000 0.000000
  CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation
  JOINT Spine
  {
    OFFSET 0.000000 10.000000 0.000000
    CHANNELS 3 Zrotation Xrotation Yrotation
    JOINT Spine1
    {
      OFFSET 0.000000 10.000000 0.000000
      CHANNELS 3 Zrotation Xrotation Yrotation
      JOINT Neck
      {
        OFFSET 0.000000 10.000000 0.000000
        CHANNELS 3 Zrotation Xrotation Yrotation
        JOINT Head
        {
          OFFSET 0.000000 10.000000 0.000000
          CHANNELS 3 Zrotation Xrotation Yrotation
          End Site
          {
            OFFSET 0.0 -5.0 0.0
          }
        }
      }
      JOINT LeftShoulder
      {
        OFFSET -5.000000 0.000000 0.000000
        CHANNELS 3 Zrotation Xrotation Yrotation
        JOINT LeftArm
        {
          OFFSET -10.000000 0.000000 0.000000
          CHANNELS 3 Zrotation Xrotation Yrotation
          JOINT LeftForeArm
          {
            OFFSET -10.000000 0.000000 0.000000
            CHANNELS 3 Zrotation Xrotation Yrotation
            JOINT LeftHand
            {
              OFFSET -5.000000 0.000000 0.000000
              CHANNELS 3 Zrotation Xrotation Yrotation
              End Site
              {
                OFFSET 0.0 -5.0 0.0
              }
            }
          }
        }
      }
      JOINT RightShoulder
      {
        OFFSET 5.000000 0.000000 0.000000
        CHANNELS 3 Zrotation Xrotation Yrotation
        JOINT RightArm
        {
          OFFSET 10.000000 0.000000 0.000000
          CHANNELS 3 Zrotation Xrotation Yrotation
          JOINT RightForeArm
          {
            OFFSET 10.000000 0.000000 0.000000
            CHANNELS 3 Zrotation Xrotation Yrotation
            JOINT RightHand
            {
              OFFSET 5.000000 0.000000 0.000000
              CHANNELS 3 Zrotation Xrotation Yrotation
              End Site
              {
                OFFSET 0.0 -5.0 0.0
              }
            }
          }
        }
      }
    }
  }
  JOINT LeftUpLeg
  {
    OFFSET -5.000000 -5.000000 0.000000
    CHANNELS 3 Zrotation Xrotation Yrotation
    JOINT LeftLeg
    {
      OFFSET 0.000000 -20.000000 0.000000
      CHANNELS 3 Zrotation Xrotation Yrotation
      JOINT LeftFoot
      {
        OFFSET 0.000000 -20.000000 0.000000
        CHANNELS 3 Zrotation Xrotation Yrotation
        End Site
        {
          OFFSET 0.0 -5.0 0.0
        }
      }
    }
  }
  JOINT RightUpLeg
  {
    OFFSET 5.000000 -5.000000 0.000000
    CHANNELS 3 Zrotation Xrotation Yrotation
    JOINT RightLeg
    {
      OFFSET 0.000000 -20.000000 0.000000
      CHANNELS 3 Zrotation Xrotation Yrotation
      JOINT RightFoot
      {
        OFFSET 0.000000 -20.000000 0.000000
        CHANNELS 3 Zrotation Xrotation Yrotation
        End Site
        {
          OFFSET 0.0 -5.0 0.0
        }
      }
    }
  }
}
MOTION
""")
            f.write(f"Frames: {len(frames)}\n")
            f.write(f"Frame Time: {1.0/fps:.8f}\n")
            
            for frame in frames:
                f.write(" ".join([f"{v:.6f}" for v in frame]) + "\n")

def main():
    parser = argparse.ArgumentParser(description='Robust MediaPipe to BVH converter')
    parser.add_argument('--video', help='Input video file')
    parser.add_argument('-o', '--output', help='Output BVH file path')
    parser.add_argument('-p', '--preview', action='store_true', help='Show preview window')
    parser.add_argument('--fps', type=float, default=30.0, help='Default FPS if cannot be read from video')
    
    args = parser.parse_args()
    
    converter = RobustBVHConverter(default_fps=args.fps)
    success = converter.process_video(args.video, args.output, args.preview)
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()