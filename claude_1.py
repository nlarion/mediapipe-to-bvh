import cv2
import mediapipe as mp
import numpy as np
import argparse
from typing import List, Tuple, Dict
import math
import os

class MediaPipeToBVH:
    def __init__(self, visualize=False):
        # Initialize MediaPipe
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        self.visualize = visualize
        
        # Define joint hierarchy for BVH
        # Map MediaPipe landmarks to BVH joints
        self.joint_hierarchy = {
            'Hips': {
                'parent': None,
                'offset': [0, 0, 0],
                'channels': ['Xposition', 'Yposition', 'Zposition', 'Zrotation', 'Xrotation', 'Yrotation'],
                'mp_indices': [23, 24]  # Left and right hip
            },
            'Spine': {
                'parent': 'Hips',
                'offset': [0, 10, 0],
                'channels': ['Zrotation', 'Xrotation', 'Yrotation'],
                'mp_indices': [11, 12]  # Left and right shoulder
            },
            'Neck': {
                'parent': 'Spine',
                'offset': [0, 25, 0],
                'channels': ['Zrotation', 'Xrotation', 'Yrotation'],
                'mp_indices': [11, 12]
            },
            'Head': {
                'parent': 'Neck',
                'offset': [0, 15, 0],
                'channels': ['Zrotation', 'Xrotation', 'Yrotation'],
                'mp_indices': [0]  # Nose
            },
            'LeftShoulder': {
                'parent': 'Spine',
                'offset': [-5, 20, 0],
                'channels': ['Zrotation', 'Xrotation', 'Yrotation'],
                'mp_indices': [11]
            },
            'LeftArm': {
                'parent': 'LeftShoulder',
                'offset': [-15, 0, 0],
                'channels': ['Zrotation', 'Xrotation', 'Yrotation'],
                'mp_indices': [13]  # Left elbow
            },
            'LeftForeArm': {
                'parent': 'LeftArm',
                'offset': [-25, 0, 0],
                'channels': ['Zrotation', 'Xrotation', 'Yrotation'],
                'mp_indices': [15]  # Left wrist
            },
            'LeftHand': {
                'parent': 'LeftForeArm',
                'offset': [-7, 0, 0],
                'channels': ['Zrotation', 'Xrotation', 'Yrotation'],
                'mp_indices': [19, 17, 15]  # Pinky, index, wrist
            },
            'RightShoulder': {
                'parent': 'Spine',
                'offset': [5, 20, 0],
                'channels': ['Zrotation', 'Xrotation', 'Yrotation'],
                'mp_indices': [12]
            },
            'RightArm': {
                'parent': 'RightShoulder',
                'offset': [15, 0, 0],
                'channels': ['Zrotation', 'Xrotation', 'Yrotation'],
                'mp_indices': [14]  # Right elbow
            },
            'RightForeArm': {
                'parent': 'RightArm',
                'offset': [25, 0, 0],
                'channels': ['Zrotation', 'Xrotation', 'Yrotation'],
                'mp_indices': [16]  # Right wrist
            },
            'RightHand': {
                'parent': 'RightForeArm',
                'offset': [7, 0, 0],
                'channels': ['Zrotation', 'Xrotation', 'Yrotation'],
                'mp_indices': [20, 18, 16]  # Pinky, index, wrist
            },
            'LeftUpLeg': {
                'parent': 'Hips',
                'offset': [-5, 0, 0],
                'channels': ['Zrotation', 'Xrotation', 'Yrotation'],
                'mp_indices': [23]  # Left hip
            },
            'LeftLeg': {
                'parent': 'LeftUpLeg',
                'offset': [0, -40, 0],
                'channels': ['Zrotation', 'Xrotation', 'Yrotation'],
                'mp_indices': [25]  # Left knee
            },
            'LeftFoot': {
                'parent': 'LeftLeg',
                'offset': [0, -40, 0],
                'channels': ['Zrotation', 'Xrotation', 'Yrotation'],
                'mp_indices': [27, 29, 31]  # Ankle, heel, foot index
            },
            'RightUpLeg': {
                'parent': 'Hips',
                'offset': [5, 0, 0],
                'channels': ['Zrotation', 'Xrotation', 'Yrotation'],
                'mp_indices': [24]  # Right hip
            },
            'RightLeg': {
                'parent': 'RightUpLeg',
                'offset': [0, -40, 0],
                'channels': ['Zrotation', 'Xrotation', 'Yrotation'],
                'mp_indices': [26]  # Right knee
            },
            'RightFoot': {
                'parent': 'RightLeg',
                'offset': [0, -40, 0],
                'channels': ['Zrotation', 'Xrotation', 'Yrotation'],
                'mp_indices': [28, 30, 32]  # Ankle, heel, foot index
            }
        }
        
        self.frames_data = []
        
    def process_video(self, video_path: str, output_path: str):
        """Process video and generate BVH file"""
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = 0
        
        # Get video properties for visualization
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Create video writer if visualization is enabled
        if self.visualize:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out_video_path = output_path.replace('.bvh', '_pose.mp4')
            out = cv2.VideoWriter(out_video_path, fourcc, fps, (width, height))
        
        print(f"Processing video: {video_path}")
        print(f"FPS: {fps}")
        print(f"Resolution: {width}x{height}")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process the frame
            results = self.pose.process(rgb_frame)
            
            if results.pose_world_landmarks:
                # Extract frame data
                frame_data = self.extract_frame_data(results.pose_world_landmarks)
                self.frames_data.append(frame_data)
                frame_count += 1
                
                # Visualize if enabled
                if self.visualize:
                    # Draw pose landmarks
                    self.mp_drawing.draw_landmarks(
                        frame,
                        results.pose_landmarks,
                        self.mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
                    )
                    
                    # Add frame info
                    cv2.putText(frame, f'Frame: {frame_count}', (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    
                    # Write frame
                    out.write(frame)
                
                if frame_count % 30 == 0:
                    print(f"Processed {frame_count} frames...")
            else:
                print(f"Warning: No pose detected in frame {frame_count}")
        
        cap.release()
        if self.visualize:
            out.release()
            print(f"Visualization saved to: {out_video_path}")
        
        print(f"Total frames processed: {frame_count}")
        print(f"Frames with pose data: {len(self.frames_data)}")
        
        # Generate BVH file
        if self.frames_data:
            self.write_bvh(output_path, fps)
            print(f"BVH file saved to: {output_path}")
        else:
            print("No pose data detected in video!")
    
    def extract_frame_data(self, landmarks) -> Dict:
        """Extract pose data for a single frame"""
        frame_data = {}
        
        # Get landmark positions
        landmark_positions = []
        for landmark in landmarks.landmark:
            landmark_positions.append([landmark.x, landmark.y, landmark.z])
        
        # Calculate joint positions and rotations
        for joint_name, joint_info in self.joint_hierarchy.items():
            if joint_name == 'Hips':
                # For root joint, calculate position
                hip_indices = joint_info['mp_indices']
                hip_pos = np.mean([landmark_positions[i] for i in hip_indices], axis=0)
                # Scale to reasonable units (multiply by 100 for cm)
                frame_data[joint_name] = {
                    'position': hip_pos * 100,
                    'rotation': [0, 0, 0]  # Will be calculated based on orientation
                }
            else:
                # For other joints, calculate rotation
                rotation = self.calculate_joint_rotation(joint_name, landmark_positions)
                frame_data[joint_name] = {
                    'rotation': rotation
                }
        
        return frame_data
    
    def calculate_joint_rotation(self, joint_name: str, landmarks: List) -> List[float]:
        """Calculate joint rotation from landmarks"""
        joint_info = self.joint_hierarchy[joint_name]
        parent_name = joint_info['parent']
        
        if parent_name is None:
            # Root joint - calculate orientation based on hips
            left_hip = np.array(landmarks[23])
            right_hip = np.array(landmarks[24])
            
            # Calculate hip orientation
            hip_vector = right_hip - left_hip
            hip_vector = hip_vector / np.linalg.norm(hip_vector)
            
            # Calculate rotation to align with world coordinates
            rotation = self.vector_to_euler(hip_vector, np.array([1, 0, 0]))
            return rotation
        
        # Get joint positions
        if len(joint_info['mp_indices']) == 1:
            joint_pos = np.array(landmarks[joint_info['mp_indices'][0]])
        else:
            # Average multiple landmarks
            joint_pos = np.mean([landmarks[i] for i in joint_info['mp_indices']], axis=0)
        
        # Get parent position
        parent_info = self.joint_hierarchy[parent_name]
        if len(parent_info['mp_indices']) == 1:
            parent_pos = np.array(landmarks[parent_info['mp_indices'][0]])
        else:
            parent_pos = np.mean([landmarks[i] for i in parent_info['mp_indices']], axis=0)
        
        # Calculate bone vector
        bone_vector = joint_pos - parent_pos
        if np.linalg.norm(bone_vector) > 0:
            bone_vector = bone_vector / np.linalg.norm(bone_vector)
        else:
            return [0, 0, 0]
        
        # Calculate rotation based on bone orientation
        # This is still simplified - a full implementation would need
        # to consider the entire kinematic chain
        reference_vector = np.array([0, -1, 0])  # Default bone direction
        rotation = self.vector_to_euler(bone_vector, reference_vector)
        
        return rotation
    
    def vector_to_euler(self, vector: np.ndarray, reference: np.ndarray) -> List[float]:
        """Convert a vector to Euler angles relative to reference"""
        # Normalize vectors
        vector = vector / np.linalg.norm(vector)
        reference = reference / np.linalg.norm(reference)
        
        # Calculate rotation axis and angle
        rotation_axis = np.cross(reference, vector)
        if np.linalg.norm(rotation_axis) > 0:
            rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
            angle = np.arccos(np.clip(np.dot(reference, vector), -1.0, 1.0))
            
            # Convert to Euler angles (simplified)
            # This is a basic conversion - proper implementation would use
            # rotation matrices or quaternions
            euler_x = angle * rotation_axis[0]
            euler_y = angle * rotation_axis[1]
            euler_z = angle * rotation_axis[2]
            
            return [math.degrees(euler_x), math.degrees(euler_y), math.degrees(euler_z)]
        else:
            return [0, 0, 0]
    
    def write_bvh(self, output_path: str, fps: float):
        """Write BVH file"""
        with open(output_path, 'w') as f:
            # Write header
            f.write("HIERARCHY\n")
            self._write_joint_hierarchy(f, 'Hips', 0)
            
            # Write motion data
            f.write(f"MOTION\n")
            f.write(f"Frames: {len(self.frames_data)}\n")
            f.write(f"Frame Time: {1.0/fps:.6f}\n")
            
            # Write frame data
            for frame_data in self.frames_data:
                frame_values = []
                
                # Add root position and rotation
                if 'Hips' in frame_data:
                    pos = frame_data['Hips']['position']
                    rot = frame_data['Hips']['rotation']
                    frame_values.extend([pos[0], pos[1], pos[2]])
                    frame_values.extend([rot[2], rot[0], rot[1]])
                
                # Add other joint rotations
                for joint_name in self.joint_hierarchy:
                    if joint_name != 'Hips' and joint_name in frame_data:
                        rot = frame_data[joint_name]['rotation']
                        frame_values.extend([rot[2], rot[0], rot[1]])
                
                # Write frame line
                f.write(' '.join([f"{v:.6f}" for v in frame_values]) + '\n')
    
    def _write_joint_hierarchy(self, f, joint_name: str, indent: int):
        """Recursively write joint hierarchy"""
        indent_str = '  ' * indent
        joint_info = self.joint_hierarchy[joint_name]
        
        if joint_info['parent'] is None:
            f.write(f"{indent_str}ROOT {joint_name}\n")
        else:
            f.write(f"{indent_str}JOINT {joint_name}\n")
        
        f.write(f"{indent_str}{{\n")
        
        # Write offset
        offset = joint_info['offset']
        f.write(f"{indent_str}  OFFSET {offset[0]:.6f} {offset[1]:.6f} {offset[2]:.6f}\n")
        
        # Write channels
        channels = joint_info['channels']
        f.write(f"{indent_str}  CHANNELS {len(channels)} {' '.join(channels)}\n")
        
        # Write children
        children = [j for j, info in self.joint_hierarchy.items() 
                   if info['parent'] == joint_name]
        
        if children:
            for child in children:
                self._write_joint_hierarchy(f, child, indent + 1)
        else:
            # End effector
            f.write(f"{indent_str}  End Site\n")
            f.write(f"{indent_str}  {{\n")
            f.write(f"{indent_str}    OFFSET 0.0 -10.0 0.0\n")
            f.write(f"{indent_str}  }}\n")
        
        f.write(f"{indent_str}}}\n")

def main():
    parser = argparse.ArgumentParser(description='Convert video to BVH using MediaPipe')
    parser.add_argument('--video', help='Path to input video file')
    parser.add_argument('--output', help='Path to output BVH file')
    parser.add_argument('--visualize', '-v', action='store_true',
                       help='Generate visualization video with pose overlay')
    
    args = parser.parse_args()
    
    # Validate input
    if not os.path.exists(args.video):
        print(f"Error: Input video '{args.video}' not found!")
        return
    
    # Create converter and process video
    converter = MediaPipeToBVH(visualize=args.visualize)
    converter.process_video(args.video, args.output)

if __name__ == "__main__":
    main()

"""
USAGE EXAMPLES:

1. Basic conversion:
   python video_to_bvh.py input_video.mp4 output_motion.bvh

2. With visualization:
   python video_to_bvh.py input_video.mp4 output_motion.bvh --visualize

REQUIREMENTS:
- pip install opencv-python mediapipe numpy

NOTES:
- The script uses MediaPipe's 33-point pose model
- BVH joint hierarchy is mapped from MediaPipe landmarks
- Rotation calculations are simplified and may need refinement for production use
- The generated BVH file can be imported into 3D software like Blender, Maya, etc.

LIMITATIONS:
- Single person detection only (MediaPipe limitation)
- Simplified rotation calculations (full inverse kinematics would be more accurate)
- Fixed skeleton proportions (could be improved with calibration)
- No hand/finger articulation (would require MediaPipe Hands)

IMPROVEMENTS TO CONSIDER:
1. Add smoothing filters to reduce jitter
2. Implement proper inverse kinematics for accurate joint rotations
3. Add support for multiple people
4. Include hand and face tracking
5. Calibrate skeleton proportions based on detected person
6. Add support for batch processing multiple videos
"""