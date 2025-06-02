import cv2
import mediapipe as mp
import numpy as np
import argparse
import os
from typing import List, Tuple, Dict
from dataclasses import dataclass, field

@dataclass
class BVHJoint:
    """Simple BVH joint representation"""
    name: str
    offset: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    channels: List[str] = field(default_factory=list)
    children: List['BVHJoint'] = field(default_factory=list)
    parent: 'BVHJoint' = None

class MediaPipeToBVH:
    def __init__(self):
        # Initialize MediaPipe Pose
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Define the skeleton structure for BVH
        self.joint_hierarchy = self._create_joint_hierarchy()
        
    def _create_joint_hierarchy(self) -> Dict:
        """Create a mapping between MediaPipe landmarks and BVH joint hierarchy"""
        # MediaPipe provides 33 landmarks, we'll map them to a standard BVH skeleton
        hierarchy = {
            'Hips': {
                'landmark_ids': [23, 24],  # Left and right hip
                'children': {
                    'Spine': {
                        'landmark_ids': [11, 12],  # Left and right shoulder
                        'children': {
                            'Chest': {
                                'landmark_ids': [11, 12],
                                'children': {
                                    'Neck': {
                                        'landmark_ids': [0],  # Nose as head reference
                                        'children': {
                                            'Head': {
                                                'landmark_ids': [0],
                                                'children': {}
                                            }
                                        }
                                    },
                                    'LeftShoulder': {
                                        'landmark_ids': [11],
                                        'children': {
                                            'LeftArm': {
                                                'landmark_ids': [13],
                                                'children': {
                                                    'LeftForeArm': {
                                                        'landmark_ids': [15],
                                                        'children': {
                                                            'LeftHand': {
                                                                'landmark_ids': [17, 19, 21],  # Wrist and fingers
                                                                'children': {}
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    },
                                    'RightShoulder': {
                                        'landmark_ids': [12],
                                        'children': {
                                            'RightArm': {
                                                'landmark_ids': [14],
                                                'children': {
                                                    'RightForeArm': {
                                                        'landmark_ids': [16],
                                                        'children': {
                                                            'RightHand': {
                                                                'landmark_ids': [18, 20, 22],  # Wrist and fingers
                                                                'children': {}
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    },
                    'LeftUpLeg': {
                        'landmark_ids': [23],
                        'children': {
                            'LeftLeg': {
                                'landmark_ids': [25],
                                'children': {
                                    'LeftFoot': {
                                        'landmark_ids': [27],
                                        'children': {
                                            'LeftToeBase': {
                                                'landmark_ids': [29, 31],
                                                'children': {}
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    },
                    'RightUpLeg': {
                        'landmark_ids': [24],
                        'children': {
                            'RightLeg': {
                                'landmark_ids': [26],
                                'children': {
                                    'RightFoot': {
                                        'landmark_ids': [28],
                                        'children': {
                                            'RightToeBase': {
                                                'landmark_ids': [30, 32],
                                                'children': {}
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        return hierarchy
    
    def _create_bvh_joints(self, hierarchy: Dict, parent: BVHJoint = None) -> BVHJoint:
        """Recursively create BVH joint structure"""
        for joint_name, joint_data in hierarchy.items():
            # Create joint
            joint = BVHJoint(name=joint_name)
            
            # Set parent
            if parent is not None:
                joint.parent = parent
                parent.children.append(joint)
            
            # Set channels - root has position and rotation, others just rotation
            if joint_name == 'Hips':
                joint.channels = ['Xposition', 'Yposition', 'Zposition', 'Xrotation', 'Yrotation', 'Zrotation']
            else:
                joint.channels = ['Xrotation', 'Yrotation', 'Zrotation']
            
            # Create children
            if 'children' in joint_data and joint_data['children']:
                for child_name, child_data in joint_data['children'].items():
                    child_hierarchy = {child_name: child_data}
                    self._create_bvh_joints(child_hierarchy, joint)
            
            if parent is None:  # This is the root joint
                return joint
    
    def _write_bvh_header(self, f, joint: BVHJoint, level: int = 0):
        """Write BVH hierarchy section"""
        indent = "  " * level
        
        if level == 0:
            f.write("HIERARCHY\n")
            f.write(f"ROOT {joint.name}\n")
        else:
            f.write(f"{indent}JOINT {joint.name}\n")
        
        f.write(f"{indent}{{\n")
        f.write(f"{indent}  OFFSET {joint.offset[0]:.6f} {joint.offset[1]:.6f} {joint.offset[2]:.6f}\n")
        
        if joint.channels:
            f.write(f"{indent}  CHANNELS {len(joint.channels)} ")
            f.write(" ".join(joint.channels) + "\n")
        
        # Write children
        for child in joint.children:
            self._write_bvh_header(f, child, level + 1)
        
        # End Site for leaf joints
        if not joint.children:
            f.write(f"{indent}  End Site\n")
            f.write(f"{indent}  {{\n")
            f.write(f"{indent}    OFFSET 0.000000 -10.000000 0.000000\n")
            f.write(f"{indent}  }}\n")
        
        f.write(f"{indent}}}\n")
    
    def _landmark_to_position(self, landmark, image_width: int, image_height: int, scale: float = 100.0) -> List[float]:
        """Convert MediaPipe landmark to 3D position"""
        # MediaPipe provides normalized coordinates (0-1), we need to scale them
        # Also convert from image space to 3D world space
        x = (landmark.x - 0.5) * scale
        y = (landmark.y - 0.5) * -scale  # Flip Y for correct orientation
        z = landmark.z * scale
        return [x, y, z]
    
    def _calculate_joint_position(self, landmarks, landmark_ids: List[int], image_width: int, image_height: int) -> List[float]:
        """Calculate average position for joints that use multiple landmarks"""
        positions = []
        for lid in landmark_ids:
            if lid < len(landmarks.landmark):
                pos = self._landmark_to_position(landmarks.landmark[lid], image_width, image_height)
                positions.append(pos)
        
        if positions:
            # Return average position
            return [sum(p[i] for p in positions) / len(positions) for i in range(3)]
        return [0.0, 0.0, 0.0]
    
    def _calculate_rotation(self, parent_pos: List[float], child_pos: List[float]) -> List[float]:
        """Calculate rotation angles between parent and child joint"""
        # Calculate direction vector
        direction = np.array(child_pos) - np.array(parent_pos)
        
        # Normalize
        length = np.linalg.norm(direction)
        if length > 0:
            direction = direction / length
        
        # Calculate Euler angles (simplified approach)
        # In a production system, you'd want more sophisticated rotation calculation
        x_rot = np.arctan2(direction[1], direction[2]) * 180 / np.pi
        y_rot = np.arctan2(-direction[0], np.sqrt(direction[1]**2 + direction[2]**2)) * 180 / np.pi
        z_rot = 0.0  # Simplified - no roll
        
        return [x_rot, y_rot, z_rot]
    
    def _extract_frame_data(self, landmarks, hierarchy: Dict, image_width: int, image_height: int, 
                           parent_pos: List[float] = None) -> Dict[str, List[float]]:
        """Extract position and rotation data for all joints in a frame"""
        frame_data = {}
        
        for joint_name, joint_data in hierarchy.items():
            # Get position for this joint
            pos = self._calculate_joint_position(landmarks, joint_data['landmark_ids'], image_width, image_height)
            
            # For root joint (Hips), store position
            if joint_name == 'Hips':
                frame_data[f'{joint_name}_Xposition'] = pos[0]
                frame_data[f'{joint_name}_Yposition'] = pos[1]
                frame_data[f'{joint_name}_Zposition'] = pos[2]
                rotation = [0.0, 0.0, 0.0]  # Default rotation for root
            else:
                # Calculate rotation relative to parent
                if parent_pos is not None:
                    rotation = self._calculate_rotation(parent_pos, pos)
                else:
                    rotation = [0.0, 0.0, 0.0]
            
            # Store rotation
            frame_data[f'{joint_name}_Xrotation'] = rotation[0]
            frame_data[f'{joint_name}_Yrotation'] = rotation[1]
            frame_data[f'{joint_name}_Zrotation'] = rotation[2]
            
            # Process children
            if 'children' in joint_data and joint_data['children']:
                child_data = self._extract_frame_data(landmarks, joint_data['children'], 
                                                     image_width, image_height, pos)
                frame_data.update(child_data)
        
        return frame_data
    
    def _collect_motion_values(self, joint: BVHJoint, frame_data: Dict[str, float], values: List[float]):
        """Collect motion values in the correct order for BVH"""
        for channel in joint.channels:
            key = f"{joint.name}_{channel}"
            if key in frame_data:
                values.append(frame_data[key])
            else:
                values.append(0.0)
        
        # Process children
        for child in joint.children:
            self._collect_motion_values(child, frame_data, values)
    
    def process_video(self, video_path: str, output_path: str, skip_frames: int = 1):
        """Process video and generate BVH file"""
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Unable to open video file: {video_path}")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"Video info: {width}x{height}, {fps} FPS, {frame_count} frames")
        
        # Create BVH structure
        root_joint = self._create_bvh_joints(self.joint_hierarchy)
        
        # Process frames
        motion_data = []
        frame_idx = 0
        processed_frames = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Skip frames if specified
            if frame_idx % skip_frames != 0:
                frame_idx += 1
                continue
            
            # Convert to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process with MediaPipe
            results = self.pose.process(rgb_frame)
            
            if results.pose_landmarks:
                # Extract motion data for this frame
                frame_data = self._extract_frame_data(results.pose_landmarks, 
                                                     self.joint_hierarchy, 
                                                     width, height)
                
                # Convert to list in correct order
                motion_values = []
                self._collect_motion_values(root_joint, frame_data, motion_values)
                motion_data.append(motion_values)
                
                processed_frames += 1
                print(f"Processed frame {frame_idx}/{frame_count}", end='\r')
            
            frame_idx += 1
        
        cap.release()
        print(f"\nProcessed {processed_frames} frames with pose data")
        
        # Write BVH file
        with open(output_path, 'w') as f:
            # Write hierarchy
            self._write_bvh_header(f, root_joint)
            
            # Write motion section
            f.write("MOTION\n")
            f.write(f"Frames: {processed_frames}\n")
            f.write(f"Frame Time: {skip_frames / fps:.6f}\n")
            
            # Write motion data
            for frame_values in motion_data:
                f.write(" ".join(f"{v:.6f}" for v in frame_values) + "\n")
        
        print(f"BVH file saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Convert video to BVH using MediaPipe pose detection')
    parser.add_argument('--video', help='Path to input video file')
    parser.add_argument('--output', help='Path to output BVH file')
    parser.add_argument('--skip-frames', type=int, default=1, 
                       help='Process every N frames (default: 1)')
    
    args = parser.parse_args()
    
    # Create converter
    converter = MediaPipeToBVH()
    
    # Process video
    try:
        converter.process_video(args.video, args.output, args.skip_frames)
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())