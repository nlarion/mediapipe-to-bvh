"""
Skeleton mapping module for MediaPipe to BVH conversion.
Maps MediaPipe's 33 landmarks to hierarchical BVH skeleton.
"""

import numpy as np
import mediapipe as mp

mp_pose = mp.solutions.pose


class BVHJoint:
    """Represents a joint in the BVH skeleton hierarchy."""
    
    def __init__(self, name, parent=None):
        self.name = name
        self.parent = parent
        self.children = []
        self.offset = np.array([0.0, 0.0, 0.0])
        self.channels = []
        self.positions = []  # For root joint
        self.rotations = []  # For all joints
        
        if parent:
            parent.add_child(self)
    
    def add_child(self, child):
        """Add a child joint to this joint."""
        self.children.append(child)
        child.parent = self
    
    def set_offset(self, offset):
        """Set the offset from parent joint."""
        self.offset = np.array(offset)
    
    def is_root(self):
        """Check if this is the root joint."""
        return self.parent is None
    
    def get_chain_to_root(self):
        """Get the chain of joints from this joint to root."""
        chain = []
        joint = self
        while joint:
            chain.append(joint)
            joint = joint.parent
        return list(reversed(chain))


class SkeletonMapper:
    """Maps MediaPipe landmarks to BVH skeleton structure."""
    
    def __init__(self):
        self.skeleton = self._create_bvh_skeleton()
        self.joint_mapping = self._create_joint_mapping()
        self.default_offsets = self._get_default_offsets()
        
    def _create_bvh_skeleton(self):
        """Create the BVH skeleton hierarchy."""
        # Root (Hips)
        hips = BVHJoint("Hips")
        hips.channels = ['Xposition', 'Yposition', 'Zposition', 
                        'Xrotation', 'Yrotation', 'Zrotation']  # XYZ order
        
        # Spine chain
        spine = BVHJoint("Spine", hips)
        spine.channels = ['Xrotation', 'Yrotation', 'Zrotation']
        
        chest = BVHJoint("Chest", spine)
        chest.channels = ['Xrotation', 'Yrotation', 'Zrotation']
        
        neck = BVHJoint("Neck", chest)
        neck.channels = ['Xrotation', 'Yrotation', 'Zrotation']
        
        head = BVHJoint("Head", neck)
        head.channels = ['Xrotation', 'Yrotation', 'Zrotation']
        
        # Left arm chain
        left_shoulder = BVHJoint("LeftShoulder", chest)
        left_shoulder.channels = ['Xrotation', 'Yrotation', 'Zrotation']
        
        left_arm = BVHJoint("LeftArm", left_shoulder)
        left_arm.channels = ['Xrotation', 'Yrotation', 'Zrotation']
        
        left_forearm = BVHJoint("LeftForeArm", left_arm)
        left_forearm.channels = ['Xrotation', 'Yrotation', 'Zrotation']
        
        left_hand = BVHJoint("LeftHand", left_forearm)
        left_hand.channels = ['Xrotation', 'Yrotation', 'Zrotation']
        
        # Right arm chain
        right_shoulder = BVHJoint("RightShoulder", chest)
        right_shoulder.channels = ['Xrotation', 'Yrotation', 'Zrotation']
        
        right_arm = BVHJoint("RightArm", right_shoulder)
        right_arm.channels = ['Xrotation', 'Yrotation', 'Zrotation']
        
        right_forearm = BVHJoint("RightForeArm", right_arm)
        right_forearm.channels = ['Xrotation', 'Yrotation', 'Zrotation']
        
        right_hand = BVHJoint("RightHand", right_forearm)
        right_hand.channels = ['Xrotation', 'Yrotation', 'Zrotation']
        
        # Left leg chain
        left_upleg = BVHJoint("LeftUpLeg", hips)
        left_upleg.channels = ['Xrotation', 'Yrotation', 'Zrotation']
        
        left_leg = BVHJoint("LeftLeg", left_upleg)
        left_leg.channels = ['Xrotation', 'Yrotation', 'Zrotation']
        
        left_foot = BVHJoint("LeftFoot", left_leg)
        left_foot.channels = ['Xrotation', 'Yrotation', 'Zrotation']
        
        # Right leg chain
        right_upleg = BVHJoint("RightUpLeg", hips)
        right_upleg.channels = ['Xrotation', 'Yrotation', 'Zrotation']
        
        right_leg = BVHJoint("RightLeg", right_upleg)
        right_leg.channels = ['Xrotation', 'Yrotation', 'Zrotation']
        
        right_foot = BVHJoint("RightFoot", right_leg)
        right_foot.channels = ['Xrotation', 'Yrotation', 'Zrotation']
        
        return hips  # Return root joint
    
    def _create_joint_mapping(self):
        """Create mapping from BVH joint names to MediaPipe landmark indices."""
        return {
            'Hips': [mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.RIGHT_HIP],
            'Spine': [mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.RIGHT_HIP,
                     mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.RIGHT_SHOULDER],
            'Chest': [mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.RIGHT_SHOULDER],
            'Neck': [mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.RIGHT_SHOULDER,
                    mp_pose.PoseLandmark.LEFT_EAR, mp_pose.PoseLandmark.RIGHT_EAR],
            'Head': [mp_pose.PoseLandmark.NOSE, mp_pose.PoseLandmark.LEFT_EAR, 
                    mp_pose.PoseLandmark.RIGHT_EAR],
            
            # Arms
            'LeftShoulder': [mp_pose.PoseLandmark.LEFT_SHOULDER],
            'LeftArm': [mp_pose.PoseLandmark.LEFT_ELBOW],
            'LeftForeArm': [mp_pose.PoseLandmark.LEFT_WRIST],
            'LeftHand': [mp_pose.PoseLandmark.LEFT_WRIST, mp_pose.PoseLandmark.LEFT_PINKY,
                        mp_pose.PoseLandmark.LEFT_INDEX],
            
            'RightShoulder': [mp_pose.PoseLandmark.RIGHT_SHOULDER],
            'RightArm': [mp_pose.PoseLandmark.RIGHT_ELBOW],
            'RightForeArm': [mp_pose.PoseLandmark.RIGHT_WRIST],
            'RightHand': [mp_pose.PoseLandmark.RIGHT_WRIST, mp_pose.PoseLandmark.RIGHT_PINKY,
                         mp_pose.PoseLandmark.RIGHT_INDEX],
            
            # Legs
            'LeftUpLeg': [mp_pose.PoseLandmark.LEFT_HIP],
            'LeftLeg': [mp_pose.PoseLandmark.LEFT_KNEE],
            'LeftFoot': [mp_pose.PoseLandmark.LEFT_ANKLE, mp_pose.PoseLandmark.LEFT_HEEL,
                        mp_pose.PoseLandmark.LEFT_FOOT_INDEX],
            
            'RightUpLeg': [mp_pose.PoseLandmark.RIGHT_HIP],
            'RightLeg': [mp_pose.PoseLandmark.RIGHT_KNEE],
            'RightFoot': [mp_pose.PoseLandmark.RIGHT_ANKLE, mp_pose.PoseLandmark.RIGHT_HEEL,
                         mp_pose.PoseLandmark.RIGHT_FOOT_INDEX],
        }
    
    def _get_default_offsets(self):
        """Get default bone offsets for initial skeleton setup."""
        return {
            'Spine': np.array([0, 10, 0]),
            'Chest': np.array([0, 15, 0]),
            'Neck': np.array([0, 5, 0]),
            'Head': np.array([0, 10, 0]),
            'LeftShoulder': np.array([-8, 2, 0]),
            'RightShoulder': np.array([8, 2, 0]),
            'LeftArm': np.array([-15, -5, 0]),
            'RightArm': np.array([15, -5, 0]),
            'LeftForeArm': np.array([-12, -8, 0]),
            'RightForeArm': np.array([12, -8, 0]),
            'LeftHand': np.array([-5, -5, 0]),
            'RightHand': np.array([5, -5, 0]),
            'LeftUpLeg': np.array([-5, -10, 0]),
            'RightUpLeg': np.array([5, -10, 0]),
            'LeftLeg': np.array([0, -20, 0]),
            'RightLeg': np.array([0, -20, 0]),
            'LeftFoot': np.array([0, -5, 5]),
            'RightFoot': np.array([0, -5, 5]),
        }
    
    def get_joint_position(self, joint_name, landmarks, scale=100.0):
        """Get 3D position of a BVH joint from MediaPipe landmarks.
        
        Args:
            joint_name: Name of the BVH joint
            landmarks: MediaPipe pose landmarks
            scale: Scale factor (default 100 for meters to cm)
            
        Returns:
            3D position as numpy array or None if not available
        """
        landmark_indices = self.joint_mapping.get(joint_name, [])
        positions = []
        
        for idx in landmark_indices:
            if idx < len(landmarks):
                lm = landmarks[idx]
                if hasattr(lm, 'visibility') and lm.visibility > 0.5:
                    # MediaPipe Y is down, BVH Y is up
                    pos = np.array([lm.x, -lm.y, lm.z]) * scale
                    positions.append(pos)
        
        if positions:
            return np.mean(positions, axis=0)
        return None
    
    def calculate_bone_offsets(self, reference_landmarks, scale=100.0):
        """Calculate bone offsets from a reference pose.
        
        Args:
            reference_landmarks: MediaPipe landmarks from reference frame
            scale: Scale factor for unit conversion
        """
        def set_joint_offset(joint, parent_pos, joint_pos):
            """Helper to set joint offset from positions."""
            if parent_pos is not None and joint_pos is not None:
                offset = joint_pos - parent_pos
                # Ensure minimum bone length
                length = np.linalg.norm(offset)
                if length < 1.0:  # Minimum 1cm
                    default = self.default_offsets.get(joint.name, np.array([0, 5, 0]))
                    offset = default
                joint.set_offset(offset)
            else:
                joint.set_offset(self.default_offsets.get(joint.name, np.array([0, 5, 0])))
        
        def process_joint(joint):
            """Recursively process joint hierarchy."""
            joint_pos = self.get_joint_position(joint.name, reference_landmarks, scale)
            
            if joint.parent:
                parent_pos = self.get_joint_position(joint.parent.name, reference_landmarks, scale)
                set_joint_offset(joint, parent_pos, joint_pos)
            
            for child in joint.children:
                process_joint(child)
        
        # Start from root
        process_joint(self.skeleton)
    
    def get_all_joints(self):
        """Get a flat list of all joints in hierarchical order."""
        joints = []
        
        def collect_joints(joint):
            joints.append(joint)
            for child in joint.children:
                collect_joints(child)
        
        collect_joints(self.skeleton)
        return joints
    
    def get_joint_by_name(self, name):
        """Find a joint by name."""
        for joint in self.get_all_joints():
            if joint.name == name:
                return joint
        return None