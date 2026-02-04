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
        """Create the BVH skeleton hierarchy using Mixamo naming convention.

        This matches the standard Mixamo skeleton for automatic retargeting:
        - All bones prefixed with 'mixamorig:'
        - Spine chain: Hips -> Spine -> Spine2 -> Neck -> Head
        - Arms attach to Spine2 (no separate collar bone)
        - Legs include ToeBase bones
        """
        # Root (Hips)
        hips = BVHJoint("mixamorig:Hips")
        hips.channels = ['Xposition', 'Yposition', 'Zposition',
                        'Xrotation', 'Yrotation', 'Zrotation']  # XYZ order

        # Spine chain (simplified: Spine -> Spine2, skipping Spine1)
        spine = BVHJoint("mixamorig:Spine", hips)
        spine.channels = ['Xrotation', 'Yrotation', 'Zrotation']

        spine2 = BVHJoint("mixamorig:Spine2", spine)
        spine2.channels = ['Xrotation', 'Yrotation', 'Zrotation']

        neck = BVHJoint("mixamorig:Neck", spine2)
        neck.channels = ['Xrotation', 'Yrotation', 'Zrotation']

        head = BVHJoint("mixamorig:Head", neck)
        head.channels = ['Xrotation', 'Yrotation', 'Zrotation']

        # Left arm chain (Shoulder is clavicle in Mixamo)
        left_shoulder = BVHJoint("mixamorig:LeftShoulder", spine2)
        left_shoulder.channels = ['Xrotation', 'Yrotation', 'Zrotation']

        left_arm = BVHJoint("mixamorig:LeftArm", left_shoulder)
        left_arm.channels = ['Xrotation', 'Yrotation', 'Zrotation']

        left_forearm = BVHJoint("mixamorig:LeftForeArm", left_arm)
        left_forearm.channels = ['Xrotation', 'Yrotation', 'Zrotation']

        left_hand = BVHJoint("mixamorig:LeftHand", left_forearm)
        left_hand.channels = ['Xrotation', 'Yrotation', 'Zrotation']

        # Right arm chain
        right_shoulder = BVHJoint("mixamorig:RightShoulder", spine2)
        right_shoulder.channels = ['Xrotation', 'Yrotation', 'Zrotation']

        right_arm = BVHJoint("mixamorig:RightArm", right_shoulder)
        right_arm.channels = ['Xrotation', 'Yrotation', 'Zrotation']

        right_forearm = BVHJoint("mixamorig:RightForeArm", right_arm)
        right_forearm.channels = ['Xrotation', 'Yrotation', 'Zrotation']

        right_hand = BVHJoint("mixamorig:RightHand", right_forearm)
        right_hand.channels = ['Xrotation', 'Yrotation', 'Zrotation']

        # Left leg chain (with ToeBase)
        left_upleg = BVHJoint("mixamorig:LeftUpLeg", hips)
        left_upleg.channels = ['Xrotation', 'Yrotation', 'Zrotation']

        left_leg = BVHJoint("mixamorig:LeftLeg", left_upleg)
        left_leg.channels = ['Xrotation', 'Yrotation', 'Zrotation']

        left_foot = BVHJoint("mixamorig:LeftFoot", left_leg)
        left_foot.channels = ['Xrotation', 'Yrotation', 'Zrotation']

        left_toe = BVHJoint("mixamorig:LeftToeBase", left_foot)
        left_toe.channels = ['Xrotation', 'Yrotation', 'Zrotation']

        # Right leg chain (with ToeBase)
        right_upleg = BVHJoint("mixamorig:RightUpLeg", hips)
        right_upleg.channels = ['Xrotation', 'Yrotation', 'Zrotation']

        right_leg = BVHJoint("mixamorig:RightLeg", right_upleg)
        right_leg.channels = ['Xrotation', 'Yrotation', 'Zrotation']

        right_foot = BVHJoint("mixamorig:RightFoot", right_leg)
        right_foot.channels = ['Xrotation', 'Yrotation', 'Zrotation']

        right_toe = BVHJoint("mixamorig:RightToeBase", right_foot)
        right_toe.channels = ['Xrotation', 'Yrotation', 'Zrotation']

        return hips  # Return root joint
    
    def _create_joint_mapping(self):
        """Create mapping from BVH joint names (Mixamo convention) to MediaPipe landmark indices."""
        return {
            'mixamorig:Hips': [mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.RIGHT_HIP],
            'mixamorig:Spine': [mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.RIGHT_HIP,
                     mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.RIGHT_SHOULDER],  # Mid-Torso
            'mixamorig:Spine2': [mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.RIGHT_SHOULDER],  # Upper-Torso
            'mixamorig:Neck': [mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.RIGHT_SHOULDER,
                    mp_pose.PoseLandmark.LEFT_EAR, mp_pose.PoseLandmark.RIGHT_EAR],
            'mixamorig:Head': [mp_pose.PoseLandmark.NOSE, mp_pose.PoseLandmark.LEFT_EAR,
                    mp_pose.PoseLandmark.RIGHT_EAR],

            # Left arm (Shoulder = clavicle in Mixamo)
            'mixamorig:LeftShoulder': [mp_pose.PoseLandmark.LEFT_SHOULDER],
            'mixamorig:LeftArm': [mp_pose.PoseLandmark.LEFT_ELBOW],
            'mixamorig:LeftForeArm': [mp_pose.PoseLandmark.LEFT_WRIST],
            'mixamorig:LeftHand': [mp_pose.PoseLandmark.LEFT_WRIST, mp_pose.PoseLandmark.LEFT_PINKY,
                        mp_pose.PoseLandmark.LEFT_INDEX],

            # Right arm
            'mixamorig:RightShoulder': [mp_pose.PoseLandmark.RIGHT_SHOULDER],
            'mixamorig:RightArm': [mp_pose.PoseLandmark.RIGHT_ELBOW],
            'mixamorig:RightForeArm': [mp_pose.PoseLandmark.RIGHT_WRIST],
            'mixamorig:RightHand': [mp_pose.PoseLandmark.RIGHT_WRIST, mp_pose.PoseLandmark.RIGHT_PINKY,
                         mp_pose.PoseLandmark.RIGHT_INDEX],

            # Left leg (with ToeBase)
            'mixamorig:LeftUpLeg': [mp_pose.PoseLandmark.LEFT_HIP],
            'mixamorig:LeftLeg': [mp_pose.PoseLandmark.LEFT_KNEE],
            'mixamorig:LeftFoot': [mp_pose.PoseLandmark.LEFT_ANKLE, mp_pose.PoseLandmark.LEFT_HEEL],
            'mixamorig:LeftToeBase': [mp_pose.PoseLandmark.LEFT_FOOT_INDEX],

            # Right leg (with ToeBase)
            'mixamorig:RightUpLeg': [mp_pose.PoseLandmark.RIGHT_HIP],
            'mixamorig:RightLeg': [mp_pose.PoseLandmark.RIGHT_KNEE],
            'mixamorig:RightFoot': [mp_pose.PoseLandmark.RIGHT_ANKLE, mp_pose.PoseLandmark.RIGHT_HEEL],
            'mixamorig:RightToeBase': [mp_pose.PoseLandmark.RIGHT_FOOT_INDEX],
        }
    
    def _get_default_offsets(self):
        """Get default bone offsets for initial skeleton setup (Mixamo naming)."""
        return {
            'mixamorig:Spine': np.array([0, 10, 0]),
            'mixamorig:Spine2': np.array([0, 15, 0]),
            'mixamorig:Neck': np.array([0, 5, 0]),
            'mixamorig:Head': np.array([0, 10, 0]),
            # Shoulder (clavicle) - offset from Spine2
            'mixamorig:LeftShoulder': np.array([5, 2, 0]),
            'mixamorig:RightShoulder': np.array([-5, 2, 0]),
            # Arm (upper arm) - offset from Shoulder
            'mixamorig:LeftArm': np.array([10, -2, 0]),
            'mixamorig:RightArm': np.array([-10, -2, 0]),
            'mixamorig:LeftForeArm': np.array([15, 0, 0]),
            'mixamorig:RightForeArm': np.array([-15, 0, 0]),
            'mixamorig:LeftHand': np.array([10, 0, 0]),
            'mixamorig:RightHand': np.array([-10, 0, 0]),
            # Legs
            'mixamorig:LeftUpLeg': np.array([5, -5, 0]),
            'mixamorig:RightUpLeg': np.array([-5, -5, 0]),
            'mixamorig:LeftLeg': np.array([0, -20, 0]),
            'mixamorig:RightLeg': np.array([0, -20, 0]),
            'mixamorig:LeftFoot': np.array([0, -20, 0]),
            'mixamorig:RightFoot': np.array([0, -20, 0]),
            'mixamorig:LeftToeBase': np.array([0, -2, 5]),
            'mixamorig:RightToeBase': np.array([0, -2, 5]),
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
        # Calculate average shoulder Y offset from MediaPipe data
        # This ensures symmetric Y offsets while being data-driven
        left_sh_pos = self.get_joint_position('mixamorig:LeftShoulder', reference_landmarks, scale)
        right_sh_pos = self.get_joint_position('mixamorig:RightShoulder', reference_landmarks, scale)
        spine2_pos = self.get_joint_position('mixamorig:Spine2', reference_landmarks, scale)

        level_shoulder_y = self.default_offsets.get('mixamorig:LeftShoulder', np.array([0, 2, 0]))[1]
        if left_sh_pos is not None and right_sh_pos is not None and spine2_pos is not None:
            # Calculate Y offsets from Spine2 to each shoulder
            left_y_offset = left_sh_pos[1] - spine2_pos[1]
            right_y_offset = right_sh_pos[1] - spine2_pos[1]
            # Use average of absolute values to get a positive offset
            avg_abs_y = (abs(left_y_offset) + abs(right_y_offset)) / 2.0
            # Use this average if it's reasonable, otherwise fall back to default
            if avg_abs_y > 0.5:  # At least 0.5 units
                level_shoulder_y = avg_abs_y

        def set_joint_offset(joint, parent_pos, joint_pos):
            """Helper to set joint offset from positions."""
            if parent_pos is not None and joint_pos is not None:
                offset = joint_pos - parent_pos
                # Ensure minimum bone length
                length = np.linalg.norm(offset)
                if length < 1.0:  # Minimum 1cm
                    default = self.default_offsets.get(joint.name, np.array([0, 5, 0]))
                    offset = default

                # CRITICAL FIX: Force Neck and Head offsets to be vertical in rest pose
                # MediaPipe ear positions are behind shoulders (large negative Z),
                # which causes tilted bone offsets. In BVH rest pose, these bones
                # should point straight up.
                if joint.name in ('mixamorig:Neck', 'mixamorig:Head'):
                    # Preserve only the vertical (Y) component, zero out X and Z
                    # Use the Y height or a reasonable default if Y is too small
                    y_height = abs(offset[1])
                    if y_height < 3.0:  # Minimum 3cm for neck/head bones
                        y_height = self.default_offsets.get(joint.name, np.array([0, 5, 0]))[1]
                    offset = np.array([0.0, y_height, 0.0])

                # SHOULDER LEVELING: Make shoulder Y offsets symmetric when asymmetry is small
                if joint.name in ('mixamorig:LeftShoulder', 'mixamorig:RightShoulder'):
                    if level_shoulder_y is not None:
                        offset[1] = level_shoulder_y

                # ARM OFFSETS: Force T-pose direction for arm bones
                # The reference frame may have arms in any pose (guard, etc.), but
                # BVH rest pose should be T-pose with arms pointing to the side
                if joint.name in ('mixamorig:LeftArm', 'mixamorig:LeftForeArm', 'mixamorig:LeftHand'):
                    bone_length = np.linalg.norm(offset)
                    default_dir = self.default_offsets.get(joint.name, np.array([10, -2, 0]))
                    default_dir_norm = default_dir / np.linalg.norm(default_dir)
                    offset = default_dir_norm * bone_length

                if joint.name in ('mixamorig:RightArm', 'mixamorig:RightForeArm', 'mixamorig:RightHand'):
                    bone_length = np.linalg.norm(offset)
                    default_dir = self.default_offsets.get(joint.name, np.array([-10, -2, 0]))
                    default_dir_norm = default_dir / np.linalg.norm(default_dir)
                    offset = default_dir_norm * bone_length

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