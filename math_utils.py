"""
Mathematical utilities for coordinate transformations and rotations.
Implements position-to-rotation conversion for BVH output.
"""

import numpy as np
import math
from scipy.spatial.transform import Rotation as R
from typing import Tuple, Optional

from config import JOINT_CONSTRAINTS, SMOOTHING_CONFIG


def calculate_rotation_between_vectors(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    """Calculate Euler rotation that transforms v1 to v2.
    
    Args:
        v1: Source vector (rest pose)
        v2: Target vector (current pose)
        
    Returns:
        Euler angles in degrees (ZYX order for BVH)
    """
    # Normalize vectors
    v1_norm = v1 / (np.linalg.norm(v1) + 1e-10)
    v2_norm = v2 / (np.linalg.norm(v2) + 1e-10)
    
    # Check if vectors are parallel
    dot_product = np.clip(np.dot(v1_norm, v2_norm), -1.0, 1.0)
    if abs(dot_product) > 0.9999:
        # Vectors are nearly parallel
        if dot_product > 0:
            return np.zeros(3)  # No rotation needed
        else:
            # 180 degree rotation around perpendicular axis
            perpendicular = np.array([1, 0, 0]) if abs(v1_norm[0]) < 0.9 else np.array([0, 1, 0])
            axis = np.cross(v1_norm, perpendicular)
            axis = axis / np.linalg.norm(axis)
            return axis * 180.0
    
    # Calculate rotation axis (cross product)
    axis = np.cross(v1_norm, v2_norm)
    axis_length = np.linalg.norm(axis)
    
    if axis_length < 1e-10:
        return np.zeros(3)
    
    axis = axis / axis_length
    
    # Calculate angle
    angle = np.arccos(dot_product)
    
    # Create rotation using scipy
    rotation = R.from_rotvec(axis * angle)
    
    # Convert to Euler angles (ZYX order for BVH)
    euler = rotation.as_euler('ZYX', degrees=True)
    
    return euler


def position_to_rotation(joint_pos: np.ndarray, 
                        parent_pos: np.ndarray,
                        rest_offset: np.ndarray,
                        child_pos: Optional[np.ndarray] = None) -> np.ndarray:
    """Convert joint positions to rotation angles.
    
    Args:
        joint_pos: Current joint position
        parent_pos: Parent joint position
        rest_offset: Rest pose offset from parent
        child_pos: Optional child position for better orientation
        
    Returns:
        Rotation angles in degrees (ZYX order)
    """
    if joint_pos is None or parent_pos is None:
        return np.zeros(3)
    
    # Current vector from parent to joint
    current_vector = joint_pos - parent_pos
    
    # Check for zero-length vectors
    if np.linalg.norm(current_vector) < 0.01 or np.linalg.norm(rest_offset) < 0.01:
        return np.zeros(3)
    
    # Calculate rotation
    rotation = calculate_rotation_between_vectors(rest_offset, current_vector)
    
    # Apply constraints
    rotation = apply_joint_constraints(rotation)
    
    return rotation


def axis_angle_to_euler(axis_angle: np.ndarray, order: str = 'XYZ') -> np.ndarray:
    """Convert axis-angle rotation to Euler angles.
    
    This is the method from untitled9.py that produces good results.
    
    Args:
        axis_angle: Axis-angle representation of rotation
        order: Euler angle order ('XYZ' or 'ZYX')
        
    Returns:
        Euler angles in degrees
    """
    angle = np.linalg.norm(axis_angle)
    
    if angle < 1e-10:
        return np.zeros(3)
    
    axis = axis_angle / angle
    
    # Convert to rotation matrix using Rodrigues' formula
    c = math.cos(angle)
    s = math.sin(angle)
    t = 1.0 - c
    x, y, z = axis
    
    # Compute rotation matrix elements
    r00 = c + x*x*t
    r01 = x*y*t - z*s
    r02 = x*z*t + y*s
    r10 = y*x*t + z*s
    r11 = c + y*y*t
    r12 = y*z*t - x*s
    r20 = z*x*t - y*s
    r21 = z*y*t + x*s
    r22 = c + z*z*t
    
    # Convert rotation matrix to Euler angles based on order
    if order == 'XYZ':
        # Check for gimbal lock
        sy = math.sqrt(r00*r00 + r10*r10)
        
        if sy > 1e-6:
            # Normal case - not at singularity
            x = math.atan2(r21, r22)
            y = math.atan2(-r20, sy)
            z = math.atan2(r10, r00)
        else:
            # At singularity (gimbal lock)
            x = math.atan2(-r12, r11)
            y = math.atan2(-r20, sy)
            z = 0
    elif order == 'ZYX':
        # Check for gimbal lock
        sy = math.sqrt(r00*r00 + r01*r01)
        
        if sy > 1e-6:
            # Normal case
            z = math.atan2(r10, r00)
            y = math.atan2(-r20, sy)
            x = math.atan2(r21, r22)
        else:
            # Gimbal lock
            z = math.atan2(-r01, r11)
            y = math.atan2(-r20, sy)
            x = 0
    else:
        # Default fallback
        return np.zeros(3)
    
    return np.array([x, y, z]) * (180.0 / math.pi)  # Convert to degrees


def calculate_rotation_from_directions(rest_direction: np.ndarray, 
                                      current_direction: np.ndarray,
                                      order: str = 'XYZ') -> np.ndarray:
    """Calculate rotation from rest pose direction to current direction.
    
    This uses the untitled9.py approach with cross product and axis-angle.
    
    Args:
        rest_direction: Normalized rest pose direction
        current_direction: Normalized current pose direction
        order: Euler angle order
        
    Returns:
        Euler angles in degrees
    """
    # Calculate rotation axis and angle
    cross = np.cross(rest_direction, current_direction)
    dot = np.dot(rest_direction, current_direction)
    
    if np.linalg.norm(cross) > 1e-10:
        # Calculate angle between vectors
        angle = math.acos(np.clip(dot, -1.0, 1.0))
        # Create axis-angle representation
        axis_angle = (cross / np.linalg.norm(cross)) * angle
        # Convert to Euler angles
        return axis_angle_to_euler(axis_angle, order)
    else:
        # Vectors are parallel or anti-parallel
        if dot > 0:
            return np.zeros(3)  # Same direction
        else:
            # Opposite directions - 180 degree rotation
            # Find a perpendicular axis
            if abs(rest_direction[0]) < 0.9:
                axis = np.cross(rest_direction, np.array([1, 0, 0]))
            else:
                axis = np.cross(rest_direction, np.array([0, 1, 0]))
            axis = axis / np.linalg.norm(axis)
            axis_angle = axis * math.pi
            return axis_angle_to_euler(axis_angle, order)


def apply_joint_constraints(rotation: np.ndarray, joint_type: str = None) -> np.ndarray:
    """Apply anatomical constraints to joint rotations.
    
    Args:
        rotation: Unconstrained rotation angles in degrees
        joint_type: Type of joint for specific constraints
        
    Returns:
        Constrained rotation angles
    """
    # Basic clamping to prevent extreme rotations
    rotation = np.clip(rotation, -180, 180)
    
    # Apply specific joint constraints if available
    if joint_type and joint_type in JOINT_CONSTRAINTS:
        min_angle, max_angle = JOINT_CONSTRAINTS[joint_type]
        # For simplicity, apply to primary rotation axis
        rotation[0] = np.clip(rotation[0], min_angle, max_angle)
    
    return rotation


def smooth_rotations(rotations_sequence: np.ndarray, 
                    window_size: int = 3,
                    preserve_dynamics: bool = True) -> np.ndarray:
    """Apply temporal smoothing to rotation sequence.
    
    Args:
        rotations_sequence: Array of rotations over time (frames x 3)
        window_size: Size of smoothing window
        preserve_dynamics: Whether to preserve motion dynamics
        
    Returns:
        Smoothed rotations
    """
    if len(rotations_sequence) < window_size:
        return rotations_sequence
    
    smoothed = np.zeros_like(rotations_sequence)
    half_window = window_size // 2
    
    for i in range(len(rotations_sequence)):
        start = max(0, i - half_window)
        end = min(len(rotations_sequence), i + half_window + 1)
        window = rotations_sequence[start:end]
        
        if preserve_dynamics and len(window) > 2:
            # Weighted average that preserves dynamics
            weights = np.exp(-0.5 * np.arange(len(window))**2)
            weights = weights / weights.sum()
            smoothed[i] = np.average(window, axis=0, weights=weights)
        else:
            # Simple moving average
            smoothed[i] = np.mean(window, axis=0)
    
    return smoothed


def smooth_positions(positions_sequence: np.ndarray,
                    window_size: int = 3,
                    preserve_dynamics: bool = True,
                    preserve_y_axis: bool = True) -> np.ndarray:
    """Apply temporal smoothing to position sequence.
    
    Specifically designed for hip positions to reduce jitter while preserving movement.
    
    Args:
        positions_sequence: Array of positions over time (frames x 3)
        window_size: Size of smoothing window
        preserve_dynamics: Whether to preserve motion dynamics
        preserve_y_axis: Whether to apply less smoothing to Y (vertical) axis
        
    Returns:
        Smoothed positions
    """
    if len(positions_sequence) < window_size:
        return positions_sequence
    
    smoothed = np.zeros_like(positions_sequence)
    half_window = window_size // 2
    
    for i in range(len(positions_sequence)):
        start = max(0, i - half_window)
        end = min(len(positions_sequence), i + half_window + 1)
        window = positions_sequence[start:end]
        
        if preserve_dynamics and len(window) > 2:
            # Weighted average that preserves dynamics
            # Use Gaussian-like weights centered on current frame
            center = len(window) // 2
            weights = np.exp(-0.5 * ((np.arange(len(window)) - center) / (half_window + 0.1))**2)
            weights = weights / weights.sum()
            smoothed[i] = np.average(window, axis=0, weights=weights)
        else:
            # Simple moving average
            smoothed[i] = np.mean(window, axis=0)
        
        # Optionally preserve Y-axis (vertical) more strictly
        if preserve_y_axis and i > 0:
            # Apply less smoothing to Y-axis to maintain ground contact
            y_weight = 0.7  # How much to preserve original Y value
            smoothed[i][1] = smoothed[i][1] * (1 - y_weight) + positions_sequence[i][1] * y_weight
    
    return smoothed


def calculate_bone_length(pos1: np.ndarray, pos2: np.ndarray) -> float:
    """Calculate distance between two joint positions.
    
    Args:
        pos1: First joint position
        pos2: Second joint position
        
    Returns:
        Distance in the same units as positions
    """
    return np.linalg.norm(pos2 - pos1)


def enforce_bone_length(joint_pos: np.ndarray,
                       parent_pos: np.ndarray,
                       target_length: float) -> np.ndarray:
    """Enforce consistent bone length between joints.
    
    Args:
        joint_pos: Current joint position
        parent_pos: Parent joint position
        target_length: Desired bone length
        
    Returns:
        Adjusted joint position with correct bone length
    """
    current_vector = joint_pos - parent_pos
    current_length = np.linalg.norm(current_vector)
    
    if current_length < 0.001:
        # Degenerate case - place joint above parent
        return parent_pos + np.array([0, target_length, 0])
    
    # Scale vector to target length
    scaled_vector = current_vector * (target_length / current_length)
    return parent_pos + scaled_vector


def quaternion_to_euler(quaternion: np.ndarray, order: str = 'ZYX') -> np.ndarray:
    """Convert quaternion to Euler angles.
    
    Args:
        quaternion: Quaternion (w, x, y, z)
        order: Euler angle order
        
    Returns:
        Euler angles in degrees
    """
    rotation = R.from_quat(quaternion)
    return rotation.as_euler(order, degrees=True)


def euler_to_quaternion(euler: np.ndarray, order: str = 'ZYX') -> np.ndarray:
    """Convert Euler angles to quaternion.
    
    Args:
        euler: Euler angles in degrees
        order: Euler angle order
        
    Returns:
        Quaternion (w, x, y, z)
    """
    rotation = R.from_euler(order, euler, degrees=True)
    return rotation.as_quat()


def interpolate_rotations(rot1: np.ndarray, rot2: np.ndarray, t: float) -> np.ndarray:
    """Interpolate between two rotations using SLERP.
    
    Args:
        rot1: First rotation (Euler angles in degrees)
        rot2: Second rotation (Euler angles in degrees)
        t: Interpolation parameter (0 to 1)
        
    Returns:
        Interpolated rotation (Euler angles in degrees)
    """
    # Convert to quaternions for proper interpolation
    q1 = euler_to_quaternion(rot1)
    q2 = euler_to_quaternion(rot2)
    
    # Use scipy's slerp
    r1 = R.from_quat(q1)
    r2 = R.from_quat(q2)
    times = [0, 1]
    rotations = R.from_quat([q1, q2])
    slerp = R.Slerp(times, rotations)
    
    interpolated = slerp(t)
    return interpolated.as_euler('ZYX', degrees=True)


def calculate_joint_angle(parent_pos: np.ndarray,
                         joint_pos: np.ndarray,
                         child_pos: np.ndarray) -> float:
    """Calculate the angle at a joint given three positions.
    
    Args:
        parent_pos: Parent joint position
        joint_pos: Current joint position
        child_pos: Child joint position
        
    Returns:
        Angle in degrees
    """
    vec1 = parent_pos - joint_pos
    vec2 = child_pos - joint_pos
    
    # Normalize vectors
    vec1_norm = vec1 / (np.linalg.norm(vec1) + 1e-10)
    vec2_norm = vec2 / (np.linalg.norm(vec2) + 1e-10)
    
    # Calculate angle
    cos_angle = np.clip(np.dot(vec1_norm, vec2_norm), -1.0, 1.0)
    angle = np.degrees(np.arccos(cos_angle))
    
    return angle


def apply_ik_constraint(joint_pos: np.ndarray,
                       parent_pos: np.ndarray,
                       child_pos: np.ndarray,
                       angle_limit: Tuple[float, float]) -> np.ndarray:
    """Apply inverse kinematics constraint to maintain valid joint angles.
    
    Args:
        joint_pos: Current joint position
        parent_pos: Parent joint position
        child_pos: Child joint position
        angle_limit: Min and max allowed angles in degrees
        
    Returns:
        Adjusted joint position
    """
    current_angle = calculate_joint_angle(parent_pos, joint_pos, child_pos)
    min_angle, max_angle = angle_limit
    
    if min_angle <= current_angle <= max_angle:
        return joint_pos  # No adjustment needed
    
    # Calculate target angle
    target_angle = np.clip(current_angle, min_angle, max_angle)
    angle_diff = np.radians(target_angle - current_angle)
    
    # Rotate child vector around axis perpendicular to plane
    vec1 = parent_pos - joint_pos
    vec2 = child_pos - joint_pos
    axis = np.cross(vec1, vec2)
    
    if np.linalg.norm(axis) < 0.001:
        return joint_pos  # Vectors are parallel, can't adjust
    
    axis = axis / np.linalg.norm(axis)
    
    # Apply rotation
    rotation = R.from_rotvec(axis * angle_diff)
    vec2_rotated = rotation.apply(vec2)
    
    # Calculate new joint position
    # This is simplified - proper IK would adjust the entire chain
    return child_pos - vec2_rotated


def calculate_depth_from_projected_length(observed_length: float, 
                                        actual_length: float, 
                                        focal_length: float) -> float:
    """Calculate depth (Z-distance) based on projected length of a known object.
    
    Uses similar triangles principle:
    depth = (actual_length * focal_length) / observed_length
    
    Args:
        observed_length: Length of object in the image (pixels or normalized units)
        actual_length: Actual physical length of the object
        focal_length: Focal length of the camera (in same units as observed_length)
        
    Returns:
        Calculated depth
    """
    if observed_length < 1e-6:
        return 0.0
        
    return (actual_length * focal_length) / observed_length