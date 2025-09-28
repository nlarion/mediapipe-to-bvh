"""
Configuration settings for MediaPipe to BVH converter.
Based on lessons learned from experiments.
"""

# MediaPipe Settings
MEDIAPIPE_CONFIG = {
    'static_image_mode': False,
    'model_complexity': 1,  # Balanced accuracy/speed
    'smooth_landmarks': True,
    'enable_segmentation': False,
    'min_detection_confidence': 0.5,
    'min_tracking_confidence': 0.5
}

# Processing Settings (from untitled9.py success)
PROCESSING_CONFIG = {
    'sample_rate': 2,  # Process every 2nd frame (key to reducing drift)
    'scale_factor': 100.0,  # Convert meters to centimeters
    'movement_scale': 90000.0,  # Reduced from 100000 to reduce jitter while maintaining movement
    'min_bone_length': 1.0,  # Minimum bone length in cm
    'confidence_threshold': 0.5
}

# Smoothing Settings (gentle approach learned from experiments)
SMOOTHING_CONFIG = {
    'enable_kalman': False,  # Avoid over-processing
    'enable_temporal_smoothing': True,
    'temporal_window_size': 3,  # Small window to preserve dynamics
    'preserve_dynamics': True
}

# Ground Contact Settings (learned: aggressive locking is bad)
GROUND_CONTACT_CONFIG = {
    'enable_ground_contact': False,  # Disabled based on experiments
    'ground_threshold': 5.0,  # cm from ground
    'velocity_threshold': 2.0,  # cm/frame
    'lock_threshold': 3.0  # cm horizontal movement
}

# BVH Output Settings
BVH_CONFIG = {
    'fps': 12,  # Effective FPS after sampling every 2 frames from 24 FPS video
    'rotation_order': 'XYZ',  # Match untitled9.py
    'root_height': 60.0,  # Default hip height in cm
}

# Joint Angle Constraints (degrees)
JOINT_CONSTRAINTS = {
    'elbow': (0, 145),
    'knee': (0, 140),
    'shoulder_flexion': (-180, 180),
    'shoulder_abduction': (-90, 180),
    'hip_flexion': (-30, 120),
    'hip_abduction': (-45, 45),
    'neck': (-60, 60),
    'spine': (-45, 45)
}

# Quality Thresholds
QUALITY_THRESHOLDS = {
    'min_detection_score': 6,  # Minimum number of key joints detected
    'max_position_jump': 50.0,  # Maximum cm jump between frames
    'min_frames_for_reference': 30  # Frames to check for best reference
}