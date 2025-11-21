# MediaPipe to BVH Converter

Convert videos to BVH (Biovision Hierarchy) motion capture files using MediaPipe pose estimation.

## Main Scripts

### 1. **bvh_converter.py**
- **Main Converter Script**
- Uses MediaPipe to extract pose landmarks.
- Converts landmarks to BVH motion data.
- Includes IK (Inverse Kinematics) for better foot locking and ground contact.
- **Usage:**
  ```python
  from bvh_converter import ImprovedBVHConverter
  
  converter = ImprovedBVHConverter(enable_ik=True)
  converter.convert(pose_frames, "output.bvh")
  ```

### 2. **automated_bvh_accuracy_tester.py**
- **Accuracy Analysis**
- Compares generated BVH motion against the original MediaPipe landmarks.
- Calculates metrics for:
  - Overall Accuracy
  - Visual Naturalness
  - Foot Skate / Ground Contact
  - Knee Stability
  - Temporal Drift

### 3. **verify_improvements.py**
- **Verification Script**
- Runs the full pipeline: Video -> Extraction -> Conversion -> Analysis.
- Useful for verifying that changes to the converter haven't regressed quality.
- **Usage:**
  ```bash
  python verify_improvements.py
  ```

## Installation

```bash
# Activate the virtual environment
source /home/nlarion/Desktop/motion/motion_env/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Directory Structure

```
mediapipe-to-bvh/
├── bvh_converter.py                 # Main converter logic
├── automated_bvh_accuracy_tester.py # Accuracy analysis
├── verify_improvements.py           # End-to-end verification script
├── mediapipe_extractor.py           # MediaPipe interaction
├── skeleton_mapper.py               # Mapping MP landmarks to BVH skeleton
├── ik_foot_lock.py                  # Inverse Kinematics for feet
├── math_utils.py                    # Geometry/Math helpers
├── videos/                          # Input video files
├── bvh/                             # Output BVH files
└── test_output/                     # Generated test files
```