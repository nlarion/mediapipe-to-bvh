# MediaPipe to BVH Converter

Convert videos to BVH (Biovision Hierarchy) motion capture files using pose estimation.

## Main Implementations

### 1. **mediapipe_bvh_plotly.ipynb** (Recommended)
- **Best working implementation** with interactive Plotly 3D preview
- MediaPipe-based pose detection
- Real-time visualization of BVH output
- Includes both BVH parsing and generation functions

### 2. **mediapipe_to_bvh_complete.py**
- Comprehensive command-line MediaPipe implementation
- Global position tracking and motion smoothing
- Supports video preview during processing
- Most feature-complete standalone script
```bash
python mediapipe_to_bvh_complete.py --video input.mp4 --output output.bvh --preview
```

### 3. **multi_backend_bvh.py**
- Supports multiple pose estimation backends:
  - ViTPose (most accurate)
  - MMPose
  - MediaPipe (fallback)
- Automatically selects best available backend
- Good for comparing different pose estimation methods

### 4. **vitpose_math_bvh.py**
- Mathematical approach using ViTPose
- Uses bvhio library for proper BVH handling
- Advanced 3D pose lifting techniques
- Best for high-accuracy requirements

## Directory Structure

```
mediapipe-to-bvh/
├── mediapipe_bvh_plotly.ipynb    # Main notebook with Plotly preview
├── mediapipe_to_bvh_complete.py  # Complete CLI implementation
├── multi_backend_bvh.py          # Multi-backend support
├── vitpose_math_bvh.py           # Mathematical approach
├── requirements.txt              # Python dependencies
├── model/                        # Pose detection models
│   ├── pose_landmarker_*.task   # MediaPipe models
│   └── vitpose-*.pth            # ViTPose models
├── videos/                       # Input video files
├── bvh/                         # Output BVH files
└── easy_ViTPose/                # ViTPose implementation

```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Notebook (Recommended for visualization)
1. Open `mediapipe_bvh_plotly.ipynb` in Jupyter
2. Run the cells to process videos and preview BVH output in 3D

### Command Line
```bash
# Basic conversion
python mediapipe_to_bvh_complete.py --video videos/walking.mp4 --output bvh/walking.bvh

# With preview window
python mediapipe_to_bvh_complete.py --video videos/walking.mp4 --output bvh/walking.bvh --preview

# Using multi-backend (auto-selects best available)
python multi_backend_bvh.py --input videos/walking.mp4 --output bvh/walking.bvh
```

## Features

- **MediaPipe Integration**: Reliable human pose detection
- **ViTPose Support**: State-of-the-art pose estimation
- **Plotly 3D Visualization**: Interactive preview of BVH animations
- **Multiple Backends**: Choose between different pose estimation methods
- **Motion Smoothing**: Temporal filtering for smoother animations
- **Global Position Tracking**: Maintains character movement in world space

## Requirements

- Python 3.7+
- OpenCV
- MediaPipe
- NumPy
- Plotly (for visualization)
- PyTorch (for ViTPose backend)