# BVH Accuracy Testing System

A comprehensive testing and validation framework for MediaPipe to BVH conversion that focuses on **visual correctness**, not just numerical metrics.

## 🚀 Quick Start

### Basic Accuracy Testing
```bash
# Test a single BVH file against source video
python bvh_accuracy_tester.py --bvh output.bvh --video input.mp4 --verbose

# With ground truth comparison
python bvh_accuracy_tester.py --bvh output.bvh --video input.mp4 --ground-truth gt_data.npy
```

### Visual Overlay Validation
```bash
# Create overlay video showing BVH skeleton on original video
python bvh_overlay_visualizer.py --video input.mp4 --bvh output.bvh --output overlay.mp4

# Without preview window (faster)
python bvh_overlay_visualizer.py --video input.mp4 --bvh output.bvh --no-preview
```

### Cross-Model Validation
```bash
# Compare with multiple pose estimation models
python cross_model_validator.py --video input.mp4 --models mediapipe openpose

# Sample every 10th frame for faster processing
python cross_model_validator.py --video input.mp4 --sample-rate 10 --visualize
```

### Comprehensive Test Suite
```bash
# Run all tests
python bvh_test_runner.py run

# Run specific test
python bvh_test_runner.py run --test walking_motion

# Run tests by tag
python bvh_test_runner.py run --tags motion complex

# Regression testing
python bvh_test_runner.py regression --old old.bvh --new new.bvh --video input.mp4

# Create new test case
python bvh_test_runner.py create --name my_test --video test_video.mp4
```

## 📊 Core Metrics

### Position Metrics
- **MPJPE** (Mean Per Joint Position Error): Average 3D distance between joints
- **PA-MPJPE** (Procrustes-aligned MPJPE): Position error after alignment
- **PCK** (Percentage of Correct Keypoints): % of joints within threshold

### Rotation Metrics
- **Angular Error**: Average rotation difference in degrees
- **Geodesic Distance**: Rotation distance on SO(3) manifold

### Temporal Metrics
- **Jitter**: Frame-to-frame smoothness measurement
- **Velocity/Acceleration Errors**: Motion derivative comparison
- **DTW Distance**: Temporal alignment comparison

### Visual Correctness (Critical!)
- **Re-projection Error**: 2D projection accuracy
- **Limb Length Consistency**: Bone stretching detection
- **Joint Angle Limits**: Anatomical plausibility
- **Global Orientation**: 90-degree rotation detection

## 🛠️ Components

### 1. `bvh_accuracy_tester.py`
Core accuracy testing framework with comprehensive metrics.

**Key Features:**
- Multiple accuracy metrics (MPJPE, PA-MPJPE, PCK)
- Temporal consistency analysis
- Visual correctness validation
- Anatomical constraint checking
- Detailed reporting

**Usage Example:**
```python
from bvh_accuracy_tester import AccuracyTester

tester = AccuracyTester(verbose=True)
results = tester.run_full_validation(
    bvh_path="output.bvh",
    video_path="input.mp4",
    ground_truth_path="gt.npy"  # Optional
)

# Generate report
report = tester.generate_report(results, "report.txt")
print(report)
```

### 2. `bvh_overlay_visualizer.py`
Visual validation tool for detecting issues metrics might miss.

**Key Features:**
- Overlays BVH skeleton on original video
- Side-by-side MediaPipe comparison
- Real-time alignment metrics
- Interactive controls during preview

**Controls (during preview):**
- `q` - Quit
- `m` - Toggle MediaPipe skeleton
- `b` - Toggle BVH skeleton
- `c` - Toggle comparison metrics
- `Space` - Pause/Resume

### 3. `cross_model_validator.py`
Multi-model validation to identify systematic biases.

**Supported Models:**
- MediaPipe (always available)
- OpenPose (if installed)
- MMPose (if installed)
- MoveNet (coming soon)
- BlazePose (coming soon)

**Key Features:**
- Consensus keypoint calculation
- Model disagreement detection
- Outlier identification
- Comparative visualizations

### 4. `bvh_test_runner.py`
Comprehensive test orchestration and regression testing.

**Key Features:**
- Test case management
- Automated regression detection
- Batch testing with tags
- Performance tracking
- Detailed reporting

## 📁 Test Configuration

Create a test configuration file (`test_config.yaml`):

```yaml
output_dir: test_output
enable_visual_validation: true
enable_cross_model: false
regression_threshold: 0.05
test_cases_dir: test_cases
mediapipe_script: mediapipe_to_bvh_complete.py
save_artifacts: true
parallel_tests: false
```

Define test cases (`test_cases/walking.json`):

```json
{
  "name": "walking_motion",
  "video_path": "videos/walking_00001.mp4",
  "expected_metrics": {
    "mpjpe": 0.08,
    "temporal_jitter": 0.03
  },
  "thresholds": {
    "mpjpe": 0.15,
    "temporal_jitter": 0.1
  },
  "tags": ["motion", "walking"],
  "description": "Basic walking motion validation"
}
```

## 🎯 Detecting Common Issues

### 90-Degree Rotation Errors
The tester specifically checks for orientation issues:
```python
# Global orientation check
result = tester.check_global_orientation(bvh_data)
if not result.passed:
    print(f"Rotation error detected: {result.details}")
```

### Temporal Jitter
Detects unstable or jittery motion:
```python
jitter_metrics = tester.temporal_jitter(joint_sequence)
if jitter_metrics["max_jitter"] > 0.1:
    print("High jitter detected!")
```

### Limb Stretching
Verifies bone lengths remain consistent:
```python
limb_check = tester.check_limb_lengths(joints, tolerance=0.1)
if not limb_check.passed:
    print(f"Limb stretching detected: {limb_check.value}")
```

## 📈 Interpreting Results

### Good Results
- MPJPE < 0.1 (10cm for meter-scale data)
- PA-MPJPE < 0.05
- PCK@5cm > 90%
- Temporal jitter < 0.02
- No limb stretching
- No joint angle violations

### Warning Signs
- High disagreement between models
- Visual misalignment > 50 pixels
- Temporal jitter > 0.05
- Any joint angle violations
- Sudden orientation changes

### Critical Issues
- 90-degree rotation errors
- Flipped left/right joints
- Severe temporal drift
- Limb stretching > 10%

## 🔧 Installation

### Required Dependencies
```bash
pip install numpy opencv-python mediapipe scipy matplotlib
```

### Optional Dependencies
```bash
# For BVH parsing
pip install bvh

# For DTW metrics
pip install fastdtw

# For PyTorch-based models
pip install torch torchvision

# For configuration files
pip install pyyaml
```

### OpenPose Setup (Optional)
Follow OpenPose installation guide, then:
```bash
export OPENPOSE_PATH=/usr/local/openpose
export PYTHONPATH=$OPENPOSE_PATH/python:$PYTHONPATH
```

## 🚨 Troubleshooting

### "BVH parser not available"
Install the bvh library:
```bash
pip install bvh
```

### High visual alignment error
- Check coordinate system transformations
- Verify camera parameters
- Ensure proper scale factors

### Cross-model validation fails
- Ensure models are properly installed
- Check CUDA availability for GPU models
- Verify model checkpoints are downloaded

## 📊 Example Output

```
====================================================
BVH ACCURACY VALIDATION REPORT
====================================================
Timestamp: 2024-01-15T10:30:00
BVH File: output.bvh
Video File: input.mp4

----------------------------------------
METRICS
----------------------------------------
mpjpe: 0.0823
pa_mpjpe: 0.0456
pck_5cm: 92.3456
temporal_jitter:
  mean_jitter: 0.0234
  max_jitter: 0.0567
limb_consistency:
  passed: True
  value: 0.0234
visual_alignment_error: 23.4567

----------------------------------------
WARNINGS
----------------------------------------
⚠ High jitter detected on frame 234
⚠ Potential rotation issue: Chest orientation

====================================================
```

## 🎓 Best Practices

1. **Always run visual validation** - Numbers can be misleading
2. **Use multiple test videos** - Include edge cases
3. **Set appropriate thresholds** - Based on your use case
4. **Monitor regressions** - Track metrics over time
5. **Cross-validate when possible** - Use multiple models
6. **Document test cases** - Include expected behaviors
7. **Save artifacts** - Keep BVH and overlay videos for review

## 📝 Key Takeaways

> **The goal is not to maximize accuracy scores, but to produce visually correct and anatomically plausible BVH files that maintain temporal consistency.**

Remember: A high numerical accuracy score doesn't guarantee visual correctness. Always validate with:
1. Visual overlay comparison
2. Anatomical constraint checking
3. Temporal consistency analysis
4. Cross-model validation when critical

## 🤝 Contributing

When adding new metrics or test cases:
1. Focus on visual correctness
2. Include ground truth when possible
3. Document expected ranges
4. Add regression tests
5. Update this README

## 📄 License

This testing framework is part of the MediaPipe to BVH project.