# BVH Accuracy Tester Development TODO

## 🎯 Goal
Build a robust accuracy testing system for MediaPipe to BVH pipeline that properly validates visual correctness, not just numerical metrics.

## 📊 Core Accuracy Metrics to Implement

### 1. Joint Position Error Metrics
- [ ] **Mean Per Joint Position Error (MPJPE)** - Average Euclidean distance between corresponding joints
- [ ] **Procrustes-aligned MPJPE (PA-MPJPE)** - Align poses using Procrustes analysis to remove global rotation/translation
- [ ] **Percentage of Correct Keypoints (PCK)** - Measure joints within threshold distance of ground truth

### 2. Rotation-Based Metrics
- [ ] **Mean Angular Error** - Compare joint rotations in quaternion or Euler angle space
- [ ] **Geodesic Distance** - More accurate rotation comparisons than simple Euclidean distance
- [ ] **Per-Joint Rotation Analysis** - Identify which specific rotations are problematic

### 3. Temporal Consistency Metrics
- [ ] **Jitter Measurement** - Analyze frame-to-frame smoothness
- [ ] **Velocity/Acceleration Errors** - Compare motion derivatives
- [ ] **Dynamic Time Warping (DTW)** - Handle temporal misalignments

### 4. Visual Correctness Metrics (CRITICAL!)
- [ ] **Re-projection Error** - Project BVH back to 2D and compare with video
- [ ] **Limb Length Consistency** - Ensure bones don't stretch/compress
- [ ] **Joint Angle Limits** - Verify anatomically plausible ranges
- [ ] **Global Orientation Check** - Detect 90-degree rotation errors like we just experienced

## 🛠️ Implementation Steps

### Phase 1: Core Framework
```python
class AccuracyTester:
    def __init__(self):
        self.metrics = {}

    def mpjpe(self, pred_joints, gt_joints):
        """Mean Per Joint Position Error"""
        return np.mean(np.linalg.norm(pred_joints - gt_joints, axis=-1))

    def pa_mpjpe(self, pred_joints, gt_joints):
        """Procrustes-aligned MPJPE"""
        pred_aligned = self.procrustes_align(pred_joints, gt_joints)
        return self.mpjpe(pred_aligned, gt_joints)

    def pck(self, pred_joints, gt_joints, threshold=0.05):
        """Percentage of Correct Keypoints"""
        distances = np.linalg.norm(pred_joints - gt_joints, axis=-1)
        return np.mean(distances < threshold) * 100

    def angular_error(self, pred_rotations, gt_rotations):
        """Mean angular error for rotations"""
        errors = []
        for pred_rot, gt_rot in zip(pred_rotations, gt_rotations):
            error = Rotation.from_quat(pred_rot).inv() * Rotation.from_quat(gt_rot)
            errors.append(error.magnitude())
        return np.mean(errors)

    def temporal_consistency(self, joint_sequence):
        """Measure jitter/smoothness"""
        velocities = np.diff(joint_sequence, axis=0)
        accelerations = np.diff(velocities, axis=0)
        jitter = np.std(accelerations)
        return jitter
```

### Phase 2: Multi-Stage Validation Pipeline

#### Stage 1: Direct Comparison
- [ ] Compare MediaPipe output directly with BVH joints
- [ ] Validate coordinate system transformations
- [ ] Check scale factors

#### Stage 2: Re-projection Validation
- [ ] Re-project BVH skeleton to 2D space
- [ ] Compare with original video keypoints
- [ ] Calculate pixel-space errors

#### Stage 3: Cross-Model Validation
- [ ] Implement OpenPose comparison
- [ ] Add MMPose validation
- [ ] Use multiple models to identify systematic biases

### Phase 3: Visual Validation Tools
- [ ] **2D Overlay Tool** - Overlay BVH skeleton on original video
- [ ] **3D Viewer** - Interactive 3D skeleton visualization
- [ ] **Difference Heatmaps** - Visual error representation per joint
- [ ] **Motion Trajectories** - Plot joint paths over time

## 🔄 Alternative Models for Cross-Validation

### 1. OpenPose Integration
- [ ] Install and setup OpenPose
- [ ] Create comparison pipeline
- [ ] Identify MediaPipe vs OpenPose differences

### 2. MMPose Integration
- [ ] Setup MMPose with multiple architectures
- [ ] Create benchmarking suite
- [ ] Compare accuracy across models

### 3. SMPL/SMPL-X Models
- [ ] Integrate parametric body models
- [ ] Use for anatomical plausibility checks
- [ ] Generate synthetic ground truth

### 4. MotionBERT/MHFormer
- [ ] Setup state-of-the-art 3D pose estimation
- [ ] Use as high-quality reference
- [ ] Validate 2D-to-3D lifting

## 📚 Required Libraries

### Essential
- [ ] `numpy` - Numerical operations
- [ ] `scipy` - Procrustes analysis, statistical metrics
- [ ] `pymo` or `bvhtoolbox` - BVH parsing and manipulation
- [ ] `opencv-python` - Video processing
- [ ] `matplotlib` - 2D visualization

### Advanced
- [ ] `pytorch3d` - Advanced 3D transformations
- [ ] `Open3D` - 3D visualization
- [ ] `fastdtw` - Dynamic Time Warping
- [ ] `scikit-learn` - Statistical analysis
- [ ] `motion-metrics` - Specialized motion evaluation

## 🎬 Ground Truth Dataset Creation

### Option 1: Use Existing Datasets
- [ ] **Human3.6M** - Large-scale dataset with ground truth 3D poses
- [ ] **AMASS** - Archive of motion capture sequences
- [ ] **3DPW** - 3D poses in the wild
- [ ] **MPI-INF-3DHP** - 3D human pose dataset

### Option 2: Create Test Suite
- [ ] Simple motions (T-pose, A-pose)
- [ ] Basic actions (wave, walk, jump)
- [ ] Complex sequences (dance, sports)
- [ ] Edge cases (occlusion, fast motion)

### Option 3: Synthetic Validation
- [ ] Generate known BVH files
- [ ] Render to video using Blender/Unity
- [ ] Test round-trip accuracy

## ⚠️ Critical Issues to Detect

### High Priority
- [ ] **90-degree rotation errors** - Like the Chest/Head bug we just fixed
- [ ] **Flipped joints** - Left/right confusion
- [ ] **Temporal drift** - Accumulating errors over time
- [ ] **Scale inconsistencies** - Wrong unit conversions

### Medium Priority
- [ ] **Jitter/noise** - Excessive frame-to-frame variation
- [ ] **Foot sliding** - Ground contact violations
- [ ] **Bone stretching** - Non-rigid deformations
- [ ] **Gimbal lock** - Rotation singularities

### Low Priority
- [ ] **Minor angle errors** - Small rotation inaccuracies
- [ ] **Smoothing artifacts** - Over/under smoothing
- [ ] **Edge case handling** - Occlusions, partial visibility

## 📈 Reporting and Analysis

### Metrics Dashboard
- [ ] Overall accuracy score (weighted composite)
- [ ] Per-joint error breakdown
- [ ] Temporal consistency graphs
- [ ] Visual correctness indicators

### Comparison Reports
- [ ] Before/after comparisons for changes
- [ ] Cross-model validation results
- [ ] Performance vs accuracy trade-offs

### Debug Visualizations
- [ ] Error heatmaps
- [ ] Motion trajectory plots
- [ ] Frame-by-frame difference videos
- [ ] 3D skeleton overlays

## 🚀 Implementation Priority

### Week 1: Foundation
1. Fix current tester to detect visual errors (not just numerical)
2. Add re-projection validation
3. Implement basic visual checks

### Week 2: Core Metrics
1. Implement MPJPE and PA-MPJPE
2. Add rotation error metrics
3. Create temporal consistency checks

### Week 3: Visualization
1. Build 2D overlay tool
2. Create 3D viewer
3. Generate error heatmaps

### Week 4: Cross-Validation
1. Integrate at least one alternative model
2. Create comparison pipeline
3. Build reporting system

## 📝 Lessons Learned from Current Issues

1. **Accuracy scores can be misleading** - High scores don't guarantee visual correctness
2. **Always validate visually** - Check the actual BVH output in a 3D viewer
3. **Test specific joint chains** - Torso chain (Chest→Neck→Head) needs special attention
4. **Rotation order matters** - XYZ vs ZYX can produce very different results
5. **Scale and coordinate systems** - Ensure consistent transformations throughout pipeline

## 🔧 Quick Testing Commands

```bash
# Visual validation
python bvh_viewer.py --bvh output.bvh --video input.mp4 --overlay

# Numerical validation
python accuracy_tester.py --bvh output.bvh --video input.mp4 --metrics all

# Cross-model validation
python cross_validate.py --video input.mp4 --models mediapipe,openpose,mmpose

# Regression testing
python regression_test.py --before old.bvh --after new.bvh --video input.mp4
```

## 📌 Key Takeaway

**The goal is not to maximize accuracy scores, but to produce visually correct and anatomically plausible BVH files that maintain temporal consistency.**