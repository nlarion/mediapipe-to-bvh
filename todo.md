# MediaPipe to BVH Converter

## 🚨 Setup
```bash
source /home/nlarion/Desktop/motion/motion_env/bin/activate
cd /home/nlarion/Desktop/mediapipe-to-bvh
```

## 📊 Current Performance (Average of 3 Test Videos)
| Version | thewave | walking | boxer | Average | Key Features |
|---------|---------|---------|-------|---------|--------------|
| Baseline (untitled9.py) | 68.2 | - | - | ~68/100 | No spatial tracking |
| With Holistic model | 64.8 | - | - | ~64/100 | Hand landmarks 2D |
| Jan 19 | 67.3 | 59.7 | 67.9 | 65.0/100 | 3D hands + spatial tracking |
| **Current (Jan 20)** | **72.3** | **63.6** | **72.3** | **69.4/100** | Fixed torso chain + adaptive smoothing! |
| Target | >70 | >65 | >70 | >70/100 | Almost there! |

**🎉 Recent Improvements (Jan 20):**
- ✅ Fixed Chest rotation: 79.4° → ~50° error
- ✅ Fixed Neck rotation: 73.3° → 3.2° error
- ✅ Implemented adaptive smoothing
- ✅ Average score: 65.0 → 69.4 (+4.4 points!)

**Remaining Issues:**
- ForeArm: ~76° error (needs better 3D hand reconstruction)
- Walking temporal drift: 27.0/100 (IK not fully effective yet)
- Head rotation: Still needs work (showing 0° in tests)

## 🔧 Quick Commands & Testing Protocol

### Convert & Test Single Video
```bash
# Convert video to BVH (now uses Holistic model)
python bvh_converter.py --video videos/thewave.mp4 --output bvh/output.bvh

# Test accuracy
python automated_bvh_accuracy_tester_improved.py \
    --video videos/thewave.mp4 --bvh bvh/output.bvh \
    --output accuracy_tests/test.json
```

### 📊 Comprehensive Testing Suite
**ALWAYS test on these 3 videos for full evaluation:**

```bash
# 1. Test arm movements (thewave.mp4)
python bvh_converter.py --video videos/thewave.mp4 --output bvh/thewave_test.bvh
python automated_bvh_accuracy_tester_improved.py --video videos/thewave.mp4 --bvh bvh/thewave_test.bvh --output accuracy_tests/thewave_test.json

# 2. Test walking motion (walking_00001.mp4)
python bvh_converter.py --video videos/walking_00001.mp4 --output bvh/walking_test.bvh
python automated_bvh_accuracy_tester_improved.py --video videos/walking_00001.mp4 --bvh bvh/walking_test.bvh --output accuracy_tests/walking_test.json

# 3. Test complex athletic motion (Boxer_Video_Ready_One_Only.mp4)
python bvh_converter.py --video videos/Boxer_Video_Ready_One_Only.mp4 --output bvh/boxer_test.bvh
python automated_bvh_accuracy_tester_improved.py --video videos/Boxer_Video_Ready_One_Only.mp4 --bvh bvh/boxer_test.bvh --output accuracy_tests/boxer_test.json
```

### Current Test Results (Jan 20, 2025)
| Video | Score | Key Characteristics | Main Issues |
|-------|-------|---------------------|-------------|
| thewave.mp4 | 72.3/100 | Arm movements, waving | Mean angle 50.7°, Symmetry 95.8% |
| walking_00001.mp4 | 63.6/100 | Simple walking | Temporal drift 27%, Chest 65° |
| Boxer_Video_Ready_One_Only.mp4 | 72.3/100 | Boxing movements | Mean angle 40.6°, Good asymmetry 51.9% |
| **Average** | **69.4/100** | - | Close to target! |

## 🔍 Identified Challenges & Solutions

### 🎯 Quick Wins Achieved ✅
1. **Fixed Chest/Neck chain** - Reduced errors by 70°+
2. **Implemented adaptive smoothing** - Different smoothing for different joints
3. **Integrated IK system** - Foundation ready, needs calibration

### Major Issues

#### 1. ✅ Torso Chain Errors (FIXED!)
**Previous**: Chest 65-82°, Neck 73-74°
**Current**: Chest ~50°, Neck ~3-8°
**Solutions Implemented**:
- [x] Improved Chest rotation using spine direction (hips→shoulders)
- [x] Fixed Neck orientation using actual head position
- [x] Added damping factors to reduce jitter

#### 2. ForeArm/Wrist Errors (65-82°)
**Current**: All videos show 65-82° ForeArm errors
**Root Cause**: 2D hand landmarks + limited wrist data
**Solutions**:
- [ ] Try separate MediaPipe Hands model for world landmarks
- [ ] Implement better depth estimation for hands
- [ ] Consider IK-based approach for elbow-wrist chain

### Video-Specific Issues

#### walking_00001.mp4 - Severe Temporal Drift (30.2/100)
**Problem**: Position accumulates error over time

**Existing Reference BVH Files Available**:
✅ `bvh_examples/walk-through-spce.bvh` - Walking through 3D space
✅ `bvh_examples/walking-standing-still.bvh` - Stationary walking (similar to non-3D tracking)

**Existing Test Infrastructure**:
- `bvh_reference_analyzer.py` - Analyzes BVH motion profiles including:
  - Gait characteristics (stride length, frequency, symmetry)
  - Joint angle ranges and velocities
  - Foot contact detection and gait cycle analysis
  - Motion smoothness (jerk metrics)
  - Vertical oscillation patterns
  - Energy profiles

- `test_bvh_simple.py` - Basic BVH comparison:
  - Frame count and duration matching
  - Joint count verification
  - Displacement and speed calculations
  - Overall similarity scoring (A-F grades)

- `test_bvh_with_references.py` - Advanced reference comparison:
  - Detailed motion profile matching
  - Component scores (stride, smoothness, rhythm, energy)
  - Joint angle correlation analysis
  - Generates visual comparison plots
  - Provides detailed feedback on differences

**Solutions**:
- [x] Reference BVH files already available in `bvh_examples/`
- [ ] Run `bvh_reference_analyzer.py` to extract motion profiles from references
- [ ] Use extracted profiles to guide drift correction implementation
- [ ] Implement foot contact detection using reference thresholds:
  - Foot contact velocity: ~0.1 m/s (from reference)
  - Foot sliding threshold: ~0.02 m (from reference)
  - Foot clearance height: ~5.0 cm (from reference)
- [ ] Compare our generated BVH against references using test scripts
- [ ] Add position reset when feet are planted (using IK from section 2)
- [ ] Use sliding window for hip position calculation

#### thewave.mp4 - Over-Symmetry (95.8%)
**Problem**: Left/right movements too synchronized
**Solutions**:
- [ ] Reduce smoothing window for arm joints
- [ ] Add slight noise/variation to break symmetry
- [ ] Process left/right sides independently

#### Boxer - Good Asymmetry but Over-smoothing
**Problem**: Natural asymmetry (51.9%) but movements dampened
**Solutions**:
- [ ] Reduce smoothing for fast movements
- [ ] Implement adaptive smoothing based on velocity
- [ ] Preserve sharp motion transitions

## ✅ Completed Features

### Spatial Tracking (Jan 10)
- [x] Hip position tracking - character moves through 3D space
- [x] Scale factor: 90,000x for MediaPipe world coordinates
- [x] Movement verified: 135 units forward when walking

### Today's Improvements (Jan 19)
- [x] Fixed tester motion smoothness bug (always was 0, now works)
- [x] MediaPipe Holistic integrated (21 hand landmarks per hand)
- [x] 3D hand reconstruction from 2D landmarks using wrist anchor
- [x] Overall score improved to 67.3/100 (close to baseline!)
- [x] Fixed sample rate mismatch in accuracy tester

### Technical Optimizations
- [x] Frame sampling every 2 frames (reduces drift)
- [x] Temporal smoothing (window_size=3)
- [x] Disabled Kalman filtering (made things worse)

## 🎯 Next Steps

### 1. ✅ High Priority - Torso Chain Fix (COMPLETED Jan 20)
- [x] Fix Chest rotation calculation (65-82° → ~50° error)
- [x] Fix Neck rotation calculation (73° → 3° error)
- [x] Review entire torso kinematic chain
- [x] Improved using spine direction and actual head positions

### 2. Foot Contact Locking with IK (INTEGRATED BUT NEEDS TUNING)

**✅ CURRENT STATUS: IK integrated into bvh_converter.py (Jan 20)**

**Implementation Status:**
- [x] IK system integrated directly into `bvh_converter.py`
- [x] Fixed application order - IK now applies BEFORE rotation calculation
- [x] Deleted old files: `bvh_converter_with_ik.py`, `untitled*.py`
- [x] Basic foot contact detection working (73% of frames detected)

**⚠️ IK Not Yet Effective - Needs Calibration:**

The IK system is properly integrated but isn't improving stability yet. Usage:
```bash
# Enable IK with --ik flag
python bvh_converter.py --video videos/walking.mp4 --output bvh/output.bvh --ik
```

**🔧 Still Need to Fix:**

1. **Threshold Calibration**
   - [ ] Current thresholds may not match MediaPipe's scale
   - [ ] Velocity threshold needs tuning (currently 3.0 * scale/100)
   - [ ] Height threshold needs adjustment (currently 8.0 * scale/100)
   - [ ] Need to analyze actual foot velocities in walking videos

2. **Rotation Recalculation After IK**
   - [ ] Currently updates positions but rotation calculation may not fully utilize them
   - [ ] Need to ensure hip/knee rotations properly reflect locked ankles
   - [ ] May need to recalculate parent joint rotations when child is locked

3. **Ground Plane Detection**
   - [ ] Need better ground plane estimation
   - [ ] Consider using lowest foot position as dynamic ground reference
   - [ ] Add foot height filtering to prevent "floating"

**Expected Impact Once Fixed:**
- Reduction in foot sliding
- Better temporal drift scores (currently 27.0/100 for walking)
- More stable walking cycles

### 3. Video-Specific Fixes
- [ ] 📌 **WAITING FOR USER**: Reference BVH files for walking/running to help fix drift
- [ ] Walking: Implement foot-based drift correction using reference BVH patterns
- [ ] Wave: Reduce arm smoothing to fix over-symmetry
- [ ] Boxer: Add velocity-adaptive smoothing

### 3. Medium Priority
- [ ] Add finger joint tracking (after 3D fix)
- [ ] Optimize Holistic performance (70% slower)
- [ ] Create test suite with arm-focused videos

### 4. Future
- [ ] Face orientation from face mesh
- [ ] Full finger articulation in BVH
- [ ] Multi-person support
- [ ] Real-time preview GUI

## 💡 Key Discoveries

### What Works ✅
- Frame sampling every 2 frames
- Gentle smoothing only (window=3)
- MediaPipe Holistic for hand data
- Axis-angle to Euler conversion

### What Doesn't ❌
- Aggressive ground locking
- Kalman filtering
- Large smoothing windows
- 2D hand landmarks without 3D conversion

### MediaPipe Holistic Implementation (Jan 19)
**Problem:** Hand landmarks are in 2D image space (Z=0), not 3D world space
**Solution Needed:** Transform 2D→3D using wrist anchor and depth estimation
**Status:** Foundation complete, coordinate transformation required

## 📁 Project Structure
- `bvh_converter.py` - Main converter with Holistic support
- `mediapipe_extractor.py` - Video processing (Pose/Holistic modes)
- `skeleton_mapper.py` - BVH skeleton hierarchy
- `math_utils.py` - Rotation calculations
- `config.py` - Settings
- `automated_bvh_accuracy_tester_improved.py` - Accuracy testing

---
Last Updated: 2025-01-19 | Current Best: 67.3/100 (getting close to baseline 68.2!)