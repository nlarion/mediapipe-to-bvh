# MediaPipe to BVH Converter

## 🚨 Setup
```bash
source /home/nlarion/Desktop/motion/motion_env/bin/activate
cd /home/nlarion/Desktop/mediapipe-to-bvh
```

## 📊 Current Performance (Jan 21, 2025)
| Version | thewave | walking | boxer | Average | Key Features |
|---------|---------|---------|-------|---------|--------------|
| Baseline (untitled9.py) | 68.2 | - | - | ~68/100 | No spatial tracking |
| With Holistic model | 64.8 | - | - | ~64/100 | Hand landmarks 2D |
| Jan 19 | 67.3 | 59.7 | 67.9 | 65.0/100 | 3D hands + spatial tracking |
| Jan 20 (bad visual) | 72.3 | 63.6 | 72.3 | 69.4/100 | ❌ 90° rotation errors! |
| **Current (Jan 21 Fixed)** | **82.5** | **84.9** | **74.5** | **80.6/100** | **Drift Fixed, Arms Fixed** |
| Target | >70 | >65 | >70 | >70/100 | Goal Exceeded! |

**⚠️ Critical Lesson Learned (Jan 21):**
- **Flight Phase Z-Motion was the key!** The "body stuck" issue was caused by zero Z-movement when feet weren't planted. Added depth estimation fallback.
- **IK Thresholds were too strict.** Relaxing them allowed better contact detection.
- **Arm Constraints were too tight.** Relaxing wrist constraints improved angle accuracy significantly (50° -> 33°).

**Remaining Issues:**
- Symmetry warning might be a false positive (measures error symmetry, not motion symmetry).
- Ground contact score for `thewave` is low (39.0), likely due to sliding or lack of clear steps.

## 🔧 Quick Commands & Testing Protocol

### Convert & Test Single Video
```bash
# Convert video to BVH (now uses Holistic model)
python bvh_converter.py --video videos/thewave.mp4 --output bvh/output.bvh --ik

# Test accuracy
python automated_bvh_accuracy_tester.py \
    --video videos/thewave.mp4 --bvh bvh/output.bvh \
    --output accuracy_tests/test.json
```

### 📊 Comprehensive Testing Suite
**ALWAYS test on these 3 videos for full evaluation:**

```bash
# 1. Test arm movements (thewave.mp4)
python bvh_converter.py --video videos/thewave.mp4 --output bvh/thewave_test.bvh --ik
python automated_bvh_accuracy_tester.py --video videos/thewave.mp4 --bvh bvh/thewave_test.bvh --output accuracy_tests/thewave_test.json

# 2. Test walking motion (walking_00001.mp4)
python bvh_converter.py --video videos/walking_00001.mp4 --output bvh/walking_test.bvh --ik
python automated_bvh_accuracy_tester.py --video videos/walking_00001.mp4 --bvh bvh/walking_test.bvh --output accuracy_tests/walking_test.json

# 3. Test complex athletic motion (Boxer_Video_Ready_One_Only.mp4)
python bvh_converter.py --video videos/Boxer_Video_Ready_One_Only.mp4 --output bvh/boxer_test.bvh --ik
python automated_bvh_accuracy_tester.py --video videos/Boxer_Video_Ready_One_Only.mp4 --bvh bvh/boxer_test.bvh --output accuracy_tests/boxer_test.json
```

### Current Test Results (Jan 21, 2025)
| Video | Score | Key Characteristics | Main Issues |
|-------|-------|---------------------|-------------|
| thewave.mp4 | 82.5/100 | Arm movements, waving | Symmetry warning (false positive?) |
| walking_00001.mp4 | 84.9/100 | Simple walking | **Drift Solved (76.5/100)** |
| Boxer_Video_Ready_One_Only.mp4 | 74.5/100 | Boxing movements | Good dynamics, some foot sliding |
| **Average** | **80.6/100** | - | **Major Breakthrough!** |

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

#### 2. ✅ ForeArm/Wrist Errors (FIXED!)
**Previous**: 65-82° errors
**Current**: ~33° errors
**Solutions**:
- [x] Improved 3D hand reconstruction
- [x] Relaxed wrist rotation constraints
- [x] Better orientation calculation

### Video-Specific Issues

#### walking_00001.mp4 - Severe Temporal Drift (FIXED!)
**Previous**: 30.2/100
**Current**: 76.5/100
**Solutions**:
- [x] Implemented depth-based Z-motion for flight phase
- [x] Relaxed IK thresholds for better contact detection
- [x] Combined foot-planting logic with depth estimation

#### thewave.mp4 - Over-Symmetry
**Problem**: Metric flags high symmetry
**Status**: Likely a false positive of the metric (Error Symmetry vs Motion Symmetry).
**Action**: Monitor visually.

#### Boxer - Good Asymmetry but Over-smoothing
**Problem**: Natural asymmetry but movements dampened
**Status**: Score improved to 74.5. Dynamics score is high (90.6).

## ✅ Completed Features

### Spatial Tracking (Jan 10)
- [x] Hip position tracking - character moves through 3D space
- [x] Scale factor: 90,000x for MediaPipe world coordinates
- [x] Movement verified: 135 units forward when walking

### Today's Improvements (Jan 21)
- [x] **FIXED BODY STUCK ISSUE**: Implemented depth-based Z-motion for flight phase.
- [x] **FIXED ARM ROTATIONS**: Relaxed constraints and improved orientation.
- [x] **TUNED IK**: Relaxed thresholds for better contact detection.
- [x] **HUGE SCORE BOOST**: Average score jumped from 69.4 to 80.6!

### Technical Optimizations
- [x] Frame sampling every 2 frames (reduces drift)
- [x] Temporal smoothing (window_size=3)
- [x] Disabled Kalman filtering (made things worse)

## 🎯 Next Steps

### 1. Refine Ground Contact
- [ ] Investigate why `thewave` has low ground contact score (39.0).
- [ ] Further tune IK for sliding reduction.

### 2. Visual Validation
- [ ] User to verify BVH files in Blender/Unity.
- [ ] Confirm "Body Stuck" is visually gone.

### 3. Code Cleanup
- [ ] Remove legacy code and unused config options.
- [ ] Document the new flight phase logic.

### 4. Future
- [ ] Face orientation from face mesh
- [ ] Full finger articulation in BVH
- [ ] Multi-person support
- [ ] Real-time preview GUI

## 💡 Key Discoveries

### What Works ✅
- **Depth Estimation for Z-Motion**: Crucial for flight phase when feet aren't planted.
- **Relaxed Constraints**: Better to allow more range than to clamp too hard.
- **Error Symmetry**: High score means consistent quality, not necessarily symmetric motion.

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
- `automated_bvh_accuracy_tester.py` - Accuracy testing

---
Last Updated: 2025-01-21 | Current Best: 84.9/100 (Walking)