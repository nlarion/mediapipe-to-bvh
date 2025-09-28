# BVH Accuracy Tester Analysis

## How the Tester Works

The `automated_bvh_accuracy_tester_improved.py` evaluates BVH accuracy by:

1. **Extracting Data**:
   - Reads BVH file and extracts joint positions/rotations for each frame
   - Processes the same video with MediaPipe to get ground truth poses
   - Aligns the two datasets frame by frame

2. **Calculating Metrics** (weights in overall score):
   - **Angle Accuracy** (25%): Joint angle differences between BVH and MediaPipe
   - **Position Accuracy** (15%): Relative position errors (hip-relative)
   - **Bone Consistency** (15%): How consistent bone lengths are across frames
   - **Motion Smoothness** (10%): Jerk metric - currently broken!
   - **Visual Quality** (35%): Naturalness, drift, smoothing, ground contact

3. **Overall Score**: Weighted average of all metrics (0-100 scale)

## Issues Found

### 1. Motion Smoothness Always Returns 0
**Problem**: The jerk calculation produces very high values, causing the score to be negative, then clamped to 0.

**Line 716**:
```python
smoothness_score = max(0, 100 - np.mean(position_jitter_scores) * 10)
```

**Fix Needed**:
- Scale the jerk values appropriately
- Or use a different smoothness metric (e.g., acceleration variance)

### 2. Rotation Calculation is Simplified
**Line 368**: Rotations are set to identity matrices
```python
rotations[joint_name] = np.eye(3)  # Identity for now
```

**Impact**: Not actually comparing rotations, only positions

### 3. Missing Implementations
- `acceleration_consistency` (line 759): Always 0
- `angular_velocity_correlation` (line 761): Always 0

### 4. Potential Issues

#### Sample Rate Mismatch
- Tester uses `sample_rate=1` (every frame)
- BVH converter uses `sample_rate=2` (every other frame)
- This could cause frame alignment issues

#### Joint Mapping Assumptions
- Assumes certain joints exist in BVH that might not
- No handling for missing joints in comparison

#### Confidence Calculation
- Visual confidence is arbitrary (80% if >60 frames)
- Doesn't account for actual detection quality

## Recommended Improvements

### High Priority
1. **Fix Motion Smoothness Calculation**:
   - Normalize jerk values before scoring
   - Consider using velocity variance instead
   - Add proper scaling factor based on data analysis

2. **Fix Frame Alignment**:
   - Ensure sample rates match between BVH and MediaPipe
   - Add frame interpolation if needed

3. **Implement Rotation Comparison**:
   - Actually calculate rotations from MediaPipe poses
   - Compare rotation matrices properly

### Medium Priority
1. **Add Missing Metrics**:
   - Implement acceleration consistency
   - Add angular velocity correlation

2. **Improve Joint Mapping**:
   - Handle missing joints gracefully
   - Add confidence weighting per joint

3. **Better Error Reporting**:
   - Show which specific joints/frames are problematic
   - Add per-joint confidence scores

### Low Priority
1. **Visualization Improvements**:
   - Add 3D pose comparison plots
   - Show frame-by-frame error graphs

2. **Performance Optimization**:
   - Cache MediaPipe results for repeated tests
   - Parallelize frame processing

## Quick Fixes We Can Make Now

1. **Smoothness Score Scaling**:
```python
# Replace line 716
smoothness_score = max(0, 100 - min(100, np.mean(position_jitter_scores) * 0.1))
```

2. **Sample Rate Fix**:
```python
# When calling extract_mediapipe_data, use sample_rate=2
positions_mp, rotations_mp = analyzer.extract_mediapipe_data(video_path, sample_rate=2)
```

3. **Add Debug Output**:
```python
# Add after jerk calculation
print(f"Debug: Jerk values range: {np.min(jerk):.3f} to {np.max(jerk):.3f}")
```

## Conclusion

The tester provides valuable metrics but has several issues that affect accuracy:
- Motion smoothness is broken (always 0)
- Rotations aren't actually compared
- Frame alignment might be off

Fixing the smoothness calculation and frame alignment would immediately improve testing reliability.