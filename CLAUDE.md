# MediaPipe to BVH Converter - Project Notes

## Environment Setup

**Python Virtual Environment:**
```bash
source /home/nlarion/Desktop/nlp_html_ads/nlp_html_env/bin/activate
```

Always activate this venv before running any Python scripts in this project.

## Project Goal

Build an asset pipeline for video game animations:
1. **Body animation** (current focus) - Video → MediaPipe → BVH
2. **Facial animation** (future) - Video → MediaPipe Face Mesh → Blend Shapes

See `FACIAL.md` for the facial animation roadmap.

## Key Files

- `bvh_converter.py` - Main converter (video → BVH)
- `batch_convert.py` - Process all videos in videos/ directory
- `mediapipe_extractor.py` - MediaPipe pose detection
- `skeleton_mapper.py` - Maps MediaPipe landmarks to Mixamo skeleton
- `math_utils.py` - Rotation and smoothing utilities
- `config.py` - Configuration settings

## Usage

**Single video:**
```bash
python bvh_converter.py --video videos/example.mp4 --output bvh/example.bvh
```

**All videos:**
```bash
python batch_convert.py
```

## Current Accuracy Improvements

- Head rotation clamping (physical limits: pitch ±45°, yaw ±80°, roll ±35°)
- Landmark sanity checking (ear distance, nose position validation)
- Rotation outlier rejection (interpolates frames with >45° sudden jumps)
- Shoulder/hip Y leveling in raw landmarks (fixes MediaPipe bias)

## Known Limitations

- fighter_001.mp4 has persistent tracking issues due to pose complexity
- MediaPipe struggles with extreme angles, occlusion, and fast movement
- Some videos may need manual cleanup in Blender

## Output

- BVH files go to `bvh/` directory
- Compatible with Unity, Godot, Blender
- Uses Mixamo skeleton hierarchy
