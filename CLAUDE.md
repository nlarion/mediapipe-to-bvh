# MediaPipe to BVH Converter - Project Notes

## Environment Setup

**Python Virtual Environment:**
```bash
source /home/nlarion/Desktop/nlp_html_ads/nlp_html_env/bin/activate
```

Always activate this venv before running any Python scripts in this project.

## Key Files

- `bvh_converter.py` - Main converter from video to BVH
- `visual_accuracy_tester.py` - Visual comparison of video frames vs BVH skeleton
- `render_bvh_only.py` - Side-by-side comparison of two BVH files
- `automated_tuner.py` - Numerical metrics for BVH quality analysis
- `skeleton_mapper.py` - Maps MediaPipe landmarks to Mixamo skeleton

## Testing Videos

Videos are in `videos/` directory. Generated BVH files go to `bvh/` directory.

## Known Issues Being Worked On

- Head tracking issues (head tilt, flopping)
- Shoulder alignment (uneven shoulders)
- Frame analysis outputs in `frame_analysis/` directory
