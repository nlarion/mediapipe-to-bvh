# MediaPipe to BVH Converter

Convert videos to BVH (Biovision Hierarchy) motion capture files using MediaPipe pose estimation. Outputs a Mixamo-compatible skeleton suitable for use in Unity, Godot, or Blender.

## Installation

```bash
pip install -r requirements.txt
```

**Dependencies:** mediapipe, numpy, opencv-python

**Optional (for accuracy testing):** openai or google-generativeai, Pillow

## Usage

### Single Video

```bash
python bvh_converter.py --video videos/example.mp4 --output bvh/example.bvh
```

**Options:**
- `--video` (required) - Path to input video
- `--output` (required) - Path to output BVH file
- `--preview` - Show a pose detection preview window during processing
- `--sample-rate N` - Process every Nth frame (default: 1)
- `--face` - Enable FaceMesh for more accurate head orientation

### Batch Conversion

Process all `.mp4` files in the `videos/` directory:

```bash
python batch_convert.py
```

**Options:**
- `--compare` - Generate side-by-side comparison videos (original + BVH skeleton)
- `--compare-only` - Only generate comparison videos, skip BVH conversion

### Accuracy Testing

Compare a BVH file against its source video using AI vision (GPT-4o or Gemini):

```bash
python bvh_accuracy_tester.py --video videos/example.mp4 --bvh bvh/example.bvh --provider openai
```

**Options:**
- `--video` (required) - Source video
- `--bvh` (required) - BVH file to evaluate
- `--provider` - `openai` (default) or `gemini`
- `--model` - Override the default model (gpt-4o / gemini-2.5-flash)
- `--sample-rate N` - Analyze every Nth frame
- `--max-frames N` - Cap the number of frames analyzed
- `--output-dir DIR` - Save comparison images
- `--api-key KEY` - API key (or set `OPENAI_API_KEY` / `GEMINI_API_KEY` env vars)

## Files

```
mediapipe-to-bvh/
├── bvh_converter.py         # Main pipeline: video -> pose extraction -> BVH
├── mediapipe_extractor.py   # MediaPipe pose detection & frame interpolation
├── skeleton_mapper.py       # Mixamo skeleton hierarchy & landmark mapping
├── math_utils.py            # Rotation math, smoothing, joint constraints
├── config.py                # Processing, smoothing, and output settings
├── batch_convert.py         # Batch-convert all videos/ to bvh/
├── bvh_accuracy_tester.py   # AI vision-based accuracy analysis
├── requirements.txt         # Python dependencies
├── videos/                  # Input video files
├── bvh/                     # Output BVH files
└── comparison_videos/       # Side-by-side comparison renders
```

## Pipeline

```
Video (.mp4)
  └─ mediapipe_extractor.py  (pose detection per frame)
       └─ bvh_converter.py   (rotation calculation + BVH writing)
            ├─ skeleton_mapper.py  (joint hierarchy & offsets)
            ├─ math_utils.py       (vector/rotation math)
            └─ config.py           (tuning parameters)
```

## Output

- Standard BVH text format with Mixamo-named skeleton (`mixamorig:` prefix)
- 24-bone hierarchy: Hips, Spine, Spine2, Neck, Head, L/R Shoulder + Arm + ForeArm + Hand, L/R UpLeg + Leg + Foot + ToeBase
- Euler XYZ rotations, configurable FPS (default 12)
