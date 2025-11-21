
import os
import sys
from pathlib import Path
import numpy as np

# Add current directory to path
sys.path.append(os.getcwd())

from bvh_converter import ImprovedBVHConverter
from automated_bvh_accuracy_tester import ImprovedBVHAccuracyAnalyzer
from mediapipe_extractor import MediaPipeExtractor

def verify():
    test_videos = [
        "walking_00002.mp4"
    ]
    
    results = []

    print(f"Starting verification on {len(test_videos)} videos...")
    print("-" * 60)

    for video_file in test_videos:
        video_path = os.path.join("videos", video_file)
        output_path = os.path.join("test_output", f"{os.path.splitext(video_file)[0]}_improved.bvh")
        
        if not os.path.exists(video_path):
            print(f"Warning: Video file not found: {video_path}")
            continue

        print(f"\nProcessing {video_file}...")
        
        # 1. Extract Pose Frames
        try:
            with MediaPipeExtractor() as extractor:
                pose_frames = extractor.extract_from_video(video_path)
        except Exception as e:
            print(f"  Error extracting frames: {e}")
            continue
        
        if not pose_frames:
            print("  Failed to extract pose frames")
            continue

        # 2. Convert to BVH with Improvements
        print("  Converting to BVH...")
        converter = ImprovedBVHConverter(enable_ik=True)
        success = converter.convert(pose_frames, output_path)
        
        if not success:
            print("  Conversion failed")
            continue
            
        # 3. Analyze Accuracy
        print("  Analyzing accuracy...")
        analyzer = ImprovedBVHAccuracyAnalyzer()
        
        try:
            # Extract MediaPipe data for comparison
            mp_positions, mp_rotations = analyzer.extract_mediapipe_data(video_path)
            
            # Parse generated BVH
            bvh_data = analyzer.parse_bvh(output_path)
            
            # Compare
            metrics = analyzer.compare_motion_improved(bvh_data, mp_positions, mp_rotations)
            
            results.append({
                "video": video_file,
                "metrics": metrics,
                "foot_contacts": len(converter.foot_contact_frames)
            })
            print(f"  > Score: {metrics.overall_accuracy_score:.2f} | Skate: {metrics.ground_contact_score:.2f}")
            
        except Exception as e:
            print(f"  Error during analysis: {e}")

    # Print Summary Table
    print("\n" + "="*85)
    print(f"{'Video':<30} | {'Overall':<8} | {'Natural':<8} | {'Skate':<8} | {'Knee':<8} | {'Traj':<8}")
    print("-" * 85)
    
    for res in results:
        m = res["metrics"]
        print(f"{res['video']:<30} | {m.overall_accuracy_score:<8.2f} | {m.visual_naturalness_score:<8.2f} | {m.ground_contact_score:<8.2f} | {m.knee_stability_score:<8.2f} | {m.trajectory_score:<8.2f}")
    print("="*85)

if __name__ == "__main__":
    verify()
