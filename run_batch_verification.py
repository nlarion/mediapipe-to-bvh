
import os
import sys
import glob
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# Add current directory to path
sys.path.append(os.getcwd())

from bvh_converter import ImprovedBVHConverter
from automated_bvh_accuracy_tester import ImprovedBVHAccuracyAnalyzer
from mediapipe_extractor import MediaPipeExtractor

def run_batch_verification(video_patterns):
    """Run verification on multiple videos matching patterns"""
    
    # Find all matching videos
    video_files = []
    for pattern in video_patterns:
        matches = glob.glob(os.path.join("videos", pattern))
        video_files.extend(matches)
    
    # Remove duplicates and sort
    video_files = sorted(list(set(video_files)))
    
    if not video_files:
        print("No videos found matching patterns")
        return

    print(f"Found {len(video_files)} videos to process:")
    for v in video_files:
        print(f"  - {v}")
        
    results = []
    
    for video_path in video_files:
        print(f"\n{'='*50}")
        print(f"Processing {video_path}...")
        print(f"{'='*50}")
        
        video_name = Path(video_path).stem
        output_path = f"test_output/{video_name}_improved.bvh"
        
        try:
            # 1. Extract Pose Frames
            print("Extracting poses...")
            with MediaPipeExtractor() as extractor:
                pose_frames = extractor.extract_from_video(video_path)
            
            if not pose_frames:
                print(f"Failed to extract pose frames for {video_name}")
                continue

            # 2. Convert to BVH
            print("Converting to BVH...")
            converter = ImprovedBVHConverter(enable_ik=True)
            success = converter.convert(pose_frames, output_path)
            
            if not success:
                print(f"Conversion failed for {video_name}")
                continue
                
            # 3. Analyze Accuracy
            print("Analyzing accuracy...")
            analyzer = ImprovedBVHAccuracyAnalyzer()
            mp_positions, mp_rotations = analyzer.extract_mediapipe_data(video_path)
            bvh_data = analyzer.parse_bvh(output_path)
            metrics = analyzer.compare_motion_improved(bvh_data, mp_positions, mp_rotations)
            
            # Store results
            results.append({
                'Video': video_name,
                'Overall Score': metrics.overall_accuracy_score,
                'Trajectory': metrics.trajectory_score,
                'Knee Stability': metrics.knee_stability_score,
                'Foot Skate': metrics.ground_contact_score,
                'Temporal Drift': metrics.temporal_drift_score,
                'Visual Naturalness': metrics.visual_naturalness_score
            })
            
            print(f"Done! Score: {metrics.overall_accuracy_score:.2f}")
            
        except Exception as e:
            print(f"Error processing {video_name}: {str(e)}")
            import traceback
            traceback.print_exc()
            
    # Create summary report
    if results:
        df = pd.DataFrame(results)
        print("\n" + "="*50)
        print("BATCH VERIFICATION SUMMARY")
        print("="*50)
        print(df.to_string(index=False, float_format=lambda x: "{:.2f}".format(x)))
        
        # Save to CSV
        df.to_csv("test_output/batch_results.csv", index=False)
        print(f"\nResults saved to test_output/batch_results.csv")
        
        # Calculate averages
        print("\nAVERAGE METRICS:")
        print(df.mean(numeric_only=True).apply(lambda x: "{:.2f}".format(x)))

if __name__ == "__main__":
    # List of videos to test
    patterns = [
        "fighter_001.mp4",
        "fighter_002.mp4",
        "Boxer_Video_Ready_One_Only.mp4",
        "thewave.mp4",
        "walking_00001.mp4" # Baseline
    ]
    run_batch_verification(patterns)
