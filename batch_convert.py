import os
import glob
import subprocess
from pathlib import Path

def batch_convert():
    """
    Convert all .mp4 videos in the 'videos' directory to BVH format.
    Uses the improved bvh_converter.py with IK enabled.
    """
    # Ensure output directory exists
    output_dir = "bvh"
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all MP4 videos
    video_files = glob.glob("videos/*.mp4")
    
    if not video_files:
        print("No .mp4 files found in 'videos' directory.")
        return

    print(f"Found {len(video_files)} videos to process.")
    print("-" * 50)

    success_count = 0
    fail_count = 0

    for video_path in video_files:
        video_name = Path(video_path).stem
        output_path = os.path.join(output_dir, f"{video_name}.bvh")
        
        print(f"Processing: {video_name}...")
        
        # Construct command
        # Using --ik flag as per latest improvements
        cmd = [
            "python", "bvh_converter.py",
            "--video", video_path,
            "--output", output_path,
            "--ik"
        ]
        
        try:
            # Run converter
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                print(f"✅ Success: {output_path}")
                success_count += 1
            else:
                print(f"❌ Failed: {video_name}")
                print(f"Error output:\n{result.stderr}")
                fail_count += 1
                
        except Exception as e:
            print(f"❌ Error running command for {video_name}: {e}")
            fail_count += 1
            
        print("-" * 50)

    print(f"\nBatch processing complete.")
    print(f"Successful: {success_count}")
    print(f"Failed: {fail_count}")

if __name__ == "__main__":
    batch_convert()
