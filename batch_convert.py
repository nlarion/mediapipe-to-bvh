import os
import sys
import glob
import subprocess
from pathlib import Path
import argparse

def batch_convert():
    """
    Convert all .mp4 videos in the 'videos' directory to BVH format.
    Uses the improved bvh_converter.py with IK enabled.
    """
    parser = argparse.ArgumentParser(description="Batch convert videos/*.mp4 to BVH")
    parser.add_argument("--face", action="store_true", help="Enable MediaPipe FaceMesh head tracking (passes --face to bvh_converter.py)")
    parser.add_argument("--no-ik", action="store_true", help="Disable IK (do not pass --ik)")
    args = parser.parse_args()

    output_dir = "bvh"
    os.makedirs(output_dir, exist_ok=True)

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

        cmd = [
            sys.executable, "bvh_converter.py",
            "--video", video_path,
            "--output", output_path,
        ]

        if not args.no_ik:
            cmd.append("--ik")

        if args.face:
            cmd.append("--face")

        try:
            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode == 0:
                print(f"✅ Success: {output_path}")
                success_count += 1
            else:
                print(f"❌ Failed: {video_name}")
                if result.stdout:
                    print(f"Stdout:\n{result.stdout}")
                if result.stderr:
                    print(f"Stderr:\n{result.stderr}")
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
