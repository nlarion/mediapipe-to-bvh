#!/usr/bin/env python3
"""
Batch convert all videos in videos/ directory to BVH format.
"""

import os
import sys
import glob
import subprocess
from pathlib import Path


def batch_convert():
    """Convert all .mp4 videos in the 'videos' directory to BVH format."""
    output_dir = "bvh"
    os.makedirs(output_dir, exist_ok=True)

    video_files = sorted(glob.glob("videos/*.mp4"))

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

        print(f"\nProcessing: {video_name}...")

        cmd = [
            sys.executable, "bvh_converter.py",
            "--video", video_path,
            "--output", output_path,
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

            if result.returncode == 0:
                # Extract key info from output
                outliers = ""
                leveled = ""
                for line in result.stdout.split('\n'):
                    if "outlier" in line.lower():
                        outliers = line.strip()
                    if "Leveled" in line:
                        leveled = line.strip()

                if outliers:
                    print(f"  {outliers}")
                if leveled:
                    print(f"  {leveled}")
                print(f"  ✅ Success: {output_path}")
                success_count += 1
            else:
                print(f"  ❌ Failed: {video_name}")
                if result.stderr:
                    err_lines = result.stderr.strip().split('\n')[-3:]
                    for line in err_lines:
                        print(f"    {line}")
                fail_count += 1

        except subprocess.TimeoutExpired:
            print(f"  ❌ Timeout: {video_name} (>10 min)")
            fail_count += 1
        except Exception as e:
            print(f"  ❌ Error: {e}")
            fail_count += 1

    print("-" * 50)
    print(f"\nBatch processing complete!")
    print(f"  Successful: {success_count}/{len(video_files)}")
    if fail_count > 0:
        print(f"  Failed: {fail_count}")


if __name__ == "__main__":
    batch_convert()
