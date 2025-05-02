#!/usr/bin/env python3
"""
Test script for the MediaPipe to BVH converter
This script demonstrates how to use the converter with a sample video
"""

import os
import sys
import argparse
from mediapipe_to_bvh import MediaPipeToBVH

def main():
    parser = argparse.ArgumentParser(description='Test the MediaPipe to BVH converter')
    parser.add_argument('--video', default=None, help='Path to input video file (if not provided, uses webcam)')
    parser.add_argument('--output', default=None, help='Path to output BVH file (default: output.bvh)')
    parser.add_argument('--fps', type=float, default=30.0, help='Output BVH frame rate (default: 30)')
    
    args = parser.parse_args()
    
    if args.video is None:
        print("No video file provided. Please provide a video file with --video")
        return 1
        
    output_path = args.output or f"{os.path.splitext(os.path.basename(args.video))[0]}.bvh"
    
    try:
        print(f"Processing video: {args.video}")
        print(f"Output will be saved to: {output_path}")
        
        converter = MediaPipeToBVH(fps=args.fps)
        converter.process_video(args.video)
        converter.save_bvh(output_path)
        
        print("Conversion completed successfully!")
        print(f"BVH file saved to: {output_path}")
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
        
    return 0

if __name__ == "__main__":
    sys.exit(main())