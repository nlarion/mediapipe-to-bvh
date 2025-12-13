import os
import sys
from bvh_converter import ImprovedBVHConverter
from mediapipe_extractor import MediaPipeExtractor

def test_shrug():
    video_path = "videos/shrug.mp4"
    output_path = "bvh/shrug_test.bvh"
    
    if not os.path.exists(video_path):
        print(f"Video not found: {video_path}")
        return

    print(f"Converting {video_path}...")
    
    # Extract
    with MediaPipeExtractor(use_holistic=True) as extractor:
        pose_frames = extractor.extract_from_video(video_path)
    
    # Convert
    converter = ImprovedBVHConverter(enable_ik=True)
    success = converter.convert(pose_frames, output_path)
    
    if success:
        print(f"Conversion successful: {output_path}")
        
        # Verify Collars in output
        with open(output_path, 'r') as f:
            content = f.read()
            if "JOINT LeftCollar" in content and "JOINT RightCollar" in content:
                print("✅ LeftCollar and RightCollar found in BVH.")
            else:
                print("❌ Collars NOT found in BVH.")
                
            if "JOINT LeftShoulder" in content:
                 print("✅ LeftShoulder found in BVH.")
    else:
        print("❌ Conversion failed.")

if __name__ == "__main__":
    test_shrug()
