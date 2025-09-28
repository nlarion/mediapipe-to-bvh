#!/usr/bin/env python3
"""
Test script to demonstrate using MediaPipe Holistic for better hand tracking.
This could solve the ForeArm rotation issue by providing full hand landmarks.
"""

import cv2
import mediapipe as mp
import numpy as np

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

def calculate_hand_orientation(hand_landmarks):
    """Calculate hand orientation from 21 landmarks."""
    if hand_landmarks is None:
        return None

    # Get key points for hand plane
    wrist = np.array([hand_landmarks.landmark[0].x,
                      hand_landmarks.landmark[0].y,
                      hand_landmarks.landmark[0].z])

    index_mcp = np.array([hand_landmarks.landmark[5].x,
                          hand_landmarks.landmark[5].y,
                          hand_landmarks.landmark[5].z])

    pinky_mcp = np.array([hand_landmarks.landmark[17].x,
                          hand_landmarks.landmark[17].y,
                          hand_landmarks.landmark[17].z])

    # Calculate hand plane normal
    v1 = index_mcp - wrist
    v2 = pinky_mcp - wrist
    normal = np.cross(v1, v2)

    if np.linalg.norm(normal) > 0:
        normal = normal / np.linalg.norm(normal)

        # This normal vector gives us the hand orientation
        # Can be used to calculate proper ForeArm rotation
        return normal

    return None

def test_holistic_on_video(video_path):
    """Test holistic model on a video."""
    cap = cv2.VideoCapture(video_path)

    with mp_holistic.Holistic(
        model_complexity=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as holistic:

        frame_count = 0
        while cap.isOpened() and frame_count < 10:  # Just test first 10 frames
            success, image = cap.read()
            if not success:
                break

            frame_count += 1

            # Process with holistic model
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = holistic.process(image_rgb)

            # Check what we get
            if frame_count == 1:
                print(f"\nFrame {frame_count} results:")
                print(f"- Pose landmarks: {results.pose_landmarks is not None}")
                print(f"- Left hand landmarks: {results.left_hand_landmarks is not None}")
                print(f"- Right hand landmarks: {results.right_hand_landmarks is not None}")

                if results.left_hand_landmarks:
                    print(f"  Left hand has {len(results.left_hand_landmarks.landmark)} landmarks")
                    orientation = calculate_hand_orientation(results.left_hand_landmarks)
                    if orientation is not None:
                        print(f"  Left hand orientation: {orientation}")

                if results.right_hand_landmarks:
                    print(f"  Right hand has {len(results.right_hand_landmarks.landmark)} landmarks")
                    orientation = calculate_hand_orientation(results.right_hand_landmarks)
                    if orientation is not None:
                        print(f"  Right hand orientation: {orientation}")

    cap.release()
    print("\nHolistic model can provide full hand tracking for accurate ForeArm rotations!")

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        video_path = sys.argv[1]
    else:
        video_path = "videos/thewave.mp4"

    print("Testing MediaPipe Holistic Model")
    print("=" * 50)
    test_holistic_on_video(video_path)