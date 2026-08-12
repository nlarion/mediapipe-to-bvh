#!/usr/bin/env python3
"""Build a side-by-side contact sheet: video+MediaPipe overlay vs BVH skeleton.

Reuses AccuracyTester's renderers but skips the vision-API call, so it is free
and offline. Use it to see whether a bad BVH is a tracking failure (overlay is
already wrong on the video) or a conversion failure (overlay is right, skeleton
is not).

Usage:
  python make_side_by_side.py --video videos/x.mp4 --bvh bvh/x.bvh \
      --output sheet.png [--frames 8]
"""
import argparse
import cv2
import mediapipe as mp
import numpy as np

from bvh_accuracy_tester import AccuracyTester, BVHParser


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--video', required=True)
    ap.add_argument('--bvh', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--frames', type=int, default=8)
    ap.add_argument('--cols', type=int, default=4)
    args = ap.parse_args()

    # Bypass __init__ so no vision-API key is needed, wiring up only the
    # MediaPipe pieces the renderers actually use.
    tester = AccuracyTester.__new__(AccuracyTester)
    tester.provider = None
    tester.mp_pose = mp.solutions.pose
    tester.pose = tester.mp_pose.Pose(
        static_image_mode=True, model_complexity=2, min_detection_confidence=0.5)

    bvh = BVHParser()
    bvh.parse_file(args.bvh)

    cap = cv2.VideoCapture(args.video)
    video_frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        video_frames.append(frame)
    cap.release()

    # The BVH is trimmed relative to the video, so anchor both at their ends
    # and step through in proportion rather than assuming index alignment.
    n_bvh = bvh.frames
    n_vid = len(video_frames)
    picks = np.linspace(0, 1, args.frames + 2)[1:-1]

    tiles = []
    for t in picks:
        vf = video_frames[min(int(t * n_vid), n_vid - 1)]
        overlay, detected = tester.draw_mediapipe_overlay(vf.copy())
        if not detected:
            cv2.putText(overlay, "NO POSE DETECTED", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        render = tester.render_bvh_frame(bvh, min(int(t * n_bvh), n_bvh - 1))
        tiles.append(tester.create_comparison_image(overlay, render))

    h = min(t.shape[0] for t in tiles)
    tiles = [cv2.resize(t, (int(t.shape[1] * h / t.shape[0]), h)) for t in tiles]
    w = min(t.shape[1] for t in tiles)
    tiles = [t[:, :w] for t in tiles]

    rows = [np.hstack(tiles[i:i + args.cols])
            for i in range(0, len(tiles), args.cols)]
    rw = min(r.shape[1] for r in rows)
    sheet = np.vstack([r[:, :rw] for r in rows])

    cv2.imwrite(args.output, sheet)
    print(f'wrote {args.output}  ({len(tiles)} pairs, bvh={n_bvh}f video={n_vid}f)')


main()
