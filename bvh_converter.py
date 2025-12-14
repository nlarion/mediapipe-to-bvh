#!/usr/bin/env python3
"""
MediaPipe to BVH Converter

This file was previously replaced by a placeholder and is now restored to a working implementation.

Key features:
- Converts a video to BVH using MediaPipe Pose (optionally Holistic if available later).
- Stabilizes head/neck rotation using a torso basis and a damped neck delta:
  - Chest global rotation is derived from a stable torso basis (shoulders + hips).
  - Head global rotation is derived from a stabilized face-forward vector (nose relative to shoulder center),
    blended with torso forward to reduce jitter.
  - Neck local rotation is computed as (Chest^-1 * Head) and damped.

Notes:
- This implementation uses SkeletonMapper + math_utils from this repo.
- It does not implement IK foot locking (not provided in chat). It focuses on correctness and head/neck stability.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np
from scipy.spatial.transform import Rotation as R

from skeleton_mapper import SkeletonMapper, BVHJoint
from math_utils import calculate_rotation_from_directions, smooth_rotations, smooth_positions
from config import BVH_CONFIG, PROCESSING_CONFIG, SMOOTHING_CONFIG

mp_pose = mp.solutions.pose


@dataclass
class PoseFrame:
    """Minimal pose frame container (world landmarks only)."""
    world_landmarks: Optional[List]
    valid: bool

    def is_valid(self) -> bool:
        return self.valid and self.world_landmarks is not None


class ImprovedBVHConverter:
    """
    Converter that produces BVH from MediaPipe Pose world landmarks.

    Head/Neck stabilization:
    - Chest: torso basis (left/up/forward) -> global rotation
    - Head: face forward (nose - shoulder_center) blended with torso forward -> global rotation
    - Neck: local = inv(chest_global) * head_global, damped
    """

    def __init__(self):
        self.skeleton_mapper = SkeletonMapper()
        self.scale = float(PROCESSING_CONFIG.get("scale_factor", 100.0))
        self.fps = float(BVH_CONFIG.get("fps", 12))
        self.frame_time = 1.0 / self.fps
        self.rotation_order = str(BVH_CONFIG.get("rotation_order", "XYZ"))

    # ----------------------------
    # Public API
    # ----------------------------
    def convert_video_to_bvh(self, video_path: str, output_path: str, sample_rate: int = 2, preview: bool = False) -> bool:
        frames = self._extract_pose_frames(video_path, sample_rate=sample_rate, preview=preview)
        if not frames:
            print("Error: no frames extracted")
            return False

        # Find a reference frame for offsets
        ref_idx = self._find_reference_frame(frames)
        if ref_idx is None:
            print("Error: no valid reference frame found")
            return False

        ref_landmarks = frames[ref_idx].world_landmarks
        self.skeleton_mapper.calculate_bone_offsets(ref_landmarks, scale=self.scale)

        # Root positions (simple: use hips world position, smoothed)
        hip_positions = self._calculate_root_positions(frames)

        # Rotations per frame
        all_rotations = []
        for fr in frames:
            if fr.is_valid():
                rots = self._calculate_frame_rotations(fr.world_landmarks)
            else:
                rots = self._zero_rotations()
            all_rotations.append(rots)

        # Optional smoothing
        if SMOOTHING_CONFIG.get("enable_temporal_smoothing", True):
            all_rotations = self._smooth_rotations_dict(all_rotations)

        return self._write_bvh(output_path, all_rotations, hip_positions)

    # ----------------------------
    # Extraction
    # ----------------------------
    def _extract_pose_frames(self, video_path: str, sample_rate: int = 2, preview: bool = False) -> List[PoseFrame]:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        pose = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=int(PROCESSING_CONFIG.get("model_complexity", 2)),
            smooth_landmarks=bool(PROCESSING_CONFIG.get("smooth_landmarks", True)),
            min_detection_confidence=float(PROCESSING_CONFIG.get("min_detection_confidence", 0.5)),
            min_tracking_confidence=float(PROCESSING_CONFIG.get("min_tracking_confidence", 0.5)),
        )

        frames: List[PoseFrame] = []
        idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if idx % sample_rate != 0:
                idx += 1
                continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = pose.process(rgb)

            if res.pose_world_landmarks and res.pose_world_landmarks.landmark:
                frames.append(PoseFrame(world_landmarks=res.pose_world_landmarks.landmark, valid=True))
            else:
                frames.append(PoseFrame(world_landmarks=None, valid=False))

            if preview:
                vis = frame.copy()
                if res.pose_landmarks:
                    mp.solutions.drawing_utils.draw_landmarks(
                        vis, res.pose_landmarks, mp_pose.POSE_CONNECTIONS
                    )
                cv2.imshow("MediaPipe Preview", vis)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            idx += 1

        cap.release()
        pose.close()
        if preview:
            cv2.destroyAllWindows()

        return frames

    def _find_reference_frame(self, frames: List[PoseFrame]) -> Optional[int]:
        # Prefer first valid frame with visible shoulders/hips
        required = [
            mp_pose.PoseLandmark.LEFT_HIP,
            mp_pose.PoseLandmark.RIGHT_HIP,
            mp_pose.PoseLandmark.LEFT_SHOULDER,
            mp_pose.PoseLandmark.RIGHT_SHOULDER,
        ]
        for i, fr in enumerate(frames):
            if not fr.is_valid():
                continue
            ok = True
            for idx in required:
                lm = fr.world_landmarks[idx]
                if hasattr(lm, "visibility") and lm.visibility < 0.5:
                    ok = False
                    break
            if ok:
                return i
        # fallback: first valid
        for i, fr in enumerate(frames):
            if fr.is_valid():
                return i
        return None

    # ----------------------------
    # Root motion
    # ----------------------------
    def _calculate_root_positions(self, frames: List[PoseFrame]) -> List[np.ndarray]:
        positions: List[np.ndarray] = []
        for fr in frames:
            if not fr.is_valid():
                if positions:
                    positions.append(positions[-1].copy())
                else:
                    positions.append(np.array([0.0, float(BVH_CONFIG.get("root_height", 60.0)), 0.0], dtype=float))
                continue

            hips = self.skeleton_mapper.get_joint_position("Hips", fr.world_landmarks, scale=self.scale)
            if hips is None:
                if positions:
                    positions.append(positions[-1].copy())
                else:
                    positions.append(np.array([0.0, float(BVH_CONFIG.get("root_height", 60.0)), 0.0], dtype=float))
            else:
                positions.append(hips.astype(float))

        if SMOOTHING_CONFIG.get("enable_temporal_smoothing", True):
            arr = np.array(positions, dtype=float)
            arr = smooth_positions(
                arr,
                window_size=int(SMOOTHING_CONFIG.get("temporal_window_size", 3)),
                preserve_dynamics=bool(SMOOTHING_CONFIG.get("preserve_dynamics", True)),
                preserve_y_axis=False,
            )
            positions = [p for p in arr]
        return positions

    # ----------------------------
    # Rotation helpers
    # ----------------------------
    @staticmethod
    def _safe_normalize(v: np.ndarray, eps: float = 1e-10) -> Optional[np.ndarray]:
        n = float(np.linalg.norm(v))
        if n < eps:
            return None
        return v / n

    def _calculate_torso_basis(self, landmarks) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Build a stable torso basis in BVH-like space:
        - left axis: +X (left)
        - up axis: +Y (up)
        - forward axis: +Z (forward)
        """
        try:
            l_sh = np.array([landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x,
                             -landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y,
                             landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].z], dtype=float)
            r_sh = np.array([landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x,
                             -landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y,
                             landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].z], dtype=float)
            l_hip = np.array([landmarks[mp_pose.PoseLandmark.LEFT_HIP].x,
                              -landmarks[mp_pose.PoseLandmark.LEFT_HIP].y,
                              landmarks[mp_pose.PoseLandmark.LEFT_HIP].z], dtype=float)
            r_hip = np.array([landmarks[mp_pose.PoseLandmark.RIGHT_HIP].x,
                              -landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y,
                              landmarks[mp_pose.PoseLandmark.RIGHT_HIP].z], dtype=float)

            sh_center = (l_sh + r_sh) / 2.0
            hip_center = (l_hip + r_hip) / 2.0

            left_axis = self._safe_normalize(l_sh - r_sh)  # +X = left
            up_axis = self._safe_normalize(sh_center - hip_center)  # +Y = up
            if left_axis is None or up_axis is None:
                return None

            forward_axis = self._safe_normalize(np.cross(left_axis, up_axis))
            if forward_axis is None:
                return None

            # Re-orthogonalize up
            up_axis = self._safe_normalize(np.cross(forward_axis, left_axis))
            if up_axis is None:
                return None

            return left_axis, up_axis, forward_axis
        except Exception:
            return None

    def _rotation_from_basis(self, left_axis: np.ndarray, up_axis: np.ndarray, forward_axis: np.ndarray) -> Optional[np.ndarray]:
        """
        Return global Euler XYZ (degrees) from basis vectors.
        Basis columns correspond to BVH axes: X=Left, Y=Up, Z=Forward.
        """
        try:
            rot_mat = np.column_stack((left_axis, up_axis, forward_axis))
            euler = R.from_matrix(rot_mat).as_euler("XYZ", degrees=True)
            return euler.astype(float)
        except Exception:
            return None

    def _calculate_head_global_euler(self, landmarks) -> Optional[np.ndarray]:
        """
        Head global rotation:
        - Use torso basis for stability.
        - Use face forward (nose - shoulder_center) blended with torso forward.
        """
        torso_basis = self._calculate_torso_basis(landmarks)
        if torso_basis is None:
            return None
        torso_left, torso_up, torso_forward = torso_basis

        l_sh = np.array([landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x,
                         -landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y,
                         landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].z], dtype=float)
        r_sh = np.array([landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x,
                         -landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y,
                         landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].z], dtype=float)
        sh_center = (l_sh + r_sh) / 2.0

        nose = np.array([landmarks[mp_pose.PoseLandmark.NOSE].x,
                         -landmarks[mp_pose.PoseLandmark.NOSE].y,
                         landmarks[mp_pose.PoseLandmark.NOSE].z], dtype=float)

        face_forward = self._safe_normalize(nose - sh_center)
        if face_forward is None:
            return self._rotation_from_basis(torso_left, torso_up, torso_forward)

        blended_forward = self._safe_normalize(face_forward * 0.6 + torso_forward * 0.4)
        if blended_forward is None:
            blended_forward = torso_forward

        head_left = self._safe_normalize(np.cross(torso_up, blended_forward))
        if head_left is None:
            head_left = torso_left
        head_up = self._safe_normalize(np.cross(blended_forward, head_left))
        if head_up is None:
            head_up = torso_up

        return self._rotation_from_basis(head_left, head_up, blended_forward)

    # ----------------------------
    # Frame rotations
    # ----------------------------
    def _calculate_frame_rotations(self, landmarks) -> Dict[str, np.ndarray]:
        """
        Compute local Euler rotations for all joints.
        For most joints: direction-based global rotation then converted to local.
        For Chest/Neck/Head: stabilized method described above.
        """
        rotations: Dict[str, np.ndarray] = {j.name: np.zeros(3, dtype=float) for j in self.skeleton_mapper.get_all_joints()}
        root = self.skeleton_mapper.skeleton

        torso_basis = self._calculate_torso_basis(landmarks)
        chest_global_euler = None
        chest_global_rot = None
        if torso_basis is not None:
            chest_global_euler = self._rotation_from_basis(*torso_basis)
            if chest_global_euler is not None:
                chest_global_rot = R.from_euler("XYZ", chest_global_euler, degrees=True)

        head_global_euler = self._calculate_head_global_euler(landmarks)
        head_global_rot = None
        if head_global_euler is not None:
            head_global_rot = R.from_euler("XYZ", head_global_euler, degrees=True)

        def get_dir(parent_name: str, child_name: str) -> Optional[np.ndarray]:
            p = self.skeleton_mapper.get_joint_position(parent_name, landmarks, scale=self.scale)
            c = self.skeleton_mapper.get_joint_position(child_name, landmarks, scale=self.scale)
            if p is None or c is None:
                return None
            d = c - p
            n = np.linalg.norm(d)
            if n < 1e-10:
                return None
            return d / n

        def recurse(joint: BVHJoint, parent_global: R):
            local_euler = np.zeros(3, dtype=float)
            global_rot = parent_global

            if joint.name == "Chest" and chest_global_rot is not None:
                # Convert global chest to local
                local_rot = parent_global.inv() * chest_global_rot
                local_euler = local_rot.as_euler("XYZ", degrees=True)
                global_rot = chest_global_rot

            elif joint.name == "Head" and head_global_rot is not None:
                local_rot = parent_global.inv() * head_global_rot
                local_euler = local_rot.as_euler("XYZ", degrees=True)
                global_rot = head_global_rot

            elif joint.name == "Neck" and chest_global_rot is not None and head_global_rot is not None:
                # Neck local = inv(chest_global) * head_global, damped
                neck_local = chest_global_rot.inv() * head_global_rot
                neck_local_euler = neck_local.as_euler("XYZ", degrees=True)
                neck_local_euler *= 0.5  # damping
                local_euler = neck_local_euler
                global_rot = parent_global * R.from_euler("XYZ", local_euler, degrees=True)

            else:
                # Generic: use first child direction if available
                if joint.children:
                    child = joint.children[0]
                    direction = get_dir(joint.name, child.name)
                    if direction is not None and np.linalg.norm(child.offset) > 1e-10:
                        rest_dir = child.offset / (np.linalg.norm(child.offset) + 1e-10)
                        global_euler = calculate_rotation_from_directions(rest_dir, direction, order="XYZ")
                        global_rot_obj = R.from_euler("XYZ", global_euler, degrees=True)
                        local_rot = parent_global.inv() * global_rot_obj
                        local_euler = local_rot.as_euler("XYZ", degrees=True)
                        global_rot = global_rot_obj

            rotations[joint.name] = local_euler.astype(float)

            for ch in joint.children:
                recurse(ch, global_rot)

        recurse(root, R.identity())
        return rotations

    def _zero_rotations(self) -> Dict[str, np.ndarray]:
        return {j.name: np.zeros(3, dtype=float) for j in self.skeleton_mapper.get_all_joints()}

    def _smooth_rotations_dict(self, all_rotations: List[Dict[str, np.ndarray]]) -> List[Dict[str, np.ndarray]]:
        if not all_rotations:
            return all_rotations

        joint_names = list(all_rotations[0].keys())
        out: List[Dict[str, np.ndarray]] = [dict() for _ in range(len(all_rotations))]

        base_window = int(SMOOTHING_CONFIG.get("temporal_window_size", 3))
        preserve = bool(SMOOTHING_CONFIG.get("preserve_dynamics", True))

        # Slightly less smoothing for head/neck to preserve responsiveness
        per_joint_window = {
            "Head": max(1, base_window - 1),
            "Neck": max(1, base_window - 1),
            "Chest": base_window,
        }

        for jn in joint_names:
            seq = np.array([fr[jn] for fr in all_rotations], dtype=float)
            w = per_joint_window.get(jn, base_window)
            sm = smooth_rotations(seq, window_size=w, preserve_dynamics=preserve)
            for i in range(len(all_rotations)):
                out[i][jn] = sm[i]
        return out

    # ----------------------------
    # BVH writing
    # ----------------------------
    def _write_bvh(self, output_path: str, all_rotations: List[Dict[str, np.ndarray]], hip_positions: List[np.ndarray]) -> bool:
        try:
            with open(output_path, "w") as f:
                f.write("HIERARCHY\n")
                self._write_hierarchy(f, self.skeleton_mapper.skeleton, 0)

                f.write("MOTION\n")
                f.write(f"Frames: {len(all_rotations)}\n")
                f.write(f"Frame Time: {self.frame_time:.6f}\n")

                for i in range(len(all_rotations)):
                    frame_data: List[float] = []
                    hip = hip_positions[i]
                    frame_data.extend([float(hip[0]), float(hip[1]), float(hip[2])])

                    self._write_joint_rotations(self.skeleton_mapper.skeleton, all_rotations[i], frame_data)
                    f.write(" ".join(f"{v:.6f}" for v in frame_data) + "\n")
            return True
        except Exception as e:
            print(f"Error writing BVH: {e}")
            return False

    def _write_hierarchy(self, f, joint: BVHJoint, level: int):
        indent = "  " * level
        if level == 0:
            f.write(f"{indent}ROOT {joint.name}\n")
        else:
            f.write(f"{indent}JOINT {joint.name}\n")
        f.write(f"{indent}{{\n")
        f.write(f"{indent}  OFFSET {joint.offset[0]:.6f} {joint.offset[1]:.6f} {joint.offset[2]:.6f}\n")
        if joint.channels:
            f.write(f"{indent}  CHANNELS {len(joint.channels)} {' '.join(joint.channels)}\n")
        for child in joint.children:
            self._write_hierarchy(f, child, level + 1)
        if not joint.children:
            f.write(f"{indent}  End Site\n")
            f.write(f"{indent}  {{\n")
            end_offset = joint.offset * 0.3 if np.linalg.norm(joint.offset) > 0 else np.array([0.0, -5.0, 0.0])
            f.write(f"{indent}    OFFSET {end_offset[0]:.6f} {end_offset[1]:.6f} {end_offset[2]:.6f}\n")
            f.write(f"{indent}  }}\n")
        f.write(f"{indent}}}\n")

    def _write_joint_rotations(self, joint: BVHJoint, frame_rotations: Dict[str, np.ndarray], frame_data: List[float]):
        if joint.name in frame_rotations:
            r = frame_rotations[joint.name]
            if "Xrotation" in joint.channels:
                frame_data.append(float(r[0]))
            if "Yrotation" in joint.channels:
                frame_data.append(float(r[1]))
            if "Zrotation" in joint.channels:
                frame_data.append(float(r[2]))
        for child in joint.children:
            self._write_joint_rotations(child, frame_rotations, frame_data)


def main():
    parser = argparse.ArgumentParser(description="Convert video to BVH using MediaPipe Pose")
    parser.add_argument("--video", required=True, help="Path to input video")
    parser.add_argument("--output", required=True, help="Path to output BVH")
    parser.add_argument("--preview", action="store_true", help="Show MediaPipe preview")
    parser.add_argument("--sample-rate", type=int, default=int(PROCESSING_CONFIG.get("sample_rate", 2)), help="Process every Nth frame")
    args = parser.parse_args()

    converter = ImprovedBVHConverter()
    ok = converter.convert_video_to_bvh(args.video, args.output, sample_rate=args.sample_rate, preview=args.preview)
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
