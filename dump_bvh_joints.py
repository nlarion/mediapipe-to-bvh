"""Dump BVH joint world positions per frame to JSON (run inside Blender).

Companion to overlay_debug.py, which compares these against the MediaPipe
landmarks the BVH was built from, so capture error and translation error can
be told apart.

Usage:
  blender --background --python dump_bvh_joints.py -- --input motion.bvh \
      --output joints.json
"""
import bpy
import json
import sys
import argparse


JOINTS = [
    'Hips', 'Spine', 'Spine1', 'Spine2', 'Neck', 'Head',
    'LeftShoulder', 'LeftArm', 'LeftForeArm', 'LeftHand',
    'RightShoulder', 'RightArm', 'RightForeArm', 'RightHand',
    'LeftUpLeg', 'LeftLeg', 'LeftFoot',
    'RightUpLeg', 'RightLeg', 'RightFoot',
]


def parse_args():
    argv = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []
    p = argparse.ArgumentParser()
    p.add_argument('--input', required=True)
    p.add_argument('--output', required=True)
    return p.parse_args(argv)


def main():
    args = parse_args()

    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_anim.bvh(filepath=args.input, update_scene_fps=False,
                            update_scene_duration=True)
    arm = next(o for o in bpy.context.scene.objects if o.type == 'ARMATURE')

    # BVH bone names may or may not carry the mixamorig: prefix.
    resolved = {}
    for j in JOINTS:
        for cand in (j, f'mixamorig:{j}'):
            if cand in arm.pose.bones:
                resolved[j] = cand
                break

    scene = bpy.context.scene
    f0, f1 = scene.frame_start, scene.frame_end
    frames = []
    for f in range(f0, f1 + 1):
        scene.frame_set(f)
        pose = {}
        for j, bone in resolved.items():
            pb = arm.pose.bones[bone]
            head = arm.matrix_world @ pb.head
            pose[j] = [head.x, head.y, head.z]
        frames.append(pose)

    with open(args.output, 'w') as fh:
        json.dump({'joints': list(resolved), 'frames': frames}, fh)
    print(f'wrote {len(frames)} frames, {len(resolved)} joints -> {args.output}')


main()
