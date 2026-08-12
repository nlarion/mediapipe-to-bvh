#!/usr/bin/env python3
"""Build a QA gallery: one side-by-side sheet per clip plus a browsable index.

For every video that has a matching BVH, renders a video+MediaPipe vs BVH
skeleton sheet and measures left/right arm symmetry, then writes an index.html
sorted worst-first so the clips most worth looking at are at the top.

Usage:
  python qa_gallery.py --output-dir qa [--clips walking_010 walking_004]
"""
import argparse
import glob
import html
import os
import subprocess
import sys


def arm_offsets(bvh_path):
    """Return (left_total, right_total) arm bone length from the BVH header."""
    want = {f'mixamorig:{s}{b}' for s in ('Left', 'Right')
            for b in ('ForeArm', 'Hand')}
    found, name = {}, None
    with open(bvh_path) as fh:
        for line in fh:
            s = line.strip()
            if s.startswith('JOINT') or s.startswith('ROOT'):
                name = s.split()[1]
            elif s.startswith('OFFSET') and name in want:
                x, y, z = (float(v) for v in s.split()[1:4])
                found[name] = (x * x + y * y + z * z) ** 0.5
                name = None
    left = found.get('mixamorig:LeftForeArm', 0) + found.get('mixamorig:LeftHand', 0)
    right = found.get('mixamorig:RightForeArm', 0) + found.get('mixamorig:RightHand', 0)
    return left, right


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output-dir', default='qa')
    ap.add_argument('--clips', nargs='*', default=None)
    ap.add_argument('--frames', type=int, default=6)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    names = args.clips or sorted(
        os.path.splitext(os.path.basename(p))[0] for p in glob.glob('videos/*.mp4'))

    rows = []
    for name in names:
        video, bvh = f'videos/{name}.mp4', f'bvh/{name}.bvh'
        if not (os.path.exists(video) and os.path.exists(bvh)):
            continue
        sheet = os.path.join(args.output_dir, f'{name}.png')
        r = subprocess.run(
            [sys.executable, 'make_side_by_side.py', '--video', video,
             '--bvh', bvh, '--output', sheet, '--frames', str(args.frames)],
            capture_output=True, text=True)
        if r.returncode != 0:
            print(f'  {name}: sheet FAILED', r.stderr.strip().splitlines()[-1:])
            continue
        left, right = arm_offsets(bvh)
        ratio = max(left, right) / min(left, right) if min(left, right) > 0 else 0
        rows.append((ratio, name, left, right))
        print(f'  {name}: asymmetry {ratio:.2f}x')

    rows.sort(reverse=True)

    cards = []
    for ratio, name, left, right in rows:
        cls = 'bad' if ratio > 1.15 else 'ok'
        cards.append(f'''
  <section>
    <h2>{html.escape(name)}
      <span class="{cls}">arm symmetry {ratio:.2f}x</span>
      <small>L {left:.1f} / R {right:.1f}</small>
    </h2>
    <img src="{html.escape(name)}.png" alt="{html.escape(name)}" loading="lazy">
  </section>''')

    worst = f'{rows[0][0]:.2f}x' if rows else 'n/a'
    broken = sum(1 for r in rows if r[0] > 1.15)
    doc = f'''<!doctype html>
<meta charset="utf-8">
<title>BVH QA gallery</title>
<style>
  body {{ font: 15px/1.5 system-ui, sans-serif; margin: 24px; background: #14161a; color: #e8e8e8; }}
  h1 {{ font-size: 20px; }}
  section {{ margin: 28px 0; border-top: 1px solid #333; padding-top: 12px; }}
  h2 {{ font-size: 16px; display: flex; gap: 12px; align-items: baseline; }}
  small {{ color: #8a8a8a; font-weight: normal; }}
  .bad {{ color: #ff6b6b; }} .ok {{ color: #51cf66; }}
  img {{ width: 100%; height: auto; background: #fff; border-radius: 4px; }}
  .sum {{ color: #8a8a8a; }}
</style>
<h1>BVH QA — video + MediaPipe vs converted skeleton</h1>
<p class="sum">{len(rows)} clips. {broken} with arm asymmetry &gt; 1.15x. Worst: {worst}.
Left panel is the source video with the MediaPipe overlay (what was captured);
right panel is the BVH skeleton (what the converter produced). Sorted worst-first.</p>
{''.join(cards)}
'''
    index = os.path.join(args.output_dir, 'index.html')
    with open(index, 'w') as fh:
        fh.write(doc)
    print(f'\nwrote {index}  ({len(rows)} clips, {broken} still asymmetric)')


if __name__ == '__main__':
    main()
