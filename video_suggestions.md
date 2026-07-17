# Video Reshoot Guide

Everything below comes from what actually went wrong (or right) in the
2026-07-16 living-room batch. The single biggest finding: **the arm
artifacts are a camera-angle problem, not a lighting or quality problem.**
Every clip where the camera saw your back had an arm stuck out horizontal
in the BVH; the one fully frontal clip (wave_002) tracked almost
perfectly.

## The one rule that matters most

**Never show the camera your back.**

- **Walks / runs / moonwalk:** move **side-on**, perpendicular to the
  camera, crossing the frame left-to-right or right-to-left. Profile view
  is what MediaPipe tracks best for legs AND arms, and it's the only way
  the converter captures forward travel (depth motion toward/away from
  the camera gets dampened to almost nothing).
- **Upper-body moves (wave, aiming, robot, gestures):** face the camera.
  wave_002 was the cleanest clip of the whole batch for exactly this
  reason.

## Camera & framing

- Camera at roughly **chest height, level** — not low on a couch/table.
- Back it up far enough that your **head and feet have clear margin** for
  the entire move. Several clips had the head grazing or past the top
  edge.
- **Clear the floor path.** The couch arm and fireplace edge occluded
  feet/lower legs in the left third of the room — occluded ankles are a
  big source of foot jitter and skate.
- **Stay fully in frame for the whole take.** walking_006 ended with 0.6s
  of empty room; aiming_001 had 1.5s before you entered. The converter
  now survives this (it trims no-person frames), but any partial-body
  entry/exit frames still produce garbage poses at the boundaries.

## Background & clothing

- **TV off is fine.** A black screen is a *static* background — that's
  strictly better than moving footage, which can distract detection
  (especially while you're entering frame). MediaPipe doesn't care that
  it's dark; it cares that it doesn't move and doesn't contain people.
  Bonus: your white shirt against a black screen is maximum contrast.
- **Contrast with the wall matters more than the wall itself.** White tee
  against the white wall was low-contrast; either wear a dark top or
  favor the part of the room where you contrast most. Bare legs against
  the wood floor were fine. Solid colors beat patterns.
- The bookshelf clutter is survivable, but plainer is better.

## Light & speed

- running_003 and fighter_007 were killed by **motion blur** — indoor
  light forces a slow shutter. For anything fast: every light on, or
  shoot daytime near the window. If it's still blurry, do the move at
  ~80% speed; a slightly slow run converts better than a smeared one.

## Per-take protocol

1. Hit record.
2. Step to your mark, **stand still ~1 second** (full body visible,
   arms relaxed).
3. Do the move.
4. **Hold still ~1 second.**
5. Stop recording.

The still holds give the converter a high-confidence reference frame and
give us clean loop/trim points. Keep takes short and single-purpose —
one motion per clip beats a 10s medley (aiming_002's multiple actions in
one take made its rhythm unusable).

## Format

1080p / 30fps is exactly right — no change needed. Barefoot is fine.

## What the 2026-07-16 batch taught us, clip by clip

| Worked | Why |
|---|---|
| wave_002 | frontal, full body, moderate speed → near-perfect tracking |
| robot_001 | frontal, arms away from torso |
| walking_002 (legs) | the segment where you were side-on reads as a proper walk |

| Failed | Why |
|---|---|
| every walk's arms | rear/diagonal view → arm stuck horizontal |
| running_003, fighter_007 | motion blur from fast moves in low light |
| walking_006 tail | walked out of frame |
| aiming_002 | 10s multi-action take, partially out of frame |
