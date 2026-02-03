# Facial Animation Pipeline - Future Implementation

## Overview

This document outlines the plan for adding facial animation capture to complement the body motion capture system. Together, these form a complete animation asset pipeline for video games.

## Current State

**Body Animation (this project):**
- MediaPipe pose detection → BVH skeletal animation
- Works with Godot, Unity, Blender
- Handles: walking, running, fighting, gestures, full body movement

**What's Missing:**
- Facial expressions (smile, frown, surprise, etc.)
- Lip sync for dialogue
- Eye tracking / gaze direction

## Facial Animation Approach

### Technology: MediaPipe Face Mesh

MediaPipe provides 468 facial landmarks that track:
- Eyebrows (raise, furrow)
- Eyes (open, close, squint, gaze direction)
- Nose
- Mouth (open, smile, frown, pucker, etc.)
- Jaw movement
- Cheeks

### Output Format: Blend Shapes / Morph Targets

Unlike body animation (bones/skeleton), facial animation uses **blend shapes**:
- Pre-defined facial poses (e.g., "smile", "blink_left", "mouth_open")
- Values from 0.0 to 1.0 representing intensity
- Multiple blend shapes combine for complex expressions

**Standard blend shape sets:**
- ARKit (Apple) - 52 blend shapes, widely supported
- Oculus/Meta - similar to ARKit
- FACS (Facial Action Coding System) - academic standard

### Output Formats for Games

1. **JSON/CSV** - Simple, parse in any engine
   ```json
   {
     "frame": 0,
     "timestamp": 0.0,
     "blendshapes": {
       "jawOpen": 0.3,
       "mouthSmile_L": 0.7,
       "mouthSmile_R": 0.7,
       "eyeBlink_L": 0.0,
       "eyeBlink_R": 0.0
     }
   }
   ```

2. **FBX** - Industry standard, includes blend shape animation tracks

3. **GLTF/GLB** - Modern, works great with Godot 4

## Implementation Plan

### Phase 1: Blend Shape Extraction
- Use MediaPipe Face Mesh to get 468 landmarks per frame
- Map landmarks to standard blend shape values (ARKit compatible)
- Output as JSON for testing

### Phase 2: Synchronization with Body
- Align facial animation timestamps with BVH body animation
- Same video source = automatic sync
- Handle different frame rates if needed

### Phase 3: Export Formats
- JSON for custom engine integration
- FBX via Blender scripting for Unity
- GLTF for Godot 4

### Phase 4: Lip Sync (Optional Enhancement)
- Extract audio from video
- Use speech-to-phoneme mapping
- Map phonemes to mouth blend shapes (visemes)
- Tools: Rhubarb Lip Sync, Wav2Lip, or custom

## Game Engine Integration

### Unity Workflow
1. Import character with blend shape-enabled face mesh
2. Import BVH for body animation → Humanoid retargeting
3. Import blend shape animation (FBX or scripted)
4. Blend layers: Body (base) + Face (additive)

### Godot Workflow
1. Character with face mesh blend shapes (GLTF)
2. Import body animation (BVH → Blender → GLTF, or direct)
3. Apply blend shape tracks via AnimationPlayer
4. AnimationTree for blending body + face

### Blender as Hub
Blender can combine everything:
- Import BVH body animation
- Import/apply facial blend shapes
- Retarget to game-ready character
- Export unified FBX or GLTF

## Complete Asset Pipeline Vision

```
VIDEO INPUT
    │
    ├─── MediaPipe Pose ──────► BVH ──────► Body Animation
    │
    ├─── MediaPipe Face Mesh ──► Blend Shapes ──► Facial Animation
    │
    └─── Audio Track ─────────► Lip Sync ──────► Mouth Animation
                                    │
                                    ▼
                            ┌───────────────┐
                            │   BLENDER     │
                            │  (optional)   │
                            │  - Cleanup    │
                            │  - Retarget   │
                            │  - Combine    │
                            └───────┬───────┘
                                    │
                    ┌───────────────┼───────────────┐
                    ▼               ▼               ▼
                 UNITY          GODOT          UNREAL
                 (FBX)         (GLTF)          (FBX)
```

## Character Requirements

For full pipeline support, game characters need:
1. **Humanoid skeleton** - Standard bone hierarchy (Mixamo compatible)
2. **Face mesh** - Separate or part of body mesh
3. **Blend shapes** - ARKit-compatible set recommended
4. **Proper rigging** - Skeleton weights + blend shape setup

## Tools & Resources

**Facial Capture:**
- MediaPipe Face Mesh (free, what we'll use)
- Live Link Face (iOS app, ARKit)
- Faceware, Rokoko (professional, paid)

**Blend Shape Standards:**
- [ARKit Blend Shapes](https://developer.apple.com/documentation/arkit/arfaceanchor/blendshapelocation)
- [Ready Player Me](https://docs.readyplayer.me/) - Free avatars with blend shapes

**Lip Sync:**
- Rhubarb Lip Sync (open source)
- SALSA LipSync (Unity, paid)
- Godot LipSync plugins

## Timeline

1. **Now:** Finish body animation accuracy improvements
2. **Next:** Implement facial blend shape extraction
3. **Then:** Synchronization and export formats
4. **Later:** Lip sync integration (if needed)

## Notes

- Facial animation is computationally lighter than body (just blend shape values, no IK)
- Can process face and body from same video simultaneously
- Face mesh quality degrades at angles > 45° from camera
- Good lighting on face is critical for accuracy
