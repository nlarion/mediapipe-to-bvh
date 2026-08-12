todo:

[] remove the bvh_samples dir it's taken from idk where and delete history/push
[] look into what we were doing and get back to grinding on this thing. getting close


Your simple version is all you need. The durable context is already saved where the next session will find it automatically: the workflow (BVH-first, strips before mesh) is in my memory files and on the Trello card, and the shooting checklist is committed in mediapipe-to-bvh/video_suggestions.md. A fresh session loads that memory index on startup, so "got new videos, dial in the BVHs, judge them first before any rigged-mesh run" plus maybe "check the trello card" will land it exactly where we are now — no need to hand-craft a longer jumping-off doc.