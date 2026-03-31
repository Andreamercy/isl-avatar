# ISL Avatar — Video → Avatar Motion

Convert Indian Sign Language videos into avatar animation using MediaPipe landmark extraction + a Three.js GLB viewer.

---

## How it works

```
your_video.mp4
      │
      ▼
extract_motion.py          ← MediaPipe Holistic (Pose + Hands + Face)
      │
      ▼
motion_data/WORD.json      ← 543 landmarks × every frame
      │
      ▼
index.html                 ← Load GLB + JSON → animated avatar in browser
```

---

## Step 1 — Install dependencies

```bash
pip install -r requirements.txt
```

---

## Step 2 — Extract motion from a video

```bash
# Single video
python extract_motion.py --video "/path/to/ISL video/Allergies.mp4"

# All 16 videos at once
python extract_motion.py --video "/path/to/ISL video/Allergies.mp4" --all
```

This will:
- Auto-download MediaPipe model files into `./models/` on first run (~150MB total)
- Extract pose (33 pts) + left hand (21 pts) + right hand (21 pts) + face (478 pts) per frame
- Save each word as `./motion_data/ALLERGIES.json`

---

## Step 3 — View in browser

Open `index.html` in any modern browser (Chrome / Firefox / Edge).

1. Click **"Drop or click to load GLB"** → select your `.glb` avatar file
2. Click **"Drop or click to load JSON"** → select the extracted `ALLERGIES.json`
3. Hit **▶ Play**

You can also **drag & drop** both files directly onto the page.

---

## Controls

| Control | Description |
|---|---|
| ▶ Play | Start animation |
| ⏸ Pause | Pause |
| ⏮ Reset | Go to frame 0 |
| Scrubber | Drag to any frame |
| Speed buttons | 0.5× / 1× / 1.5× / 2× |
| Show skeleton | Overlay bone rig |
| Loop | Repeat animation |

---

## Landmark mapping (MediaPipe → GLB bones)

| MediaPipe | GLB Bone |
|---|---|
| Pose[11] LEFT_SHOULDER → [13] LEFT_ELBOW | LeftArm |
| Pose[13] LEFT_ELBOW → [15] LEFT_WRIST | LeftForeArm |
| Pose[12] RIGHT_SHOULDER → [14] RIGHT_ELBOW | RightArm |
| Pose[14] RIGHT_ELBOW → [16] RIGHT_WRIST | RightForeArm |
| Left Hand [1→4] | LeftHandThumb1–4 |
| Left Hand [5→8] | LeftHandIndex1–4 |
| Left Hand [9→12] | LeftHandMiddle1–4 |
| Left Hand [13→16] | LeftHandRing1–4 |
| Left Hand [17→20] | LeftHandPinky1–4 |
| Right Hand (mirror) | RightHand* |
| Pose[0] NOSE vs shoulders | Head + Neck |
| Spine landmarks | Spine / Spine1 / Spine2 |

---

## File structure

```
isl-avatar/
├── extract_motion.py   ← run this on your video
├── index.html          ← open this in browser
├── requirements.txt
├── README.md
├── models/             ← auto-created, MediaPipe .task files downloaded here
└── motion_data/        ← auto-created, one JSON per word
    ├── ALLERGIES.json
    └── ...
```

---

## Requirements

- Python 3.9+
- Modern browser (Chrome recommended)
- No GPU needed — runs on CPU
- Internet required on first run (to download MediaPipe models ~150MB)

---

## Tested with

- Ready Player Me GLB avatars
- MediaPipe 0.10.x Tasks API
- Three.js r160
