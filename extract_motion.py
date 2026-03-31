"""
ISL Avatar - Motion Extraction Script
======================================
Extracts full upper body motion from a sign language video using MediaPipe.
Captures: pose (body), both hands (21 pts each, all finger joints), face (468 pts)

Usage:
    python extract_motion.py --video /path/to/video.mp4
    python extract_motion.py --video /path/to/video.mp4 --all   (process all mp4s in same folder)
"""

import cv2
import mediapipe as mp
import json
import os
import argparse
import urllib.request

from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.core import base_options as base_options_module

OUTPUT_DIR = "./motion_data"

# MediaPipe model URLs
MODELS = {
    "pose":  "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task",
    "hand":  "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task",
    "face":  "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task",
}

MODEL_PATHS = {
    "pose": "./models/pose_landmarker.task",
    "hand": "./models/hand_landmarker.task",
    "face": "./models/face_landmarker.task",
}


def download_models():
    """Download MediaPipe task models if not present."""
    os.makedirs("./models", exist_ok=True)
    for key, url in MODELS.items():
        path = MODEL_PATHS[key]
        if not os.path.exists(path) or os.path.getsize(path) < 1000:
            print(f"  Downloading {key} model...")
            urllib.request.urlretrieve(url, path)
            print(f"  ✅ {key} model saved to {path}")
        else:
            print(f"  ✅ {key} model already present")


def lm_to_dict(lm, include_visibility=False):
    d = {"x": round(lm.x, 6), "y": round(lm.y, 6), "z": round(lm.z, 6)}
    if include_visibility and hasattr(lm, "visibility"):
        d["v"] = round(lm.visibility, 4)
    return d


def extract_motion(video_path: str) -> dict:
    """Extract pose + both hands + face landmarks frame by frame."""

    cap = cv2.VideoCapture(video_path)
    fps      = cap.get(cv2.CAP_PROP_FPS)
    width    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"  Video: {width}x{height} @ {fps:.1f}fps — {total_frames} frames ({total_frames/fps:.1f}s)")

    # ── Build landmarker instances ──────────────────────────────────────────
    RunningMode = mp_vision.RunningMode

    pose_opts = mp_vision.PoseLandmarkerOptions(
        base_options=base_options_module.BaseOptions(model_asset_path=MODEL_PATHS["pose"]),
        running_mode=RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    hand_opts = mp_vision.HandLandmarkerOptions(
        base_options=base_options_module.BaseOptions(model_asset_path=MODEL_PATHS["hand"]),
        running_mode=RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    face_opts = mp_vision.FaceLandmarkerOptions(
        base_options=base_options_module.BaseOptions(model_asset_path=MODEL_PATHS["face"]),
        running_mode=RunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    frames_data = []
    frame_idx   = 0

    with (
        mp_vision.PoseLandmarker.create_from_options(pose_opts) as pose_lm,
        mp_vision.HandLandmarker.create_from_options(hand_opts) as hand_lm,
        mp_vision.FaceLandmarker.create_from_options(face_opts) as face_lm,
    ):
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # MediaPipe expects RGB
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))

            pose_result = pose_lm.detect_for_video(mp_image, timestamp_ms)
            hand_result = hand_lm.detect_for_video(mp_image, timestamp_ms)
            face_result = face_lm.detect_for_video(mp_image, timestamp_ms)

            frame_data = {
                "frame":      frame_idx,
                "pose":       [],
                "left_hand":  [],
                "right_hand": [],
                "face":       [],
            }

            # Pose — 33 landmarks
            if pose_result.pose_landmarks:
                frame_data["pose"] = [
                    lm_to_dict(lm, include_visibility=True)
                    for lm in pose_result.pose_landmarks[0]
                ]

            # Hands — 21 landmarks each, labelled Left/Right
            if hand_result.hand_landmarks:
                for i, hand_lms in enumerate(hand_result.hand_landmarks):
                    label = hand_result.handedness[i][0].category_name  # "Left" or "Right"
                    key   = "left_hand" if label == "Left" else "right_hand"
                    frame_data[key] = [lm_to_dict(lm) for lm in hand_lms]

            # Face — 478 landmarks (468 mesh + 10 iris)
            if face_result.face_landmarks:
                frame_data["face"] = [
                    lm_to_dict(lm) for lm in face_result.face_landmarks[0]
                ]

            frames_data.append(frame_data)
            frame_idx += 1

            if frame_idx % 30 == 0:
                pct = frame_idx / total_frames * 100
                print(f"  Processing... {pct:.0f}% ({frame_idx}/{total_frames})", end="\r")

    cap.release()
    print(f"\n  ✅ Done — {frame_idx} frames extracted")

    word = os.path.splitext(os.path.basename(video_path))[0].upper()
    return {
        "word":         word,
        "fps":          fps,
        "total_frames": frame_idx,
        "width":        width,
        "height":       height,
        "landmark_counts": {
            "pose":       33,
            "left_hand":  21,
            "right_hand": 21,
            "face":       478,
        },
        "frames": frames_data,
    }


def process_video(video_path: str):
    word = os.path.splitext(os.path.basename(video_path))[0].upper()
    out_file = os.path.join(OUTPUT_DIR, f"{word}.json")
    print(f"\n🎬 Processing: {video_path}")
    data = extract_motion(video_path)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(data, f)  # compact for smaller file size
    size_mb = os.path.getsize(out_file) / 1024 / 1024
    print(f"  💾 Saved → {out_file}  ({size_mb:.1f} MB)")
    return out_file


def main():
    parser = argparse.ArgumentParser(description="ISL Video → Motion JSON extractor")
    parser.add_argument("--video",  required=True, help="Path to .mp4 video file")
    parser.add_argument("--all",    action="store_true", help="Process all .mp4 files in same folder")
    args = parser.parse_args()

    print("🔽 Checking MediaPipe models...")
    download_models()

    if args.all:
        folder = os.path.dirname(os.path.abspath(args.video))
        videos = [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(".mp4")]
        print(f"\n📂 Found {len(videos)} videos in {folder}")
        for v in sorted(videos):
            process_video(v)
    else:
        process_video(args.video)

    print("\n✅ All done! JSON files saved in ./motion_data/")


if __name__ == "__main__":
    main()
