"""
Extract 55 Whole-Body Pose Keypoints (13 upper-body + 42 hand joints) per video.
Normalizes coordinates to [-1, 1], samples uniformly to T=64 frames.
"""

import argparse
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm


def sample_linear_indices(total_frames: int, target_frames: int = 64) -> np.ndarray:
    if total_frames <= 0:
        return np.zeros((target_frames,), dtype=int)
    if total_frames >= target_frames:
        idxs = np.linspace(0, total_frames - 1, target_frames)
        return np.rint(idxs).astype(int)
    pad = np.full((target_frames - total_frames,), total_frames - 1, dtype=int)
    return np.concatenate([np.arange(total_frames, dtype=int), pad])


def extract_mediapipe_video(video_path: str, mp_holistic, timesteps: int = 64) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total_frames <= 0:
        return np.zeros((timesteps, 55, 2), dtype=np.float32)

    indices = set(sample_linear_indices(total_frames, timesteps))
    frames_dict = {}
    current_idx = 0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 1)

    while cap.isOpened() and len(frames_dict) < len(indices):
        ret, frame = cap.read()
        if not ret:
            break
        if current_idx in indices:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = mp_holistic.process(rgb)

            pts = []
            # Upper body: 13 joints (nose, eyes, ears, shoulders, elbows, wrists, hips)
            body_landmarks = res.pose_landmarks.landmark if res.pose_landmarks else None
            for idx in range(13):
                if body_landmarks and idx < len(body_landmarks):
                    pts.append([body_landmarks[idx].x, body_landmarks[idx].y])
                else:
                    pts.append([0.5, 0.5])

            # Left hand: 21 joints
            left_landmarks = res.left_hand_landmarks.landmark if res.left_hand_landmarks else None
            for idx in range(21):
                if left_landmarks and idx < len(left_landmarks):
                    pts.append([left_landmarks[idx].x, left_landmarks[idx].y])
                else:
                    pts.append([0.5, 0.5])

            # Right hand: 21 joints
            right_landmarks = res.right_hand_landmarks.landmark if res.right_hand_landmarks else None
            for idx in range(21):
                if right_landmarks and idx < len(right_landmarks):
                    pts.append([right_landmarks[idx].x, right_landmarks[idx].y])
                else:
                    pts.append([0.5, 0.5])

            # Scale to [-1, 1]
            pts = (np.clip(np.array(pts, dtype=np.float32), 0.0, 1.0) - 0.5) * 2.0
            frames_dict[current_idx] = pts
        current_idx += 1
    cap.release()

    # Reconstruct 64 frames in order
    ordered_indices = sample_linear_indices(total_frames, timesteps)
    out_seq = []
    last_pts = np.zeros((55, 2), dtype=np.float32)
    for idx in ordered_indices:
        if idx in frames_dict:
            last_pts = frames_dict[idx]
        out_seq.append(last_pts)
    return np.array(out_seq, dtype=np.float32)


def main():
    parser = argparse.ArgumentParser(description="Extract 55 whole-body pose keypoints")
    parser.add_argument("--csv_path", type=str, required=True, help="Manifest CSV")
    parser.add_argument("--video_dir", type=str, required=True, help="Video directory")
    parser.add_argument("--out_dir", type=str, required=True, help="Output directory for .npz files")
    parser.add_argument("--frames", type=int, default=64)
    args = parser.parse_args()

    import mediapipe as mp
    mp_holistic = mp.solutions.holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
    )

    out_path = Path(args.out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.csv_path)

    print(f"Extracting 55 keypoints for {len(df)} videos...")
    for _, row in tqdm(df.iterrows(), total=len(df)):
        vid_id = str(row["ID_video"]).replace(".mp4", "").replace(".webm", "")
        out_file = out_path / f"{vid_id}.npz"
        if out_file.exists():
            continue

        video_path = None
        for ext in [".mp4", ".webm", "_yolo.mp4"]:
            candidate = Path(args.video_dir) / f"{vid_id}{ext}"
            if candidate.exists():
                video_path = candidate
                break

        if video_path is None:
            pose_seq = np.zeros((args.frames, 55, 2), dtype=np.float32)
        else:
            pose_seq = extract_mediapipe_video(str(video_path), mp_holistic, args.frames)

        np.savez_compressed(out_file, x=pose_seq)

    mp_holistic.close()
    print(f"Pose extraction finished. Saved to {args.out_dir}")


if __name__ == "__main__":
    main()
