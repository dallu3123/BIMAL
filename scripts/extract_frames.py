"""
LeRobot v3 비디오를 프레임 JPG로 사전 추출.

출력 구조:
    <data_dir>/cache/<camera_key>/ep{episode_index:04d}/f{local_frame:04d}.jpg

순차 디코딩이기 때문에 학습 시 랜덤 시크 비용이 사라집니다.
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import cv2
import pandas as pd
from tqdm import tqdm

DEFAULT_CAMERAS = (
    "observation.images.cam_high",
    "observation.images.cam_left_wrist",
    "observation.images.cam_right_wrist",
)


def extract(data_dir: str, image_size: int, jpeg_quality: int, cameras: tuple[str, ...]) -> None:
    with open(os.path.join(data_dir, "meta", "info.json")) as f:
        info = json.load(f)
    fps = float(info["fps"])

    meta_files = sorted(
        glob.glob(os.path.join(data_dir, "meta", "episodes", "chunk-*", "file-*.parquet"))
    )
    meta_df = pd.concat([pd.read_parquet(p) for p in meta_files], ignore_index=True)

    cache_root = os.path.join(data_dir, "cache")
    os.makedirs(cache_root, exist_ok=True)
    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)]

    for cam in cameras:
        print(f"\n=== {cam} ===")
        # (file_idx) 별 에피소드 묶기 → 파일 1개만 열고 순차 디코딩
        file_idx_col = f"videos/{cam}/file_index"
        ts_col = f"videos/{cam}/from_timestamp"
        grouped: dict[int, list[tuple[int, float, int]]] = {}
        for _, row in meta_df.iterrows():
            fi = int(row[file_idx_col])
            grouped.setdefault(fi, []).append(
                (int(row["episode_index"]), float(row[ts_col]), int(row["length"]))
            )

        for file_idx, eps in grouped.items():
            # 동일 파일 내에서는 from_timestamp 오름차순 보장
            eps.sort(key=lambda x: x[1])
            video_path = os.path.join(data_dir, "videos", cam, "chunk-000", f"file-{file_idx:03d}.mp4")
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise RuntimeError(f"Cannot open {video_path}")

            current_pos = 0  # 디코더가 읽은 다음 프레임 인덱스 (파일 기준)
            for ep_idx, from_ts, length in tqdm(eps, desc=f"{cam} file{file_idx:03d}"):
                start_in_file = int(round(from_ts * fps))

                # 이미 지나쳤으면 seek; 아니면 skip-read (순차가 더 빠름)
                if start_in_file < current_pos:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, float(start_in_file))
                    current_pos = start_in_file
                else:
                    while current_pos < start_in_file:
                        ok = cap.grab()
                        if not ok:
                            raise RuntimeError(f"grab() failed at {current_pos} in {video_path}")
                        current_pos += 1

                ep_dir = os.path.join(cache_root, cam, f"ep{ep_idx:04d}")
                os.makedirs(ep_dir, exist_ok=True)

                for local in range(length):
                    ok, frame = cap.read()
                    if not ok or frame is None:
                        raise RuntimeError(
                            f"read() failed at ep{ep_idx} local{local} file{file_idx} pos{current_pos}"
                        )
                    current_pos += 1
                    if frame.shape[0] != image_size or frame.shape[1] != image_size:
                        frame = cv2.resize(frame, (image_size, image_size), interpolation=cv2.INTER_AREA)
                    out = os.path.join(ep_dir, f"f{local:04d}.jpg")
                    cv2.imwrite(out, frame, encode_params)

            cap.release()

    print(f"\nDone. Cache root: {cache_root}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./source/lerobot_bearing4_v3")
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--jpeg_quality", type=int, default=90)
    args = parser.parse_args()
    extract(args.data_dir, args.image_size, args.jpeg_quality, DEFAULT_CAMERAS)
