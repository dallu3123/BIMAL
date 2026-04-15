"""
LeRobot v3.0 형식의 실제 Bimanual 데이터셋 로더.

데이터 루트 구조:
    root/
      data/chunk-000/file-*.parquet           (observation.state, action, index, ...)
      videos/<video_key>/chunk-000/file-*.mp4  (카메라별 영상)
      meta/info.json
      meta/episodes/chunk-000/file-*.parquet   (에피소드별 길이, 비디오 파일 인덱스/타임스탬프)
"""

from __future__ import annotations

import glob
import json
import os
from dataclasses import dataclass

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


DEFAULT_CAMERAS = (
    "observation.images.cam_high",
    "observation.images.cam_left_wrist",
    "observation.images.cam_right_wrist",
)


@dataclass
class _EpisodeMeta:
    episode_index: int
    length: int
    data_from: int  # 글로벌 프레임 인덱스 (parquet row) 시작
    # 카메라별 (file_index, from_timestamp) — 영상 파일 내 시작 위치 계산용
    video_info: dict[str, tuple[int, float]]


class BimalRealDataset(Dataset):
    """
    각 샘플은 (카메라 프레임 1장 per 카메라, 미래 seq_len action 시퀀스) 쌍입니다.

    반환 형식:
        images_dict: {camera_key: Tensor(3, H, W) in [0, 1]}
        actions:     Tensor(seq_len, action_dim)
    """

    def __init__(
        self,
        data_dir: str,
        seq_len: int = 16,
        image_size: int = 224,
        camera_keys: tuple[str, ...] = DEFAULT_CAMERAS,
        use_cache: bool = False,
        cache_dir: str | None = None,
    ) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.seq_len = seq_len
        self.image_size = image_size
        self.camera_keys = tuple(camera_keys)
        self.use_cache = use_cache
        self.cache_dir = cache_dir or os.path.join(data_dir, "cache")

        with open(os.path.join(data_dir, "meta", "info.json")) as f:
            self.info = json.load(f)
        self.fps = float(self.info["fps"])

        self._load_action_table()
        self._load_episode_meta()
        self._build_index()

        # 비디오 캡처 캐시 (worker 별 lazy open)
        self._video_cache: dict[tuple[str, int], cv2.VideoCapture] = {}

    # --- 초기화 헬퍼 ---

    def _load_action_table(self) -> None:
        data_files = sorted(glob.glob(os.path.join(self.data_dir, "data", "chunk-*", "file-*.parquet")))
        if not data_files:
            raise FileNotFoundError(f"No data parquet found under {self.data_dir}/data")
        frames = [pd.read_parquet(p, columns=["episode_index", "frame_index", "action"]) for p in data_files]
        df = pd.concat(frames, ignore_index=True)
        df = df.sort_values(["episode_index", "frame_index"]).reset_index(drop=True)
        self._actions_by_ep: dict[int, np.ndarray] = {}
        for ep, g in df.groupby("episode_index"):
            # action 컬럼은 list-like; 2D float32 배열로 정규화
            acts = np.stack([np.asarray(a, dtype=np.float32) for a in g["action"].to_list()], axis=0)
            self._actions_by_ep[int(ep)] = acts

    def _load_episode_meta(self) -> None:
        meta_files = sorted(
            glob.glob(os.path.join(self.data_dir, "meta", "episodes", "chunk-*", "file-*.parquet"))
        )
        if not meta_files:
            raise FileNotFoundError(f"No episodes meta under {self.data_dir}/meta/episodes")
        meta_df = pd.concat([pd.read_parquet(p) for p in meta_files], ignore_index=True)

        self._episodes: list[_EpisodeMeta] = []
        for _, row in meta_df.iterrows():
            ep_idx = int(row["episode_index"])
            length = int(row["length"])
            video_info: dict[str, tuple[int, float]] = {}
            for cam in self.camera_keys:
                file_idx_col = f"videos/{cam}/file_index"
                ts_col = f"videos/{cam}/from_timestamp"
                if file_idx_col not in meta_df.columns:
                    raise KeyError(f"Missing column {file_idx_col} in episodes meta")
                video_info[cam] = (int(row[file_idx_col]), float(row[ts_col]))
            self._episodes.append(
                _EpisodeMeta(
                    episode_index=ep_idx,
                    length=length,
                    data_from=int(row["dataset_from_index"]),
                    video_info=video_info,
                )
            )

    def _build_index(self) -> None:
        """(episode_slot, local_start_frame) 페어의 플랫 리스트 구성."""
        self._samples: list[tuple[int, int]] = []
        for i, ep in enumerate(self._episodes):
            usable = ep.length - self.seq_len
            if usable <= 0:
                continue
            for start in range(usable):
                self._samples.append((i, start))

    # --- 비디오 접근 ---

    def _get_capture(self, cam: str, file_idx: int) -> cv2.VideoCapture:
        key = (cam, file_idx)
        cap = self._video_cache.get(key)
        if cap is not None:
            return cap
        # 경로: videos/<cam>/chunk-000/file-000.mp4 (chunk는 항상 0 가정 — info.json 기준)
        path = os.path.join(self.data_dir, "videos", cam, "chunk-000", f"file-{file_idx:03d}.mp4")
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {path}")
        self._video_cache[key] = cap
        return cap

    def _read_frame(self, cam: str, file_idx: int, frame_in_file: int) -> np.ndarray:
        cap = self._get_capture(cam, file_idx)
        cap.set(cv2.CAP_PROP_POS_FRAMES, float(frame_in_file))
        ok, frame = cap.read()
        if not ok or frame is None:
            raise RuntimeError(f"Failed to read frame {frame_in_file} from {cam} file {file_idx}")
        # BGR -> RGB
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def _to_tensor(self, frame_rgb: np.ndarray) -> torch.Tensor:
        if frame_rgb.shape[0] != self.image_size or frame_rgb.shape[1] != self.image_size:
            frame_rgb = cv2.resize(
                frame_rgb, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA
            )
        tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1).contiguous().float() / 255.0
        return tensor

    # --- Dataset 프로토콜 ---

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        ep_slot, start = self._samples[idx]
        ep = self._episodes[ep_slot]

        actions = self._actions_by_ep[ep.episode_index][start : start + self.seq_len]
        actions_t = torch.from_numpy(actions)  # (seq_len, action_dim)

        images_dict: dict[str, torch.Tensor] = {}
        for cam in self.camera_keys:
            if self.use_cache:
                path = os.path.join(
                    self.cache_dir, cam, f"ep{ep.episode_index:04d}", f"f{start:04d}.jpg"
                )
                frame_bgr = cv2.imread(path, cv2.IMREAD_COLOR)
                if frame_bgr is None:
                    raise FileNotFoundError(f"Cache miss: {path}")
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            else:
                file_idx, from_ts = ep.video_info[cam]
                frame_in_file = int(round(from_ts * self.fps)) + start
                frame_rgb = self._read_frame(cam, file_idx, frame_in_file)
            images_dict[cam] = self._to_tensor(frame_rgb)

        return images_dict, actions_t

    def __del__(self) -> None:  # best-effort cleanup
        for cap in getattr(self, "_video_cache", {}).values():
            try:
                cap.release()
            except Exception:
                pass
