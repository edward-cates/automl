from pathlib import Path

import torch
import torchvision
from einops import rearrange

from src.datasets.local_dataset import LocalDataset

class VideoDataset(LocalDataset):
    def __init__(
            self,
            path: Path,
            file_manifest: list[Path] | None = None,
            img_size: int = 128, # assume square video for now.
            labels: list[str] | None = None, # assumed to be none for now - starting with an unsupervised task.
    ):
        super().__init__(path, file_manifest)
        self.img_size = img_size
        self.labels = labels

    def __getitem__(self, index: int) -> torch.Tensor:
        video_data = VideoDataset._get_video_data(self.file_manifest[index])
        video_data = self._preprocess_video(video_data)
        return video_data

    def _preprocess_video(self, video_data: torch.Tensor) -> torch.Tensor:
        video_data = VideoDataset._normalize_video(video_data)
        video_data = rearrange(video_data, "t h w c -> c t h w")
        video_data = VideoDataset._resize_video(video_data, self.img_size)
        return video_data

    @staticmethod
    def _get_video_data(video_path: Path) -> torch.Tensor:
        video_data, audio_data, metadata = torchvision.io.read_video(str(video_path))
        # print("[debug:video_dataset.py]", video_path, video_data.shape)
        return video_data

    @staticmethod
    def _normalize_video(video_data: torch.Tensor) -> torch.Tensor:
        return video_data.float() / 255.0

    @staticmethod
    def _resize_video(video_data: torch.Tensor, img_size: int) -> torch.Tensor:
        transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize((img_size, img_size)),
        ])
        return transforms(video_data)
