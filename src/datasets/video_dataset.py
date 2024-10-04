from pathlib import Path

from src.datasets.local_dataset import LocalDataset

class VideoDataset(LocalDataset):
    def __init__(self, path: Path):
        super().__init__(path)
