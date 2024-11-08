import random
from collections.abc import Sequence

class AutomlDataset:
    """
    Generalize the task of loading and splitting a dataset.
    """
    def __init__(self, samples: Sequence, indexes: list[int] | None = None):
        self._samples = samples
        if indexes is None:
            self._indexes = list(range(len(samples)))
        else:
            self._indexes = indexes

    def __len__(self) -> int:
        return len(self._indexes)

    def __getitem__(self, idx: int):
        return self._samples[self._indexes[idx]]

    def split(self, split_ratio: float, shuffle: bool) -> tuple["AutomlDataset", "AutomlDataset"]:
        indexes_copy = self._indexes.copy()
        if shuffle:
            random.shuffle(indexes_copy)
        split_idx = int(len(indexes_copy) * split_ratio)
        return (
            AutomlDataset(self._samples, indexes_copy[:split_idx]),
            AutomlDataset(self._samples, indexes_copy[split_idx:]),
        )
