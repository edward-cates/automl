import random

from automl.training.automl_sample import AutomlSample

class AutomlDataset:
    """
    Generalize the task of loading and splitting a dataset.
    """
    def __init__(self, samples: list[AutomlSample]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        return self.samples[idx].get_model_input()

    def split(self, split_ratio: float, shuffle: bool) -> tuple["AutomlDataset", "AutomlDataset"]:
        if shuffle:
            random.shuffle(self.samples)
        split_idx = int(len(self.samples) * split_ratio)
        return (
            AutomlDataset(self.samples[:split_idx]),
            AutomlDataset(self.samples[split_idx:]),
        )
