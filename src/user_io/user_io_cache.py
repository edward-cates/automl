
from src.datasets.local_dataset import LocalDataset

class UserIoCache:

    def __init__(self):
        self.dataset: LocalDataset | None = None
        self.train_dataset: LocalDataset | None = None
        self.test_dataset: LocalDataset | None = None
        self.holdout_dataset: LocalDataset | None = None


    def __str__(self) -> str:
        message = ""

        if self.dataset is not None:
            message += f"Dataset: {self.dataset.name}\n"

        if self.train_dataset is not None:
            message += f"(train len={len(self.train_dataset)})"

        if self.test_dataset is not None:
            message += f"(test len={len(self.test_dataset)})"

        return message
