from abc import ABC, abstractmethod
from pathlib import Path

class LocalDataset(ABC):
    """
    Dataset base class.
    """
    def __init__(
            self,
            path: Path,
            file_manifest: list[Path] | None = None,
    ):
        assert path.exists(), f"Path does not exist: {path}"
        assert (path / "data").exists(), f"Data folder does not exist: {path / "data"}"
        self.path = path
        self.file_manifest: list[Path] = list((self.path / "data").glob("**/*")) \
            if file_manifest is None else file_manifest
    @property
    def name(self) -> str:
        return self.path.name

    def __len__(self) -> int:
        return len(self.file_manifest)

    def split(
            self,
            train_samples: int,
            how: str = "file_name",
            holdout_set: bool = False, # For now, not implemented.
    ) -> tuple["LocalDataset", "LocalDataset"]:
        if how == "file_name":
            return self._split_by_file_name(train_samples=train_samples)
        raise ValueError(f"Unknown split method: {how}")

    # Private.

    def _split_by_file_name(self, train_samples: int) -> tuple["LocalDataset", "LocalDataset"]:
        file_names_sorted = sorted(self.file_manifest, key=lambda x: x.name)
        train_set = LocalDataset(
            path=self.path,
            file_manifest=file_names_sorted[:train_samples],
        )
        test_set = LocalDataset(
            path=self.path,
            file_manifest=file_names_sorted[train_samples:],
        )
        return train_set, test_set
