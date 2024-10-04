from pathlib import Path

from src.files_and_folders.folder_type_determiner import FolderTypeDeterminer
from src.datasets.file_extension_to_dataset_type_map import FileExtensionToDatasetTypeMap
from src.files_and_folders.file_extension_enum import FileExtensionEnum
from src.datasets.dataset_type_enum import DatasetType
from src.datasets.video_dataset import VideoDataset
from src.datasets.local_dataset import LocalDataset

class DatasetFinder:
    """
    Find and list datasets in the cache.
    Create new ones by transformerming files/folders.
    """
    cache_dir: Path = Path("cache")

    def __init__(self):
        pass

    @staticmethod
    def list_datasets() -> list[Path]:
        """
        Find all datasets in the cache.
        This means find directories in the cache_dir (but not their subdirectories).
        """
        return [
            path
            for path in DatasetFinder.cache_dir.iterdir()
            if path.is_dir() and not path.name.startswith(".")
        ]

    @staticmethod
    def check_name_exists(name: str) -> bool:
        """
        Check if a dataset name exists.
        """
        return (DatasetFinder.cache_dir / name).exists()

    @staticmethod
    def load_dataset(name: str) -> LocalDataset:
        """
        Load a dataset from a path.
        """
        path = DatasetFinder.cache_dir / name
        assert path.exists(), f"Dataset does not exist: {path}"
        file_extension = FolderTypeDeterminer(path / "data").determine_folder_type()
        dataset_type = FileExtensionToDatasetTypeMap[file_extension]
        if dataset_type == DatasetType.VIDEO:
            return VideoDataset(path)
        else:
            raise ValueError(f"Dataset type not yet implemented: {dataset_type}")

    @staticmethod
    def create_new_dataset(
        name: str,
        path: Path,
    ) -> LocalDataset:
        """
        Create a new dataset.
        """
        assert path.exists(), f"Path does not exist: {path}"
        assert path.is_dir(), f"Path is not a directory: {path}"
        assert " " not in name, "Dataset name cannot contain spaces."

        local_path = DatasetFinder.cache_dir / name
        assert not local_path.exists(), "Dataset already exists."
        local_path.mkdir(exist_ok=False, parents=False)

        # just make a symlink
        (local_path / "data").symlink_to(path, target_is_directory=True)
        return DatasetFinder.load_dataset(name)

