from src.files_and_folders.file_extension_enum import FileExtensionEnum
from src.datasets.dataset_type_enum import DatasetType

FileExtensionToDatasetTypeMap = {
    FileExtensionEnum.MP4: DatasetType.VIDEO,
    FileExtensionEnum.MOV: DatasetType.VIDEO,
    FileExtensionEnum.AVI: DatasetType.VIDEO,
    FileExtensionEnum.JPEG: DatasetType.IMAGE,
    FileExtensionEnum.JPG: DatasetType.IMAGE,
    FileExtensionEnum.PNG: DatasetType.IMAGE,
    FileExtensionEnum.TXT: DatasetType.TEXT,
    FileExtensionEnum.WAV: DatasetType.AUDIO,
}
