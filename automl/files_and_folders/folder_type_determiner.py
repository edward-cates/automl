from pathlib import Path

from src.files_and_folders.file_extension_enum import FileExtensionEnum

class FolderTypeDeterminer:
    """
    Determine which dataset type a folder is based on its contents.
    """
    def __init__(self, path: Path):
        self.path = path

    def determine_folder_type(self) -> FileExtensionEnum:
        """
        - get all files/folders in the folder,
        - make sure they're all the same time (raise an error if not or if subdirectories are present)
        - get the shared extension,
        - return the file extension (or throw error if not recognized).
        """
        items = list(self.path.iterdir())
        
        if not items:
            raise ValueError("The folder is empty.")
        
        if any(item.is_dir() for item in items):
            raise ValueError("Subdirectories are not allowed.")
        
        extensions = set(item.suffix.lower() for item in items)
        
        if len(extensions) != 1:
            raise ValueError("All files must have the same extension.")
        
        extension = extensions.pop()

        return FileExtensionEnum(extension)
