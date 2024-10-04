from pathlib import Path

from src.datasets.dataset_finder import DatasetFinder
from src.user_io.user_io_state import UserIoState
from src.files_and_folders.folder_type_determiner import FolderTypeDeterminer
from src.files_and_folders.file_extension_enum import FileExtensionEnum
from src.datasets.dataset_type_enum import DatasetType
from src.datasets.file_extension_to_dataset_type_map import FileExtensionToDatasetTypeMap
from src.user_io.states.states_enum import StatesEnum
from src.user_io.states.state_error_message import StateErrorMessage
from src.user_io.user_io_cache import UserIoCache
from src.user_io.states.dataset_manipulation.load_dataset_state import LoadDatasetState

class CreateDatasetStartState(UserIoState):

    def __init__(self, cache: UserIoCache):
        super().__init__(cache)

    def build_prompt(self) -> str:
        return "Please input the path to the directory containing the data you want to use to create the dataset."        

    def handle_response(self, response: str) -> StatesEnum | StateErrorMessage:
        path = Path(response)
        if not path.exists():
            print(f"\nThe path {path} does not exist, try again.\n")
            return StateErrorMessage(message=f"The path {path} does not exist, try again.")
        elif not path.is_dir():
            print(f"\nThe path {path} is not a directory, try again.\n")
            return StateErrorMessage(message=f"The path {path} is not a directory, try again.")
        else:
            print("\nDetermining dataset type...\n")
            folder_type_determiner = FolderTypeDeterminer(path)
            try:
                file_extension: FileExtensionEnum = folder_type_determiner.determine_folder_type()
                dataset_type: DatasetType = FileExtensionToDatasetTypeMap[file_extension]
                dataset_name = self.get_dataset_name()
                """
                Figuring out this stuff now validates the directory.
                """
                print(f"Creating {dataset_type} dataset \"{dataset_name}\" from extension {file_extension}.")
                self.cache.dataset = DatasetFinder.create_new_dataset(name=dataset_name, path=path)
                return StatesEnum.LOAD_DATASET_STATE
            except ValueError as e:
                print(f"\nError: {e}\n")
                return StateErrorMessage(message=f"Error: {e}")

    @staticmethod
    def get_dataset_name() -> str:
        dataset_name: str = " "
        while " " in dataset_name:
            dataset_name = input("What do you want to call this dataset?\n")
            if " " in dataset_name:
                print("\nDataset name cannot contain spaces, try again.\n")
        return dataset_name
