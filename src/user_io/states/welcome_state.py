
from src.user_io.user_io_state import UserIoState
from src.user_io.states.states_enum import StatesEnum
from src.user_io.states.state_error_message import StateErrorMessage
from src.user_io.user_io_cache import UserIoCache
from src.datasets.dataset_finder import DatasetFinder

class WelcomeState(UserIoState):
    def __init__(self, cache: UserIoCache):
        super().__init__(cache)

    def build_prompt(self) -> str:
        return "Welcome to the dataset creator! What would you like to do?" + \
            "\n1. Create a new dataset" + \
            "\n2. Load an existing dataset"

    def handle_response(self, response: str) -> StatesEnum | StateErrorMessage:
        if "1" in response:
            return StatesEnum.CREATE_DATASET_START_STATE
        elif "2" in response:
            dataset_name = WelcomeState.get_dataset_name()
            self.cache.dataset = DatasetFinder.load_dataset(dataset_name)
            return StatesEnum.LOAD_DATASET_STATE
        else:
            return StateErrorMessage(message="Unsure what you meant by that. Please try again.")

    @staticmethod
    def get_dataset_name() -> str:
        # loop until dataset exists.
        dataset_name = input("Enter the dataset name:\n")
        while not DatasetFinder.check_name_exists(dataset_name):
            print("Dataset name does not exist, try again.")
            dataset_name = input("Enter the dataset name:\n")
        return dataset_name
