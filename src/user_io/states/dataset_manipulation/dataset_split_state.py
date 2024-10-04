

from src.user_io.user_io_state import UserIoState
from src.user_io.states.states_enum import StatesEnum
from src.user_io.states.state_error_message import StateErrorMessage
from src.user_io.user_io_cache import UserIoCache
from src.datasets.local_dataset import LocalDataset

class DatasetSplitState(UserIoState):

    def __init__(self, cache: UserIoCache):
        super().__init__(cache)

    def build_prompt(self) -> str:
        return f"How to split dataset \"{self.cache.dataset.name}\"?" + \
            "\n1. File name,"

    def handle_response(self, response: str) -> StatesEnum | StateErrorMessage:
        if "1" in response:
            train_samples: float = self.get_train_samples()
            dataset_splits: tuple[LocalDataset] = self.cache.dataset.split(
                how="file_name",
                train_samples=train_samples,
                holdout_set=False,
            )
            self.cache.train_dataset = dataset_splits[0]
            self.cache.test_dataset = dataset_splits[1]
            return StatesEnum.LOAD_DATASET_STATE
        return StateErrorMessage("Not implemented yet.")

    def get_train_samples(self) -> int:
        dataset_length = len(self.cache.dataset)

        response = input(
            f"The dataset has {dataset_length} samples. How many samples to use for training?" + \
                "\n  - Enter an integer for the number of samples OR a fraction between 0 and 1.\n"
        )

        def parse_response(response: str) -> int | None:
            try:
                value = float(response)
                if 0 < value < 1:
                    return int(value * dataset_length)
                elif 1 <= value <= dataset_length:
                    return int(value)
            except ValueError:
                pass
            return None

        train_samples = parse_response(response)
        while train_samples is None:
            response = input("Invalid input. Please try again.\n")
            train_samples = parse_response(response)

        return train_samples
