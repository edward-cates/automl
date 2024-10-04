
from src.user_io.user_io_state import UserIoState
from src.user_io.states.states_enum import StatesEnum
from src.user_io.states.state_error_message import StateErrorMessage
from src.user_io.user_io_cache import UserIoCache

class LoadDatasetState(UserIoState):

    def __init__(self, cache: UserIoCache):
        super().__init__(cache)

    def build_prompt(self) -> str:
        return f"Welcome to dataset \"{self.cache.dataset.name}\"." + \
            "\n1. Split dataset,"

    def handle_response(self, response: str) -> StatesEnum | StateErrorMessage:
        if "1" in response:
            return StatesEnum.DATASET_SPLIT_STATE
        return StateErrorMessage("Not implemented yet.")

