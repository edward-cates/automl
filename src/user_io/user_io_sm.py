
from src.user_io.states.states_enum import StatesEnum
from src.user_io.user_io_state import UserIoState
from src.user_io.states.welcome_state import WelcomeState
from src.user_io.states.dataset_creation.create_dataset_start_state import CreateDatasetStartState
from src.user_io.user_io_cache import UserIoCache
from src.user_io.states.dataset_manipulation.load_dataset_state import LoadDatasetState
from src.user_io.states.dataset_manipulation.dataset_split_state import DatasetSplitState

class UserIoSm:

    def __init__(self) -> None:
        self.cache = UserIoCache()
        self.current_state: UserIoState = WelcomeState(cache=self.cache)

    def step(self) -> None:
        next_state_enum = self.current_state.run()

        if next_state_enum == StatesEnum.WELCOME_STATE:
            self.current_state = WelcomeState(cache=self.cache)
        elif next_state_enum == StatesEnum.CREATE_DATASET_START_STATE:
            self.current_state = CreateDatasetStartState(cache=self.cache)
        elif next_state_enum == StatesEnum.LOAD_DATASET_STATE:
            self.current_state = LoadDatasetState(cache=self.cache)
        elif next_state_enum == StatesEnum.DATASET_SPLIT_STATE:
            self.current_state = DatasetSplitState(cache=self.cache)
        else:
            raise ValueError(f"Invalid state: {next_state_enum}")
