from enum import Enum

class StatesEnum(Enum):
    WELCOME_STATE = "welcome_state"
    CREATE_DATASET_START_STATE = "create_dataset_start_state"
    LOAD_DATASET_STATE = "load_dataset_state"
    DATASET_SPLIT_STATE = "dataset_split_state"
