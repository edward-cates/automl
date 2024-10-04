from abc import ABC, abstractmethod

from src.user_io.states.states_enum import StatesEnum
from src.user_io.states.state_error_message import StateErrorMessage
from src.user_io.user_io_cache import UserIoCache

class UserIoState(ABC):

    def __init__(self, cache: UserIoCache):
        self.cache = cache

    def run(self) -> StatesEnum:
        print("\n" + str(self.cache) + "\n" + self.build_prompt())
        next_state = StateErrorMessage(message="")
        while isinstance(next_state, StateErrorMessage):
            response = input()
            print()
            next_state: StatesEnum | StateErrorMessage = self.handle_response(response)
            if isinstance(next_state, StateErrorMessage):
                print(next_state.message)
        return next_state

    @abstractmethod
    def build_prompt(self) -> str:
        pass

    @abstractmethod
    def handle_response(self, response: str) -> StatesEnum | StateErrorMessage:
        pass

