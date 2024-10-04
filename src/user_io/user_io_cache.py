
from src.datasets.local_dataset import LocalDataset
from src.llm.chatgpt import Tools, ToolDescriptor, ToolArgument

class UserIoCache(Tools):

    class Config:
        arbitrary_types_allowed = True

    dataset: LocalDataset | None = None
    train_dataset: LocalDataset | None = None
    test_dataset: LocalDataset | None = None
    holdout_dataset: LocalDataset | None = None

    @property
    def describe_state_descriptor(self) -> ToolDescriptor:
        return ToolDescriptor(
            name="describe_state",
            description="Describe the current state of the cache.",
        )
    def describe_state(self) -> dict[str, bool]:
        """
        Return a dictionary describing which properties are set.
        """
        return {
            field: (value is not None)
            for field, value in self.model_dump().items()
        }

    @property
    def tool_descriptors(self) -> list[ToolDescriptor]:
        return [
            self.describe_state_descriptor,
        ]


if __name__ == "__main__":
    cache = UserIoCache()
    cache.dataset = 1
    print(cache.describe_state())

