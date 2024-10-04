
from src.datasets.local_dataset import LocalDataset
from src.datasets.dataset_finder import DatasetFinder
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
    def describe_state(self) -> dict[str, str]:
        """
        Return a dictionary describing which properties are set.
        """
        return {
            "dataset": self.dataset.name,
            "train_dataset": len(self.train_dataset),
            "test_dataset": len(self.test_dataset),
        }

    @property
    def list_datasets_descriptor(self) -> ToolDescriptor:
        return ToolDescriptor(
            name="list_datasets",
            description="List all available datasets in the cache.",
        )

    def list_datasets(self) -> str:
        """
        List all datasets found in the cache.
        """
        datasets = DatasetFinder.list_datasets()
        if not datasets:
            return "No datasets found in the cache."
        dataset_list = "\n".join([str(dataset) for dataset in datasets])
        return f"Available datasets:\n{dataset_list}"

    @property
    def choose_dataset_descriptor(self) -> ToolDescriptor:
        return ToolDescriptor(
            name="choose_dataset",
            description="Choose a dataset and set it as the current dataset.",
            arguments=[
                ToolArgument(
                    name="dataset_name",
                    description="The name of the dataset to choose",
                    type="string",
                ),
            ],
        )
    def choose_dataset(self, dataset_name: str) -> str:
        """
        Set the dataset based on the provided name.
        """
        self.dataset = DatasetFinder.load_dataset(dataset_name)
        return f"Dataset '{dataset_name}' has been selected." + \
            f"\nNew state: {self.describe_state()}"

    @property
    def tool_descriptors(self) -> list[ToolDescriptor]:
        return [
            self.describe_state_descriptor,
            self.list_datasets_descriptor,
            self.choose_dataset_descriptor,
        ]
    
    @property
    def split_dataset_descriptor(self) -> ToolDescriptor:
        return ToolDescriptor(
            name="split_dataset",
            description="Split the current dataset into train and test sets.",
            arguments=[
                ToolArgument(
                    name="train_samples_or_frac",
                    description="Number of samples or fraction for the training set. If between 0 and 1, it's treated as a fraction.",
                    type="number",
                ),
            ],
        )
    def split_dataset(self, train_samples_or_frac: float) -> str:
        """
        Split the current dataset into train and test sets.
        """
        if self.dataset is None:
            return "No dataset is currently selected. Please choose a dataset first."

        dataset_length = len(self.dataset)
        
        if 0 < train_samples_or_frac < 1:
            train_samples = int(train_samples_or_frac * dataset_length)
        else:
            train_samples = int(train_samples_or_frac)

        if train_samples <= 0 or train_samples >= dataset_length:
            return f"Invalid number of samples. Please provide a value between 1 and {dataset_length - 1}."

        self.train_dataset, self.test_dataset = self.dataset.split(
            train_samples=train_samples,
        )
        return f"Dataset split into train ({train_samples} samples) and test ({dataset_length - train_samples} samples) sets."

    @property
    def tool_descriptors(self) -> list[ToolDescriptor]:
        return [
            self.describe_state_descriptor,
            self.list_datasets_descriptor,
            self.choose_dataset_descriptor,
            self.split_dataset_descriptor,
        ]


if __name__ == "__main__":
    cache = UserIoCache()
    cache.dataset = 1
    print(cache.describe_state())

