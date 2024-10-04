from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.datasets.local_dataset import LocalDataset
from src.datasets.dataset_finder import DatasetFinder
from src.models.model_finder import ModelFinder
from src.llm.chatgpt import Tools, ToolDescriptor, ToolArgument

class UserIoCache(Tools):

    class Config:
        arbitrary_types_allowed = True

    dataset: LocalDataset | None = None
    train_dataset: LocalDataset | None = None
    test_dataset: LocalDataset | None = None

    train_dataloader: DataLoader | None = None
    test_dataloader: DataLoader | None = None

    models: dict[str, torch.nn.Module] = dict()

    @property
    def describe_state_descriptor(self) -> ToolDescriptor:
        return ToolDescriptor(
            name="describe_state",
            description="Describe the current state of the cache.",
        )
    def describe_state(self) -> dict:
        """
        Return a dictionary describing which properties are set.
        """
        return {
            "dataset": f"{self.dataset.name} ({type(self.dataset).__name__}, len={len(self.dataset)})" if self.dataset else "not set",
            "train_dataset": len(self.train_dataset) if self.train_dataset else "not set",
            "test_dataset": len(self.test_dataset) if self.test_dataset else "not set",
            "train_dataloader": f"batch_size={self.train_dataloader.batch_size}" if self.train_dataloader else "not set",
            "test_dataloader": f"batch_size={self.test_dataloader.batch_size}" if self.test_dataloader else "not set",
            "models": self.models.keys(),
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
    def list_models_descriptor(self) -> ToolDescriptor:
        return ToolDescriptor(
            name="list_models",
            description="List all available models in the cache.",
        )

    def list_models(self) -> str:
        """
        List all models found in the cache.
        """
        models = ModelFinder.list_models()
        if not models:
            return "No models found in the cache."
        model_list = "\n".join([str(model) for model in models])
        return f"Available models:\n{model_list}"

    @property
    def load_model_descriptor(self) -> ToolDescriptor:
        return ToolDescriptor(
            name="load_model",
            description="Load a model from the cache and add it to the current models.",
            arguments=[
                ToolArgument(
                    name="model_name",
                    description="The name of the model to load",
                    type="string",
                ),
            ],
        )

    def load_model(self, model_name: str) -> str:
        """
        Load a model from the cache and add it to the current models.
        """
        model = ModelFinder.load_model(model_name)
        self.models[model_name] = model
        return f"Model '{model_name}' has been loaded and added to the current models." + \
            f"\nNew state: {self.describe_state()}"

    @property
    def create_new_dataset_descriptor(self) -> ToolDescriptor:
        return ToolDescriptor(
            name="create_new_dataset",
            description="Create a new dataset from a local path and add it to the cache.",
            arguments=[
                ToolArgument(
                    name="name",
                    description="The name for the new dataset",
                    type="string",
                ),
                ToolArgument(
                    name="path",
                    description="The local path to the existing data directory. It should contain files of the same type.",
                    type="string",
                ),
            ],
        )
    def create_new_dataset(self, name: str, path: str) -> str:
        """
        Create a new dataset from a local path and add it to the cache.
        """
        new_dataset = DatasetFinder.create_new_dataset(name=name, path=Path(path))
        self.dataset = new_dataset
        return f"New dataset '{name}' has been created and set as the current dataset." + \
            f"\nNew state: {self.describe_state()}"

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
    def create_dataloader_descriptor(self) -> ToolDescriptor:
        return ToolDescriptor(
            name="create_dataloader",
            description="Create a DataLoader for the train or test dataset.",
            arguments=[
                ToolArgument(
                    name="is_train",
                    description="Whether to create a DataLoader for the training set (True) or test set (False).",
                    type="boolean",
                ),
                ToolArgument(
                    name="batch_size",
                    description="The batch size for the DataLoader.",
                    type="integer",
                ),
                ToolArgument(
                    name="shuffle",
                    description="Whether to shuffle the data in the DataLoader.",
                    type="boolean",
                ),
            ],
        )
    def create_dataloader(self, is_train: bool, batch_size: int, shuffle: bool) -> str:
        """
        Create a DataLoader for the train or test dataset.
        """
        if is_train and self.train_dataset is None:
            return "No training dataset is available. Please split the dataset first."
        elif not is_train and self.test_dataset is None:
            return "No test dataset is available. Please split the dataset first."

        dataset = self.train_dataset if is_train else self.test_dataset

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle
        )

        set_type = "training" if is_train else "test"
        self.dataloader = dataloader
        
        return f"Created DataLoader for the {set_type} set with batch size {batch_size} and shuffle={shuffle}."

    @property
    def tool_descriptors(self) -> list[ToolDescriptor]:
        return [
            getattr(self, attr) for attr in dir(self)
            if attr.endswith("_descriptor")
        ]


if __name__ == "__main__":
    cache = UserIoCache()
    cache.dataset = 1
    print(cache.describe_state())

