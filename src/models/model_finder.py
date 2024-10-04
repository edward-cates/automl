from pathlib import Path
import importlib

import torch

class ModelFinder:
    """
    To detect and load PyTorch model objects (torch.nn.Module)
    """

    cache_dir: Path = Path("cache/models")

    @staticmethod
    def list_models() -> list[str]:
        """
        List all models in the cache.
        """
        return [
            path.name
            for path in ModelFinder.cache_dir.iterdir()
            if path.is_dir() and not path.name.startswith(".")
        ]

    @staticmethod
    def load_model(name: str) -> torch.nn.Module:
        """
        Load a model from a path.
        """
        path = ModelFinder.cache_dir / name
        assert path.exists(), f"Model does not exist: {path}"
        # there's an __init__.py in the directory.
        # dynamically import the model, figure out the class name, and instantiate it.
        module = importlib.import_module(f"cache.models.{name}.__init__")
        class_name = module.__all__[0]
        model_class = getattr(module, class_name)
        print(f"[debug:model_finder.py] instantiating model: {model_class}")
        return model_class()
