from pathlib import Path
from abc import ABC, abstractmethod
from typing import Any

import torch

class AutomlModel(torch.nn.Module, ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, *args, **kwargs) -> Any:
        pass

    def save(self, path: Path) -> None:
        torch.save(self.state_dict(), path)

    def load(self, path: Path) -> None:
        self.load_state_dict(torch.load(path))
