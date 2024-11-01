from pathlib import Path
from typing import Any

import torch

class AutomlModel(torch.nn.Module):
    def __init__(self, torch_model: torch.nn.Module):
        super().__init__()
        self.torch_model = torch_model

    def forward(self, *args, **kwargs) -> Any:
        return self.torch_model(*args, **kwargs)

    def save(self, path: Path) -> None:
        torch.save(self.torch_model.state_dict(), path)

    def load(self, path: Path) -> None:
        self.torch_model.load_state_dict(torch.load(path))
