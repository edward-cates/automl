from typing import Any, Callable

from tqdm.auto import tqdm
import numpy as np
import torch

from automl.training.automl_model import AutomlModel
from automl.training.automl_sample import AutomlSample
from automl.training.automl_dataset import AutomlDataset

class AutomlTrainer:
    """
    Generalize the task of training (anything).
    """
    def __init__(
            self,
            model: torch.nn.Module,
            samples: list[AutomlSample],
            forward: Callable[[Any, torch.nn.Module], torch.Tensor],
            loss_fn: Callable[[Any, Any], torch.Tensor],
            eval_fn: Callable[[list[np.ndarray]], dict[str, float]],
            batch_size: int = 2,
            data_split_ratio: float = 0.8,
            optimizer_kwargs: dict = dict(
                lr=0.001,
            ),
    ):
        """
        :forward: Takes (a) data batch, (b) model, returns the model's output.
        :loss_fn: Takes (a) data batch, (b) model's output, returns the loss.
        :eval_fn: Takes (a) list of all model's outputs for the epoch (as numpy arrays), returns a dictionary of metrics.
        """
        self.model = AutomlModel(model)
        self.dataset = AutomlDataset(samples)
        self.functions = dict(
            forward=forward,
            loss_fn=loss_fn,
            eval_fn=eval_fn,
        )
        # Prepare.
        self.optimizer = torch.optim.AdamW(self.model.parameters(), **optimizer_kwargs)
        self.train_dataset, self.test_dataset = self.dataset.split(data_split_ratio, shuffle=True)
        self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)
        self.test_loader = torch.utils.data.DataLoader(self.test_dataset, batch_size=batch_size, shuffle=True)
        self.epoch_number = 0
        self.metrics_over_time = dict(
            train=list(),
            test=list(),
        )

    def run_full_epoch(self) -> None:
        self.run_train_epoch()
        self.run_test_epoch()
        self.epoch_number += 1

    def run_train_epoch(self) -> None:
        self.metrics_over_time["train"].append(
            self._epoch_inner(self.train_loader, is_train=True),
        )

    def run_test_epoch(self) -> None:
        self.metrics_over_time["test"].append(
            self._epoch_inner(self.test_loader, is_train=False),
        )

    def _epoch_inner(self, dataloader: torch.utils.data.DataLoader, is_train: bool) -> dict[str, float]:
        all_outputs = []
        total_loss = 0
        train_or_test = "Train" if is_train else "Test"
        pbar = tqdm(dataloader, desc=f"{train_or_test} Epoch {self.epoch_number}", total=len(dataloader))
        for batch in pbar:
            outputs = self.functions["forward"](batch, self.model)
            all_outputs.append(
                outputs.detach().cpu().numpy(),
            )
            loss = self.functions["loss_fn"](batch, outputs)
            total_loss += loss.item()
            if is_train:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            pbar.set_postfix(loss=total_loss / len(dataloader))
        return dict(
            loss=total_loss / len(dataloader),
            **self.functions["eval_fn"](all_outputs),
        )
