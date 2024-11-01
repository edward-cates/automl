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
            forward_fn: Callable[[Any, torch.nn.Module], torch.Tensor],
            loss_fn: Callable[[Any, torch.Tensor], torch.Tensor],
            step_save_fn: Callable[[Any, torch.Tensor], Any],
            eval_fn: Callable[[Any], dict[str, float]],
            test_samples: list[AutomlSample] | None = None,
            batch_size: int = 2,
            data_split_ratio: float = 0.8,
            shuffle_before_split: bool = True,
            optimizer_kwargs: dict = dict(
                lr=0.001,
            ),
    ):
        """
        :forward_fn: Takes (a) data batch, (b) model, returns the model's output.
        :loss_fn: Takes (a) data batch, (b) model's output, returns the loss.
        :step_save_fn: Takes (a) data batch, (b) model's output, returns anything - will be appended to a list.
                        This list is passed to the eval function at the end of the epoch.
        :eval_fn: Takes the list of `step_save_fn` outputs, returns a dictionary of metrics.
        :test_samples: List of samples to use for testing. If test split is created using `data_split_ratio`.
        """
        self.model = AutomlModel(model)
        self.samples = samples
        self.test_samples = test_samples
        self.forward_fn = forward_fn
        self.loss_fn = loss_fn
        self.step_save_fn = step_save_fn
        self.eval_fn = eval_fn
        # Prepare.
        self.optimizer = torch.optim.AdamW(self.model.parameters(), **optimizer_kwargs)
        if test_samples is not None:
            self.train_dataset = AutomlDataset(self.samples)
            self.test_dataset = AutomlDataset(self.test_samples)
        else:
            dataset = AutomlDataset(self.samples)
            self.train_dataset, self.test_dataset = dataset.split(data_split_ratio, shuffle=shuffle_before_split)
        self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)
        self.test_loader = torch.utils.data.DataLoader(self.test_dataset, batch_size=batch_size, shuffle=True)
        self.epoch_number = 0
        self.metrics_over_time = list()

    def run_full_epoch(self) -> dict[str, float]:
        self.metrics_over_time.append(dict(
            train=self.run_train_epoch(),
            test=self.run_test_epoch(),
        ))
        self.epoch_number += 1
        return self.metrics_over_time[-1]

    def run_train_epoch(self) -> dict[str, float]:
        return self._epoch_inner(self.train_loader, is_train=True)

    def run_test_epoch(self) -> dict[str, float]:
        with torch.no_grad():
            return self._epoch_inner(self.test_loader, is_train=False)

    def _epoch_inner(self, dataloader: torch.utils.data.DataLoader, is_train: bool) -> dict[str, float]:
        if is_train:
            self.model.train()
        else:
            self.model.eval()
        step_save_outputs = []
        total_loss = 0
        train_or_test = "Train" if is_train else "Test"
        pbar = tqdm(dataloader, desc=f"{train_or_test} Epoch {self.epoch_number}", total=len(dataloader))
        for batch in pbar:
            outputs = self.forward_fn(batch, self.model)
            loss = self.loss_fn(batch, outputs)
            total_loss += loss.item()
            if is_train:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            step_save_outputs.append(self.step_save_fn(batch, outputs))
            pbar.set_postfix(loss=total_loss / len(step_save_outputs))
        return dict(
            loss=total_loss / len(dataloader),
            **self.eval_fn(step_save_outputs),
        )
