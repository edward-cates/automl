from collections import deque
from collections.abc import Sequence
from typing import Any, Callable

from tqdm.auto import tqdm
import numpy as np
import torch
from torch.optim.adamw import AdamW
from torch.utils.data import DataLoader

from automl.training.automl_model import AutomlModel
from automl.training.automl_dataset import AutomlDataset

class AutomlTrainer:
    """
    Generalize the task of training (anything).
    """
    def __init__(
            self,
            model: torch.nn.Module,
            samples: Sequence[Any],
            forward_fn: Callable[[Any, torch.nn.Module], torch.Tensor],
            loss_fn: Callable[[Any, torch.Tensor], torch.Tensor],
            step_save_fn: Callable[[Any, torch.Tensor], Any],
            eval_fn: Callable[[Any], dict[str, float]],
            epoch_compare_fn: Callable[[dict[str, float] | None, dict[str, float]], bool],
            test_samples: Sequence | None = None,
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
        :epoch_compare_fn: Given (best epoch results) and (most recent epoch results), returns true if the most recent epoch is better.
                        Called even after first epoch, when `best_epoch_results` is `None`.
        :test_samples: List of samples to use for testing. If test split is created using `data_split_ratio`.

            Note that this class does not handle the device - let the caller do that.
        """
        self.model = AutomlModel(model)
        self.samples = samples
        self.test_samples = test_samples
        self.forward_fn = forward_fn
        self.loss_fn = loss_fn
        self.step_save_fn = step_save_fn
        self.eval_fn = eval_fn
        self.epoch_compare_fn = epoch_compare_fn
        self.batch_size = batch_size
        # Prepare.
        self.optimizer = AdamW(self.model.parameters(), **optimizer_kwargs)
        if test_samples is not None:
            self.train_dataset = AutomlDataset(self.samples)
            self.test_dataset = AutomlDataset(self.test_samples)
        else:
            dataset = AutomlDataset(self.samples)
            self.train_dataset, self.test_dataset = dataset.split(data_split_ratio, shuffle=shuffle_before_split)
        self.train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True) # type: ignore
        self.test_loader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=True) # type: ignore
        self.epoch_number = 0
        self.metrics_over_time = list()
        self.best_epoch_results = None
        self.best_model_state_dict = None

    def run_over_data(self, samples: Sequence[Any], break_at: int | None = None) -> dict[str, float]:
        dataset = AutomlDataset(samples)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False) # type: ignore
        with torch.no_grad():
            return self._epoch_inner(dataloader, is_train=False, break_at=break_at)

    def run_full_epoch(self, break_at: int | tuple[int, int] | None = None, use_tqdm: bool = True) -> dict[str, float]:
        break_at_train = break_at[0] if isinstance(break_at, tuple) else break_at
        break_at_test = break_at[1] if isinstance(break_at, tuple) else break_at
        epoch_results = dict(
            train=self.run_train_epoch(break_at=break_at_train, use_tqdm=use_tqdm),
            test=self.run_test_epoch(break_at=break_at_test, use_tqdm=use_tqdm),
        )
        self.metrics_over_time.append(epoch_results)
        if self.epoch_compare_fn(
            self.best_epoch_results["test"] if self.best_epoch_results is not None else None,
            epoch_results["test"],
        ):
            self.best_epoch_results = epoch_results
            self.best_model_state_dict = {k: v.clone() for k, v in self.model.torch_model.state_dict().items()}
            # print(f"Updated best model state dict at epoch {self.epoch_number}.")
        self.epoch_number += 1
        return self.metrics_over_time[-1]

    def run_train_epoch(self, break_at: int | None = None, use_tqdm: bool = True) -> dict[str, float]:
        return self._epoch_inner(self.train_loader, is_train=True, break_at=break_at, use_tqdm=use_tqdm)

    def run_test_epoch(self, break_at: int | None = None, use_tqdm: bool = True) -> dict[str, float]:
        with torch.no_grad():
            return self._epoch_inner(self.test_loader, is_train=False, break_at=break_at, use_tqdm=use_tqdm)

    def reload_best_checkpoint(self):
        assert self.best_model_state_dict is not None
        self.model.torch_model.load_state_dict(self.best_model_state_dict)

    def _epoch_inner(
            self,
            dataloader: DataLoader,
            is_train: bool,
            break_at: int | None = None,
            use_tqdm: bool = True,
    ) -> dict[str, float]:
        if is_train:
            self.model.train()
        else:
            self.model.eval()
        step_save_outputs = []
        total_loss = 0
        loss_tail = deque(maxlen=100)
        train_or_test = "Train" if is_train else "Test"
        pbar = tqdm(dataloader, desc=f"{train_or_test} Epoch {self.epoch_number}", total=len(dataloader)) if use_tqdm else dataloader
        for batch in pbar:
            outputs = self.forward_fn(batch, self.model)
            loss = self.loss_fn(batch, outputs)
            total_loss += loss.item()
            loss_tail.append(loss.item())
            if is_train:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            step_save_outputs.append(self.step_save_fn(batch, outputs))
            if use_tqdm:
                pbar.set_postfix(loss=total_loss / len(step_save_outputs), loss_tail=np.mean(loss_tail))
            if break_at is not None and len(step_save_outputs) >= break_at:
                print(f"WARNING: Breaking at {len(step_save_outputs)} steps.")
                break
        return dict(
            epoch_number=self.epoch_number,
            loss=total_loss / len(dataloader),
            **self.eval_fn(step_save_outputs),
        )
