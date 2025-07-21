# automl

## Quick Start

😉 The only kind of start there is with this package.

```python
import torch
from automl import AutomlTrainer

# train_samples: list[tuple[torch.Tensor, torch.Tensor]]
# test_samples: list[tuple[torch.Tensor, torch.Tensor]]

def forward_fn(batch, model):
    inputs, labels = batch
    return model(inputs)

def loss_fn(batch, outputs):
    input_sequences, labels = batch
    return torch.abs(outputs - labels).mean()

def step_save_fn(batch, outputs) -> dict:
    input_sequences, labels = batch
    predictions = outputs
    return dict(
        labels=labels.numpy(),
        predictions=predictions.detach().cpu().numpy(),
    )

def eval_fn(step_results: list[dict]) -> dict:
    labels = np.concatenate([result["labels"] for result in step_results])
    predictions = np.concatenate([result["predictions"] for result in step_results])
    return dict(
        mean_error=np.mean(np.abs(labels - predictions)),
    )

def epoch_compare_fn(best_results: list[dict] | None, step_results: list[dict]) -> bool:
    return best_results is None or step_results["mean_error"] < best_results["mean_error"]

trainer = AutomlTrainer(
    model=model,
    samples=train_samples,
    forward_fn=forward_fn,
    loss_fn=loss_fn,
    step_save_fn=step_save_fn,
    eval_fn=eval_fn,
    epoch_compare_fn=epoch_compare_fn,
    test_samples=test_samples,
    batch_size=2,
    optimizer_kwargs=dict(
        lr=1e-4,
    ),
)

trainer.run_test_epoch()

for _ in range(100):
    trainer.run_full_epoch()

train_losses = [
    f'{d["train"]["loss"]:.2f}' for d in trainer.metrics_over_time
]
test_losses = [
    f'{d["test"]["loss"]:.2f}' for d in trainer.metrics_over_time
]
# TODO: plot loss over epoch.

trainer.best_epoch_results

torch.save(trainer.best_model_state_dict, "best_model_state_dict.pt")
```

