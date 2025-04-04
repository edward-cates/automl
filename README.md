# automl

## TODO:

- Tokenization and move batch to device - figure that out, not great right now.

## Usage

```python
from automl import VideoDirectory, AutomlTrainer

directory = VideoDirectory(
    image_dir="train_frames",
    labels=labels,
    image_padding=(2, 0),
    label_padding=(0, 0),
    file_stem_formatter="frame_{frame_number:04d}",
    extension="jpg",
    image_resize=None,
)
samples = directory.get_samples()

device = "cuda:0"
model.to(device)
print()

import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

loss_weights = torch.tensor([1.0, 10.0, 10.0], device=device)

def forward_fn(batch, model):
    images, labels = batch
    return model(images.to(device))

def loss_fn(batch, outputs):
    images, labels = batch
    return torch.nn.functional.cross_entropy(outputs, labels.to(device), weight=loss_weights)

def step_save_fn(batch, outputs) -> dict:
    images, labels = batch
    predictions = torch.argmax(outputs, dim=1)
    return dict(
        labels=labels.numpy(),
        predictions=predictions.detach().cpu().numpy(),
    )

def eval_fn(step_results: list[dict]) -> dict:
    labels = np.concatenate([result["labels"] for result in step_results])
    predictions = np.concatenate([result["predictions"] for result in step_results])
    return dict(
        accuracy=accuracy_score(labels, predictions),
        f1=f1_score(labels, predictions, average="macro"),
    )

trainer = AutomlTrainer(
    model=model,
    samples=samples,
    forward_fn=forward_fn,
    loss_fn=loss_fn,
    step_save_fn=step_save_fn,
    eval_fn=eval_fn,
    batch_size=2,
    data_split_ratio=0.75,
    shuffle_before_split=True,
    optimizer_kwargs=dict(
        lr=1e-4,
    ),
)

trainer.run_test_epoch()

trainer.run_full_epoch()

# save the model to model.pth.
torch.save(model.state_dict(), "model.pth") # 0.90
```

And

```python
from automl import AutomlTrainer, TimeSeriesDataFrame

...

import torch
import numpy as np
from sklearn.metrics import r2_score
import plotly.graph_objects as go

loss_criterion = torch.nn.HuberLoss(delta=1.0)
# loss_criterion = nn.MSELoss()
device = "cuda:0"
transformer_model.to(device)

def move_batch_to_device_fn(batch):
    sequences, y = batch
    return sequences.to(device), y.to(device)

def forward_fn(batch, model):
    sequences, y = batch
    return model(sequences)

def loss_fn(batch, output):
    sequences, y = batch
    # return (output - y).abs().mean()
    return loss_criterion(output, y)

def step_save_fn(batch, output):
    # Anything to save per step?
    sequences, y = batch
    return {
        "targets": y.detach().cpu().numpy().flatten(),
        "preds": output.detach().cpu().numpy().flatten(),
        "target_by_ix": {
            ix: y[:, ix].detach().cpu().numpy().flatten()
            for ix in range(y.shape[1])
        },
        "preds_by_ix": {
            ix: output[:, ix].detach().cpu().numpy().flatten()
            for ix in range(output.shape[1])
        },
    }

def eval_fn(outputs):
    all_targets = np.concatenate([o["targets"] for o in outputs])
    all_preds = np.concatenate([o["preds"] for o in outputs])
    all_targets_by_ix = {
        ix: np.concatenate([o["target_by_ix"][ix] for o in outputs])
        for ix in range(len(outputs[0]["target_by_ix"]))
    }
    all_preds_by_ix = {
        ix: np.concatenate([o["preds_by_ix"][ix] for o in outputs])
        for ix in range(len(outputs[0]["preds_by_ix"]))
    }
    # just plot 200 random points.
    idx = np.random.choice(len(all_targets), 200)
    fig = go.Figure([go.Scatter(x=all_targets[idx], y=all_preds[idx], mode="markers")])
    # label the axes.
    fig.update_xaxes(title="Target")
    fig.update_yaxes(title="Predicted")
    return dict(
        r2_score=r2_score(all_targets, all_preds),
        fig=fig,
        r2_by_ix={
            ix: r2_score(all_targets_by_ix[ix], all_preds_by_ix[ix])
            for ix in range(len(all_targets_by_ix))
        },
    )

trainer = AutomlTrainer(
    model=transformer_model,
    samples=tsdf_train,
    test_samples=tsdf_test,
    move_batch_to_device_fn=move_batch_to_device_fn,
    forward_fn=forward_fn,
    loss_fn=loss_fn,
    step_save_fn=step_save_fn,
    eval_fn=eval_fn,
    batch_size=256,
    shuffle_before_split=False,
    optimizer_kwargs=dict(
        lr=1e-4,
        # weight_decay=1e-6,
    ),
)

eval_results = trainer.run_test_epoch(break_at=10)
import json
print(eval_results["r2_score"])
print(json.dumps(eval_results["r2_by_ix"], indent=2))

for epoch in range(200):
    eval_results = trainer.run_full_epoch((None, 200))
    print(eval_results["train"]["r2_score"])
    # eval_results["train"]["fig"].show()
    print(eval_results["test"]["r2_score"])
    # eval_results["test"]["fig"].show()
    print(json.dumps(eval_results["test"]["r2_by_ix"], indent=2))
    # break
    # save model
    torch.save(trainer.model.torch_model.state_dict(), f"transformer_model_epoch_{epoch}.pth")

eval_results["test"]["fig"].show()

# save model
# torch.save(trainer.model.torch_model.state_dict(), "transformer_model.pth")
```
