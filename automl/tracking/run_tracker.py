import hashlib
import json
from pathlib import Path

import plotly.graph_objects as go

def hash_str(to_hash: str) -> str:
    """
    Create a unique hex hash of any string.
    
        Outputs should all be the same length.
    """
    return hashlib.sha256(to_hash.encode("utf-8")).hexdigest()

class RunTracker:
    """
    Track one run of anything.
    Save:
    1. Hyperparameters / config variables. Runs will be grouped by these.
    2. Metrics. Each run has many metrics.
    3. Plotly HTML graphs. Each run can have many graphs.
    """
    data_dir: Path = Path("cache")

    def __init__(self, config: dict):
        self.data_dir.mkdir(parents=False, exist_ok=True)
        self.experiment_dir = self.data_dir / hash_str(json.dumps(config))
        self.experiment_dir.mkdir(parents=False, exist_ok=True)
        self._write_config(config)

    def log_metrics(self, metrics: dict):
        existing_metrics = self._read_metrics()
        existing_metrics.update(metrics)
        metrics_path = self.experiment_dir / "metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(existing_metrics, f, indent=4)
            f.write("\n")

    def log_graph(self, name: str, graph: go.Figure):
        graph_dir = self.experiment_dir / "graphs"
        graph_dir.mkdir(parents=False, exist_ok=True)
        graph_path = graph_dir / f"{name}.html"
        graph.write_html(graph_path)

    def _write_config(self, config: dict):
        config_path = self.experiment_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump(config, f, indent=4)
            f.write("\n")

    def _read_metrics(self) -> dict:
        metrics_path = self.experiment_dir / "metrics.json"
        if not metrics_path.exists():
            return dict()
        with open(metrics_path, "r") as f:
            return json.load(f)
