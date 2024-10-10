from pathlib import Path
from pydantic import BaseModel

from datasets.dataset_type_enum import DatasetType

class DatasetDescriptor(BaseModel):
    path: Path
    dataset_type: DatasetType
