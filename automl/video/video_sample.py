from pathlib import Path

import einops
import torch
import torchvision

from automl.training.automl_sample import AutomlSample

class VideoSample(AutomlSample):
    def __init__(
            self,
            image_dir: Path,
            frame_number: int,
            label: int,
            pad_before: int,
            pad_after: int,
            file_stem_formatter: str,
            extension: str,
            image_resize: tuple[int, int] | None,
    ):
        super().__init__()
        self.image_dir = image_dir
        assert self.image_dir.exists(), f"Image directory {self.image_dir} does not exist"
        self.frame_number = frame_number
        self.label = label
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.file_stem_formatter = file_stem_formatter
        self.extension = extension
        self.image_resize = image_resize

    def get_model_input(self) -> tuple[torch.Tensor, int]:
        return self._get_image_stack(), self.label

    def _get_image_stack(self) -> torch.Tensor:
        image_stack = torch.stack(list(map(self._path_to_tensor, self._get_image_paths())))
        # swap axes from (T, C, H, W) to have shape (C, T, H, W)
        return einops.rearrange(image_stack, "t c h w -> c t h w")

    def _get_image_paths(self) -> list[Path]:
        return [
            self.image_dir / f"{self.file_stem_formatter.format(frame_number=n)}.{self.extension}"
            for n in range(self.frame_number - self.pad_before, self.frame_number + self.pad_after + 1)
        ]

    def _path_to_tensor(self, path: Path) -> torch.Tensor:
        assert path.exists(), f"Image path {path} does not exist"
        rgb_image = torchvision.io.read_image(path.as_posix())
        if self.image_resize is not None:
            rgb_image = torchvision.transforms.Resize(self.image_resize)(rgb_image)
        return rgb_image.float() / 255.0
