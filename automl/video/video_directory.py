from collections.abc import Sequence
from pathlib import Path

import torch
from automl.video.video_sample import VideoSample

class VideoDirectory(Sequence):
    """
    A directory containing video frames.
    """
    def __init__(
            self,
            image_dir: Path | str,
            labels: dict[int, int],
            image_padding: tuple[int, int] = (4, 4),
            label_padding: tuple[int, int] = (1, 1),
            file_stem_formatter: str = "image_{frame_number:04d}",
            extension: str = "jpg",
            image_resize: tuple[int, int] | None = (512, 512),
    ):
        """
        :labels: A dictionary mapping frame numbers to labels.
        """
        self.image_dir = Path(image_dir)
        assert self.image_dir.exists(), f"Image directory {self.image_dir} does not exist"
        self.labels = labels
        self.image_padding = image_padding
        self.label_padding = label_padding
        self.file_stem_formatter = file_stem_formatter
        assert "frame_number" in file_stem_formatter, "File stem formatter must contain 'frame_number'"
        self.extension = extension
        self.image_resize = image_resize
        # Prepare.
        self.frame_count = len(list(self.image_dir.glob(f"*.{extension}")))
        assert self.label_padding[0] <= self.image_padding[0], "Label padding before must be <= image padding before"
        assert self.label_padding[1] <= self.image_padding[1], "Label padding after must be <= image padding after"
        self._frame_numbers = list(range(
            self.image_padding[0],
            self.frame_count - self.image_padding[1],
        ))

    def __len__(self) -> int:
        return len(self._frame_numbers)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        return self._create_video_sample(self._frame_numbers[idx]).get_torch_data()

    def _create_video_sample(self, frame_number: int) -> VideoSample:
        return VideoSample(
            image_dir=self.image_dir,
            frame_number=frame_number,
            label=self._figure_out_label(frame_number),
            pad_before=self.image_padding[0],
            pad_after=self.image_padding[1],
            file_stem_formatter=self.file_stem_formatter,
            extension=self.extension,
            image_resize=self.image_resize,
        )

    def _figure_out_label(self, frame_number: int) -> int:
        if frame_number in self.labels:
            return self.labels[frame_number]
        for i in range(1, max(self.label_padding) + 1):
            if i < self.label_padding[0] and frame_number - i in self.labels:
                return self.labels[frame_number - i]
            if i < self.label_padding[1] and frame_number + i in self.labels:
                return self.labels[frame_number + i]
        # No-class label.
        return 0
