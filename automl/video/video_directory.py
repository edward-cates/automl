from pathlib import Path

from automl.video.video_sample import VideoSample

class VideoDirectory:
    """
    A directory containing video frames.
    """
    def __init__(
            self,
            image_dir: Path,
            labels: dict[int, int],
            image_padding: tuple[int, int] = (0, 0),
            label_padding: tuple[int, int] = (0, 0),
            file_stem_formatter: str = "image_{frame_number:05d}",
            extension: str = "png",
    ):
        """
        :labels: A dictionary mapping frame numbers to labels.
        """
        self.image_dir = image_dir
        assert self.image_dir.exists(), f"Image directory {self.image_dir} does not exist"
        self.labels = labels
        self.image_padding = image_padding
        self.label_padding = label_padding
        self.file_stem_formatter = file_stem_formatter
        self.extension = extension
        self.frame_count = len(list(self.image_dir.glob(f"*.{extension}")))
        # assert that all label values are >0.
        assert all(label > 0 for label in self.labels.values()), "All label values must be >0 (0 is reserved for no-class)"
        assert self.label_padding[0] <= self.image_padding[0], "Label padding before must be <= image padding before"
        assert self.label_padding[1] <= self.image_padding[1], "Label padding after must be <= image padding after"

    def get_samples(self) -> list[VideoSample]:
        return [
            self._create_video_sample(frame_number)
            for frame_number in range(
                self.image_padding[0],
                self.frame_count - self.image_padding[1],
            )
        ]

    def _create_video_sample(self, frame_number: int) -> VideoSample:
        return VideoSample(
            image_dir=self.image_dir,
            frame_number=frame_number,
            label=self._figure_out_label(frame_number),
            pad_before=self.image_padding[0],
            pad_after=self.image_padding[1],
            file_stem_formatter=self.file_stem_formatter,
            extension=self.extension,
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
