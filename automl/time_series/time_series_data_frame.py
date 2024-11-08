import itertools
from tqdm.auto import tqdm
import pandas as pd
import torch
class TimeSeriesDataFrame:
    """
    A class to help with operations on a time series dataframe,
    which can get confusing real quick.

        Can be used as a torch dataset too.
    """
    def __init__(
            self,
            df: pd.DataFrame,
            time_col: str,
            time_increment: int | pd.Timedelta,
            group_col: str,
            y_col: str,
            feature_cols: list[str],
            fillna: float = -1.0,
            min_seq_len: int = 10, # Sets size of smallest sample that can be made.
            max_seq_len: int | None = None,
    ):
        self.df = df.sort_values(by=[group_col, time_col]).reset_index(drop=True)
        self.time_col = time_col
        self.time_increment = time_increment
        self.group_col = group_col
        self.y_col = y_col
        self.feature_cols = feature_cols
        self.fillna = fillna
        self.min_seq_len = min_seq_len
        self.max_seq_len = max_seq_len
        if self.max_seq_len is not None:
            assert self.max_seq_len >= self.min_seq_len, f"Internal bug: {self.max_seq_len=} < {self.min_seq_len=}"
        self._validate()
        self._groups: list[pd.DataFrame] = self._split_groups()
        self._group_indexes, self._sample_indexes = self._map_indexes()

    def _map_indexes(self) -> tuple[list[int], list[int]]:
        """
        Returns a list of length (num_samples),
        where each value is the index of the group that the sample belongs to
        and a list of length (num_samples),
        where each value is the index of the sample within its group.
        """
        group_indexes = []
        sample_indexes = []

        for group_idx, group in tqdm(enumerate(self._groups), total=len(self._groups), desc="Mapping indexes"):
            target_values = group[self.y_col].values
            for sample_idx in range(0, len(group) - self.min_seq_len + 1):
                if pd.notna(target_values[sample_idx + self.min_seq_len - 1]):
                    group_indexes.append(group_idx)
                    sample_indexes.append(sample_idx)

        assert len(group_indexes) == len(sample_indexes), f"Internal bug: {len(group_indexes)=} != {len(sample_indexes)=}"
        return group_indexes, sample_indexes

    def __len__(self) -> int:
        return len(self._group_indexes)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns (
            time-series frame of (seq_len, num_features),
            target value
        )
        """
        group_idx = self._group_indexes[idx]
        sample_idx = self._sample_indexes[idx]
        end_idx = self.min_seq_len + sample_idx
        start_idx = max(0, end_idx - self.max_seq_len) if self.max_seq_len is not None else 0
        data = self._groups[group_idx].iloc[start_idx:end_idx]
        assert len(data) >= self.min_seq_len, f"Internal bug: {len(data)=} < {self.min_seq_len=}"
        if self.max_seq_len is not None:
            assert len(data) <= self.max_seq_len, f"Internal bug: {len(data)=} > {self.max_seq_len=}"
        target_value = data.iloc[-1][self.y_col]
        assert pd.notna(target_value), "Target value is NaN."
        return (
            torch.tensor(data[self.feature_cols].fillna(self.fillna).values),
            torch.tensor(target_value),
        )

    # Private.

    def _validate(self):
        assert self.time_col in self.df.columns, "Time column not in dataframe."
        assert self.group_col in self.df.columns, "Group column not in dataframe."
        assert self.y_col in self.df.columns, "Y column not in dataframe."
        assert all(col in self.df.columns for col in self.feature_cols), "All feature columns must be in dataframe."

    def _split_groups(self) -> list[pd.DataFrame]:
        """
        Some groups may be missing timestamps. Fill them in (the middle ones - don't need to extend the start or end).

            Don't assume that time columns are any kind of datetime.
        """
        num_groups = self.df[self.group_col].nunique()
        return [
            self._perform_group_operations(group)
            for group_id, group in tqdm(self.df.groupby(self.group_col), total=num_groups, desc="Splitting groups")
        ]

    def _perform_group_operations(self, group: pd.DataFrame):
        return self._fill_time_gaps_for_group(group)

    def _fill_time_gaps_for_group(self, group: pd.DataFrame) -> pd.DataFrame:
        """
        Fill time gaps for a single group.
        """
        min_time = group[self.time_col].min()
        max_time = group[self.time_col].max()
        # if increment is an integer, just use a range.
        all_times = list(range(min_time, max_time + self.time_increment, self.time_increment)) \
            if isinstance(self.time_increment, int) \
            else pd.date_range(start=min_time, end=max_time, freq=self.time_increment)
        group.set_index(self.time_col, inplace=True)
        reindexed_group = group.reindex(all_times)
        reindexed_group[self.group_col] = group[self.group_col].iloc[0]
        return reindexed_group.reset_index(drop=False)
