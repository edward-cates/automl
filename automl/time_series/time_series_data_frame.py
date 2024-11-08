from tqdm.auto import tqdm
import pandas as pd

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
            min_seq_len: int = 10, # Sets size of smallest sample that can be made.
    ):
        self.df = df.sort_values(by=[group_col, time_col]).reset_index(drop=True)
        self.time_col = time_col
        self.time_increment = time_increment
        self.group_col = group_col
        self.y_col = y_col
        self.feature_cols = feature_cols
        self.min_seq_len = min_seq_len
        self._fill_time_gaps()
        self._validate()

    def __len__(self) -> int:
        """
        Length 
        """
        return len(self.df)

    # Private.

    def _validate(self):
        assert self.time_col in self.df.columns, "Time column not in dataframe."
        assert self.group_col in self.df.columns, "Group column not in dataframe."
        assert self.y_col in self.df.columns, "Y column not in dataframe."
        assert all(col in self.df.columns for col in self.feature_cols), "All feature columns must be in dataframe."

    def _fill_time_gaps(self):
        """
        Some groups may be missing timestamps. Fill them in (the middle ones - don't need to extend the start or end).

            Don't assume that time columns are any kind of datetime.
        """
        # Use progress apply.
        tqdm.pandas()
        self.df = self.df.groupby(self.group_col).progress_apply(self._fill_time_gaps_for_group) \
            .reset_index(drop=True)

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
        reindexed_group[self.group_col] = group.name
        return reindexed_group.reset_index(drop=False)
