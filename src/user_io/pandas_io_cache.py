from pathlib import Path

import pandas as pd
from src.llm.chatgpt import Tools, ToolDescriptor, ToolArgument

class PandasIoCache(Tools):
    """
    Cache for pandas dataframes.
    """

    dataframes: dict[str, pd.DataFrame] = dict()

    @property
    def describe_current_state_descriptor(self) -> ToolDescriptor:
        return ToolDescriptor(
            name="describe_current_state",
            description="Describe the current state of the cache.",
        )
    def describe_current_state(self) -> dict:
        """
        For each dataframe, return
        - the key,
        - the shape,
        - the columns.
        """
        return {
            key: {
                "shape": df.shape,
                "columns": df.columns.tolist(),
            }
            for key, df in self.dataframes.items()
        }

    @property
    def read_dataframe_descriptor(self) -> ToolDescriptor:
        return ToolDescriptor(
            name="read_dataframe",
            description="Read a dataframe from a file and store it in the cache.",
            arguments=[
                ToolArgument(
                    name="filepath",
                    description="Path to the file to read. Supported formats: .feather, .parquet, .csv",
                    type="string"
                ),
                ToolArgument(
                    name="key",
                    description="Key to store the dataframe under in the cache",
                    type="string"
                )
            ]
        )
    def read_dataframe(self, filepath: str, key: str) -> str:
        path = Path(filepath)
        if not path.exists():
            return f"Error: File {filepath} does not exist."

        try:
            if path.suffix == '.feather':
                df = pd.read_feather(path)
            elif path.suffix == '.parquet':
                df = pd.read_parquet(path)
            elif path.suffix == '.csv':
                df = pd.read_csv(path)
            else:
                return f"Error: Unsupported file format. Supported formats are .feather, .parquet, and .csv"

            self.dataframes[key] = df
            return f"Successfully read dataframe from {filepath} and stored it under key '{key}'"
        except Exception as e:
            return f"Error reading file: {str(e)}"

    @property
    def group_by_and_aggregate_descriptor(self) -> ToolDescriptor:
        return ToolDescriptor(
            name="group_by_and_aggregate",
            description="Group a dataframe by specified columns and perform aggregations.",
            arguments=[
                ToolArgument(
                    name="key",
                    description="Key of the dataframe in the cache to perform operations on",
                    type="string"
                ),
                ToolArgument(
                    name="group_by",
                    description="List of column names to group by",
                    type="array"
                ),
                ToolArgument(
                    name="agg_dict",
                    description="""Dictionary specifying the aggregation operations to perform.
                    Keys are column names, values are aggregation functions as strings.
                    Examples:
                    - {'column1': 'sum', 'column2': 'mean'}
                    - {'sales': 'sum', 'price': 'max', 'quantity': 'mean'}
                    Supported aggregation functions: 'sum', 'mean', 'median', 'min', 'max', 'count', 'std', 'var'""",
                    type="object"
                )
            ]
        )
    def group_by_and_aggregate(
            self,
            key: str,
            group_by: list[str],
            agg_dict: dict[str, str],
    ) -> str:
        """
        Group by the given columns and aggregate using the given dictionary.
        """
        df = self.dataframes[key]
        grouped = df.groupby(group_by).agg(agg_dict)
        self.dataframes[key] = grouped
        return f"Successfully grouped by {group_by} and aggregated using {agg_dict}"


