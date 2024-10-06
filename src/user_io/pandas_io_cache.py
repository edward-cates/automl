from pathlib import Path
import json

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
            key: self._describe_dataframe(df)
            for key, df in self.dataframes.items()
        }
    
    @staticmethod
    def _describe_dataframe(df: pd.DataFrame) -> dict:
        return {
            "shape": df.shape,
            "columns": df.columns.tolist(),
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
            return f"Successfully read dataframe from {filepath} and stored it under key '{key}': {self._describe_dataframe(df)}"
        except Exception as e:
            return f"Error reading file: {str(e)}"

    @property
    def preview_dataframe_descriptor(self) -> ToolDescriptor:
        return ToolDescriptor(
            name="preview_dataframe",
            description="Get a text-based preview of a dataframe stored in the cache.",
            arguments=[
                ToolArgument(
                    name="key",
                    description="Key of the dataframe in the cache to preview",
                    type="string"
                ),
                ToolArgument(
                    name="num_rows",
                    description="Number of rows to preview (default is 5)",
                    type="integer"
                ),
                ToolArgument(
                    name="num_cols",
                    description="Number of columns to preview (default is 10)",
                    type="integer"
                )
            ]
        )
    def preview_dataframe(self, key: str, num_rows: int = 5, num_cols: int = 10) -> str:
        if key not in self.dataframes:
            return f"Error: No dataframe found with key '{key}'"

        df = self.dataframes[key]

        cols = df.columns.tolist()[:num_cols]

        preview = df.head(num_rows)[cols].to_string()
        
        return f"Preview of dataframe '{key}' (first {num_rows} rows):\n\n{preview}"

    @property
    def group_by_and_simple_aggregate_descriptor(self) -> ToolDescriptor:
        return ToolDescriptor(
            name="group_by_and_simple_aggregate",
            description="Group a dataframe by specified columns and perform simple aggregations (simple means can be described just with a string).",
            arguments=[
                ToolArgument(
                    name="key",
                    description="Key of the dataframe in the cache to perform operations on",
                    type="string"
                ),
                ToolArgument(
                    name="group_by",
                    description="List of column names to group by",
                    type="list[str]"
                ),
                ToolArgument(
                    name="agg_dict",
                    description="""Dictionary (as JSON!) specifying the aggregation operations to perform.
                    Keys are column names, values are aggregation functions as strings.
                    Examples:
                    - {'column1': 'sum', 'column2': 'mean'}
                    - {'sales': 'sum', 'price': 'max', 'quantity': 'mean'}
                    Supported aggregation functions: 'sum', 'mean', 'median', 'min', 'max', 'count', 'std', 'var', 'nunique', and any other pandas aggregation function.""",
                    type="string"
                )
            ]
        )
    def group_by_and_simple_aggregate(
            self,
            key: str,
            group_by: list[str],
            # agg_dict: dict[str, str],
            agg_dict: str,
    ) -> str:
        """
        Group by the given columns and aggregate using the given dictionary.
        """
        df = self.dataframes[key]
        grouped = df.groupby(group_by).agg(json.loads(agg_dict)).reset_index()
        new_key = f"{key}_grouped_by_"
        for col in group_by:
            new_key += f"{col}_"
        new_key = new_key[:-1]
        self.dataframes[new_key] = grouped
        return f"Successfully grouped by {group_by} and aggregated using {agg_dict}, " \
               f"results stored in dataframe '{new_key}': {self._describe_dataframe(grouped)}"

    @property
    def value_counts_descriptor(self) -> ToolDescriptor:
        return ToolDescriptor(
            name="value_counts",
            description="Compute value counts for a specified column in a dataframe.",
            arguments=[
                ToolArgument(
                    name="key",
                    description="Key of the dataframe in the cache to perform operations on",
                    type="string"
                ),
                ToolArgument(
                    name="column",
                    description="Name of the column to compute value counts for",
                    type="string"
                ),
                ToolArgument(
                    name="normalize",
                    description="If True, returns proportions instead of frequencies",
                    type="boolean"
                ),
                ToolArgument(
                    name="sort",
                    description="If True, sort the result in descending order",
                    type="boolean"
                )
            ]
        )

    def value_counts(
            self,
            key: str,
            column: str,
            normalize: bool = False,
            sort: bool = True
    ) -> str:
        """
        Compute value counts for a specified column in a dataframe.
        """
        if key not in self.dataframes:
            return f"Error: No dataframe found with key '{key}'"

        df = self.dataframes[key]
        if column not in df.columns:
            return f"Error: Column '{column}' not found in dataframe '{key}'"

        try:
            value_counts = df[column].value_counts(normalize=normalize, sort=sort)
            result_df = value_counts.reset_index()
            result_df.columns = [column, 'count']
            
            self.dataframes[f"{key}_value_counts_{column}"] = result_df
            
            return f"Successfully computed value counts for column '{column}' in dataframe '{key}'. " \
                   f"Results stored in new dataframe '{key}_value_counts_{column}': \n{result_df.head().to_string()}"
        except Exception as e:
            return f"Error computing value counts: {str(e)}"

