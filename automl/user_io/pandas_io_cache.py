from pathlib import Path
import json
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from src.user_io.session_cache import SessionCache, CacheObject
from src.llm.chatgpt import Tools, ToolDescriptor, ToolArgument

class PandasIoCache(Tools):
    """
    Cache for pandas dataframes.
    """

    cache: SessionCache = SessionCache()

    @property
    def describe_cache_descriptor(self) -> ToolDescriptor:
        return ToolDescriptor(
            name="describe_cache",
            description="Describe the contents of the cache",
            arguments=[],
        )
    def describe_cache(self) -> str:
        return self.cache.describe()

    @property
    def preview_dataframe_descriptor(self) -> ToolDescriptor:
        return ToolDescriptor(
            name="preview_dataframe",
            description="Preview a dataframe stored in the cache and return it as a string",
            arguments=[
                ToolArgument(
                    name="cache_key",
                    description="The cache key of the dataframe to preview",
                    type="string",
                ),
                ToolArgument(
                    name="num_rows",
                    description="Number of rows to preview (default: 5)",
                    type="integer",
                ),
                ToolArgument(
                    name="num_cols",
                    description="Number of columns to preview (default: 10)",
                    type="integer",
                ),
            ],
        )
    def preview_dataframe(self, cache_key: str, num_rows: int = 5, num_cols: int = 10) -> str:
        if cache_key not in self.cache.cache:
            return f"Error: No dataframe found in cache with key '{cache_key}'"
        
        df = self.cache.get(cache_key).value
        if not isinstance(df, pd.DataFrame):
            return f"Error: The object with key '{cache_key}' is not a pandas DataFrame"
        
        preview_df = df.head(num_rows)
        if num_cols is not None:
            preview_df = preview_df.iloc[:, :num_cols]
        
        preview = preview_df.to_string()
        total_rows, total_cols = df.shape
        cols_shown = preview_df.shape[1]
        
        print (f"Preview of '{cache_key}' ({total_rows} rows x {total_cols} columns total):\n"
                f"Showing {num_rows} rows and {cols_shown} columns\n\n{preview}")
        
        return f"Preview of '{cache_key}' ({total_rows} rows x {total_cols} columns total) has been shown to the user."

    @property
    def save_dataframe_as_csv_descriptor(self) -> ToolDescriptor:
        return ToolDescriptor(
            name="save_dataframe_as_csv",
            description="Save a dataframe from the cache as a CSV file",
            arguments=[
                ToolArgument(
                    name="cache_key",
                    description="The cache key of the dataframe to save",
                    type="string",
                ),
                ToolArgument(
                    name="file_path",
                    description="The file path where the CSV should be saved",
                    type="string",
                ),
            ],
        )

    def save_dataframe_as_csv(self, cache_key: str, file_path: str) -> str:
        if cache_key not in self.cache.cache:
            return f"Error: No dataframe found in cache with key '{cache_key}'"
        
        df = self.cache.get(cache_key).value
        if not isinstance(df, pd.DataFrame):
            return f"Error: The object with key '{cache_key}' is not a pandas DataFrame"
        
        try:
            df.to_csv(file_path, index=False)
            return f"Successfully saved dataframe '{cache_key}' to {file_path}"
        except Exception as e:
            return f"Error saving dataframe to CSV: {str(e)}"

    @property
    def exec_python_code_descriptor(self) -> ToolDescriptor:
        return ToolDescriptor(
            name="exec_python_code",
            description="""Execute Python code and assign the result to a variable in the cache.
            BE SURE THE GIVEN CODE ASSIGNS THE SPECIFIED OUTPUT VARIABLE!!! (e.g. <output_variable_name> = <result>)
            If the code does not assign the specified output variable, an error will be thrown.
            If the code does not assign an output variable (e.g. inplace dataframe edits), pass 'NOT_APPLICABLE' as the output_variable_name.""",
            arguments=[
                ToolArgument(
                    name="code",
                    description="The Python code to execute.",
                    type="string",
                ),
                ToolArgument(
                    name="output_variable_name",
                    description="""If applicable, The name of the variable to store the result of the execution.
                    Be specific to avoid later conflicts or ambiguity.
                    If not applicable (e.g. inplace dataframe edits), pass "NOT_APPLICABLE" here.""",
                    type="string",
                ),
                ToolArgument(
                    name="output_variable_description",
                    description="A description of the variable to store the result of the execution. Be specific to avoid later conflicts or ambiguity.",
                    type="string",
                ),
            ],
        )
    def exec_python_code(self, code: str, output_variable_name: str, output_variable_description: str) -> str:
        # It's important that any exceptions get raised, so don't catch them here.
        if output_variable_name != "NOT_APPLICABLE":
            assert f"{output_variable_name} = " in code, f"Error: The code does not assign the output variable {output_variable_name}"
        locals_copy = locals()
        locals_copy.update({ key: item.value for key, item in self.cache.cache.items() })
        exec(
            code,
            globals(),
            locals_copy,
        )
        if output_variable_name == "NOT_APPLICABLE":
            return "Successfully executed the code."
        result = locals_copy[output_variable_name]
        self.cache.set(output_variable_name, CacheObject(
            description=output_variable_description,
            value=result,
            type_name=type(result).__name__,
        ))
        msg = f"Successfully stored the result of the execution in '{output_variable_name}'"
        print(f"[debug:pandas_io_cache] {msg}")
        return msg

