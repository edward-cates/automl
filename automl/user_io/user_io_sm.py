from pathlib import Path

import pandas as pd
from pydantic import BaseModel

from src.user_io.user_io_cache_project_specific import UserIoCacheProjectSpecific
from src.user_io.pandas_io_cache import PandasIoCache, CacheObject
from src.llm.chatgpt import ChatGPT, Prompt

class UserPrompt(BaseModel):
    prompt: str
    terminate: bool

class UserIoSm:
    def __init__(self, csv_path: str) -> None:
        self.llm = ChatGPT()
        self.user_cache = PandasIoCache()
        assert Path(csv_path).exists(), f"CSV path {csv_path} does not exist"
        print(f"[debug:user_io_sm] csv_path found: {csv_path}")
        self.user_cache.cache.set("starting_df", CacheObject(
            description="The base dataframe",
            value=pd.read_csv(csv_path),
            type_name="pd.DataFrame",
        ))

    def run(self) -> None:
        gpt_prompt = Prompt()
        gpt_prompt.add("system", f"""You a pandas data analysis assistant.
        You can executed Python code and store variables of any type in a cache dict.
        Starting state:
        {self.user_cache.describe_cache()}
        """)
        gpt_prompt.add("system", """If you need input from the user to complete a task,
        ask for the needed information. Start the conversation by describing the cache state
        and then asking the user what they want to do.""")
        gpt_prompt.add("system", """If you don't know how to fix an error,
        just quit - don't try the same thing repeatedly.""")
        user_prompt: BaseModel = self.llm.ask(
            prompt=gpt_prompt,
            response_format=UserPrompt,
        )
        print(user_prompt.prompt)

        while not user_prompt.terminate:
            response = input()
            gpt_prompt.add("system", f"""CACHE STATE:
{self.user_cache.describe_cache()}
END CACHE STATE
""")
            gpt_prompt.add("user", response)
            user_prompt = self.llm.ask(
                prompt=gpt_prompt,
                response_format=UserPrompt,
                tools=self.user_cache,
                debug=False,
            )
            print("\n" + user_prompt.prompt)

