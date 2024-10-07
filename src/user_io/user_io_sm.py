
from pydantic import BaseModel

from src.user_io.user_io_cache_project_specific import UserIoCacheProjectSpecific
from src.user_io.pandas_io_cache import PandasIoCache
from src.llm.chatgpt import ChatGPT, Prompt

class UserPrompt(BaseModel):
    prompt: str
    terminate: bool

class UserIoSm:
    def __init__(self) -> None:
        self.llm = ChatGPT()
        self.user_cache = PandasIoCache()

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

