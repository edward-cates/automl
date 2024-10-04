
from pydantic import BaseModel

from src.user_io.user_io_cache_project_specific import UserIoCacheProjectSpecific
from src.llm.chatgpt import ChatGPT, Prompt

class UserPrompt(BaseModel):
    prompt: str
    terminate: bool

class UserIoSm:
    def __init__(self) -> None:
        self.llm = ChatGPT()
        self.user_cache = UserIoCacheProjectSpecific()

    def run(self) -> None:
        gpt_prompt = Prompt()
        gpt_prompt.add("system", f"""Start the chat by summarizing the available tools, then ask the user what they want to do.
Tool descriptors:
{self.user_cache.tool_descriptors}
""")
        gpt_prompt.add("system", "If you need input from the user to complete a task, ask for the needed information.")
        gpt_prompt.add("system", "Keep asking the user what they want to do until they say they want to stop/quit/exit/etc.")
        user_prompt: BaseModel = self.llm.ask(
            prompt=gpt_prompt,
            response_format=UserPrompt,
        )
        print(user_prompt.prompt)

        while not user_prompt.terminate:
            response = input()
            gpt_prompt.add("user", response)
            user_prompt = self.llm.ask(
                prompt=gpt_prompt,
                response_format=UserPrompt,
                tools=self.user_cache,
                debug=False,
            )
            print("\n" + user_prompt.prompt)

