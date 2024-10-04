
import os
from abc import ABC, abstractmethod

from pydantic import BaseModel
from openai import OpenAI

class Prompt(BaseModel):
    messages: list[dict[str, str]] = []

    @classmethod
    def from_str(cls, content: str):
        assert isinstance(content, str), f"Invalid content: {content=}, must be a string"
        return cls(messages=[{"role": "user", "content": content}])

    def add(self, role: str, content: str):
        assert isinstance(role, str), f"Invalid role: {role=}, must be a string"
        assert isinstance(content, str), f"Invalid content: {content=}, must be a string"
        valid_roles = {'system', 'assistant', 'user', 'function', 'tool'}
        assert role in valid_roles, f"Invalid role: {role=}, must be in {valid_roles=}"
        self.messages.append({"role": role, "content": content})

class ToolArgument(BaseModel):
    name: str
    description: str
    type: str

class ToolDescriptor(BaseModel):
    name: str
    description: str
    arguments: list[ToolArgument] = []

    def render(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "strict": True,
                "parameters": {
                    "type": "object",
                    "properties": {
                        arg.name: {
                            "type": arg.type,
                            "description": arg.description,
                        }
                        for arg in self.arguments
                    },
                    "required": [arg.name for arg in self.arguments],
                    "additionalProperties": False,
                }
            }
        }

class Tools(BaseModel, ABC):
    @property
    @abstractmethod
    def tool_descriptors(self) -> list[ToolDescriptor]:
        pass

    def render_tool_descriptors(self) -> list[dict]:
        return [tool.render() for tool in self.tool_descriptors]

    def call(self, function_name: str, kwargs: dict, debug: bool) -> str:
        if debug:
            print(f"[debug:chatgpt:tools] calling {function_name} with {kwargs=}")
        callable_function = getattr(self, function_name)
        try:
            return str(callable_function(**kwargs))
        except Exception as e:
            return str(e)


class ChatGPT:
    """
    A class to ask ChatGPT something.
    """
    def __init__(self, model_name: str = 'gpt-4o-mini'):
        """
        fyi:`gpt-3.5-turbo` doesn't support structured output, which will throw an error
            with this code.
        """
        api_key = os.getenv("OPENAI_API_KEY")
        assert api_key is not None, "OPENAI_API_KEY environment variable must be set"
        self.client = OpenAI(api_key=api_key)
        self.model_name = model_name

    def ask(
            self,
            prompt: Prompt, # Will modify in place.
            response_format: type[BaseModel],
            tools: Tools | None = None,
            depth: int = 0,
            debug: bool = False,
    ) -> BaseModel:
        assert depth < 10, "Depth limit reached"
        if debug:
            print(f"[debug:chatgpt ({depth=})] asking:")
            for message in prompt.messages:
                print(f"  - {message=}")
        try:
            if tools is None:
                response = self.client.beta.chat.completions.parse(
                    model=self.model_name,
                    messages=prompt.messages,
                    response_format=response_format,
                    temperature=0.0,
                )
            else:
                response = self.client.beta.chat.completions.parse(
                    model=self.model_name,
                    messages=prompt.messages,
                    response_format=response_format,
                    tools=tools.render_tool_descriptors() if tools else None,
                    tool_choice="auto" if tools else None,
                    temperature=0.0,
                )
        except Exception as e:
            print("HERE ARE THE MESSAGES THAT LED TO THE ERROR:")
            for message in prompt.messages:
                print(f"  - {message=}")
            raise e
        message = response.choices[0].message
        if debug:
            print(f"[debug:chatgpt ({depth=})] response: {message=}")
        if not message.tool_calls:
            prompt.messages.append({
                "role": "assistant",
                "content": message.content,
            })
            return message.parsed
        # We have tool calls!
        prompt.messages.append(message) # Tool message must immediately follow a message containing "tool_calls".
        for tool_call in message.tool_calls:
            prompt.messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "function_name": tool_call.function.name,
                "content": tools.call(
                    function_name=tool_call.function.name,
                    kwargs=tool_call.function.parsed_arguments,
                    debug=debug,
                ),
            })
        return self.ask(
            prompt=prompt,
            response_format=response_format,
            tools=tools,
            depth=depth+1,
            debug=debug,
        )


if __name__ == "__main__":
    # load dotenv
    from dotenv import load_dotenv
    load_dotenv()
    assert os.getenv("OPENAI_API_KEY") is not None, "OPENAI_API_KEY is not set"
    chat = ChatGPT()

    class MyTools(Tools):
        @property
        def tool_descriptors(self) -> list[ToolDescriptor]:
            return [
                ToolDescriptor(
                    name="say_hi",
                    description="Say hi to the name that's given",
                    arguments=[
                        ToolArgument(
                            name="name",
                            description="Make up a random name",
                            type="string",
                        ),
                    ],
                ),
            ]

        def say_hi(self, name: str):
            return f"Hi, {name}!"

    class MyResponse(BaseModel):
        welcome_message: str

    response = chat.ask(
        Prompt.from_str("Say hi once and quit."),
        response_format=MyResponse,
        tools=MyTools(),
    )

    print(response)



