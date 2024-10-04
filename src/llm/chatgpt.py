import os

from openai import OpenAI
from pydantic import BaseModel

class ChatGPT:
    """
    A class to ask ChatGPT something.
    """
    def __init__(self, model_name: str = 'gpt-4o-mini'):
        """
        `gpt-3.5-turbo` doesn't support structured output, which will throw an error
            with this code.
        """
        api_key = os.getenv("OPENAI_API_KEY")
        assert api_key is not None, "OPENAI_API_KEY environment variable must be set"
        self.client = OpenAI(api_key=api_key)
        self.model_name = model_name

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

    def ask(
            self,
            prompt: Prompt | str,
            response_format: type[BaseModel],
    ) -> BaseModel:
        if isinstance(prompt, str):
            prompt = self.Prompt.from_str(content=prompt)
        response = self.client.beta.chat.completions.parse(
            model=self.model_name,
            messages=prompt.messages,
            response_format=response_format,
            temperature=0.0,
        )
        # Print the AI's response
        return response.choices[0].message.parsed

if __name__ == "__main__":
    # load dotenv
    from dotenv import load_dotenv
    load_dotenv()
    assert os.getenv("OPENAI_API_KEY") is not None, "OPENAI_API_KEY is not set"
    chat = ChatGPT()
    prompt = "say hello world"
    class Response(BaseModel):
        say_hi: str
    response = chat.ask(prompt, response_format=Response)
    print(response)



