from abc import abstractmethod
from openai import OpenAI
from models.schemas import ChatMLMessage
from dotenv import load_dotenv
import os


class LlamaBaseModel():

    def __init__(self) -> None:
        self.model: str

    @abstractmethod
    def execute(self, messages: list[ChatMLMessage]) -> str: ...


class Llama70b(LlamaBaseModel):

    def __init__(self) -> None:

        load_dotenv()
        self.openai = OpenAI(
            api_key=os.environ["api_key"],
            base_url="https://api.deepinfra.com/v1/openai",
        )
        self.model = "meta-llama/Meta-Llama-3-70B-Instruct"

    def execute(self, messages: list[ChatMLMessage]) -> str:
        chat_completion: str = (
            self.openai.chat.completions.create(
                model=self.model, messages=messages  # type: ignore
            )
            .choices[0]
            .message.content
        )
        return chat_completion
