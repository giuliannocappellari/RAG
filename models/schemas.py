from abc import ABC, abstractmethod
from typing import Literal, TypedDict


class ChatMLMessage(TypedDict):
    content: str
    role: Literal["user", "assistant", "system"]

class BaseModel(ABC):

    @abstractmethod
    def execute(self, messages: list[ChatMLMessage]) -> str: ...