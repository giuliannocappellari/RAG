from pydantic import BaseModel



class TextDocument(BaseModel):
    title: str
    description: str
    html: str
    raw_text:str

    class Config:
        examples = {
            "title": "<title>",
            "description": "<page_desc>",
            "html": "<html>",
            "raw_text":"<raw_text>"
        }


class EmbeddedDocument(BaseModel):
    title: list[float]
    description: list[float]
    html_text: list[float]
