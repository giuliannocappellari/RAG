from sentence_transformers import SentenceTransformer, CrossEncoder, util
import os
import tiktoken
from schemas import TextDocument
from utils.utils import load_or_write, remove_strings
import json

bi_encoder = SentenceTransformer("multi-qa-MiniLM-L6-cos-v1")
bi_encoder.max_seq_length = 256  # Truncate long passages to 256 tokens
top_k = 32  # Number of passages we want to retrieve with the bi-encoder


def embed(document: str | list):
    return bi_encoder.encode(document, convert_to_tensor=True, show_progress_bar=True)


def doc_embed(document: TextDocument) -> None:

    load_or_write("title", embed(document.title))
    load_or_write("description", embed(document.description))

    load_or_write(
        "raw_text",
        embed(document.raw_text),
    )


def split_text(input_text, max_tokens=1000):
    # Initialize the tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")

    # Tokenize the input text
    tokens = tokenizer.encode(input_text)

    # Split tokens into chunks
    chunks = [tokens[i : i + max_tokens] for i in range(0, len(tokens), max_tokens)]

    # Convert token chunks back to text
    text_chunks = [tokenizer.decode(chunk) for chunk in chunks]

    return text_chunks


import time

if __name__ == "__main__":
    documents = json.load(open("database/documents.json", "r", encoding="latin-1"))
    new_docs = []
    for document in documents:
        document["raw_text"] = remove_strings(document["raw_text"])
        new_doc = split_text(document["raw_text"])
        for doc in new_doc:
            new_docs.append(
                {
                    "title": document["title"],
                    "description": document["description"],
                    "raw_text": doc,
                    "html": document["html"],
                }
            )
            doc_embed(TextDocument(**new_docs[-1]))
    json.dump(new_docs, open("database/documents.json", "w", encoding="latin-1"))
