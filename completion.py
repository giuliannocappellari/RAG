from embed import embed
from sentence_transformers import util
import pickle
from models.llama.model import Llama70b
from utils.prompts import llama_messages
from copy import deepcopy
import rag


def generate_completion(query: str):
    messages = deepcopy(llama_messages)
    embedded_query = embed(query)
    hits = rag.get_hists(embedded_query)
    # exit()
    messages[-1]["content"] = messages[-1]["content"].format(context=hits, query=query)
    print(Llama70b().execute(messages))


if __name__ == "__main__":
    generate_completion("O que causou a seca em Porto Alegre?")
