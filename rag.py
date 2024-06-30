from sentence_transformers import util
from typing import Union, List
from numpy import ndarray
from torch import Tensor
from utils.utils import load_database
from collections import defaultdict
import numpy as np
import json


def get_hists(query: Union[List[Tensor], ndarray, Tensor]) -> list:
    corpus = load_database()
    data = [
        util.semantic_search(query, corpus_embeddings, top_k=5)[0]
        for corpus_embeddings in corpus
    ]

    scores_dict = defaultdict(list)

    for sublist in data:
        for item in sublist:
            corpus_id = item["corpus_id"]
            score = item["score"]
            scores_dict[corpus_id].append(score)

    # Calculate the mean score for each corpus_id
    mean_scores = {
        corpus_id: np.mean(scores) for corpus_id, scores in scores_dict.items()
    }

    # Get the top 5 corpus_id based on mean scores
    top_5_corpus_ids = sorted(mean_scores, key=mean_scores.get, reverse=True)[:3]

    # print("Top 5 corpus_ids based on mean scores:", top_5_corpus_ids)
    documents = json.load(open("database/documents.json", "r"))
    return (
        [documents[i]["raw_text"] for i in top_5_corpus_ids]
        + [documents[i - 1]["raw_text"] for i in top_5_corpus_ids if i > 0]
        + [
            documents[i + 1]["raw_text"]
            for i in top_5_corpus_ids
            if i < len(documents) - 1
        ]
    )
