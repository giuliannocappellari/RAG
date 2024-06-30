import pickle
import os
from typing import Union, List
from numpy import ndarray
from torch import Tensor
import re


def load_or_write(document: str, embed: Union[List[Tensor], ndarray, Tensor]) -> None:
    if os.path.exists(f"database/{document}.pkl"):
        # Load existing database
        with open(f"database/{document}.pkl", "rb") as f:
            database = pickle.load(f)
    else:
        database = []

    database.append(embed)

    # Save the updated database
    with open(f"database/{document}.pkl", "wb") as f:
        pickle.dump(database, f)


def load_database():
    dir = "/home/giuliano/PUCRS/DL2/t3/database/"
    files = os.listdir(dir)
    return tuple(
        pickle.load(open(os.path.join(dir, file), "rb"))
        for file in files
        if file.endswith(".pkl")
    )


def remove_strings(input_string):
    # Define the regex pattern
    pattern = r"\*.*?\\n"

    # Use re.sub to find and remove the matching strings
    result = re.sub(pattern, "", input_string)

    return result


if __name__ == "__main__":
    print(load_database())
