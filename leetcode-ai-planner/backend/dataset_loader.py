import pandas as pd
import re
from .config import DATASET_PATH

def load_dataset():
    df = pd.read_csv(DATASET_PATH)

    # Normalize
    # The tags are stored as string representation of numpy array: "['Tag1' 'Tag2']"
    # We use regex to extract content within single quotes
    df["tags"] = df["tags"].apply(
        lambda x: [t.lower() for t in re.findall(r"'([^']*)'", str(x))]
    )
    df["difficulty"] = df["difficulty"].str.lower()
    df["problem_description"] = df["problem_description"].fillna("")

    return df
