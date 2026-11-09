"""Data loading and preprocessing utilities for multiclass sentiment analysis."""

from pathlib import Path
from typing import Dict, Tuple
import pandas as pd

HF_DATASET_BASE = "hf://datasets/Sp1786/multiclass-sentiment-analysis-dataset/"
SPLITS = {
    "train": "train_df.csv",
    "validation": "val_df.csv",
    "test": "test_df.csv",
}
TARGET_NAMES = ["negative", "neutral", "positive"]
LABEL_MAPPING = {0: "negative", 1: "neutral", 2: "positive"}


def load_raw_data(cache_dir: str = "dataset") -> Dict[str, pd.DataFrame]:
    """Load train, validation, and test datasets from Hugging Face or local cache.

    Drops rows with missing values.
    """
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    dfs = {}
    for split_key, filename in SPLITS.items():
        local_file = cache_path / filename
        if local_file.exists():
            df = pd.read_csv(local_file)
        else:
            url = HF_DATASET_BASE + filename
            df = pd.read_csv(url)
            # Cache locally to speed up future runs
            df.to_csv(local_file, index=False)

        # Drop null values as done in the original experiment
        df = df.dropna().reset_index(drop=True)
        dfs[split_key] = df

    return dfs


def prepare_datasets(
    cache_dir: str = "dataset",
) -> Tuple[
    Tuple[pd.Series, pd.Series],
    Tuple[pd.Series, pd.Series],
    Tuple[pd.Series, pd.Series],
]:
    """Load data and extract (X, y) for train, validation, and test splits.

    Returns:
        ((X_train, y_train), (X_val, y_val), (X_test, y_test))
    """
    dfs = load_raw_data(cache_dir=cache_dir)
    df_train = dfs["train"]
    df_val = dfs["validation"]
    df_test = dfs["test"]

    X_train, y_train = df_train["text"], df_train["label"].values
    X_val, y_val = df_val["text"], df_val["label"].values
    X_test, y_test = df_test["text"], df_test["label"].values

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)

