from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


def load_data(data_dir="data"):
    """CSV dosyalarını oku; test verisinde hedef sütunu bulunmaz."""
    data_dir = Path(data_dir)
    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test.csv")
    return train, test


def split_data(train, seed=42):
    """Aynı yorumun eğitim ve doğrulamaya birlikte girmesini engelle."""
    groups = train["body"].fillna("").str.lower().str.strip()
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    train_idx, valid_idx = next(splitter.split(train, groups=groups))
    return train.iloc[train_idx].copy(), train.iloc[valid_idx].copy()


def make_text(frame):
    """Kural ve yorumu tek bir model girdisine dönüştür."""
    return "RULE: " + frame["rule"].fillna("") + " COMMENT: " + frame["body"].fillna("")
