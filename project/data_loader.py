import os
from typing import List, Tuple

from cleaning import clean_lyrics

# Emotion labels used throughout experiments
EMOTIONS = ["Angry", "Happy", "Relaxed", "Sad"]
LABEL2ID = {label: i for i, label in enumerate(EMOTIONS)}
ID2LABEL = {i: label for label, i in LABEL2ID.items()}

# Default location of the dataset folders
DATASET_DIR = "NJU_MusicMood_v1.0"


def get_lyrics(path: str) -> str:
    with open(path, "r", encoding="utf-8") as fh:
        return clean_lyrics(fh.read())


def get_lyrics_and_labels(split: str, dataset_dir: str = DATASET_DIR) -> Tuple[List[str], List[str]]:
    texts, labels = [], []
    for emotion in EMOTIONS:
        folder = os.path.join(dataset_dir, emotion, split)
        if not os.path.isdir(folder):
            continue

        for fname in os.listdir(folder):
            if fname.lower() == "info.txt" or not fname.endswith(".txt"):
                continue

            path = os.path.join(folder, fname)
            text = get_lyrics(path)
            if text.strip():
                texts.append(text)
                labels.append(emotion)
    return texts, labels


def build_hf_dataset(train_texts: List[str], train_labels: List[str], dev_texts: List[str], dev_labels: List[str]):
    """Create Hugging Face Datasets objects from raw lists."""
    from datasets import Dataset

    train_ds = Dataset.from_dict({"text": train_texts, "label": [LABEL2ID[l] for l in train_labels]})
    dev_ds = Dataset.from_dict({"text": dev_texts, "label": [LABEL2ID[l] for l in dev_labels]})
    return train_ds, dev_ds


__all__ = ["DATASET_DIR", "EMOTIONS", "LABEL2ID", "ID2LABEL", "get_lyrics_and_labels", "build_hf_dataset"]
