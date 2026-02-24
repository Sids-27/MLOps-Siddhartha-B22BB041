"""
Data loading, sampling, splitting, encoding, and PyTorch dataset creation
for the Goodreads Genre Classification project.
"""

import gzip
import json
import random
import pickle
import os

import requests
import torch
from transformers import DistilBertTokenizerFast

from utils import (
    GENRE_URL_DICT,
    HEAD,
    SAMPLE_SIZE,
    TRAIN_PER_GENRE,
    MAX_LENGTH,
    MODEL_NAME,
    set_seed,
)


# ─── Download & sample reviews ──────────────────────────────────────────────

def load_reviews(url: str, head: int = HEAD, sample_size: int = SAMPLE_SIZE):
    """Stream reviews from a gzipped JSON URL, return a random sample."""
    reviews = []
    count = 0

    response = requests.get(url, stream=True)
    response.raise_for_status()
    with gzip.open(response.raw, "rt", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            reviews.append(d["review_text"])
            count += 1
            if head is not None and count >= head:
                break

    return random.sample(reviews, min(sample_size, len(reviews)))


def download_all_genres(genre_url_dict: dict = GENRE_URL_DICT,
                        cache_path: str = "genre_reviews_dict.pickle"):
    """Download reviews for every genre; cache to pickle."""
    if os.path.exists(cache_path):
        print(f"Loading cached reviews from {cache_path}")
        with open(cache_path, "rb") as fh:
            return pickle.load(fh)

    genre_reviews = {}
    for genre, url in genre_url_dict.items():
        print(f"Downloading reviews for genre: {genre}")
        genre_reviews[genre] = load_reviews(url)

    with open(cache_path, "wb") as fh:
        pickle.dump(genre_reviews, fh)
    print(f"Saved reviews to {cache_path}")
    return genre_reviews


# ─── Train / test split ─────────────────────────────────────────────────────

def split_data(genre_reviews: dict, train_per_genre: int = TRAIN_PER_GENRE):
    """
    From each genre take 1000 reviews: first `train_per_genre` for training,
    the rest for testing.
    """
    train_texts, train_labels = [], []
    test_texts, test_labels = [], []

    for genre, reviews in genre_reviews.items():
        sampled = random.sample(reviews, 1000)
        for review in sampled[:train_per_genre]:
            train_texts.append(review)
            train_labels.append(genre)
        for review in sampled[train_per_genre:]:
            test_texts.append(review)
            test_labels.append(genre)

    return train_texts, train_labels, test_texts, test_labels


# ─── Tokenisation & encoding ────────────────────────────────────────────────

def encode_texts(train_texts, test_texts, model_name: str = MODEL_NAME,
                 max_length: int = MAX_LENGTH):
    """Tokenise train and test texts with the DistilBERT tokenizer."""
    tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)
    train_encodings = tokenizer(train_texts, truncation=True, padding=True,
                                max_length=max_length)
    test_encodings = tokenizer(test_texts, truncation=True, padding=True,
                               max_length=max_length)
    return tokenizer, train_encodings, test_encodings


def encode_labels(train_labels, test_labels, label2id):
    """Map string labels to integer ids."""
    train_encoded = [label2id[y] for y in train_labels]
    test_encoded = [label2id[y] for y in test_labels]
    return train_encoded, test_encoded


# ─── PyTorch Dataset ─────────────────────────────────────────────────────────

class GenreDataset(torch.utils.data.Dataset):
    """Simple PyTorch wrapper around tokenized encodings + integer labels."""

    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx])
                for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


# ─── Convenience: full pipeline ─────────────────────────────────────────────

def prepare_datasets():
    """
    End-to-end data pipeline:
    download → split → encode → return datasets + metadata.
    """
    set_seed(42)

    genre_reviews = download_all_genres()
    train_texts, train_labels, test_texts, test_labels = split_data(genre_reviews)

    print(f"Train: {len(train_texts)} | Test: {len(test_texts)}")

    from utils import get_label_mappings
    label2id, id2label = get_label_mappings(train_labels)

    tokenizer, train_enc, test_enc = encode_texts(train_texts, test_texts)
    train_labels_enc, test_labels_enc = encode_labels(train_labels, test_labels,
                                                      label2id)

    train_dataset = GenreDataset(train_enc, train_labels_enc)
    test_dataset = GenreDataset(test_enc, test_labels_enc)

    return {
        "train_dataset": train_dataset,
        "test_dataset": test_dataset,
        "tokenizer": tokenizer,
        "label2id": label2id,
        "id2label": id2label,
        "train_texts": train_texts,
        "train_labels": train_labels,
        "test_texts": test_texts,
        "test_labels": test_labels,
    }


if __name__ == "__main__":
    info = prepare_datasets()
    print("Label mapping:", info["label2id"])
    print("Train dataset size:", len(info["train_dataset"]))
    print("Test  dataset size:", len(info["test_dataset"]))
