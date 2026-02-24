"""
Utility functions and configuration for the Goodreads Genre Classification project.
"""

import os
import random
import numpy as np
import torch


# ─── Configuration ───────────────────────────────────────────────────────────

# Pre-trained model: DistilBERT (cased)
# Selected because it is a distilled (smaller, faster) version of BERT,
# making it ideal for fine-tuning on modest hardware while retaining ~97%
# of BERT's language understanding. The cased variant preserves capitalization,
# which can carry genre-relevant signals in book reviews (e.g., proper nouns,
# titles).
MODEL_NAME = "distilbert-base-cased"

# CUDA if available, else CPU
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Maximum token length for BERT inputs
MAX_LENGTH = 512

# Directory to cache the fine-tuned model locally
CACHED_MODEL_DIR = "distilbert-reviews-genres"

# Training hyper-parameters
NUM_EPOCHS = 3
TRAIN_BATCH_SIZE = 10
EVAL_BATCH_SIZE = 16
LEARNING_RATE = 5e-5
WARMUP_STEPS = 100
WEIGHT_DECAY = 0.01

# Data sampling parameters
HEAD = 10000        # number of reviews to stream per genre
SAMPLE_SIZE = 2000  # random sample per genre
TRAIN_PER_GENRE = 800
TEST_PER_GENRE = 200  # remaining after TRAIN_PER_GENRE from 1000

# Output directories
RESULTS_DIR = "./results"
LOGS_DIR = "./logs"
EVAL_RESULTS_FILE = "evaluation_results.json"

# Genre URLs
GENRE_URL_DICT = {
    "poetry":                 "https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_poetry.json.gz",
    "children":               "https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_children.json.gz",
    "comics_graphic":         "https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_comics_graphic.json.gz",
    "fantasy_paranormal":     "https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_fantasy_paranormal.json.gz",
    "history_biography":      "https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_history_biography.json.gz",
    "mystery_thriller_crime": "https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_mystery_thriller_crime.json.gz",
    "romance":                "https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_romance.json.gz",
    "young_adult":            "https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_young_adult.json.gz",
}


# ─── Helpers ─────────────────────────────────────────────────────────────────

def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_label_mappings(labels):
    """Return label2id and id2label dicts from a list of labels."""
    unique_labels = sorted(set(labels))
    label2id = {label: idx for idx, label in enumerate(unique_labels)}
    id2label = {idx: label for label, idx in label2id.items()}
    return label2id, id2label
