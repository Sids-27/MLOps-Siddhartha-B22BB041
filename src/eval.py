"""
Evaluation script for the fine-tuned DistilBERT genre classifier.

Modes
-----
1. **Local evaluation** — load from `CACHED_MODEL_DIR` (default).
2. **HuggingFace evaluation** — pass --hf_repo <repo_id> to pull the
   model from your HuggingFace profile and compare metrics.
"""

import argparse
import json
import os

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
)
from transformers import (
    DistilBertForSequenceClassification,
    DistilBertTokenizerFast,
    Trainer,
    TrainingArguments,
)

from utils import CACHED_MODEL_DIR, DEVICE, EVAL_RESULTS_FILE, set_seed, MAX_LENGTH
from data import prepare_datasets, GenreDataset, encode_labels


# ─── Metrics callback ───────────────────────────────────────────────────────

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="weighted"
    )
    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


# ─── Evaluate a model ───────────────────────────────────────────────────────

def evaluate_model(model_path: str, data_info: dict, tag: str = "local"):
    """Load model from `model_path`, evaluate on test set, and return metrics."""
    print(f"\n{'=' * 60}")
    print(f"Evaluating model from: {model_path}  [{tag}]")
    print(f"{'=' * 60}")

    model = DistilBertForSequenceClassification.from_pretrained(model_path).to(DEVICE)
    tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)

    # Re-encode test texts with this tokenizer (important when loading from HF)
    test_encodings = tokenizer(
        data_info["test_texts"], truncation=True, padding=True, max_length=MAX_LENGTH
    )
    label2id = data_info["label2id"]
    test_labels_enc = [label2id[y] for y in data_info["test_labels"]]
    test_dataset = GenreDataset(test_encodings, test_labels_enc)

    training_args = TrainingArguments(
        output_dir="./eval_tmp",
        per_device_eval_batch_size=16,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        compute_metrics=compute_metrics,
    )

    # Built-in evaluation
    eval_results = trainer.evaluate(eval_dataset=test_dataset)
    print("\n--- Evaluation Metrics ---")
    for k, v in eval_results.items():
        print(f"  {k}: {v}")

    # Detailed classification report
    predictions = trainer.predict(test_dataset)
    pred_labels_int = predictions.predictions.argmax(-1).flatten().tolist()
    id2label = data_info["id2label"]
    pred_labels = [id2label[i] for i in pred_labels_int]

    report = classification_report(data_info["test_labels"], pred_labels)
    print("\n--- Classification Report ---")
    print(report)

    eval_results["classification_report"] = report
    eval_results["tag"] = tag
    return eval_results


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate DistilBERT genre classifier")
    parser.add_argument(
        "--hf_repo",
        type=str,
        default=None,
        help="HuggingFace repo id (e.g. your-username/distilbert-reviews-genres) "
             "to load and compare against the local model.",
    )
    parser.add_argument(
        "--local_model",
        type=str,
        default=CACHED_MODEL_DIR,
        help="Path to the locally saved model directory.",
    )
    args = parser.parse_args()

    set_seed(42)
    os.environ["WANDB_DISABLED"] = "true"

    # Prepare data (uses cache if available)
    data_info = prepare_datasets()

    all_results = {}

    # 1. Local evaluation
    local_metrics = evaluate_model(args.local_model, data_info, tag="local")
    all_results["local"] = {k: v for k, v in local_metrics.items()
                            if k != "classification_report"}

    # 2. HuggingFace evaluation (optional)
    if args.hf_repo:
        hf_metrics = evaluate_model(args.hf_repo, data_info, tag="huggingface")
        all_results["huggingface"] = {k: v for k, v in hf_metrics.items()
                                      if k != "classification_report"}

        # Comparison
        print("\n" + "=" * 60)
        print("COMPARISON: Local vs HuggingFace")
        print("=" * 60)
        for metric in ["eval_accuracy", "eval_precision", "eval_recall", "eval_f1", "eval_loss"]:
            local_val = all_results["local"].get(metric, "N/A")
            hf_val = all_results["huggingface"].get(metric, "N/A")
            print(f"  {metric:25s}  Local: {local_val}  |  HF: {hf_val}")

    # Save results
    with open(EVAL_RESULTS_FILE, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {EVAL_RESULTS_FILE}")


if __name__ == "__main__":
    main()
