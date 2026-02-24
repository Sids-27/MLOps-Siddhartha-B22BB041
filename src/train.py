"""
Training script for fine-tuning DistilBERT on Goodreads genre classification.

Covers:
 - Baseline logistic regression model
 - DistilBERT fine-tuning via the HuggingFace Trainer API
 - Metric logging and model saving
"""

import os
import json

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

from transformers import (
    DistilBertForSequenceClassification,
    TrainingArguments,
    Trainer,
)

from utils import (
    MODEL_NAME,
    DEVICE,
    CACHED_MODEL_DIR,
    NUM_EPOCHS,
    TRAIN_BATCH_SIZE,
    EVAL_BATCH_SIZE,
    LEARNING_RATE,
    WARMUP_STEPS,
    WEIGHT_DECAY,
    RESULTS_DIR,
    LOGS_DIR,
    set_seed,
)
from data import prepare_datasets


# ─── Metrics callback used by Trainer ────────────────────────────────────────

def compute_metrics(pred):
    """Compute accuracy from Trainer predictions."""
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc}


# ─── Baseline: Logistic Regression ──────────────────────────────────────────

def run_baseline(train_texts, train_labels, test_texts, test_labels):
    """TF-IDF + Logistic Regression baseline."""
    print("\n" + "=" * 60)
    print("BASELINE — TF-IDF + Logistic Regression")
    print("=" * 60)

    vectorizer = TfidfVectorizer()
    X_train = vectorizer.fit_transform(train_texts)
    X_test = vectorizer.transform(test_texts)

    lr_model = LogisticRegression(max_iter=1000).fit(X_train, train_labels)
    predictions = lr_model.predict(X_test)

    report = classification_report(test_labels, predictions)
    print(report)
    return report


# ─── BERT Fine-tuning ───────────────────────────────────────────────────────

def train_bert(data_info: dict):
    """Fine-tune DistilBERT and save the model locally."""
    os.environ["WANDB_DISABLED"] = "true"

    id2label = data_info["id2label"]
    label2id = data_info["label2id"]

    print("\n" + "=" * 60)
    print(f"Loading pre-trained model: {MODEL_NAME}")
    print(f"Device: {DEVICE}")
    print("=" * 60)

    model = DistilBertForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(id2label),
        id2label=id2label,
        label2id=label2id,
    ).to(DEVICE)

    training_args = TrainingArguments(
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        warmup_steps=WARMUP_STEPS,
        weight_decay=WEIGHT_DECAY,
        output_dir=RESULTS_DIR,
        logging_dir=LOGS_DIR,
        logging_steps=100,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="epoch",
        report_to=[],
        load_best_model_at_end=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=data_info["train_dataset"],
        eval_dataset=data_info["test_dataset"],
        compute_metrics=compute_metrics,
    )

    print("\nStarting fine-tuning …")
    train_result = trainer.train()

    # Log training metrics
    metrics = train_result.metrics
    print("\n--- Training Metrics ---")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    # Save model & tokenizer
    trainer.save_model(CACHED_MODEL_DIR)
    data_info["tokenizer"].save_pretrained(CACHED_MODEL_DIR)

    # Also save label mappings so the model is self-contained
    with open(os.path.join(CACHED_MODEL_DIR, "label_mappings.json"), "w") as f:
        json.dump({"label2id": label2id, "id2label": {str(k): v for k, v in id2label.items()}}, f, indent=2)

    print(f"\nModel saved to ./{CACHED_MODEL_DIR}/")
    return trainer


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    set_seed(42)

    # 1. Prepare data
    data_info = prepare_datasets()

    # 2. Baseline
    run_baseline(
        data_info["train_texts"],
        data_info["train_labels"],
        data_info["test_texts"],
        data_info["test_labels"],
    )

    # 3. Fine-tune BERT
    trainer = train_bert(data_info)

    # 4. Quick evaluation summary after training
    eval_results = trainer.evaluate()
    print("\n--- Post-training Evaluation ---")
    for k, v in eval_results.items():
        print(f"  {k}: {v}")

    return trainer, data_info


if __name__ == "__main__":
    main()
