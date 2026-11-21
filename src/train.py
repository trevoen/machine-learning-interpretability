"""Fine-tuning comparativo de Transformers para detecção de toxicidade.

Diferente do baseline único do repositório de referência (unitary/toxic-bert),
aqui comparamos múltiplas arquiteturas (ver MODEL_IDS em config.py) sob o
mesmo protocolo de treino/avaliação.
"""
import argparse

import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from config import (
    BATCH_SIZE,
    EPOCHS,
    LABEL_COL,
    LEARNING_RATE,
    MAX_LEN,
    MODEL_IDS,
    OUTPUTS_DIR,
    RANDOM_SEED,
    TEXT_COL,
)
from data import load_train_test


class ToxicityDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

    def __len__(self):
        return len(self.labels)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(labels, preds, average="binary", zero_division=0)
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}


def build_datasets(tokenizer, train_df, test_df):
    train_enc = tokenizer(train_df[TEXT_COL].astype(str).tolist(), truncation=True, padding=True, max_length=MAX_LEN)
    test_enc = tokenizer(test_df[TEXT_COL].astype(str).tolist(), truncation=True, padding=True, max_length=MAX_LEN)
    train_ds = ToxicityDataset(train_enc, train_df[LABEL_COL].tolist())
    test_ds = ToxicityDataset(test_enc, test_df[LABEL_COL].tolist())
    return train_ds, test_ds


def run(model_key: str):
    model_id = MODEL_IDS[model_key]
    train_df, test_df = load_train_test()

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    train_ds, test_ds = build_datasets(tokenizer, train_df, test_df)
    model = AutoModelForSequenceClassification.from_pretrained(model_id, num_labels=2)

    run_dir = OUTPUTS_DIR / model_key
    args = TrainingArguments(
        output_dir=str(run_dir),
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_steps=50,
        seed=RANDOM_SEED,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    metrics = trainer.evaluate()
    print(f"[{model_key}] métricas finais:", metrics)

    trainer.save_model(str(run_dir / "best_model"))
    tokenizer.save_pretrained(str(run_dir / "best_model"))
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=list(MODEL_IDS.keys()), default="distilbert")
    args = parser.parse_args()
    run(args.model)
