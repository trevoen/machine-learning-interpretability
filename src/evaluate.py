"""Avalia os modelos já treinados (train.py) e monta uma tabela comparativa."""
import json

import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from config import LABEL_COL, MAX_LEN, MODEL_IDS, OUTPUTS_DIR, TEXT_COL
from data import load_train_test


@torch.no_grad()
def predict(model_dir, texts, batch_size=32):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()

    preds = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        enc = tokenizer(batch, truncation=True, padding=True, max_length=MAX_LEN, return_tensors="pt")
        logits = model(**enc).logits
        preds.extend(logits.argmax(dim=-1).tolist())
    return preds


def evaluate_all():
    _, test_df = load_train_test()
    texts = test_df[TEXT_COL].astype(str).tolist()
    y_true = test_df[LABEL_COL].tolist()

    summary = {}
    for model_key in MODEL_IDS:
        model_dir = OUTPUTS_DIR / model_key / "best_model"
        if not model_dir.exists():
            print(f"[{model_key}] ainda não treinado, pulando (rode train.py --model {model_key}).")
            continue

        y_pred = predict(str(model_dir), texts)
        report = classification_report(y_true, y_pred, target_names=["não-tóxica", "tóxica"], output_dict=True)
        cm = confusion_matrix(y_true, y_pred).tolist()
        summary[model_key] = {"report": report, "confusion_matrix": cm}
        print(f"\n=== {model_key} ===")
        print(classification_report(y_true, y_pred, target_names=["não-tóxica", "tóxica"]))

    out_path = OUTPUTS_DIR / "comparison_summary.json"
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nResumo salvo em {out_path}")
    return summary


if __name__ == "__main__":
    evaluate_all()
