"""Avalia os modelos já treinados (train.py) no teste e monta tabelas comparativas.

Saídas em outputs/:
- comparison_metrics.csv: métricas gerais por modelo
- comparison_summary.json: classification report + matriz de confusão por modelo
- predictions_test.csv: predição e probabilidade de cada modelo por mensagem (análise de erros)
- metrics_by_subgroup.csv: F1/precisão/recall por papel do campeão e relação com o ofensor
"""
import json

import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from config import ASSOCIATION_COL, CHAMPION_COL, LABEL_COL, MAX_LEN, MODEL_IDS, OUTPUTS_DIR, TEXT_COL
from data import load_train_test
from riot_api import build_lookup, get_champion_map, primary_role

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TARGET_NAMES = ["não-tóxica", "tóxica"]


@torch.no_grad()
def predict(model_dir, texts, batch_size=64):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(DEVICE).eval()

    probs = []
    for i in range(0, len(texts), batch_size):
        enc = tokenizer(texts[i : i + batch_size], truncation=True, padding=True,
                        max_length=MAX_LEN, return_tensors="pt").to(DEVICE)
        probs.extend(torch.softmax(model(**enc).logits, dim=-1)[:, 1].tolist())
    return probs


def binary_metrics(y_true, y_pred):
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    _, _, f1_macro, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    return {"precision": p, "recall": r, "f1": f1, "f1_macro": f1_macro}


def subgroup_metrics(df, model_key):
    rows = []
    for col in ["role", ASSOCIATION_COL]:
        for value, g in df.dropna(subset=[col]).groupby(col):
            rows.append({
                "model": model_key, "axis": col, "group": value, "n": len(g),
                "n_toxic": int(g[LABEL_COL].sum()),
                **binary_metrics(g[LABEL_COL], g[f"pred_{model_key}"]),
            })
    return rows


def evaluate_all():
    _, test_df = load_train_test()
    champion_map = get_champion_map()
    lookup = build_lookup(champion_map)
    test_df["role"] = test_df[CHAMPION_COL].map(lambda c: primary_role(champion_map, lookup, c))
    texts = test_df[TEXT_COL].tolist()
    y_true = test_df[LABEL_COL].tolist()

    summary, overall_rows, subgroup_rows = {}, [], []
    for model_key in MODEL_IDS:
        model_dir = OUTPUTS_DIR / model_key / "best_model"
        if not model_dir.exists():
            print(f"[{model_key}] ainda não treinado, pulando (rode train.py --model {model_key}).")
            continue

        probs = predict(str(model_dir), texts)
        y_pred = [int(p > 0.5) for p in probs]
        test_df[f"prob_{model_key}"] = probs
        test_df[f"pred_{model_key}"] = y_pred

        summary[model_key] = {
            "report": classification_report(y_true, y_pred, target_names=TARGET_NAMES, output_dict=True),
            "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        }
        overall_rows.append({"model": model_key, **binary_metrics(y_true, y_pred)})
        subgroup_rows.extend(subgroup_metrics(test_df, model_key))
        print(f"\n=== {model_key} ===")
        print(classification_report(y_true, y_pred, target_names=TARGET_NAMES, digits=4))

    if not summary:
        return summary

    (OUTPUTS_DIR / "comparison_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    overall = pd.DataFrame(overall_rows).round(4)
    overall.to_csv(OUTPUTS_DIR / "comparison_metrics.csv", index=False)
    pd.DataFrame(subgroup_rows).round(4).to_csv(OUTPUTS_DIR / "metrics_by_subgroup.csv", index=False)
    pred_cols = [TEXT_COL, LABEL_COL, CHAMPION_COL, "role", ASSOCIATION_COL] + \
        [c for c in test_df.columns if c.startswith(("prob_", "pred_"))]
    test_df[pred_cols].to_csv(OUTPUTS_DIR / "predictions_test.csv", index=False, encoding="utf-8")

    print("\n", overall.to_string(index=False))
    print(f"\nResultados salvos em {OUTPUTS_DIR}")
    return summary


if __name__ == "__main__":
    evaluate_all()
