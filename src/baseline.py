"""Baseline zero-shot: unitary/toxic-bert (melhor modelo pronto avaliado no paper de referência).

O modelo não é treinado aqui; só pontuamos todas as mensagens e avaliamos:
1. no teste oficial, para conferir que reproduzimos o número do paper (F1 ~0.59);
2. nos mesmos folds agrupados por partida usados no fine-tuning (comparação justa).

Binarização igual à do repositório de referência: score sigmoide do rótulo
"toxic" >= 0.45.
"""
import argparse
import json

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from config import LABEL_COL, MAX_LEN, OUTPUTS_DIR, TEXT_COL
from data import get_folds, load_all
from train import METRICS, metrics_from_preds

MODEL_ID = "unitary/toxic-bert"
THRESHOLD = 0.45
OUT_DIR = OUTPUTS_DIR / "baseline_toxic_bert"


@torch.no_grad()
def score_messages(texts, device, batch_size=64):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID).to(device).eval()
    toxic_idx = model.config.label2id["toxic"]
    scores = []
    for i in range(0, len(texts), batch_size):
        enc = tokenizer(texts[i : i + batch_size], truncation=True, padding=True,
                        max_length=MAX_LEN, return_tensors="pt").to(device)
        scores.extend(torch.sigmoid(model(**enc).logits[:, toxic_idx]).tolist())
    return np.array(scores)


def main(device, n_folds):
    df = load_all()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    scores_path = OUT_DIR / "scores.csv"
    if scores_path.exists():
        scores = pd.read_csv(scores_path)["score"].to_numpy()
    else:
        scores = score_messages(df[TEXT_COL].tolist(), device)
        pd.DataFrame({"id": df["id"], "score": scores}).to_csv(scores_path, index=False)

    y = df[LABEL_COL].to_numpy()
    pred = (scores >= THRESHOLD).astype(int)
    official = (df["official_split"] == "test").to_numpy()
    folds = get_folds(df, "group", n_folds)

    per_fold = pd.DataFrame([{"fold": k, **metrics_from_preds(y[folds == k], pred[folds == k])}
                             for k in range(1, n_folds + 1)])
    summary = {
        "model_id": MODEL_ID,
        "threshold": THRESHOLD,
        "official_test": metrics_from_preds(y[official], pred[official]),
        "group_cv_mean": {m: float(per_fold[m].mean()) for m in METRICS},
        "group_cv_std": {m: float(per_fold[m].std(ddof=1)) for m in METRICS},
        "pooled_all": metrics_from_preds(y, pred),
    }
    per_fold.to_csv(OUT_DIR / "folds_group.csv", index=False)
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Teste oficial (paper reporta F1=0.5900): " +
          " | ".join(f"{m}={v:.4f}" for m, v in summary["official_test"].items()))
    print("Folds agrupados: " + " | ".join(
        f"{m}={summary['group_cv_mean'][m]:.4f}±{summary['group_cv_std'][m]:.4f}" for m in METRICS))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()
    main(args.device, args.folds)
