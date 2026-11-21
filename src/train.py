"""Fine-tuning comparativo de Transformers para detecção de toxicidade.

Diferente do baseline único do repositório de referência (unitary/toxic-bert),
aqui comparamos múltiplas arquiteturas (ver MODEL_IDS em config.py) sob o
mesmo protocolo de treino/avaliação.

Modos:
- validação cruzada agrupada por partida (--folds K --protocol group): protocolo
  principal. Treino + teste oficiais são unidos e divididos em K folds
  estratificados por rótulo e agrupados por partida (chatlog_id); a validação
  interna (escolha do epoch) também é separada por partida. Nenhuma partida
  aparece em mais de um papel, eliminando o vazamento do split oficial.
- validação cruzada por mensagem (--protocol message): mesmos dados, folds
  aleatórios por mensagem como no split oficial. Serve para medir quanto o
  vazamento de partida infla as métricas.
- execução única (padrão, sem --folds): split oficial + 10% de validação; salva
  outputs/<modelo>/best_model, usado pela demo.

Em todos os modos o conjunto de teste nunca participa da seleção de modelo.
"""
import argparse
import json
import math
import shutil

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from config import (
    BATCH_SIZE,
    CHECKPOINTS_DIR,
    EPOCHS,
    LABEL_COL,
    LEARNING_RATE,
    MAX_LEN,
    MODEL_IDS,
    OUTPUTS_DIR,
    RANDOM_SEED,
    TEXT_COL,
)
from data import get_folds, inner_validation_split, load_all, load_train_val_test

METRICS = ["accuracy", "precision", "recall", "f1", "f1_macro"]
ID2LABEL = {0: "non-toxic", 1: "toxic"}


def write_model_card(best_dir, model_key, model_id, best_epoch, n_train, n_val, n_test, val_metrics, test_metrics):
    """README.md dentro do best_model, para quem receber só a pasta do modelo."""
    rows = "\n".join(f"| {m} | {val_metrics[m]:.4f} | {test_metrics[m]:.4f} |" for m in METRICS)
    card = f"""# {model_key} fine-tuned for League of Legends chat toxicity

Base model: `{model_id}`, fine-tuned for binary toxicity classification on the
English part of the L2DTnH dataset (League of Legends Tribunal chat logs;
Ave et al., ESORICS 2026).

- Labels: `0 = non-toxic`, `1 = toxic` (decision threshold 0.5 on the softmax)
- Training data: official train split minus a stratified 10% validation set
  ({n_train} training / {n_val} validation messages); best epoch chosen by
  validation F1 (epoch {best_epoch})
- Hyperparameters: lr 2e-5, batch size 32, up to 4 epochs, 10% warmup,
  weight decay 0.01, max length 64 tokens, fp16

## Metrics (toxic class)

| metric | validation | official test (n={n_test}) |
|---|---|---|
{rows}

Note: the official split is random by message, so all 101 matches appear in
both train and test (match leakage). In the project's match-grouped 5-fold CV,
DistilBERT scored 0.700 +/- 0.016 F1 against 0.710 +/- 0.026 with
message-level folds; see the project repository for the per-model numbers.

## Usage

```python
from transformers import pipeline
clf = pipeline("text-classification", model="path/to/best_model")
clf(["gg wp", "uninstall the game you trash noob"])
```
"""
    (best_dir / "README.md").write_text(card, encoding="utf-8")


class ToxicityDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item

    def __len__(self):
        return len(self.labels)


def metrics_from_preds(labels, preds):
    prec, rec, f1, _ = precision_recall_fscore_support(labels, preds, average="binary", zero_division=0)
    _, _, f1_macro, _ = precision_recall_fscore_support(labels, preds, average="macro", zero_division=0)
    return {"accuracy": accuracy_score(labels, preds), "precision": prec, "recall": rec,
            "f1": f1, "f1_macro": f1_macro}


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    return metrics_from_preds(labels, np.argmax(logits, axis=-1))


def to_dataset(tokenizer, df):
    enc = tokenizer(df[TEXT_COL].tolist(), truncation=True, max_length=MAX_LEN)
    return ToxicityDataset(enc, df[LABEL_COL].tolist())


def strip_prefix(metrics, prefix):
    return {k[len(prefix) + 1:]: float(v) for k, v in metrics.items() if k.startswith(prefix + "_")}


def train_one(model_key, train_df, val_df, test_df, run_dir, epochs, save_model):
    """Treina um modelo, escolhe o epoch pela validação e avalia no teste."""
    model_id = MODEL_IDS[model_key]
    torch.manual_seed(RANDOM_SEED)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    train_ds, val_ds, test_ds = (to_dataset(tokenizer, df) for df in (train_df, val_df, test_df))
    model = AutoModelForSequenceClassification.from_pretrained(
        model_id, num_labels=2, id2label=ID2LABEL, label2id={v: k for k, v in ID2LABEL.items()},
    )

    total_steps = math.ceil(len(train_ds) / BATCH_SIZE) * epochs
    ckpt_dir = (CHECKPOINTS_DIR / run_dir.relative_to(OUTPUTS_DIR) if CHECKPOINTS_DIR else run_dir) / "checkpoints"
    args = TrainingArguments(
        output_dir=str(ckpt_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE * 2,
        learning_rate=LEARNING_RATE,
        warmup_steps=int(0.1 * total_steps),  # 10% de warmup (warmup_ratio saiu no transformers 5)
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_steps=50,
        fp16=torch.cuda.is_available(),
        seed=RANDOM_SEED,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
    )
    trainer.train()

    val_metrics = strip_prefix(trainer.evaluate(val_ds, metric_key_prefix="val"), "val")
    test_out = trainer.predict(test_ds, metric_key_prefix="test")
    test_metrics = strip_prefix(test_out.metrics, "test")
    test_probs = torch.softmax(torch.tensor(test_out.predictions, dtype=torch.float32), dim=-1)[:, 1].numpy()
    best_epoch = next((h["epoch"] for h in trainer.state.log_history
                       if "eval_f1" in h and float(h["eval_f1"]) == trainer.state.best_metric), None)
    print(f"[{run_dir.name}] melhor epoch={best_epoch} | val F1={val_metrics['f1']:.4f} | "
          f"teste F1={test_metrics['f1']:.4f}")

    if save_model:
        best_dir = run_dir / "best_model"
        trainer.save_model(str(best_dir))
        tokenizer.save_pretrained(str(best_dir))
        write_model_card(best_dir, model_key, model_id, best_epoch, len(train_df), len(val_df),
                         len(test_df), val_metrics, test_metrics)
    shutil.rmtree(ckpt_dir, ignore_errors=True)

    results = {
        "model_key": model_key,
        "model_id": model_id,
        "best_epoch": best_epoch,
        "n_train": len(train_df),
        "n_val": len(val_df),
        "val": val_metrics,
        "test": test_metrics,
        "log_history": trainer.state.log_history,
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    del trainer, model
    torch.cuda.empty_cache()
    return results, test_probs


def run_single(model_key, epochs):
    train_df, val_df, test_df = load_train_val_test()
    results, _ = train_one(model_key, train_df, val_df, test_df, OUTPUTS_DIR / model_key, epochs, save_model=True)
    return results


def run_cv(model_key, epochs, n_folds, protocol):
    df = load_all()
    folds = get_folds(df, protocol, n_folds)
    cv_dir = OUTPUTS_DIR / model_key / f"cv_{protocol}"
    oof = np.full(len(df), np.nan)

    fold_rows = []
    for k in range(1, n_folds + 1):
        fold_dir = cv_dir / f"fold_{k}"
        test_mask = folds == k
        if (fold_dir / "results.json").exists() and (fold_dir / "test_probs.npy").exists():
            print(f"[{model_key}/{protocol}] fold {k} já concluído, reaproveitando.")
            results = json.loads((fold_dir / "results.json").read_text(encoding="utf-8"))
            probs = np.load(fold_dir / "test_probs.npy")
        else:
            print(f"\n[{model_key}/{protocol}] fold {k}/{n_folds}")
            train_df, val_df = inner_validation_split(df[~test_mask], grouped=(protocol == "group"))
            test_df = df[test_mask].reset_index(drop=True)
            if protocol == "group":
                assert not (set(train_df.group) | set(val_df.group)) & set(test_df.group)
                assert not set(train_df.group) & set(val_df.group)
            results, probs = train_one(model_key, train_df, val_df, test_df, fold_dir, epochs, save_model=False)
            np.save(fold_dir / "test_probs.npy", probs)
        oof[test_mask] = probs
        fold_rows.append({"fold": k, "n_test": int(test_mask.sum()), "best_epoch": results["best_epoch"],
                          **{f"val_{m}": results["val"][m] for m in METRICS},
                          **{f"test_{m}": results["test"][m] for m in METRICS}})

    fold_table = pd.DataFrame(fold_rows)
    fold_table.to_csv(cv_dir / "folds.csv", index=False)
    pd.DataFrame({"id": df["id"], "group": df["group"], "fold": folds, "label": df[LABEL_COL],
                  "prob": oof}).to_csv(cv_dir / "oof_predictions.csv", index=False)

    summary = {
        "model_key": model_key,
        "protocol": protocol,
        "n_folds": n_folds,
        "val_mean": {m: float(fold_table[f"val_{m}"].mean()) for m in METRICS},
        "val_std": {m: float(fold_table[f"val_{m}"].std(ddof=1)) for m in METRICS},
        "test_mean": {m: float(fold_table[f"test_{m}"].mean()) for m in METRICS},
        "test_std": {m: float(fold_table[f"test_{m}"].std(ddof=1)) for m in METRICS},
        # Métricas sobre todas as predições out-of-fold juntas (cada mensagem prevista uma vez).
        "oof_pooled": metrics_from_preds(df[LABEL_COL].to_numpy(), (oof > 0.5).astype(int)),
    }
    (cv_dir / "cv_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\n=== {model_key}: {n_folds}-fold CV ({protocol}) ===")
    print(fold_table.round(4).to_string(index=False))
    for split in ["val", "test"]:
        print(f"{split}: " + " | ".join(
            f"{m}={summary[f'{split}_mean'][m]:.4f}±{summary[f'{split}_std'][m]:.4f}" for m in METRICS))
    print("oof agregado: " + " | ".join(f"{m}={v:.4f}" for m, v in summary["oof_pooled"].items()))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=list(MODEL_IDS.keys()), default="distilbert")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--folds", type=int, default=0,
                        help="K da validação cruzada (0 = execução única no split oficial)")
    parser.add_argument("--protocol", choices=["group", "message"], default="group",
                        help="group: folds agrupados por partida (sem vazamento); message: por mensagem")
    args = parser.parse_args()
    if args.folds > 1:
        run_cv(args.model, args.epochs, args.folds, args.protocol)
    else:
        run_single(args.model, args.epochs)
