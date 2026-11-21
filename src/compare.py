"""Comparação estatística entre modelos a partir das predições out-of-fold (OOF).

Usa as predições OOF da validação cruzada (cada mensagem prevista uma única vez
por um modelo que não viu a sua partida) para:
1. bootstrap pareado por partida: intervalo de confiança de 95% da diferença de
   F1 entre pares de modelos (partidas inteiras são reamostradas, respeitando a
   dependência entre mensagens da mesma partida);
2. desempenho por subgrupo (relação com o ofensor e papel do campeão);
3. sensibilidade ao limiar de decisão (exploratório);
4. exemplos de falsos positivos e falsos negativos para a análise de erros.

Saídas em outputs/comparison/.
"""
import json

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_recall_fscore_support

from config import ASSOCIATION_COL, CHAMPION_COL, LABEL_COL, OUTPUTS_DIR, RANDOM_SEED, TEXT_COL
from data import load_all
from riot_api import build_lookup, get_champion_map, primary_role

OUT_DIR = OUTPUTS_DIR / "comparison"
N_BOOT = 2000
SYSTEMS = {
    "toxic-bert": None,  # zero-shot, scores do baseline.py
    "distilbert": "distilbert/cv_group",
    "bert": "bert/cv_group",
    "roberta": "roberta/cv_group",
    "distilbert_message": "distilbert/cv_message",
}
PAIRS = [
    ("distilbert", "toxic-bert"), ("bert", "toxic-bert"), ("roberta", "toxic-bert"),
    ("bert", "distilbert"), ("roberta", "distilbert"), ("roberta", "bert"),
    ("distilbert_message", "distilbert"),
]


def load_predictions():
    df = load_all()
    champion_map = get_champion_map()
    lookup = build_lookup(champion_map)
    df["role"] = df[CHAMPION_COL].map(lambda c: primary_role(champion_map, lookup, c))
    for name, sub in SYSTEMS.items():
        if sub is None:
            s = pd.read_csv(OUTPUTS_DIR / "baseline_toxic_bert" / "scores.csv")
            df = df.merge(s.rename(columns={"score": f"prob_{name}"}), on="id", how="left")
            df[f"pred_{name}"] = (df[f"prob_{name}"] >= 0.45).astype(int)
        else:
            o = pd.read_csv(OUTPUTS_DIR / sub / "oof_predictions.csv")[["id", "prob"]]
            df = df.merge(o.rename(columns={"prob": f"prob_{name}"}), on="id", how="left")
            df[f"pred_{name}"] = (df[f"prob_{name}"] > 0.5).astype(int)
    assert not df.filter(like="prob_").isna().any().any(), "predições faltando"
    return df


def paired_bootstrap(df, a, b, n_boot=N_BOOT):
    """IC 95% de F1(a) - F1(b), reamostrando partidas (grupos) com reposição."""
    rng = np.random.default_rng(RANDOM_SEED)
    groups = df.groupby("group").indices
    keys = list(groups)
    y, pa, pb = df[LABEL_COL].to_numpy(), df[f"pred_{a}"].to_numpy(), df[f"pred_{b}"].to_numpy()
    deltas = np.empty(n_boot)
    for i in range(n_boot):
        idx = np.concatenate([groups[keys[j]] for j in rng.integers(0, len(keys), len(keys))])
        deltas[i] = f1_score(y[idx], pa[idx], zero_division=0) - f1_score(y[idx], pb[idx], zero_division=0)
    observed = f1_score(y, pa) - f1_score(y, pb)
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    p = min(1.0, 2 * min((deltas <= 0).mean(), (deltas >= 0).mean()))
    return {"a": a, "b": b, "delta_f1": float(observed), "ci95_low": float(lo), "ci95_high": float(hi),
            "p_bootstrap": float(p)}


def subgroup_table(df, systems):
    rows = []
    for col in [ASSOCIATION_COL, "role"]:
        for value, g in df.dropna(subset=[col]).groupby(col):
            for s in systems:
                p, r, f1, _ = precision_recall_fscore_support(g[LABEL_COL], g[f"pred_{s}"], average="binary",
                                                              zero_division=0)
                rows.append({"axis": col, "group": value, "system": s, "n": len(g),
                             "toxic_rate": g[LABEL_COL].mean(), "precision": p, "recall": r, "f1": f1})
    return pd.DataFrame(rows)


def threshold_sweep(df, systems, thresholds=np.round(np.arange(0.1, 0.91, 0.05), 2)):
    rows = []
    for s in systems:
        for t in thresholds:
            p, r, f1, _ = precision_recall_fscore_support(df[LABEL_COL], (df[f"prob_{s}"] >= t).astype(int),
                                                          average="binary", zero_division=0)
            rows.append({"system": s, "threshold": t, "precision": p, "recall": r, "f1": f1})
    return pd.DataFrame(rows)


def error_examples(df, system, n=25):
    cols = [TEXT_COL, LABEL_COL, ASSOCIATION_COL, f"prob_{system}"]
    fp = df[(df[f"pred_{system}"] == 1) & (df[LABEL_COL] == 0)].sort_values(f"prob_{system}", ascending=False)
    fn = df[(df[f"pred_{system}"] == 0) & (df[LABEL_COL] == 1)].sort_values(f"prob_{system}")
    # erros em que os três modelos ajustados concordam são os mais informativos
    tuned = ["distilbert", "bert", "roberta"]
    all_fp = df[(df[[f"pred_{s}" for s in tuned]].sum(axis=1) == 3) & (df[LABEL_COL] == 0)]
    all_fn = df[(df[[f"pred_{s}" for s in tuned]].sum(axis=1) == 0) & (df[LABEL_COL] == 1)]
    return (pd.concat([fp.head(n).assign(error="FP"), fn.head(n).assign(error="FN")])[cols + ["error"]],
            {"fp_all_three": int(len(all_fp)), "fn_all_three": int(len(all_fn)),
             "fp_total": int(len(fp)), "fn_total": int(len(fn))})


if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_predictions()
    systems = list(SYSTEMS)

    overall = pd.DataFrame([
        {"system": s, **dict(zip(["precision", "recall", "f1"],
                                 precision_recall_fscore_support(df[LABEL_COL], df[f"pred_{s}"],
                                                                 average="binary")[:3]))}
        for s in systems]).round(4)
    print("=== OOF agregado (todas as 15.166 mensagens)\n", overall.to_string(index=False))

    boot = pd.DataFrame([paired_bootstrap(df, a, b) for a, b in PAIRS]).round(4)
    print(f"\n=== Bootstrap pareado por partida ({N_BOOT} reamostragens)\n", boot.to_string(index=False))

    sub = subgroup_table(df, ["toxic-bert", "distilbert", "bert", "roberta"]).round(4)
    print("\n=== F1 por subgrupo\n",
          sub.pivot_table(index=["axis", "group"], columns="system", values="f1").round(3).to_string())

    sweep = threshold_sweep(df, ["distilbert", "bert", "roberta"]).round(4)
    best = sweep.loc[sweep.groupby("system")["f1"].idxmax()]
    print("\n=== Melhor limiar (exploratório, escolhido nas próprias predições OOF)\n", best.to_string(index=False))

    examples, counts = error_examples(df, "roberta")
    print("\n=== Contagem de erros\n", counts)

    overall.to_csv(OUT_DIR / "oof_overall.csv", index=False)
    boot.to_csv(OUT_DIR / "bootstrap_pairs.csv", index=False)
    sub.to_csv(OUT_DIR / "subgroups_oof.csv", index=False)
    sweep.to_csv(OUT_DIR / "threshold_sweep.csv", index=False)
    examples.to_csv(OUT_DIR / "error_examples_roberta.csv", index=False, encoding="utf-8")
    (OUT_DIR / "error_counts.json").write_text(json.dumps(counts, indent=2), encoding="utf-8")
    print(f"\nSalvo em {OUT_DIR}")
