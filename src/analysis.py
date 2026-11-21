"""Contribuição original: toxicidade cruzada com o contexto do autor da mensagem.

Usa os rótulos reais do L2DTnH (não predições) e a Data Dragon para responder:
1. Jogadores de certos arquétipos de campeão (Assassin, Tank, Support...) escrevem
   mensagens tóxicas com mais frequência?
2. A taxa de toxicidade muda conforme a relação do autor com o jogador denunciado
   (o próprio ofensor, um aliado ou um inimigo)?

Para cada eixo: tabela de taxas, teste qui-quadrado de independência, V de Cramér
(tamanho de efeito) e resíduos padronizados ajustados (quais categorias desviam).
"""
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

from config import ASSOCIATION_COL, CHAMPION_COL, LABEL_COL, OUTPUTS_DIR
from data import load_train_test
from riot_api import build_lookup, get_champion_map, primary_role


def load_annotated():
    train_df, test_df = load_train_test()
    df = pd.concat([train_df, test_df], ignore_index=True)
    champion_map = get_champion_map()
    lookup = build_lookup(champion_map)
    df["role"] = df[CHAMPION_COL].map(lambda c: primary_role(champion_map, lookup, c))
    return df


def rate_table(df, col):
    return (
        df.groupby(col)[LABEL_COL]
        .agg(mensagens="count", tóxicas="sum")
        .assign(taxa_toxicidade=lambda t: t["tóxicas"] / t["mensagens"])
        .sort_values("taxa_toxicidade", ascending=False)
    )


def chi_square_test(df, col):
    contingency = pd.crosstab(df[col], df[LABEL_COL])
    chi2, p, dof, expected = chi2_contingency(contingency)
    n = contingency.to_numpy().sum()
    k = min(contingency.shape) - 1
    cramers_v = float(np.sqrt(chi2 / (n * k)))

    # Resíduo padronizado ajustado da célula "tóxica" por categoria: |r| > 1.96 ~ p < 0.05
    observed = contingency.to_numpy()
    row_p = observed.sum(axis=1, keepdims=True) / n
    col_p = observed.sum(axis=0, keepdims=True) / n
    adj_resid = (observed - expected) / np.sqrt(expected * (1 - row_p) * (1 - col_p))
    residuals = dict(zip(contingency.index.astype(str), adj_resid[:, 1].round(2).tolist()))

    return {
        "chi2": float(chi2),
        "p_value": float(p),
        "dof": int(dof),
        "n": int(n),
        "cramers_v": cramers_v,
        "adjusted_residuals_toxic": residuals,
    }


def plot_rates(table, overall_rate, xlabel, title, out_path):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(table.index.astype(str), table["taxa_toxicidade"], color="#4C72B0")
    ax.axhline(overall_rate, color="#C44E52", linestyle="--", linewidth=1, label=f"média geral ({overall_rate:.1%})")
    ax.set_ylabel("Taxa de mensagens tóxicas")
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


AXES = {
    "role": ("Papel do campeão jogado pelo autor (Data Dragon)", "Toxicidade por arquétipo do campeão"),
    ASSOCIATION_COL: ("Relação do autor com o jogador denunciado", "Toxicidade por relação com o ofensor"),
}


if __name__ == "__main__":
    df = load_annotated()
    overall = df[LABEL_COL].mean()
    unmatched = df["role"].isna() & df[CHAMPION_COL].notna()
    print(f"{len(df)} mensagens | taxa geral de toxicidade: {overall:.2%}")
    print(f"Campeões sem correspondência na Data Dragon: {unmatched.sum()} "
          f"{sorted(df.loc[unmatched, CHAMPION_COL].unique().tolist())}")

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    stats = {}
    for col, (xlabel, title) in AXES.items():
        sub = df.dropna(subset=[col])
        table = rate_table(sub, col)
        stats[col] = chi_square_test(sub, col)
        print(f"\n=== {title} ===\n{table}\n{json.dumps(stats[col], ensure_ascii=False, indent=2)}")
        table.to_csv(OUTPUTS_DIR / f"toxicity_by_{col}.csv")
        plot_rates(table, overall, xlabel, title, OUTPUTS_DIR / f"toxicity_by_{col}.png")

    (OUTPUTS_DIR / "analysis_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nTabelas, gráficos e estatísticas salvos em {OUTPUTS_DIR}")
