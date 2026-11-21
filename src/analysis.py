"""Contribuição original: toxicidade cruzada com o papel do campeão mencionado.

Usa os dados já rotulados (label real do L2DTnH, não predição) + a Data Dragon
para responder: mensagens que mencionam certos arquétipos de campeão (ex:
Assassin, Tank, Support) têm taxa de toxicidade diferente?
"""
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import chi2_contingency

from config import CHAMPION_COL, LABEL_COL, OUTPUTS_DIR
from data import load_train_test
from riot_api import get_champion_map, primary_role


def build_role_table():
    train_df, test_df = load_train_test()
    df = pd.concat([train_df, test_df], ignore_index=True)

    champion_map = get_champion_map()
    df["role"] = df[CHAMPION_COL].map(lambda c: primary_role(champion_map, c))
    df = df.dropna(subset=["role"])

    table = (
        df.groupby("role")[LABEL_COL]
        .agg(mensagens="count", tóxicas="sum")
        .assign(taxa_toxicidade=lambda t: t["tóxicas"] / t["mensagens"])
        .sort_values("taxa_toxicidade", ascending=False)
    )
    return df, table


def chi_square_test(df):
    contingency = pd.crosstab(df["role"], df[LABEL_COL])
    chi2, p, dof, _ = chi2_contingency(contingency)
    return {"chi2": float(chi2), "p_value": float(p), "dof": int(dof)}


def plot_role_toxicity(table, out_path):
    ax = table["taxa_toxicidade"].plot(kind="bar", figsize=(8, 4))
    ax.set_ylabel("Taxa de mensagens tóxicas")
    ax.set_xlabel("Papel do campeão (Data Dragon)")
    ax.set_title("Toxicidade por arquétipo de campeão mencionado")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


if __name__ == "__main__":
    df, table = build_role_table()
    print(table)

    test_result = chi_square_test(df)
    print("\nTeste qui-quadrado (independência entre papel e toxicidade):", test_result)

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUTPUTS_DIR / "toxicity_by_role.csv")
    plot_role_toxicity(table, OUTPUTS_DIR / "toxicity_by_role.png")
    print(f"\nTabela e gráfico salvos em {OUTPUTS_DIR}")
