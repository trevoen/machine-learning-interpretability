"""Carregamento e limpeza dos splits de treino/teste (L2DTnH, filtrado para inglês)."""
import pandas as pd

from config import TRAIN_CSV, TEST_CSV, TEXT_COL, LABEL_COL, CHAMPION_COL


def _read_csv(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Arquivo não encontrado: {path}\n"
            "Confirme se 'esorics26_toxicity' está clonado ao lado deste projeto."
        )
    with open(path, "r", encoding="latin1", newline="") as f:
        sample = f.read(4096)
    sep = ";" if sample.count(";") > sample.count(",") else ","
    return pd.read_csv(path, sep=sep, encoding="latin1")


def load_split(path):
    df = _read_csv(path)
    df = df[df[LABEL_COL] != 2].copy()
    df[LABEL_COL] = pd.to_numeric(df[LABEL_COL], errors="coerce").fillna(0).astype(int).clip(0, 1)
    df = df[df[TEXT_COL].astype(str).str.len() >= 1]
    df[CHAMPION_COL] = df[CHAMPION_COL].astype(str).str.strip()
    return df.reset_index(drop=True)


def load_train_test():
    return load_split(TRAIN_CSV), load_split(TEST_CSV)


if __name__ == "__main__":
    train_df, test_df = load_train_test()
    print("Treino:", train_df.shape, "| Teste:", test_df.shape)
    print("Distribuição (treino):", train_df[LABEL_COL].value_counts().to_dict())
    print("Distribuição (teste):", test_df[LABEL_COL].value_counts().to_dict())
    print("Campeões distintos mencionados:", train_df[CHAMPION_COL].nunique())
