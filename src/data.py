"""Carregamento, limpeza e particionamento do L2DTnH (filtrado para inglês).

Atenção ao vazamento: o split oficial do repositório de referência é aleatório
por mensagem, e as 101 partidas (chatlog_id) aparecem tanto no treino quanto no
teste. Por isso o protocolo principal do trabalho (group_folds) particiona por
partida: nenhuma partida aparece ao mesmo tempo em treino, validação e teste.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, train_test_split

from config import (
    CACHE_DIR,
    CHAMPION_COL,
    GROUP_COL,
    LABEL_COL,
    RANDOM_SEED,
    TEST_CSV,
    TEXT_COL,
    TRAIN_CSV,
    VAL_SIZE,
)


def _read_csv(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Arquivo não encontrado: {path}\n"
            "Confirme se 'esorics26_toxicity' está clonado ao lado deste projeto."
        )
    return pd.read_csv(path, encoding="utf-8")


def load_split(path):
    df = _read_csv(path)
    df = df[df[LABEL_COL] != 2].copy()
    df[LABEL_COL] = pd.to_numeric(df[LABEL_COL], errors="coerce").fillna(0).astype(int).clip(0, 1)
    df[TEXT_COL] = df[TEXT_COL].fillna("").astype(str)
    df = df[df[TEXT_COL].str.strip().str.len() >= 1]
    df[CHAMPION_COL] = df[CHAMPION_COL].astype("string").str.strip()
    return df.reset_index(drop=True)


def load_train_test():
    """Split oficial do repositório de referência (por mensagem; tem vazamento de partida)."""
    return load_split(TRAIN_CSV), load_split(TEST_CSV)


def load_train_val_test():
    """Split oficial + validação estratificada (10%) tirada do treino."""
    train_df, test_df = load_train_test()
    train_df, val_df = train_test_split(
        train_df, test_size=VAL_SIZE, stratify=train_df[LABEL_COL], random_state=RANDOM_SEED,
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df


def load_all():
    """Treino + teste oficiais juntos, com uma coluna 'group' identificando a partida.

    Mensagens sem chatlog_id (sem metadados) viram grupos unitários.
    """
    train_df, test_df = load_train_test()
    df = pd.concat([train_df.assign(official_split="train"), test_df.assign(official_split="test")],
                   ignore_index=True)
    group = df[GROUP_COL].astype("Int64").astype(str)
    missing = df[GROUP_COL].isna()
    group[missing] = "na_" + df.index[missing].astype(str)
    df["group"] = group
    return df


def _sgkf_folds(df, n_folds, seed):
    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    folds = np.zeros(len(df), dtype=int)
    for k, (_, idx) in enumerate(sgkf.split(df, df[LABEL_COL], df["group"]), start=1):
        folds[idx] = k
    return folds


def group_folds(df, n_folds, n_candidates=300):
    """Fold (1..k) de cada mensagem, estratificado por rótulo e agrupado por partida.

    Como as partidas têm tamanhos muito diferentes (até ~370 mensagens), uma única
    semente pode gerar folds desbalanceados. Testamos n_candidates sementes
    (a partir de RANDOM_SEED, determinístico) e ficamos com a que minimiza a
    variação relativa do tamanho e da taxa de toxicidade entre os folds.
    """
    labels = df[LABEL_COL].to_numpy()
    best, best_score = None, np.inf
    for seed in range(RANDOM_SEED, RANDOM_SEED + n_candidates):
        folds = _sgkf_folds(df, n_folds, seed)
        sizes = np.bincount(folds)[1:]
        rates = np.array([labels[folds == k].mean() for k in range(1, n_folds + 1)])
        score = sizes.std() / sizes.mean() + rates.std() / rates.mean()
        if score < best_score:
            best, best_score = folds, score
    return best


def message_folds(df, n_folds):
    """Fold (1..k) de cada mensagem, estratificado só por rótulo (ignora a partida; tem vazamento)."""
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_SEED)
    folds = np.zeros(len(df), dtype=int)
    for k, (_, idx) in enumerate(skf.split(df, df[LABEL_COL]), start=1):
        folds[idx] = k
    return folds


def get_folds(df, protocol, n_folds):
    """Folds de load_all() para o protocolo 'group' ou 'message', com cache em disco.

    O cache garante que todos os modelos (e o baseline) usem exatamente os mesmos folds.
    """
    cache = CACHE_DIR / f"folds_{protocol}_k{n_folds}.csv"
    if cache.exists():
        cached = pd.read_csv(cache)
        if len(cached) == len(df) and (cached["id"].to_numpy() == df["id"].to_numpy()).all():
            return cached["fold"].to_numpy()
    folds = group_folds(df, n_folds) if protocol == "group" else message_folds(df, n_folds)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"id": df["id"], "group": df["group"], "fold": folds}).to_csv(cache, index=False)
    return folds


def inner_validation_split(train_df, grouped, n_splits=8):
    """Separa ~1/n_splits do treino de um fold para validação (escolha do epoch)."""
    if grouped:
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)
        tr_idx, va_idx = next(splitter.split(train_df, train_df[LABEL_COL], train_df["group"]))
    else:
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)
        tr_idx, va_idx = next(splitter.split(train_df, train_df[LABEL_COL]))
    return train_df.iloc[tr_idx].reset_index(drop=True), train_df.iloc[va_idx].reset_index(drop=True)


if __name__ == "__main__":
    df = load_all()
    print(f"{len(df)} mensagens | {df['group'].str.startswith('na_').sum()} sem partida | "
          f"{df.loc[~df['group'].str.startswith('na_'), 'group'].nunique()} partidas")
    df["fold"] = get_folds(df, "group", 5)
    print("\nFolds agrupados por partida (k=5):")
    print(df.groupby("fold").agg(mensagens=(LABEL_COL, "size"), tóxicas=(LABEL_COL, "sum"),
                                 taxa=(LABEL_COL, "mean"), partidas=("group", "nunique")).round(4))
    for k in range(1, 6):
        tr, te = df[df.fold != k], df[df.fold == k]
        assert not set(tr.group) & set(te.group), "vazamento de partida entre treino e teste!"
    print("OK: nenhuma partida compartilhada entre treino e teste em nenhum fold.")
