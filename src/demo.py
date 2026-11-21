"""Demo interativa: classifica uma mensagem digitada e mostra o contexto do
campeão mencionado (via Data Dragon), se houver."""
import argparse
import re

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from config import MAX_LEN, MODEL_IDS, OUTPUTS_DIR
from riot_api import get_champion_map


def load_model(model_key: str):
    model_dir = OUTPUTS_DIR / model_key / "best_model"
    if not model_dir.exists():
        raise FileNotFoundError(f"Modelo '{model_key}' não treinado ainda. Rode train.py primeiro.")
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
    model.eval()
    return tokenizer, model


def classify(text, tokenizer, model):
    enc = tokenizer(text, truncation=True, padding=True, max_length=MAX_LEN, return_tensors="pt")
    with torch.no_grad():
        probs = torch.softmax(model(**enc).logits, dim=-1)[0]
    return {"toxica": bool(probs[1] > 0.5), "confianca": float(probs.max())}


def find_champion_mentions(text, champion_map):
    # Fronteira de palavra para evitar falsos positivos ("Vi" em "victory").
    lowered = text.lower()
    return [
        name for name in champion_map
        if re.search(rf"(?<![a-z0-9]){re.escape(name.lower())}(?![a-z0-9])", lowered)
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=list(MODEL_IDS.keys()), default="distilbert")
    args = parser.parse_args()

    tokenizer, model = load_model(args.model)
    champion_map = get_champion_map()

    print("Digite uma mensagem de chat (ou 'sair' para encerrar):")
    while True:
        text = input("> ").strip()
        if text.lower() in {"sair", "exit", "quit"}:
            break
        if not text:
            continue

        result = classify(text, tokenizer, model)
        label = "TÓXICA" if result["toxica"] else "não-tóxica"
        print(f"  -> {label} (confiança={result['confianca']:.2f})")

        mentions = find_champion_mentions(text, champion_map)
        for name in mentions:
            info = champion_map[name]
            print(f"     campeão mencionado: {name} ({info['title']}) — papéis: {', '.join(info['tags'])}")


if __name__ == "__main__":
    main()
