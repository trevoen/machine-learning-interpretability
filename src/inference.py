"""Inferência com os modelos treinados (usado pelo playground e pela célula do Colab)."""
import json

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from config import MAX_LEN, MODEL_IDS, OUTPUTS_DIR

BASELINE_ID = "unitary/toxic-bert"
BASELINE_THRESHOLD = 0.45  # mesmo critério do trabalho de referência
DISPLAY_NAMES = {"distilbert": "DistilBERT", "bert": "BERT", "roberta": "RoBERTa", "toxic-bert": "toxic-bert (sem fine-tuning)"}


class ToxicityModel:
    def __init__(self, key, path, device, baseline=False):
        self.key = key
        self.name = DISPLAY_NAMES.get(key, key)
        self.baseline = baseline
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForSequenceClassification.from_pretrained(path).to(device).eval()
        self.toxic_idx = self.model.config.label2id.get("toxic", 1)
        self.default_threshold = BASELINE_THRESHOLD if baseline else 0.5
        self.cv_f1 = _cv_f1(key)

    @torch.no_grad()
    def predict_proba(self, texts, batch_size=64):
        probs = []
        for i in range(0, len(texts), batch_size):
            enc = self.tokenizer(texts[i : i + batch_size], truncation=True, padding=True,
                                 max_length=MAX_LEN, return_tensors="pt").to(self.device)
            logits = self.model(**enc).logits
            if self.baseline:  # multi-rótulo (sigmoide), usamos o rótulo "toxic"
                p = torch.sigmoid(logits[:, self.toxic_idx])
            else:
                p = torch.softmax(logits, dim=-1)[:, self.toxic_idx]
            probs.extend(p.float().cpu().tolist())
        return probs


def _cv_f1(key):
    path = (OUTPUTS_DIR / "baseline_toxic_bert" / "summary.json") if key == "toxic-bert" \
        else (OUTPUTS_DIR / key / "cv_group" / "cv_summary.json")
    if not path.exists():
        return None
    s = json.loads(path.read_text(encoding="utf-8"))
    return (s["group_cv_mean"] if key == "toxic-bert" else s["test_mean"])["f1"]


def available_models():
    return [k for k in MODEL_IDS if (OUTPUTS_DIR / k / "best_model" / "config.json").exists()]


def load_models(include_baseline=False, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    models = [ToxicityModel(k, str(OUTPUTS_DIR / k / "best_model"), device) for k in available_models()]
    if include_baseline:
        models.append(ToxicityModel("toxic-bert", BASELINE_ID, device, baseline=True))
    return models
