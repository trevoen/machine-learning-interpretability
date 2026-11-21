from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PLN_ROOT = PROJECT_ROOT.parent

# Splits já preparados pelo repositório de referência (L2DTnH / ESORICS 2026).
# Reaproveitados como fonte de dados; ver README para citação.
DATASET_DIR = PLN_ROOT / "esorics26_toxicity" / "model_creation" / "1_dataset"
TRAIN_CSV = DATASET_DIR / "2_16000_chatlogs_english_only_train_split.csv"
TEST_CSV = DATASET_DIR / "2_16000_chatlogs_english_only_test_split.csv"

# Dados brutos do Tribunal (contexto extra: métricas de partida, papéis, etc.)
RAW_TRIBUNAL_DIR = PLN_ROOT / "datasetlol"
RAW_CHATLOGS_CSV = RAW_TRIBUNAL_DIR / "chatlogs.csv"

TEXT_COL = "message"
LABEL_COL = "label"
CHAMPION_COL = "champion_name"

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
CACHE_DIR = PROJECT_ROOT / "cache"

# Modelos comparados no fine-tuning (contribuição: comparação entre arquiteturas,
# diferente do baseline único usado no repositório de referência).
MODEL_IDS = {
    "distilbert": "distilbert-base-uncased",
    "bert": "bert-base-uncased",
    "roberta": "roberta-base",
}

MAX_LEN = 192
BATCH_SIZE = 32
EPOCHS = 4
LEARNING_RATE = 2e-5
RANDOM_SEED = 42

# Data Dragon (dados estáticos públicos da Riot, sem necessidade de API key)
DDRAGON_VERSIONS_URL = "https://ddragon.leagueoflegends.com/api/versions.json"
DDRAGON_CHAMPION_URL_TEMPLATE = (
    "https://ddragon.leagueoflegends.com/cdn/{version}/data/en_US/champion.json"
)
