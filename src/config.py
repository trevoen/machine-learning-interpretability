import os
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
# Campeão jogado pelo AUTOR da mensagem (não o campeão mencionado no texto).
CHAMPION_COL = "champion_name"
# Relação do autor com o jogador denunciado: offender / ally / enemy.
ASSOCIATION_COL = "association_to_offender"
# Identificador da partida (um chatlog = uma partida com um jogador denunciado).
GROUP_COL = "chatlog_id"

# Sobrescrevíveis por variável de ambiente (usado no Colab: resultados no Drive,
# checkpoints no disco local da VM, que é bem mais rápido).
OUTPUTS_DIR = Path(os.environ.get("PLN_OUTPUTS", PROJECT_ROOT / "outputs"))
CACHE_DIR = PROJECT_ROOT / "cache"
CHECKPOINTS_DIR = Path(os.environ["PLN_CHECKPOINTS"]) if os.environ.get("PLN_CHECKPOINTS") else None

# Modelos comparados no fine-tuning (contribuição: comparação entre arquiteturas,
# diferente do baseline único usado no repositório de referência).
MODEL_IDS = {
    "distilbert": "distilbert-base-uncased",
    "bert": "bert-base-uncased",
    "roberta": "roberta-base",
}

# Mensagens de chat são curtas (mediana de 2 palavras, máximo de 38), então 64
# tokens cobre tudo. O padding é dinâmico por batch (DataCollatorWithPadding).
MAX_LEN = 64
VAL_SIZE = 0.1
BATCH_SIZE = 32
EPOCHS = 4
LEARNING_RATE = 2e-5
RANDOM_SEED = 42

# Data Dragon (dados estáticos públicos da Riot, sem necessidade de API key)
DDRAGON_VERSIONS_URL = "https://ddragon.leagueoflegends.com/api/versions.json"
DDRAGON_CHAMPION_URL_TEMPLATE = (
    "https://ddragon.leagueoflegends.com/cdn/{version}/data/en_US/champion.json"
)
