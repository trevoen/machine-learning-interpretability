"""Integração com a Data Dragon (dados estáticos públicos da Riot Games, sem API key).

Usado para a análise original do trabalho: cruzar toxicidade com o papel/arquétipo
do campeão mencionado na mensagem (ex: Tank, Mage, Assassin...).
"""
import json

import requests

from config import CACHE_DIR, DDRAGON_VERSIONS_URL, DDRAGON_CHAMPION_URL_TEMPLATE

CACHE_FILE = CACHE_DIR / "champion_data.json"


def _fetch_latest_version() -> str:
    resp = requests.get(DDRAGON_VERSIONS_URL, timeout=10)
    resp.raise_for_status()
    return resp.json()[0]


def _fetch_champion_data(version: str) -> dict:
    url = DDRAGON_CHAMPION_URL_TEMPLATE.format(version=version)
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    return resp.json()["data"]


def get_champion_map(force_refresh: bool = False) -> dict:
    """Retorna {nome_do_campeao: {"tags": [...], "title": ..., "id": ...}}."""
    if CACHE_FILE.exists() and not force_refresh:
        return json.loads(CACHE_FILE.read_text(encoding="utf-8"))

    version = _fetch_latest_version()
    raw = _fetch_champion_data(version)
    champion_map = {
        name: {"tags": info["tags"], "title": info["title"], "id": info["key"]}
        for name, info in raw.items()
    }

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_FILE.write_text(json.dumps(champion_map, ensure_ascii=False, indent=2), encoding="utf-8")
    return champion_map


def primary_role(champion_map: dict, champion_name: str) -> str | None:
    entry = champion_map.get(champion_name)
    if not entry or not entry["tags"]:
        return None
    return entry["tags"][0]


if __name__ == "__main__":
    champ_map = get_champion_map()
    print(f"{len(champ_map)} campeões carregados.")
    sample = list(champ_map.items())[:5]
    for name, info in sample:
        print(name, "->", info["tags"])
