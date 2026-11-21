"""Integração com a Data Dragon (dados estáticos públicos da Riot Games, sem API key).

Usado para a análise original do trabalho: cruzar toxicidade com o papel/arquétipo
(Tank, Mage, Assassin...) do campeão jogado pelo autor da mensagem.
"""
import json
import re

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


def normalize_name(name) -> str:
    """'Kha'Zix' -> 'khazix', 'Dr. Mundo' -> 'drmundo', 'Jarvan IV' -> 'jarvaniv'."""
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def get_champion_map(force_refresh: bool = False) -> dict:
    """Retorna {nome_de_exibição: {"id", "tags", "title"}}, ex: "Lee Sin" -> {"id": "LeeSin", ...}."""
    if CACHE_FILE.exists() and not force_refresh:
        cached = json.loads(CACHE_FILE.read_text(encoding="utf-8"))
        if "_version" in cached:
            return cached["champions"]

    version = _fetch_latest_version()
    raw = _fetch_champion_data(version)
    champion_map = {
        info["name"]: {"id": info["id"], "tags": info["tags"], "title": info["title"]}
        for info in raw.values()
    }

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    payload = {"_version": version, "champions": champion_map}
    CACHE_FILE.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return champion_map


def build_lookup(champion_map: dict) -> dict:
    """Índice normalizado que aceita tanto o nome de exibição quanto o id da Data Dragon."""
    lookup = {}
    for name, info in champion_map.items():
        lookup[normalize_name(name)] = name
        lookup[normalize_name(info["id"])] = name
    lookup["nunu"] = next((n for n in champion_map if n.startswith("Nunu")), "Nunu")
    return lookup


def resolve_champion(lookup: dict, champion_name):
    if champion_name is None or str(champion_name).lower() in {"nan", "<na>", ""}:
        return None
    return lookup.get(normalize_name(champion_name))


def primary_role(champion_map: dict, lookup: dict, champion_name):
    name = resolve_champion(lookup, champion_name)
    if name is None or not champion_map[name]["tags"]:
        return None
    return champion_map[name]["tags"][0]


if __name__ == "__main__":
    champ_map = get_champion_map()
    print(f"{len(champ_map)} campeões carregados.")
    for name, info in list(champ_map.items())[:5]:
        print(name, "->", info["tags"])
