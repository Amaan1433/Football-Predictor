"""Team-name normalization: every raw spelling maps to one canonical name.

Add new spellings here as sources are added (football-data.org, Highlightly).
Unknown names raise an error instead of passing through silently.
"""

from __future__ import annotations

from collections.abc import Iterable

# raw spelling -> canonical name
ALIASES: dict[str, str] = {
    # DataHub / football-data.co.uk short names
    "Alaves": "Alaves",
    "Albacete": "Albacete",
    "Almeria": "Almeria",
    "Ath Bilbao": "Athletic Bilbao",
    "Ath Madrid": "Atletico Madrid",
    "Barcelona": "Barcelona",
    "Betis": "Real Betis",
    "Cadiz": "Cadiz",
    "Celta": "Celta Vigo",
    "Compostela": "Compostela",
    "Cordoba": "Cordoba",
    "Eibar": "Eibar",
    "Elche": "Elche",
    "Espanol": "Espanyol",
    "Extremadura": "Extremadura",
    "Getafe": "Getafe",
    "Gimnastic": "Gimnastic",
    "Girona": "Girona",
    "Granada": "Granada",
    "Hercules": "Hercules",
    "Huesca": "Huesca",
    "La Coruna": "Deportivo La Coruna",
    "Las Palmas": "Las Palmas",
    "Leganes": "Leganes",
    "Lerida": "Lleida",
    "Levante": "Levante",
    "Logrones": "Logrones",
    "Malaga": "Malaga",
    "Mallorca": "Mallorca",
    "Merida": "Merida",
    "Murcia": "Real Murcia",
    "Numancia": "Numancia",
    "Osasuna": "Osasuna",
    "Oviedo": "Real Oviedo",
    "Real Madrid": "Real Madrid",
    "Recreativo": "Recreativo Huelva",
    "Salamanca": "Salamanca",
    "Santander": "Racing Santander",
    "Sevilla": "Sevilla",
    "Sociedad": "Real Sociedad",
    "Sp Gijon": "Sporting Gijon",
    "Tenerife": "Tenerife",
    "Valencia": "Valencia",
    "Valladolid": "Real Valladolid",
    "Vallecano": "Rayo Vallecano",
    "Villareal": "Villarreal",  # typo present in some seasons
    "Villarreal": "Villarreal",
    "Xerez": "Xerez",
    "Zaragoza": "Real Zaragoza",
    # Highlightly spellings that differ from the canonical name
    "Alavés": "Alaves",
    "Athletic Club": "Athletic Bilbao",
    "Atlético Madrid": "Atletico Madrid",
    "Celta de Vigo": "Celta Vigo",
    "Deportivo La Coruña": "Deportivo La Coruna",
    "Sevilla FC": "Sevilla",
}

CANONICAL: frozenset[str] = frozenset(ALIASES.values())
# A canonical name is always a valid spelling of itself.
ALIASES.update({c: c for c in CANONICAL})


class UnmappedTeamError(KeyError):
    """Raised when a raw team name has no entry in ALIASES."""


def check_mapped(names: Iterable[str]) -> None:
    """Raise UnmappedTeamError listing every raw name missing from ALIASES."""
    missing = sorted({n for n in names if n not in ALIASES})
    if missing:
        raise UnmappedTeamError(f"Unmapped team names (add to src/names.py): {missing}")


def canonical(name: str) -> str:
    """Return the canonical name for a raw spelling, failing loudly if unknown."""
    check_mapped([name])
    return ALIASES[name]
