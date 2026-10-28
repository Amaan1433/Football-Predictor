"""Club identity across openfootball domestic leagues, Champions League files and our La Liga tables.

club_id = "<COUNTRY>:<key>", where key is the normalized name (accents, case,
punctuation, generic tokens like FC/AFC and numbers removed), after applying
EU_ALIASES for spellings that normalization cannot reconcile.
Spanish clubs use our La Liga canonical names, so both models share identities.
"""

from __future__ import annotations

import re
import unicodedata

from src.names import ALIASES as LALIGA_ALIASES

# Tokens that carry no identity ("FC Porto" == "Porto"). Numbers are dropped separately.
GENERIC_TOKENS = {
    "fc", "afc", "cf", "sc", "ac", "as", "ss", "ssc", "bc", "cfc", "club", "clube", "calcio",
    "de", "del", "futbol", "balompie", "sad", "cd", "ud", "sd", "rcd", "rc", "fk", "sk", "nk",
    "kv", "bk", "if", "ff", "pfc", "sv", "bv", "vfl", "vfb", "tsg", "fsv", "ogc", "losc", "sco",
    "us", "uc", "gd", "cs", "sl", "sbv",
}

# (country, normalized key) -> normalized key to use instead.
EU_ALIASES: dict[tuple[str, str], str] = {
    ("ITA", "internazionale milano"): "inter",
    ("ITA", "lazio roma"): "lazio",
    ("NED", "az alkmaar"): "az",
    ("NED", "feyenoord rotterdam"): "feyenoord",
    ("NED", "nec nijmegen"): "nec",
    ("NED", "psv eindhoven"): "psv",
    ("POR", "estoril praia"): "estoril",
    ("POR", "sport lisboa e benfica"): "benfica",
    ("POR", "sporting portugal"): "sporting",
    ("POR", "sporting cp"): "sporting",
    ("AZE", "qarabag agdam"): "qarabag",
    ("BEL", "royale union saint gilloise"): "union saint gilloise",
    ("CRO", "gnk dinamo zagreb"): "dinamo zagreb",
    ("AUT", "red bull salzburg"): "rb salzburg",
    ("CYP", "pafos"): "paphos",
    ("GRE", "pae olympiakos sfp"): "olympiakos piraeus",
    ("GER", "bor monchengladbach"): "borussia monchengladbach",
    ("FRA", "strasbourg alsace"): "strasbourg",
    ("FRA", "racing lens"): "lens",
    ("ROU", "steaua bucuresti"): "fcsb",  # FCSB continued Steaua's team after the 2017 name dispute
    # Highlightly spellings of clubs already in the openfootball data
    ("GER", "bayern munich"): "bayern munchen",
    ("FRA", "lille"): "lille osc",
    ("FRA", "lyon"): "olympique lyonnais",
    ("GRE", "aek athens"): "aek athen",
    ("KOS", "drita"): "kf drita",
    ("EST", "flora tallinn"): "flora",
    ("KAZ", "kairat almaty"): "kairat",
    ("ENG", "brighton"): "brighton hove albion",
    ("ENG", "coventry"): "coventry city",
    ("ENG", "ipswich"): "ipswich town",
    ("ENG", "leeds"): "leeds united",
    ("ENG", "tottenham"): "tottenham hotspur",
    ("ITA", "fiorentina"): "acf fiorentina",
    ("FRA", "auxerre"): "aj auxerre",
    ("FRA", "estac troyes"): "es troyes",
    ("FRA", "marseille"): "olympique marseille",
    ("FRA", "rennes"): "stade rennais",
    ("POR", "braga"): "sporting braga",
    ("POR", "guimaraes"): "vitoria guimaraes",
    ("NED", "cambuur"): "cambuur leeuwarden",
    ("NED", "willem ii"): "willem ii tilburg",
}

# Highlightly gives no country. Clubs already in our data are found by name (see
# highlightly_club_id); these are placed by hand: UEFA association of the club's league.
HIGHLIGHTLY_COUNTRY: dict[str, str] = {
    "Bayern Munich": "GER", "Lille": "FRA", "Lyon": "FRA", "AEK Athens FC": "GRE", "Drita": "KOS",
    "Flora Tallinn": "EST", "Kairat Almaty": "KAZ",
    # new to our data (no earlier UCL season on openfootball)
    "Aarhus GF": "DEN", "Ararat-Armenia": "ARM", "Atert Bissen": "LUX", "FC Thun": "SUI", "Floriana": "MLT",
    "Gornik Zabrze": "POL", "Gyori ETO FC": "HUN", "Hapoel Beer Sheva": "ISR", "Heart of Midlothian": "SCO",
    "Kauno Žalgiris": "LTU", "KuPS": "FIN", "LASK Linz": "AUT", "Levski Sofia": "BUL", "MKK-Dnepr": "BLR",
    "Mjällby AIF": "SWE", "Omonia Nicosia": "CYP", "Riga FC": "LVA", "Sabah FA": "AZE", "Saburtalo": "GEO",
    "Sutjeska": "MNE", "Tre Fiori": "SMR", "Universitatea Craiova": "ROU", "Vardar Skopje": "MKD", "Viking": "NOR",
}

# Common words that do not suggest two names are the same club (used by the rename checks).
SHARED_WORD_STOPLIST = {"city", "town", "united", "rovers", "wanderers", "albion", "county", "athletic"}

# UEFA association code -> country of the league the club plays in.
COUNTRY_OVERRIDES: dict[str, str] = {"MCO": "FRA"}  # AS Monaco plays in Ligue 1

# Same-country pairs that share a word but are different clubs (checked by hand).
DISTINCT_CLUBS: set[frozenset[str]] = {
    frozenset(p) for p in [
        ("CZE:slavia praha", "CZE:sparta praha"),
        ("ISR:maccabi haifa", "ISR:maccabi tel aviv"),
        ("RUS:cska moskva", "RUS:lokomotiv moskva"),
        ("RUS:cska moskva", "RUS:spartak moskva"),
        ("RUS:lokomotiv moskva", "RUS:spartak moskva"),
        ("LTU:kauno zalgiris", "LTU:zalgiris"),  # Kaunas vs Vilnius
    ]
}


class ClubNameError(ValueError):
    """Raised when club names cannot be reconciled; the message lists what to add to EU_ALIASES."""


def normalize(name: str) -> str:
    """Accent/case/punctuation-insensitive key with generic tokens and numbers removed."""
    s = unicodedata.normalize("NFKD", name)
    s = "".join(c for c in s if not unicodedata.combining(c)).casefold()
    s = s.replace("ø", "o").replace("ß", "ss").replace("ł", "l").replace("æ", "ae")
    tokens = [t for t in re.split(r"[^a-z0-9]+", s) if t]
    kept = [t for t in tokens if t not in GENERIC_TOKENS and not t.isdigit()]
    return " ".join(kept) or " ".join(tokens)


def club_id(name: str, country: str) -> str:
    """Stable club id for a raw spelling."""
    if country == "ESP":
        canon = LALIGA_ALIASES.get(name)
        if canon is None:  # e.g. "Real Madrid CF": match on normalized La Liga canonical names
            by_key = {normalize(c): c for c in set(LALIGA_ALIASES.values())}
            canon = by_key.get(EU_ALIASES.get(("ESP", normalize(name)), normalize(name)))
        if canon is None:
            raise ClubNameError(f"Spanish club '{name}' does not match any La Liga name (add to src/names.py)")
        return f"ESP:{normalize(canon)}"
    key = normalize(name)
    return f"{country}:{EU_ALIASES.get((country, key), key)}"


def highlightly_club_id(name: str, known: set[str]) -> str:
    """Club id for a Highlightly name: HIGHLIGHTLY_COUNTRY, else the one known club it normalizes to."""
    if name in HIGHLIGHTLY_COUNTRY:
        return club_id(name, HIGHLIGHTLY_COUNTRY[name])
    hits = set()
    for country in {k.split(":")[0] for k in known}:
        try:
            cid = club_id(name, country)
        except ClubNameError:  # not a La Liga name
            continue
        if cid in known:
            hits.add(cid)
    if len(hits) != 1:
        raise ClubNameError(f"Highlightly club '{name}' matches {sorted(hits) or 'no known club'}; "
                            "add it to HIGHLIGHTLY_COUNTRY (and EU_ALIASES if spelled differently)")
    return hits.pop()
