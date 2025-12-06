#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
xeno_canto_downloader_clean.py
Coleta sons curtos e mais puros de pássaros (Xeno-Canto v3) com fallback progressivo.
Filtra no cliente por duração <= 20s, tipo contendo {song, call, vocal}, e remove ruídos por rmk.
"""

import os
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Set

import requests
import pandas as pd
from tqdm import tqdm
from dateutil import parser as dateparser

# ========================
# CONFIG
# ========================

API_URL = "https://xeno-canto.org/api/3/recordings"
API_KEY = os.getenv("XENO_CANTO_API_KEY", "bd912d57358526a0925b784bf418fd04cf628a1a")

# BBOX aproximada do Amazonas (NÃO será mais usada na query, mas mantida se quiser voltar)
AMAZONAS_BOX = "-9.9,-73.9,2.3,-56.0"

TARGET_COUNT = 1000
PER_PAGE = 300            # API aceita 50–500
SLEEP_TIME = 0.35
DOWNLOAD_AUDIO = True

OUT_DIR = Path("../data/xeno_canto_audio_mp3")
CSV_PATH = Path("../data/xenocanto_meta.csv")

# filtros cliente
MAX_LEN_SECONDS = 20
TYPE_KEYWORDS = ("song", "call", "vocal")

BAD_RMK_TERMS = (
    "background", "noise", "forest", "rain", "river", "wind",
    "multiple", "human", "traffic", "talk", "songscape", "soundscape"
)

# ========================
# HELPERS
# ========================

def ensure_https(url_or_path: str) -> str:
    return "https:" + url_or_path if isinstance(url_or_path, str) and url_or_path.startswith("//") else url_or_path

def safe_filename(s: str) -> str:
    return "".join(c for c in s if c.isalnum() or c in ("-", "_", ".", " ")).rstrip()

def guess_ext_from_filename(name: str) -> str:
    if not name:
        return ".mp3"
    n = name.lower()
    for ext in (".mp3", ".wav", ".flac", ".ogg", ".m4a"):
        if n.endswith(ext):
            return ext
    return ".mp3"

def build_default_filename(rec: Dict[str, Any]) -> str:
    rid = str(rec.get("id", "")).strip()
    gen = (rec.get("gen") or "").strip() or "gen"
    sp  = (rec.get("sp")  or "").strip() or "sp"
    base = f"XC{rid}-{gen}-{sp}"
    ext  = guess_ext_from_filename(rec.get("file-name") or "")
    return safe_filename(base + ext)

def download_file(url: str, dest_path: Path, session: requests.Session) -> bool:
    try:
        with session.get(url, stream=True, timeout=60) as r:
            r.raise_for_status()
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            with open(dest_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
        return True
    except Exception:
        return False

def fetch_page(query: str, key: str, per_page: int, page: int, session: requests.Session) -> Dict[str, Any]:
    params = {"query": query, "key": key, "per_page": per_page, "page": page}
    r = session.get(API_URL, params=params, timeout=60)
    r.raise_for_status()
    return r.json()

def parse_len_to_seconds(length_field: str) -> Optional[float]:
    """Converte 'mm:ss' ou 'h:mm:ss' em segundos; aceita já numérico em string."""
    if not length_field:
        return None
    s = str(length_field).strip()
    if s.isdigit():
        return float(s)
    parts = s.split(":")
    try:
        parts = [int(p) for p in parts]
    except Exception:
        return None
    if len(parts) == 3:
        h, m, sec = parts
        return h * 3600 + m * 60 + sec
    if len(parts) == 2:
        m, sec = parts
        return m * 60 + sec
    if len(parts) == 1:
        return float(parts[0])
    return None

def normalize_recording(rec: Dict[str, Any], audio_path: Path) -> Dict[str, Any]:
    out = {k: rec.get(k, "") for k in (
        "id", "gen", "sp", "grp", "en", "rec", "cnt", "loc",
        "lat", "lon", "q", "length", "type", "rmk", "file-name", "date"
    )}
    out["file_path"] = str(audio_path)
    # parse coords
    for k in ("lat", "lon"):
        try:
            out[k] = float(out[k]) if out[k] not in ("", None) else None
        except Exception:
            out[k] = None
    # iso date
    try:
        out["date_iso"] = dateparser.parse(out.get("date") or "").date().isoformat()
    except Exception:
        out["date_iso"] = ""
    # numeric seconds
    out["length_seconds"] = parse_len_to_seconds(out.get("length") or "")
    return out

def pass_client_filters(rec: Dict[str, Any]) -> bool:
    """Aplica filtros locais (robustos)."""
    # Duração
    ls = parse_len_to_seconds(rec.get("length", ""))
    if ls is None or ls <= 0 or ls > MAX_LEN_SECONDS:
        return False

    # Tipo desejado (conteúdo textual; o campo 'type' pode trazer texto livre)
    t = (rec.get("type") or "").lower()
    if not any(k in t for k in TYPE_KEYWORDS):
        return False

    # Remover ruídos (remarks/rmk)
    rmk = (rec.get("rmk") or "").lower()
    if any(b in rmk for b in BAD_RMK_TERMS):
        return False

    return True

# ========================
# COLETOR (AGORA GLOBAL, QUALIDADE >= B)
# ========================

def coletar_amazonas(target_count: int = TARGET_COUNT) -> pd.DataFrame:
    """
    Estratégia progressiva (ajustada):
      1) grp:birds q:>=B  → pássaros do mundo com boa qualidade
      2) grp:birds q:>=C  → um pouco mais abrangente
      3) grp:birds        → fallback geral
    Filtro de duração, tipo e rmk continua sendo feito no cliente.
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []
    seen_ids: Set[str] = set()

    # 🔥 AQUI está a mudança importante:
    stages = [
        ("birds", 'grp:birds q:">=B"'),
        ("birds", 'grp:birds q:">=C"'),
        ("birds", 'grp:birds'),
    ]

    with requests.Session() as session, tqdm(
        total=target_count, desc="📥 Download total", unit="rec",
        ncols=90, colour="green", dynamic_ncols=True, leave=True
    ) as pbar:

        for grp_name, base_query in stages:
            if len(rows) >= target_count:
                break

            page = 1
            printed_header = False

            while len(rows) < target_count:
                try:
                    data = fetch_page(base_query, API_KEY, PER_PAGE, page, session)
                except Exception as e:
                    print(f"[WARN] Falha na query '{base_query}' p{page}: {e}")
                    break

                recs = data.get("recordings", []) or []
                if not printed_header:
                    total_found = int(data.get("numRecordings", 0) or 0)
                    num_pages   = int(data.get("numPages", 1) or 1)
                    print(f"\n[STAGE {grp_name}] Query: {base_query}  (found={total_found}, pages={num_pages})")
                    printed_header = True

                if not recs:
                    break

                accepted_this_page = 0

                for rec in recs:
                    if len(rows) >= target_count:
                        break

                    rid = str(rec.get("id", "")).strip()
                    if not rid or rid in seen_ids:
                        continue

                    # Filtro local (robusto)
                    if not pass_client_filters(rec):
                        continue

                    file_url = ensure_https(rec.get("file", ""))
                    if not file_url:
                        continue

                    dest_path = OUT_DIR / build_default_filename(rec)
                    ok = True
                    if DOWNLOAD_AUDIO:
                        if not dest_path.exists() or dest_path.stat().st_size == 0:
                            ok = download_file(file_url, dest_path, session)
                    if not ok:
                        continue

                    row = normalize_recording(rec, dest_path)
                    rows.append(row)
                    seen_ids.add(rid)
                    accepted_this_page += 1
                    pbar.update(1)

                # debug por página
                print(f"  p{page}: aceitos={accepted_this_page} | total={len(rows)}")

                if len(rows) >= target_count:
                    break

                page += 1
                if page > int(data.get("numPages", 1)):
                    break
                time.sleep(SLEEP_TIME)

    return pd.DataFrame(rows)

# ========================
# MAIN
# ========================

if __name__ == "__main__":
    print("🌎 Iniciando coleta GLOBAL de sons curtos e limpos de pássaros (qualidade >= B)...")
    df = coletar_amazonas(TARGET_COUNT)

    if not df.empty:
        # deduplicar por caminho ou id
        if "id" in df.columns:
            df = df.drop_duplicates(subset=["id"]).copy()
        df = df.drop_duplicates(subset=["file_path"]).copy()
        CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(CSV_PATH, index=False, encoding="utf-8")
        print(f"\nRegistros coletados: {len(df)}")
        print(f"CSV salvo em: {CSV_PATH.resolve()}")
        print(f"Áudios salvos em: {OUT_DIR.resolve()}")
    else:
        print("\nNenhum registro passou pelos filtros. Sugestões:")
        print("  - Aumente MAX_LEN_SECONDS para 25–30")
        print("  - Relaxe BAD_RMK_TERMS ou TYPE_KEYWORDS")
        print("  - Remova q:\">=B\" e teste com q:\">=C\" ou sem filtro de qualidade")
