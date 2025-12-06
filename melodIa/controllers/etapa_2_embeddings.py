#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ETAPA 2 — Extração de Embeddings com birdnetlib + BirdNET-Analyzer

Regra para encontrar o áudio (em ordem de prioridade):

1) Usa o file_path vindo do CSV (ex.: ..\\data\\amazonas_audio\\XCxxxx-Gen-sp.mp3),
   resolvendo para caminho absoluto a partir do BASE_DIR.

2) Usa o mesmo nome de arquivo (p.name) dentro de data/xeno_canto_audio_mp3.

3) Se nada disso funcionar, monta:
       base = f"XC{id}-{gen}-{sp}"
       tenta:
           data/xeno_canto_audio_mp3/base.mp3
           data/xeno_canto_audio_mp3/base.wav
"""

from pathlib import Path
import traceback

import numpy as np
import pandas as pd
from tqdm import tqdm

from birdnetlib.analyzer import Analyzer
from birdnetlib import Recording

# ---------------------------------------------------------------------
# Caminhos
# ---------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

META_CLEAN_CSV = DATA_DIR / "xenocanto_clean.csv"
AUDIO_DIR = DATA_DIR / "xeno_canto_audio_mp3"

EMB_OUT = DATA_DIR / "xenocanto_embeddings.npy"
LBL_OUT = DATA_DIR / "xenocanto_labels.csv"

# Colunas esperadas no CSV
COL_ID = "id"
COL_GEN = "gen"
COL_SP = "sp"
COL_SPECIES = "en"       # nome comum em inglês
COL_FILE_PATH = "file_path"  # caminho original vindo do xenocanto_meta


def montar_caminho_audio(row: pd.Series) -> Path | None:
    """
    Resolve o caminho do áudio nesta ordem:

    1) file_path vindo do CSV (..\data\amazonas_audio\XCxxxx-Gen-sp.mp3).
       - Se for relativo, resolvemos a partir do BASE_DIR.
    2) Mesmo nome de arquivo (basename) dentro de data/xeno_canto_audio_mp3.
    3) Nome sintetizado: XC{id}-{gen}-{sp}.mp3 ou .wav em data/xeno_canto_audio_mp3.

    Retorna o primeiro Path que existir. Caso nenhum exista, retorna None.
    """
    candidates: list[Path] = []

    # 1) Tentar o file_path vindo do CSV
    raw_fp = str(row.get(COL_FILE_PATH, "")).strip()
    if raw_fp:
        p = Path(raw_fp)

        # Se for relativo, tornar absoluto a partir do BASE_DIR
        if not p.is_absolute():
            p = (BASE_DIR / p).resolve()

        # a) Caminho original
        candidates.append(p)
        # b) Mesmo nome de arquivo na pasta xeno_canto_audio_mp3
        candidates.append(AUDIO_DIR / p.name)

    # 2) Montar a partir de id + gen + sp (fallback)
    try:
        xc_id = str(row[COL_ID]).strip()
        gen = str(row[COL_GEN]).strip()
        sp = str(row[COL_SP]).strip()
        base_name = f"XC{xc_id}-{gen}-{sp}"

        candidates.append(AUDIO_DIR / f"{base_name}.mp3")
        candidates.append(AUDIO_DIR / f"{base_name}.wav")
    except KeyError:
        # Se faltarem colunas, apenas não adiciona essas opções
        pass

    # 3) Retornar o primeiro que existir
    for c in candidates:
        if c.exists():
            return c

    return None


def main():
    print("=== ETAPA 2 — Embeddings com birdnetlib + Analyzer (usando file_path + fallback) ===")

    if not META_CLEAN_CSV.exists():
        raise FileNotFoundError(f"CSV limpo não encontrado: {META_CLEAN_CSV}")

    df = pd.read_csv(META_CLEAN_CSV)
    print(f"Lendo metadata limpa de: {META_CLEAN_CSV}")
    print(f"Total de registros no CSV limpo: {len(df)}")

    # Garante que as colunas necessárias existem
    for col in [COL_ID, COL_GEN, COL_SP, COL_SPECIES, COL_FILE_PATH]:
        if col not in df.columns:
            raise KeyError(f"Coluna obrigatória '{col}' não encontrada no CSV.")

    print(f"Coluna de espécie usada: '{COL_SPECIES}'")
    print(f"Coluna de caminho original usada: '{COL_FILE_PATH}'")

    # Carregar o Analyzer uma única vez
    print("\nCarregando modelo BirdNET-Analyzer via birdnetlib.Analyzer()...")
    analyzer = Analyzer()
    print("✔ Analyzer carregado.\n")

    embeddings_list: list[np.ndarray] = []
    labels_list: list[str] = []

    n_total = len(df)
    n_sem_arquivo = 0
    n_sem_segmentos = 0
    n_sem_vetor_1024 = 0
    n_ok = 0

    for _, row in tqdm(df.iterrows(), total=n_total, desc="Processando áudios"):
        species = str(row[COL_SPECIES])

        audio_path = montar_caminho_audio(row)
        if audio_path is None:
            n_sem_arquivo += 1
            if n_sem_arquivo <= 10:
                print(
                    f"[AVISO] Arquivo não encontrado para id={row[COL_ID]}, "
                    f"gen={row[COL_GEN]}, sp={row[COL_SP]}"
                )
            continue

        try:
            recording = Recording(
                analyzer,
                str(audio_path),
                min_conf=0.0,
            )

            recording.extract_embeddings()
            emb_obj = recording.embeddings

            # Deve ser uma lista de dicts (igual ao debug)
            if not isinstance(emb_obj, list) or len(emb_obj) == 0:
                n_sem_segmentos += 1
                if n_sem_segmentos <= 5:
                    print(
                        f"[AVISO] Sem embeddings (lista vazia) em: {audio_path.name}"
                    )
                continue

            segment_vectors = []
            for seg in emb_obj:
                # Caso normal -> dict com 'embeddings'
                if isinstance(seg, dict) and "embeddings" in seg:
                    emb_val = seg["embeddings"]
                    try:
                        v = np.array(emb_val, dtype=np.float32).ravel()
                    except Exception:
                        continue
                    if v.size == 1024:
                        segment_vectors.append(v)
                    continue

                # Fallback raro: seg não-dict
                try:
                    v = np.array(seg, dtype=np.float32).ravel()
                    if v.size == 1024:
                        segment_vectors.append(v)
                except Exception:
                    continue

            if not segment_vectors:
                n_sem_vetor_1024 += 1
                if n_sem_vetor_1024 <= 5:
                    print(
                        f"[AVISO] Não obtive nenhum vetor 1024D em: {audio_path.name}"
                    )
                continue

            # Média dos segmentos -> 1 vetor por arquivo
            mat = np.stack(segment_vectors, axis=0)
            vec = mat.mean(axis=0).astype(np.float32)

            embeddings_list.append(vec)
            labels_list.append(species)
            n_ok += 1

        except Exception as e:
            print(f"[ERRO] Falha ao processar {audio_path.name}: {e}")
            traceback.print_exc()
            continue

    if not embeddings_list:
        print("\n[ERRO] Nenhum embedding foi extraído. Verifique os caminhos dos arquivos.")
        print(f"  - Linhas sem arquivo encontrado: {n_sem_arquivo}")
        print(f"  - Linhas sem segmentos:          {n_sem_segmentos}")
        print(f"  - Linhas sem vetor 1024D:        {n_sem_vetor_1024}")
        return

    X = np.stack(embeddings_list, axis=0)
    y = np.array(labels_list, dtype=object)

    print("\nResumo da ETAPA 2:")
    print(f"  - Amostras com embedding extraído: {X.shape[0]}")
    print(f"  - Dimensão do embedding:           {X.shape[1]}")
    print(f"  - Total de linhas no CSV:          {n_total}")
    print(f"  - Linhas sem arquivo encontrado:   {n_sem_arquivo}")
    print(f"  - Linhas sem segmentos:            {n_sem_segmentos}")
    print(f"  - Linhas sem vetor 1024D:          {n_sem_vetor_1024}")
    print(f"  - Linhas OK:                       {n_ok}")

    np.save(EMB_OUT, X)
    pd.DataFrame({"label": y}).to_csv(LBL_OUT, index=False)

    print(f"\n✔ Embeddings salvos em: {EMB_OUT}")
    print(f"✔ Labels salvos em:     {LBL_OUT}")
    print("\nETAPA 2 concluída com sucesso.")


if __name__ == "__main__":
    main()
