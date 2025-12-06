#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DEBUG — Inspeção da saída de embeddings do BirdNET (birdnetlib + Analyzer)

Objetivo:
    - Pegar alguns arquivos de áudio do dataset
    - Rodar Recording.extract_embeddings()
    - Inspecionar a estrutura de `recording.embeddings`
    - Descobrir por que muitos áudios NÃO estão gerando embeddings válidos
"""

from pathlib import Path
import random
import traceback

import numpy as np
import pandas as pd

from birdnetlib.analyzer import Analyzer
from birdnetlib import Recording


# ---------------------------------------------------------------------
# Caminhos básicos (mesma lógica das outras etapas)
# ---------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

META_CLEAN_CSV = DATA_DIR / "xenocanto_clean.csv"
AUDIO_DIR = DATA_DIR / "xeno_canto_audio_mp3"


def escolher_amostras(n_amostras=10, modo="primeiros"):
    """
    Escolhe alguns arquivos do CSV limpo para debug.

    modo:
        - "primeiros": pega as primeiras N linhas
        - "random": faz amostragem aleatória
    """
    if not META_CLEAN_CSV.exists():
        raise FileNotFoundError(f"CSV limpo não encontrado: {META_CLEAN_CSV}")

    df = pd.read_csv(META_CLEAN_CSV)

    # tenta achar coluna com o nome do arquivo
    # ajuste aqui se o nome da coluna for diferente
    possiveis_colunas = ["file", "file_name", "filename", "file_path", "audio_file"]
    col_arquivo = None
    for c in possiveis_colunas:
        if c in df.columns:
            col_arquivo = c
            break

    if col_arquivo is None:
        raise KeyError(
            f"Não encontrei coluna de arquivo em {META_CLEAN_CSV}. "
            f"Tente ajustar a lista 'possiveis_colunas' no código."
        )

    if modo == "random":
        if len(df) > n_amostras:
            df = df.sample(n_amostras, random_state=42)
        else:
            df = df.copy()
    else:
        df = df.head(n_amostras).copy()

    arquivos = df[col_arquivo].astype(str).tolist()
    return arquivos


def debug_arquivo(analyzer: Analyzer, audio_path: Path, idx: int):
    """
    Roda o BirdNET em um único arquivo e imprime a estrutura de embeddings.
    """
    print("\n" + "=" * 80)
    print(f"[{idx:02d}] Arquivo: {audio_path.name}")
    print(f"Caminho completo: {audio_path}")

    if not audio_path.exists():
        print("  ❌ Arquivo NÃO encontrado no disco.")
        return

    try:
        # assinatura correta para birdnetlib 0.18.0
        recording = Recording(
            analyzer,
            str(audio_path),
            min_conf=0.0,
        )

        print("  - Chamando extract_embeddings() ...")
        recording.extract_embeddings()

        emb_obj = recording.embeddings
        print(f"  - Tipo de recording.embeddings: {type(emb_obj)}")

        # Caso 1: numpy array
        if isinstance(emb_obj, np.ndarray):
            print(f"  - Shape do array: {emb_obj.shape}")
            print(f"  - dtype: {emb_obj.dtype}")
            if emb_obj.size == 0:
                print("  ⚠ Array vazio (nenhum embedding).")
            else:
                # mostra alguns valores
                flat = emb_obj.ravel()
                print("  - Primeiros 5 valores:", flat[:5].tolist())
            return

        # Caso 2: lista
        if isinstance(emb_obj, list):
            print(f"  - Tamanho da lista: {len(emb_obj)}")
            if len(emb_obj) == 0:
                print("  ⚠ Lista vazia (nenhum segmento).")
                return

            # examina até 3 elementos
            max_seg = min(3, len(emb_obj))
            for i in range(max_seg):
                seg = emb_obj[i]
                print(f"    [seg {i}] tipo: {type(seg)}")

                if isinstance(seg, dict):
                    print(f"    [seg {i}] chaves: {list(seg.keys())}")

                    # mostra algumas chaves e infos de shape/tipo
                    for k in seg.keys():
                        val = seg[k]
                        if isinstance(val, np.ndarray):
                            print(
                                f"      - {k}: ndarray, shape={val.shape}, dtype={val.dtype}"
                            )
                        elif isinstance(val, list):
                            print(
                                f"      - {k}: list, len={len(val)} "
                                f"(primeiros elementos: {val[:3] if len(val) > 0 else []})"
                            )
                        else:
                            print(
                                f"      - {k}: {type(val)}, valor exemplo={repr(val)[:40]}"
                            )
                else:
                    print(f"    [seg {i}] valor bruto: {repr(seg)[:200]}")
            return

        # Outro tipo qualquer
        print("  ⚠ recording.embeddings tem tipo inesperado / não tratado.")

    except Exception as e:
        print("  ❌ EXCEÇÃO ao processar o arquivo:")
        print("     ", e)
        print("  ---- stack trace ----")
        traceback.print_exc()


def main():
    print("=== DEBUG — Amostras de embeddings BirdNET ===")


    # 1) Carrega o Analyzer uma vez só
    print("\nCarregando Analyzer (BirdNET)...")
    analyzer = Analyzer()
    print("✔ Analyzer carregado.")

    # 2) Escolhe alguns arquivos para inspecionar
    print("\nSelecionando amostras do CSV limpo...")
    arquivos = escolher_amostras(n_amostras=10, modo="random")
    print(f"Total de arquivos selecionados para debug: {len(arquivos)}")

    # 3) Roda debug em cada arquivo
    for idx, nome_arquivo in enumerate(arquivos, start=1):
        # Se no CSV veio só o nome (XCxxxx.mp3), monta o caminho
        # Se já veio caminho completo, esse join ainda funciona
        audio_path = AUDIO_DIR / Path(nome_arquivo).name
        debug_arquivo(analyzer, audio_path, idx)

    print("\n=== FIM DO DEBUG ===")
    print("Use a saída acima para ajustar a lógica de extração de embeddings.")


if __name__ == "__main__":
    main()
