#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Etapa 1 — Validação e preparação do dataset Xeno-Canto
para o pipeline BirdNET + KNN.

O que este script faz:

1. Lê o CSV original: data/xenocanto_meta.csv
2. Garante que exista a coluna de espécie "en".
3. Extrai o nome do arquivo da coluna "file_path".
4. Verifica se o MP3 correspondente existe em:
      data/xeno_canto_audio_mp3/
5. Remove linhas sem espécie ou sem arquivo de áudio.
6. Salva um CSV limpo em:
      data/xenocanto_clean.csv

Esse CSV limpo será usado nas próximas etapas
(geração de embeddings com BirdNET e treino do KNN).
"""

import sys
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------
# Configuração de caminhos com base na estrutura do projeto
# ---------------------------------------------------------------------

# BASE_DIR = .../amzsounds
BASE_DIR = Path(__file__).resolve().parent.parent

AUDIO_DIR = BASE_DIR / "data" / "xeno_canto_audio_mp3"
META_FILE = BASE_DIR / "data" / "xenocanto_meta.csv"
OUTPUT_FILE = BASE_DIR / "data" / "xenocanto_clean.csv"


# ---------------------------------------------------------------------
# Funções auxiliares
# ---------------------------------------------------------------------
def validar_colunas(meta_df: pd.DataFrame) -> pd.DataFrame:
    """
    Garante que o DataFrame tenha as colunas mínimas necessárias.
    - 'file_path' : caminho original do arquivo no Xeno-Canto
    - 'en'        : nome da espécie em inglês (label que vamos usar)
    """
    colunas_necessarias = ["file_path", "en"]

    for col in colunas_necessarias:
        if col not in meta_df.columns:
            raise KeyError(
                f"Coluna obrigatória '{col}' não encontrada no CSV. "
                f"Colunas disponíveis: {list(meta_df.columns)}"
            )

    return meta_df


def validar_dataset(meta_df: pd.DataFrame) -> pd.DataFrame:
    """
    Valida e filtra o dataset:
    - remove linhas sem espécie
    - extrai o nome do arquivo da coluna 'file_path'
    - checa se o arquivo existe em AUDIO_DIR
    """
    print(f"Registros totais no CSV original: {len(meta_df)}")

    # Remover espécies nulas ou vazias
    meta_df = meta_df.dropna(subset=["en"])
    meta_df = meta_df[meta_df["en"].astype(str).str.strip() != ""]
    print(f"Após remover espécies vazias: {len(meta_df)}")

    # Extrair o nome do arquivo do caminho original
    meta_df["filename"] = meta_df["file_path"].apply(
        lambda x: Path(str(x)).name
    )

    # Verificar existência do arquivo MP3
    def arquivo_existe(fname: str) -> bool:
        return (AUDIO_DIR / fname).exists()

    meta_df["exists"] = meta_df["filename"].apply(arquivo_existe)

    total_existentes = meta_df["exists"].sum()
    print(f"Arquivos de áudio encontrados em {AUDIO_DIR}: {total_existentes}")

    # Manter apenas arquivos que existem fisicamente
    meta_df = meta_df[meta_df["exists"] == True]

    print(f"Registros finais após validação: {len(meta_df)}")
    return meta_df


# ---------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------
def etapa_1():
    print("\n=== ETAPA 1 — Validação do Dataset Xeno-Canto ===")

    # Verificações básicas de estrutura
    if not META_FILE.exists():
        raise FileNotFoundError(f"Arquivo CSV não encontrado: {META_FILE}")

    if not AUDIO_DIR.exists():
        raise FileNotFoundError(
            f"Pasta de áudios não encontrada: {AUDIO_DIR}\n"
            "Verifique se os MP3 estão em 'data/xeno_canto_audio_mp3/'."
        )

    print(f"Lendo metadata de: {META_FILE}")
    meta_df = pd.read_csv(META_FILE)

    # Valida colunas obrigatórias
    meta_df = validar_colunas(meta_df)

    # Valida conteúdo e existência dos arquivos
    meta_df = validar_dataset(meta_df)

    # Remover coluna auxiliar 'exists' antes de salvar (opcional)
    if "exists" in meta_df.columns:
        meta_df = meta_df.drop(columns=["exists"])

    print(f"Salvando dataset limpo em: {OUTPUT_FILE}")
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    meta_df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")

    print("Etapa 1 concluída com sucesso.")
    print(f"Use '{OUTPUT_FILE.name}' nas próximas etapas (BirdNET + KNN).\n")


# ---------------------------------------------------------------------
# Execução direta
# ---------------------------------------------------------------------
if __name__ == "__main__":
    try:
        etapa_1()
    except Exception as e:
        print("\n[ERRO] Durante a Etapa 1:")
        print(e)
        sys.exit(1)
