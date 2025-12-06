#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Remover espécies com apenas 1 amostra (singletons) do dataset Xeno-canto.

O que este script faz:

1. Lê o CSV limpo: data/xenocanto_clean.csv
2. Conta quantas vezes cada espécie (coluna 'en') aparece.
3. Identifica as espécies que aparecem APENAS 1 vez.
4. Para cada linha dessas espécies:
   - Monta o nome de arquivo: XC{id}-{gen}-{sp}.mp3 ou .wav
   - Tenta apagar o(s) arquivo(s) em data/xeno_canto_audio_mp3
   - Remove a linha do DataFrame
5. Faz backup do CSV original e sobrescreve o xenocanto_clean.csv com a versão filtrada.

ATENÇÃO:
- O backup é salvo como: data/xenocanto_clean.backup_before_singletons.csv
"""

from pathlib import Path
import shutil

import pandas as pd
from tqdm import tqdm

# ---------------------------------------------------------------------
# Caminhos
# ---------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
AUDIO_DIR = DATA_DIR / "xeno_canto_audio_mp3"

META_CLEAN_CSV = DATA_DIR / "xenocanto_clean.csv"
BACKUP_CSV = DATA_DIR / "xenocanto_clean.backup_before_singletons.csv"

# Colunas usadas (compatíveis com o CSV que você mostrou)
COL_ID = "id"
COL_GEN = "gen"
COL_SP = "sp"
COL_SPECIES = "en"  # nome comum em inglês


def montar_caminhos_audio(row: pd.Series) -> list[Path]:
    """
    Monta possíveis caminhos de arquivo para o áudio dessa linha.

    Padrão principal:
        XC{id}-{gen}-{sp}.mp3
        XC{id}-{gen}-{sp}.wav

    Retorna uma lista de Paths (alguns podem não existir).
    """
    xc_id = str(row[COL_ID]).strip()
    gen = str(row[COL_GEN]).strip()
    sp = str(row[COL_SP]).strip()

    base_name = f"XC{xc_id}-{gen}-{sp}"

    candidatos: list[Path] = []
    for ext in [".mp3", ".wav"]:
        candidatos.append(AUDIO_DIR / f"{base_name}{ext}")

    return candidatos


def main():
    print("=== REMOVER SINGLETONS (espécies com apenas 1 amostra) ===\n")

    if not META_CLEAN_CSV.exists():
        raise FileNotFoundError(f"CSV limpo não encontrado: {META_CLEAN_CSV}")

    print(f"Lendo CSV limpo: {META_CLEAN_CSV}")
    df = pd.read_csv(META_CLEAN_CSV)
    n_total_original = len(df)
    print(f"Total de linhas no CSV (antes): {n_total_original}")

    # Garantir colunas básicas
    for col in [COL_ID, COL_GEN, COL_SP, COL_SPECIES]:
        if col not in df.columns:
            raise KeyError(f"Coluna obrigatória '{col}' não encontrada no CSV.")

    # -----------------------------------------------------------------
    # 1. Contar quantas amostras cada espécie tem
    # -----------------------------------------------------------------
    counts = df[COL_SPECIES].value_counts()
    singletons = counts[counts == 1]

    n_singleton_species = len(singletons)
    print(f"\nTotal de espécies com APENAS 1 amostra: {n_singleton_species}")

    if n_singleton_species == 0:
        print("Não há singletons. Nada para remover.")
        return

    print("\nExemplo de algumas espécies singleton:")
    print(singletons.head(15))

    # Máscara: quais linhas pertencem a espécies singleton
    mask_singleton = df[COL_SPECIES].isin(singletons.index)
    df_singletons = df[mask_singleton].copy()
    df_keep = df[~mask_singleton].copy()

    n_rows_singleton = len(df_singletons)
    n_rows_keep = len(df_keep)

    print(f"\nTotal de linhas que serão REMOVIDAS (singletons): {n_rows_singleton}")
    print(f"Total de linhas que permanecerão:                 {n_rows_keep}")

    # -----------------------------------------------------------------
    # 2. Apagar arquivos de áudio dos singletons
    # -----------------------------------------------------------------
    print("\nApagando arquivos de áudio correspondentes às espécies singleton...")

    arquivos_apagados = 0
    arquivos_nao_encontrados = 0

    for _, row in tqdm(df_singletons.iterrows(), total=n_rows_singleton, desc="Deletando áudios"):
        cand_paths = montar_caminhos_audio(row)

        for p in cand_paths:
            if p.exists():
                try:
                    p.unlink()
                    arquivos_apagados += 1
                    # Se quiser ver cada deleção, descomente:
                    # print(f"[DEL] {p}")
                except Exception as e:
                    print(f"[ERRO] Ao tentar deletar {p}: {e}")
            else:
                arquivos_nao_encontrados += 1

    # -----------------------------------------------------------------
    # 3. Backup do CSV original e sobrescrever com o filtrado
    # -----------------------------------------------------------------
    print("\nCriando backup do CSV original...")
    shutil.copy2(META_CLEAN_CSV, BACKUP_CSV)
    print(f"✔ Backup criado em: {BACKUP_CSV}")

    print("Salvando CSV filtrado (sem singletons)...")
    df_keep.to_csv(META_CLEAN_CSV, index=False)
    print(f"✔ CSV atualizado salvo em: {META_CLEAN_CSV}")

    # -----------------------------------------------------------------
    # 4. Resumo
    # -----------------------------------------------------------------
    print("\n=== RESUMO DA LIMPEZA ===")
    print(f"Linhas totais antes:               {n_total_original}")
    print(f"Espécies singleton removidas:      {n_singleton_species}")
    print(f"Linhas removidas (singletons):     {n_rows_singleton}")
    print(f"Linhas restantes:                  {n_rows_keep}")
    print(f"Arquivos de áudio apagados (OK):   {arquivos_apagados}")
    print(f"Caminhos tentados inexistentes:    {arquivos_nao_encontrados}")
    print("\nConcluído. O próximo ETAPA 2 + 3 já vão rodar só com espécies >= 2 amostras.")


if __name__ == "__main__":
    main()
