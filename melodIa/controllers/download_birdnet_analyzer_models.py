#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_birdnetlib_embeddings.py

Teste mínimo para garantir que o BirdNET está funcionando DE VERDADE
usando birdnetlib + BirdNET-Analyzer.

Passos:
1) Carrega o modelo (Analyzer).
2) Lê UM arquivo de áudio.
3) Extrai embeddings com recording.extract_embeddings().
4) Mostra informações básicas dos embeddings.
"""

import os
from datetime import datetime

import numpy as np

from birdnetlib import Recording
from birdnetlib.analyzer import Analyzer


# 1. CONFIGURAÇÕES BÁSICAS
# Ajuste esse caminho para um arquivo real do seu dataset
AUDIO_DIR = r"D:\CienciaDados\amzsounds\data\xeno_canto_audio_mp3"
AUDIO_FILE = r"XC562169-Basileuterus-delattrii.mp3"  # troque se quiser

AUDIO_PATH = os.path.join(AUDIO_DIR, AUDIO_FILE)


def main():
    print("=== TESTE BirdNETLIB + Analyzer (Embeddings) ===")

    if not os.path.isfile(AUDIO_PATH):
        print(f"[ERRO] Arquivo não encontrado:\n  {AUDIO_PATH}")
        return

    # 2. INICIALIZA O ANALYZER (BirdNET-Analyzer)
    # Por padrão, usa a versão mais recente suportada (ex.: 2.4)
    # O modelo será baixado automaticamente se ainda não existir.
    print("\n[1/3] Carregando modelo BirdNET-Analyzer via birdnetlib.Analyzer()...")
    analyzer = Analyzer()  # conforme documentação oficial
    print("✔ Modelo carregado.")

    # 3. CRIA A RECORDING E EXTRAI EMBEDDINGS
    print(f"\n[2/3] Lendo e processando o arquivo:\n  {AUDIO_PATH}")

    # lat/lon/date são opcionais — aqui vamos ignorar, já que queremos só o vetor
    recording = Recording(
        analyzer,
        AUDIO_PATH,
        # Exemplo se você quiser filtrar por localização/data:
        # lat=-3.1,
        # lon=-60.0,
        # date=datetime(year=2024, month=8, day=1),
    )

    # Chamada oficial para embeddings segundo a doc:
    # https://joeweiss.github.io/birdnetlib/api/#embeddings
    recording.extract_embeddings()

    embeddings = recording.embeddings

    print("\n[3/3] Resultado de recording.extract_embeddings():")
    print(f"  Tipo: {type(embeddings)}")

    # Tentativa segura de inspecionar o conteúdo
    if isinstance(embeddings, np.ndarray):
        print(f"  Shape do array: {embeddings.shape}")
    elif isinstance(embeddings, list):
        print(f"  Lista com {len(embeddings)} elementos.")
        if len(embeddings) > 0:
            print(f"  Tipo do primeiro elemento: {type(embeddings[0])}")
            # Se for dict com chave 'embedding', mostramos o tamanho do vetor
            first = embeddings[0]
            if isinstance(first, dict) and "embedding" in first:
                vec = np.array(first["embedding"])
                print(f"  Dimensão do vetor 'embedding': {vec.shape}")
    else:
        print("  (Formato inesperado, vamos inspecionar depois com mais calma.)")

    print("\nTeste concluído.\n"
          "Se não houve Exception e algo foi impresso acima para 'embeddings', "
          "temos um BirdNET FUNCIONANDO. A partir daqui, conectamos isso ao KNN.")


if __name__ == "__main__":
    main()
