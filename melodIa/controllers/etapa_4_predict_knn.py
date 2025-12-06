#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ETAPA 4 — Predição de espécie com BirdNET + KNN

Uso:
    - Recebe o caminho de um arquivo de áudio (.mp3 / .wav)
    - Extrai embedding com birdnetlib + Analyzer
    - Usa o KNN treinado (ETAPA 3) para prever a espécie
"""

import sys
from pathlib import Path
import numpy as np

from birdnetlib import Recording
from birdnetlib.analyzer import Analyzer

import joblib


# ---------------------------------------------------------------------
# Caminhos básicos
# ---------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_DIR = BASE_DIR / "model"

KNN_MODEL_FILE = MODEL_DIR / "knn_birdnet.pkl"
LBL_ENCODER_FILE = MODEL_DIR / "label_encoder.pkl"


# ---------------------------------------------------------------------
# Mesma lógica auxiliar da Etapa 2
# ---------------------------------------------------------------------
def descobrir_chave_embedding(seg):
    """
    Tenta descobrir a chave que contém o vetor numérico de embedding.
    """
    import numpy as _np

    if not isinstance(seg, dict):
        return None, None

    # 1) Procura por chaves com 'embed' no nome
    for key in seg.keys():
        lk = str(key).lower()
        if "embed" in lk:
            val = seg[key]
            if isinstance(val, _np.ndarray):
                vec = val.astype(_np.float32).ravel()
                if vec.size >= 8:  # tolerante
                    return key, vec
            elif isinstance(val, list):
                try:
                    vec = _np.array(val, dtype=_np.float32).ravel()
                    if vec.size >= 8:
                        return key, vec
                except Exception:
                    pass

    # 2) Se não achar por nome, tenta qualquer valor list/ndarray
    for key, val in seg.items():
        if isinstance(val, _np.ndarray):
            vec = val.astype(_np.float32).ravel()
            if vec.size >= 8:
                return key, vec
        elif isinstance(val, list):
            try:
                vec = _np.array(val, dtype=_np.float32).ravel()
                if vec.size >= 8:
                    return key, vec
            except Exception:
                pass

    return None, None


def extrair_embedding_unico(analyzer, audio_path: Path) -> np.ndarray:
    """
    Extrai UM vetor de embedding (média dos segmentos) para um único arquivo.
    Retorna um np.ndarray [D] ou lança RuntimeError se não conseguir.
    """
    if not audio_path.exists():
        raise FileNotFoundError(f"Áudio não encontrado: {audio_path}")

    print(f"\n[ETAPA 4] Extraindo embedding para: {audio_path.name}")

    # Cria Recording
    recording = Recording(
        analyzer,
        str(audio_path),
        min_conf=0.0,
    )

    # Extrai embeddings
    recording.extract_embeddings()
    emb_obj = recording.embeddings

    # Caso 1: já veio array numpy
    if isinstance(emb_obj, np.ndarray):
        if emb_obj.size == 0:
            raise RuntimeError("Array de embeddings vazio.")
        if emb_obj.ndim == 1:
            return emb_obj.astype(np.float32)
        return emb_obj.mean(axis=0).astype(np.float32)

    # Caso 2: lista de dicts / segmentos
    if isinstance(emb_obj, list) and len(emb_obj) > 0:
        segmento_vetores = []

        for seg in emb_obj:
            # pode ser dict ou lista/array direto
            if not isinstance(seg, dict):
                try:
                    vec = np.array(seg, dtype=np.float32).ravel()
                    if vec.size >= 8:
                        segmento_vetores.append(vec)
                    continue
                except Exception:
                    continue

            key, vec = descobrir_chave_embedding(seg)
            if key is None or vec is None:
                continue
            segmento_vetores.append(vec)

        if not segmento_vetores:
            raise RuntimeError("Nenhum vetor de embedding numérico válido encontrado nesse áudio.")

        mat = np.stack(segmento_vetores, axis=0)
        return mat.mean(axis=0).astype(np.float32)

    raise RuntimeError(f"Formato inesperado de recording.embeddings: {type(emb_obj)}")


# ---------------------------------------------------------------------
# Predição principal
# ---------------------------------------------------------------------
def prever_especie(audio_path: Path):
    """
    Carrega Analyzer, KNN e LabelEncoder, extrai embedding e retorna
    (label_prevista, prob_opcional).
    """
    # Carrega modelo e encoder
    if not KNN_MODEL_FILE.exists():
        raise FileNotFoundError(f"KNN não encontrado: {KNN_MODEL_FILE}")
    if not LBL_ENCODER_FILE.exists():
        raise FileNotFoundError(f"LabelEncoder não encontrado: {LBL_ENCODER_FILE}")

    print("Carregando KNN e LabelEncoder...")
    knn = joblib.load(KNN_MODEL_FILE)
    le = joblib.load(LBL_ENCODER_FILE)
    print("✔ KNN e LabelEncoder carregados.")

    # Carrega BirdNET Analyzer
    print("Carregando modelo BirdNET-Analyzer (Analyzer)...")
    analyzer = Analyzer()
    print("✔ Analyzer carregado.")

    # Extrai embedding do novo áudio
    emb_vec = extrair_embedding_unico(analyzer, audio_path)  # [D]

    # Ajusta forma para [1, D]
    emb_vec = emb_vec.reshape(1, -1)

    # Predição
    y_pred = knn.predict(emb_vec)
    label_prevista = le.inverse_transform(y_pred)[0]

    # Probabilidades (se suportado)
    prob = None
    if hasattr(knn, "predict_proba"):
        proba = knn.predict_proba(emb_vec)[0]
        idx = y_pred[0]
        prob = float(proba[idx])
        return label_prevista, prob

    return label_prevista, None


# ---------------------------------------------------------------------
# Execução via linha de comando
# ---------------------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso:")
        print("  python -m controllers.etapa_4_predict_knn <caminho_para_arquivo_de_audio>")
        sys.exit(1)

    audio_path = Path(sys.argv[1])

    try:
        especie, prob = prever_especie(audio_path)
        print("\n=== RESULTADO DA PREDIÇÃO ===")
        print(f"Espécie prevista: {especie}")
        if prob is not None:
            print(f"Confiança (aprox): {prob:.4f}")
        else:
            print("Probabilidade não disponível (KNN sem predict_proba).")

    except Exception as e:
        print("\n[ERRO] Durante a predição:")
        print(e)
        sys.exit(1)
