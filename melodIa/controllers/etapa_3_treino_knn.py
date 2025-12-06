#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ETAPA 3 — Treino do KNN usando embeddings BirdNET (birdnetlib)

Entradas:
    - data/xenocanto_embeddings.npy   (matriz [N_amostras, N_features])
    - data/xenocanto_labels.csv       (coluna 'label')

Saídas:
    - model/knn_birdnet.pkl           (modelo KNN treinado FINAL, usando TODAS as amostras)
    - model/label_encoder.pkl         (LabelEncoder das espécies)

Lógica de treino/avaliação:

1) Carrega X e y (labels de texto).
2) Codifica labels com LabelEncoder.
3) Verifica distribuição de classes:
   - Se poucas amostras (< 10) ou poucas classes (< 2) → treina KNN com todos os dados, sem avaliação.
   - Se houver classes com apenas 1 amostra:
        * Usa APENAS as classes com >= 2 amostras para gerar um conjunto de
          treino/teste estratificado e calcular métricas (relatório leve).
        * Depois treina o modelo FINAL usando TODAS as amostras (incluindo
          as classes de 1 amostra).
   - Caso contrário (todas as classes com >= 2 amostras):
        * Faz train/test split estratificado com 80/20 em TODO o conjunto.
        * Avalia e depois treina o modelo FINAL usando TODO o dataset.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix

import joblib


# ---------------------------------------------------------------------
# Caminhos
# ---------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "model"

EMB_FILE = DATA_DIR / "xenocanto_embeddings.npy"
LABELS_FILE = DATA_DIR / "xenocanto_labels.csv"

KNN_MODEL_FILE = MODEL_DIR / "knn_birdnet.pkl"
LBL_ENCODER_FILE = MODEL_DIR / "label_encoder.pkl"


def etapa_3():
    print("\n=== ETAPA 3 — Treino do KNN com embeddings BirdNET ===")

    if not EMB_FILE.exists():
        raise FileNotFoundError(f"Arquivo de embeddings não encontrado: {EMB_FILE}")
    if not LABELS_FILE.exists():
        raise FileNotFoundError(f"Arquivo de labels não encontrado: {LABELS_FILE}")

    # Cria diretório de modelos
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------
    # 1. Carregar dados
    # ------------------------------------------------------
    X = np.load(EMB_FILE)                  # [N, D]
    labels_df = pd.read_csv(LABELS_FILE)   # col 'label'

    if "label" not in labels_df.columns:
        raise KeyError("CSV de labels precisa conter a coluna 'label'.")

    y_str = labels_df["label"].astype(str).values  # rótulos como texto

    n_amostras, n_features = X.shape
    print(f"Total de amostras:  {n_amostras}")
    print(f"Dimensão do vetor:  {n_features}")

    # ------------------------------------------------------
    # 2. Codificar labels (string -> inteiro)
    # ------------------------------------------------------
    le = LabelEncoder()
    y = le.fit_transform(y_str)

    n_classes = len(le.classes_)
    print(f"Total de classes (espécies): {n_classes}")

    # Distribuição de classes (texto)
    vc = pd.Series(y_str).value_counts().sort_values(ascending=True)
    min_count = vc.min()
    print("\nDistribuição de amostras por espécie (as menores classes primeiro):")
    print(vc.head(15))  # só mostra as 15 menores
    print(f"\nMenor número de amostras em uma classe: {min_count}")

    # ------------------------------------------------------
    # 3. Estratégia de treino vs. avaliação
    # ------------------------------------------------------
    if n_amostras < 10 or n_classes < 2:
        print("\n[AVISO] Poucas amostras ou poucas classes para uma validação decente.")
        print("        Vou treinar o KNN com TODAS as amostras e NÃO farei train/test split.")
        print("        Isso é útil só para teste de pipeline. Depois, tente aumentar N.\n")

        knn_final = KNeighborsClassifier(n_neighbors=1, metric="euclidean")
        knn_final.fit(X, y)

        print("Modelo KNN treinado com todas as amostras (sem avaliação).")

    else:
        if min_count < 2:
            print("\n[AVISO] Existem classes com APENAS 1 amostra.")
            print("        Não é possível fazer split estratificado com todas as classes.")
            print("        Estratégia:")
            print("          - Usar APENAS classes com >= 2 amostras para avaliação estratificada.")
            print("          - Depois treinar o modelo FINAL com TODAS as amostras.\n")

            # Filtrar apenas classes com >= 2 amostras para avaliação
            especies_multi = vc[vc >= 2].index  # nomes (labels texto)
            mask_multi = np.isin(y_str, especies_multi)

            X_multi = X[mask_multi]
            y_multi_str = y_str[mask_multi]

            # Re-encode para avaliação
            le_eval = LabelEncoder()
            y_multi = le_eval.fit_transform(y_multi_str)

            n_multi = X_multi.shape[0]
            n_classes_multi = len(le_eval.classes_)
            print(f"Total de amostras para avaliação (classes >=2): {n_multi}")
            print(f"Total de classes usadas na avaliação:         {n_classes_multi}")

            if n_multi < 10 or n_classes_multi < 2:
                print("\n[AVISO] Mesmo restringindo a classes com >= 2 amostras,")
                print("        ainda há poucos dados para uma avaliação decente.")
                print("        Vou pular a avaliação e treinar apenas o modelo FINAL com TODO o dataset.\n")

            else:
                # Split estratificado no subconjunto "multi"
                X_train, X_test, y_train, y_test = train_test_split(
                    X_multi,
                    y_multi,
                    test_size=0.2,
                    random_state=42,
                    stratify=y_multi,
                )

                k = 5 if n_multi >= 50 else 3
                print(f"\nTreinando KNN (AVALIAÇÃO) com k={k} vizinhos no subconjunto multi-classe...")
                knn_eval = KNeighborsClassifier(n_neighbors=k, metric="euclidean")
                knn_eval.fit(X_train, y_train)

                print("\nAvaliando no conjunto de teste (apenas classes >=2 amostras)...")

                # ⬇⬇⬇ CORREÇÃO: alinhar labels numéricos e nomes das classes ⬇⬇⬇
                labels_indices = np.arange(len(le_eval.classes_))

                print("\n=== Classification Report (subconjunto multi-classe) ===")
                print(
                    classification_report(
                        y_test,
                        y_pred=knn_eval.predict(X_test),
                        labels=labels_indices,
                        target_names=le_eval.classes_,
                        zero_division=0,
                    )
                )

                print("\n=== Matriz de Confusão (subconjunto multi-classe) ===")
                print(
                    confusion_matrix(
                        y_test,
                        knn_eval.predict(X_test),
                        labels=labels_indices,
                    )
                )

            # Independente da avaliação, agora treinamos o modelo FINAL com TODAS as amostras
            k_final = 5 if n_amostras >= 50 else 3
            print(f"\nTreinando modelo FINAL de KNN com TODO o dataset (k={k_final})...")
            knn_final = KNeighborsClassifier(n_neighbors=k_final, metric="euclidean")
            knn_final.fit(X, y)

        else:
            # Caso “bonito”: todas as classes têm >= 2 amostras
            print("\nDividindo em treino/validação (80/20 estratificado) em TODO o dataset...")

            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=0.2,
                random_state=42,
                stratify=y,
            )

            k = 5 if n_amostras >= 50 else 3
            print(f"Treinando KNN com k={k} vizinhos (treino/validação)...")

            knn_eval = KNeighborsClassifier(n_neighbors=k, metric="euclidean")
            knn_eval.fit(X_train, y_train)

            print("\nAvaliando no conjunto de teste...")

            labels_indices = np.arange(len(le.classes_))

            print("\n=== Classification Report ===")
            print(
                classification_report(
                    y_test,
                    knn_eval.predict(X_test),
                    labels=labels_indices,
                    target_names=le.classes_,
                    zero_division=0,
                )
            )

            print("\n=== Matriz de Confusão ===")
            print(
                confusion_matrix(
                    y_test,
                    knn_eval.predict(X_test),
                    labels=labels_indices,
                )
            )

            # Modelo FINAL com TODO o dataset
            k_final = k
            print(f"\nTreinando modelo FINAL de KNN com TODO o dataset (k={k_final})...")
            knn_final = KNeighborsClassifier(n_neighbors=k_final, metric="euclidean")
            knn_final.fit(X, y)

    # ------------------------------------------------------
    # 4. Salvar modelo FINAL e LabelEncoder (para TODO o dataset)
    # ------------------------------------------------------
    joblib.dump(knn_final, KNN_MODEL_FILE)
    joblib.dump(le, LBL_ENCODER_FILE)

    print("\n✔ Modelo KNN FINAL salvo em:", KNN_MODEL_FILE)
    print("✔ LabelEncoder salvo em:", LBL_ENCODER_FILE)
    print("\nETAPA 3 concluída.")


if __name__ == "__main__":
    try:
        etapa_3()
    except Exception as e:
        print("\n[ERRO] Durante a ETAPA 3:")
        print(e)
        sys.exit(1)
