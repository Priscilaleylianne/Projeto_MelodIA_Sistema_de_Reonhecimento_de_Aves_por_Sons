#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ETAPA 5 — Avaliação e Ajuste Fino do KNN

Objetivo:
    - Carregar embeddings BirdNET (ETAPA 2) e labels
    - Avaliar o KNN em um cenário com dados suficientes
    - Gerar relatório de métricas e matriz de confusão
    - Opcionalmente fazer busca de melhor k (Grid Search simples)

Observação:
    - Com poucos dados (como N=2), o script apenas informa a limitação
      e imprime estatísticas básicas, sem fazer avaliação séria.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
)

import joblib


# ---------------------------------------------------------------------
# Caminhos
# ---------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "model"
REPORT_DIR = DATA_DIR / "reports"

EMB_FILE = DATA_DIR / "xenocanto_embeddings.npy"
LABELS_FILE = DATA_DIR / "xenocanto_labels.csv"

KNN_MODEL_FILE = MODEL_DIR / "knn_birdnet.pkl"
LBL_ENCODER_FILE = MODEL_DIR / "label_encoder.pkl"

REPORT_TXT = REPORT_DIR / "knn_birdnet_relatorio.txt"


def etapa_5():
    print("\n=== ETAPA 5 — Avaliação do KNN com embeddings BirdNET ===")

    # -----------------------------------------------------------------
    # 1. Verificações básicas de arquivos
    # -----------------------------------------------------------------
    if not EMB_FILE.exists():
        raise FileNotFoundError(f"Arquivo de embeddings não encontrado: {EMB_FILE}")
    if not LABELS_FILE.exists():
        raise FileNotFoundError(f"Arquivo de labels não encontrado: {LABELS_FILE}")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------
    # 2. Carregar dados
    # -----------------------------------------------------------------
    X = np.load(EMB_FILE)                  # [N, D]
    labels_df = pd.read_csv(LABELS_FILE)   # deve ter coluna 'label'

    if "label" not in labels_df.columns:
        raise KeyError("CSV de labels precisa conter a coluna 'label'.")

    y_str = labels_df["label"].astype(str).values

    n_samples, n_features = X.shape
    print(f"Total de amostras:  {n_samples}")
    print(f"Dimensão dos vetores: {n_features}")

    # Distribuição de classes
    class_counts = pd.Series(y_str).value_counts()
    print("\nDistribuição de classes (labels de texto):")
    print(class_counts)

    # -----------------------------------------------------------------
    # 3. Verificar se há dados suficientes para avaliação
    # -----------------------------------------------------------------
    if n_samples < 20 or class_counts.min() < 2:
        print("\n[AVISO] Poucos dados ou classes com menos de 2 amostras.")
        print("        A avaliação quantitativa (train/test ou cross-val) não será confiável.")
        print("        ETAPA 5 entra em modo 'relatório leve' apenas.\n")

        # Mesmo assim, salvamos um mini-relatório básico
        with open(REPORT_TXT, "w", encoding="utf-8") as f:
            f.write("=== RELATÓRIO KNN BIRDNET (ETAPA 5 - MODO LIMITADO) ===\n\n")
            f.write(f"Total de amostras:  {n_samples}\n")
            f.write(f"Dimensão dos vetores: {n_features}\n\n")
            f.write("Distribuição de classes:\n")
            f.write(class_counts.to_string())
            f.write("\n\nObservação:\n")
            f.write(
                "Poucos dados ou classes muito desbalanceadas.\n"
                "Para uma avaliação confiável, gere embeddings para mais gravações.\n"
            )

        print(f"✔ Relatório leve salvo em: {REPORT_TXT}")
        print("ETAPA 5 concluída (modo limitado).")
        return

    # -----------------------------------------------------------------
    # 4. Codificar labels e dividir treino/teste
    # -----------------------------------------------------------------
    print("\nDados suficientes. Prosseguindo com avaliação completa.")

    le = LabelEncoder()
    y = le.fit_transform(y_str)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    print(f"Tamanho treino: {X_train.shape[0]}")
    print(f"Tamanho teste:  {X_test.shape[0]}")

    # -----------------------------------------------------------------
    # 5. Busca simples do melhor k
    # -----------------------------------------------------------------
    param_grid = {
        "n_neighbors": [1, 3, 5, 7, 9],
        "metric": ["euclidean"],
    }

    print("\nExecutando GridSearchCV para escolher melhor k...")
    base_knn = KNeighborsClassifier()

    grid = GridSearchCV(
        base_knn,
        param_grid,
        scoring="accuracy",
        cv=3,
        n_jobs=-1,
        verbose=1,
    )
    grid.fit(X_train, y_train)

    print(f"\nMelhor combinação de hiperparâmetros: {grid.best_params_}")
    print(f"Melhor acurácia em CV (treino): {grid.best_score_:.4f}")

    best_knn = grid.best_estimator_

    # -----------------------------------------------------------------
    # 6. Avaliar no conjunto de teste
    # -----------------------------------------------------------------
    print("\nAvaliando o melhor KNN no conjunto de teste...")
    y_pred = best_knn.predict(X_test)

    acc_test = accuracy_score(y_test, y_pred)
    print(f"Acurácia no teste: {acc_test:.4f}")

    report = classification_report(
        y_test,
        y_pred,
        target_names=le.classes_,
        digits=4,
    )
    cm = confusion_matrix(y_test, y_pred)

    print("\n=== Classification Report ===")
    print(report)

    print("\n=== Matriz de Confusão ===")
    print(cm)

    # -----------------------------------------------------------------
    # 7. Salvar relatório e modelo refinado
    # -----------------------------------------------------------------
    # Atualizar arquivos de modelo com o best_knn e o LabelEncoder
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_knn, KNN_MODEL_FILE)
    joblib.dump(le, LBL_ENCODER_FILE)

    print("\n✔ Modelo KNN atualizado salvo em:", KNN_MODEL_FILE)
    print("✔ LabelEncoder salvo em:", LBL_ENCODER_FILE)

    # Salvar relatório em texto
    with open(REPORT_TXT, "w", encoding="utf-8") as f:
        f.write("=== RELATÓRIO KNN BIRDNET (ETAPA 5) ===\n\n")
        f.write(f"Total de amostras:  {n_samples}\n")
        f.write(f"Dimensão dos vetores: {n_features}\n\n")
        f.write("Distribuição de classes (texto):\n")
        f.write(class_counts.to_string())
        f.write("\n\nMelhor combinação (GridSearchCV):\n")
        f.write(str(grid.best_params_) + "\n")
        f.write(f"Melhor acurácia em CV (treino): {grid.best_score_:.4f}\n\n")
        f.write(f"Acurácia no teste: {acc_test:.4f}\n\n")
        f.write("=== Classification Report ===\n")
        f.write(report)
        f.write("\n=== Matriz de Confusão ===\n")
        f.write(np.array2string(cm))
        f.write("\n")

    print("\n✔ Relatório detalhado salvo em:", REPORT_TXT)
    print("ETAPA 5 concluída com sucesso.")


if __name__ == "__main__":
    try:
        etapa_5()
    except Exception as e:
        print("\n[ERRO] Durante a ETAPA 5:")
        print(e)
        sys.exit(1)
