#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path
from functools import lru_cache
from typing import List

import numpy as np
import joblib
from fastapi import FastAPI, Request, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from birdnetlib.analyzer import Analyzer
from birdnetlib import Recording

# ============================================================
#  CAMINHOS
# ============================================================
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"
TEMPLATE_DIR = BASE_DIR / "template"
STATIC_DIR = BASE_DIR / "static"
UPLOAD_DIR = STATIC_DIR / "uploads"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

KNN_MODEL_FILE = MODEL_DIR / "knn_birdnet.pkl"
LABEL_ENCODER_FILE = MODEL_DIR / "label_encoder.pkl"

# ============================================================
#  FASTAPI
# ============================================================
app = FastAPI(title="MelodIA (BirdNET + KNN)")

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory=str(TEMPLATE_DIR))


# ============================================================
#  CARGA DE MODELOS (CACHE)
# ============================================================
@lru_cache(maxsize=1)
def get_knn():
    if not KNN_MODEL_FILE.exists() or not LABEL_ENCODER_FILE.exists():
        raise RuntimeError(
            "Arquivos de modelo não encontrados. "
            f"Esperado em {KNN_MODEL_FILE} e {LABEL_ENCODER_FILE}"
        )
    knn = joblib.load(KNN_MODEL_FILE)
    le = joblib.load(LABEL_ENCODER_FILE)
    return knn, le


@lru_cache(maxsize=1)
def get_analyzer():
    return Analyzer()


# ============================================================
#  FUNÇÕES AUXILIARES
# ============================================================
def descobrir_chave_embedding(seg) -> tuple[str | None, np.ndarray | None]:
    """Tenta descobrir automaticamente qual chave contém o vetor de embedding."""
    import numpy as _np

    if not isinstance(seg, dict):
        return None, None

    # 1) tenta chaves com "embed" no nome
    for key, val in seg.items():
        lk = str(key).lower()
        if "embed" in lk:
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

    # 2) fallback: qualquer list/ndarray "grande"
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


def extract_embedding(file_path: str, analyzer: Analyzer) -> np.ndarray | None:
    """Extrai UM vetor de embedding médio a partir do áudio."""
    rec = Recording(analyzer, file_path, min_conf=0.0)
    rec.extract_embeddings()
    emb_obj = rec.embeddings

    # Caso 1: array direto
    if isinstance(emb_obj, np.ndarray):
        if emb_obj.size == 0:
            return None
        if emb_obj.ndim == 1:
            return emb_obj.astype(np.float32)
        return emb_obj.mean(axis=0).astype(np.float32)

    # Caso 2: lista de segmentos
    if isinstance(emb_obj, list) and len(emb_obj) > 0:
        vets: List[np.ndarray] = []
        for seg in emb_obj:
            # se não for dict, tenta converter direto
            if not isinstance(seg, dict):
                try:
                    v = np.array(seg, dtype=np.float32).ravel()
                    if v.size >= 8:
                        vets.append(v)
                    continue
                except Exception:
                    continue

            key, vec = descobrir_chave_embedding(seg)
            if key is None or vec is None:
                continue
            vets.append(vec)

        if not vets:
            return None

        mat = np.stack(vets, axis=0)
        return mat.mean(axis=0).astype(np.float32)

    return None


def predict_topk(embedding: np.ndarray, k_top: int = 5):
    """Retorna top-k rótulos e scores do KNN."""
    knn, le = get_knn()
    x = embedding.reshape(1, -1)

    if not hasattr(knn, "predict_proba"):
        raise RuntimeError("KNN não suporta predict_proba.")

    proba = knn.predict_proba(x)[0]
    top_idx = np.argsort(proba)[::-1][:k_top]
    top_labels = le.inverse_transform(top_idx)
    top_scores = proba[top_idx]
    return top_labels.tolist(), top_scores.tolist()


# ============================================================
#  ROTAS
# ============================================================
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("app.html", {"request": request})


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if file.content_type not in ("audio/mpeg", "audio/wav", "audio/x-wav", "audio/wave"):
        raise HTTPException(status_code=400, detail="Envie um arquivo MP3 ou WAV.")

    # Salva o arquivo em static/uploads
    suffix = Path(file.filename).suffix or ".mp3"
    dest = UPLOAD_DIR / f"audio_upload{suffix}"
    with dest.open("wb") as f:
        f.write(await file.read())

    analyzer = get_analyzer()
    emb = extract_embedding(str(dest), analyzer)

    if emb is None:
        return JSONResponse(
            {
                "ok": False,
                "message": "Não foi possível extrair embedding desse áudio. "
                "Tente outro trecho.",
            }
        )

    try:
        labels, scores = predict_topk(emb, k_top=5)
    except Exception as e:
        return JSONResponse({"ok": False, "message": f"Erro no KNN: {e}"})

    especie_main = labels[0]
    conf_main = float(scores[0])

    return JSONResponse(
        {
            "ok": True,
            "species_main": especie_main,
            "confidence_main": conf_main,
            "labels": labels,
            "scores": scores,
            "audio_url": f"/static/uploads/{dest.name}",
        }
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
