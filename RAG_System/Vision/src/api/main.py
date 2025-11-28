"""
FastAPI Backend para Detección de Ingredientes
API REST para predicción de ingredientes desde imágenes usando CLIP + FAISS + XGBoost
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from pathlib import Path
import tempfile
import os
import time
import traceback

from src.vision.inference import IngredientPredictor

app = FastAPI(
    title="Ingredient Detection API",
    description="API para detección de ingredientes desde imágenes usando Image Retrieval + ML Scoring",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ingredient_predictor: Optional[IngredientPredictor] = None

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_info: Dict[str, Any]

class IngredientPredictionResponse(BaseModel):
    success: bool
    ingredients: List[str]
    num_detected: int
    probabilities: Dict[str, float]
    metadata: Dict[str, Any]
    processing_time_ms: float

@app.on_event("startup")
async def startup_event():
    global ingredient_predictor

    print("=" * 70)
    print("   INGREDIENT DETECTION API - STARTING")
    print("=" * 70)

    try:
        print("\n[1/1] Cargando IngredientPredictor...")
        config_path = "configs/inference_config.yaml"
        ingredient_predictor = IngredientPredictor(config_path)
        print("✓ IngredientPredictor cargado exitosamente")
        print(f"  - Min K: {ingredient_predictor.min_k}")
        print(f"  - Max K: {ingredient_predictor.max_k}")
        print(f"  - Similarity threshold: {ingredient_predictor.similarity_threshold}")
        print(f"  - Prediction threshold: {ingredient_predictor.prediction_threshold}")
    except Exception as e:
        print(f"✗ ERROR cargando modelo: {e}")
        print(traceback.format_exc())
        ingredient_predictor = None

    print("\n" + "=" * 70)
    if ingredient_predictor:
        print("   API LISTA Y FUNCIONANDO")
        print("   URL: http://localhost:8000")
        print("   Documentación: http://localhost:8000/docs")
    else:
        print("   ADVERTENCIA: API iniciada PERO el modelo NO se cargó")
        print("   Verifica los paths en configs/inference_config.yaml")
    print("=" * 70 + "\n")

@app.get("/")
async def root():
    return {
        "service": "Ingredient Detection API",
        "version": "2.0.0",
        "status": "running",
        "model_loaded": ingredient_predictor is not None,
        "endpoints": {
            "health": "GET /health",
            "predict": "POST /predict"
        },
        "docs": "/docs"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    if ingredient_predictor is None:
        raise HTTPException(
            status_code=503,
            detail="Modelo no cargado. Verifica logs de inicio."
        )

    return HealthResponse(
        status="healthy",
        model_loaded=True,
        model_info={
            "min_k": ingredient_predictor.min_k,
            "max_k": ingredient_predictor.max_k,
            "similarity_threshold": ingredient_predictor.similarity_threshold,
            "prediction_threshold": ingredient_predictor.prediction_threshold,
            "num_global_ingredients": len(ingredient_predictor.global_frequencies)
        }
    )

@app.post("/predict", response_model=IngredientPredictionResponse)
async def predict_ingredients(
    file: UploadFile = File(..., description="Imagen de comida (JPG, PNG, etc.)"),
    threshold: float = Form(0.5, ge=0.0, le=1.0, description="Umbral de probabilidad (0.0-1.0)")
):
    """
    Predice ingredientes desde una imagen de comida

    **Pipeline:**
    1. Imagen → CLIP embedding (512 dims)
    2. FAISS retrieval → Top-K imágenes similares
    3. K adaptativo (10-30)
    4. Feature engineering (9 features)
    5. XGBoost scoring → Probabilidades
    6. Threshold → Ingredientes finales

    **Parámetros:**
    - file: Imagen de comida
    - threshold: Umbral de probabilidad (default: 0.5)

    **Retorna:**
    - ingredients: Lista de ingredientes detectados
    - probabilities: Probabilidad por ingrediente
    - metadata: Info del proceso (K usado, similitud top-1, etc.)
    """
    start_time = time.time()

    if ingredient_predictor is None:
        raise HTTPException(
            status_code=503,
            detail="Modelo no cargado. Verifica que el servidor haya iniciado correctamente."
        )

    # Validar que sea imagen
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail=f"Archivo debe ser una imagen. Tipo recibido: {file.content_type}"
        )

    # Validar tamaño (máx 10MB)
    MAX_SIZE = 10 * 1024 * 1024  # 10MB
    content = await file.read()
    if len(content) > MAX_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"Imagen muy grande. Máximo: 10MB, recibido: {len(content) / 1024 / 1024:.2f}MB"
        )

    tmp_path = None
    try:
        # Guardar temporalmente
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            tmp.write(content)
            tmp_path = tmp.name

        # Predicción
        result = ingredient_predictor.predict(
            image_path=tmp_path,
            threshold=threshold,
            return_probabilities=True
        )

        # Limpiar archivo temporal
        os.unlink(tmp_path)

        processing_time = (time.time() - start_time) * 1000  # ms

        return IngredientPredictionResponse(
            success=True,
            ingredients=result['ingredients'],
            num_detected=len(result['ingredients']),
            probabilities=result.get('probabilities', {}),
            metadata=result.get('metadata', {}),
            processing_time_ms=round(processing_time, 2)
        )

    except Exception as e:
        # Cleanup en caso de error
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)

        raise HTTPException(
            status_code=500,
            detail=f"Error procesando imagen: {str(e)}"
        )
