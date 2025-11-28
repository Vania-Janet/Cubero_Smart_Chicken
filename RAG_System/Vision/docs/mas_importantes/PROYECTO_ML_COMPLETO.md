# Smart Budget Kitchen - Sistema ML de Recomendación de Recetas

**Proyecto de Machine Learning Multimodal**
- **Recomendación de Recetas**: Híbrido (Content-based + Collaborative Filtering)
- **Visión Computacional**: Image Retrieval + ML Scoring (CLIP + FAISS + XGBoost)
- **Pipeline Optimizado**: ROC-AUC 0.84 en predicción de ingredientes

---

## Estado Actual del Proyecto

### Completado

#### Módulo 1: Sistema de Recomendación
- [x] Preprocesamiento de Food.com dataset
- [x] Content-based recommender (TF-IDF)
- [x] Collaborative filtering (SVD)
- [x] Sistema híbrido con ponderación
- [x] Scripts de entrenamiento y evaluación

#### Módulo 2: Visión Computacional (Image Retrieval + Scoring)
- [x] Pipeline CLIP + FAISS para retrieval
- [x] K adaptativo con elbow detection
- [x] Scoring model (XGBoost) con 9 features
- [x] Hybrid oversampling para desbalanceo de clases
- [x] **ROC-AUC: 0.84** (mejora +52.9% vs baseline 0.55)
- [x] Sistema de inferencia end-to-end

#### Experimentación y Optimización
- [x] Experimentos de balanceo de clases
- [x] Pruebas de sampling (undersample, hybrid, SMOTE)
- [x] Pruebas de data augmentation
- [x] Optimización de parámetros K adaptativo
- [x] Implementación de configuración ganadora

---

## Resultados Destacados

### Optimización de Predicción de Ingredientes

| Configuración | ROC-AUC | Mejora |
|---------------|---------|--------|
| Baseline (K=3-20, sin balanceo) | 0.5500 | - |
| K optimizado + undersample 70/30 | 0.7823 | +42.2% |
| **K optimizado + hybrid oversample 1.5x** | **0.8410** | **+52.9%** |

**Configuración Ganadora**:
- min_k=10, max_k=30, threshold=0.60
- Hybrid oversampling (1.5x positivos + undersample negativos)
- Elastic Net regularization (reg_lambda=1.0, reg_alpha=0.1)
- 9 features engineering
- Average Precision: 0.6369

---

## Arquitectura del Sistema

### Pipeline de Visión (Image Retrieval + ML Scoring)

```
Imagen Query
    ↓
[CLIP ViT-B-32] → Embedding (512-dim)
    ↓
[FAISS Index] → Top-K Imágenes Similares (K adaptativo: 10-30)
    ↓
[Feature Engineering] → 9 features por ingrediente candidato
    ↓
[XGBoost Classifier] → Probabilidad por ingrediente
    ↓
Ingredientes Predichos (threshold=0.5)
```

### Sistema de Recomendación

```
Usuario + Preferencias
    ↓
[Content-based (TF-IDF)] → Similitud por ingredientes/tags
    ↓
[Collaborative (SVD)] → Factorización de matrices
    ↓
[Híbrido] → α·content + β·collaborative + γ·popularity
    ↓
Top-N Recetas Recomendadas
```

---

## Estructura del Proyecto

```
Proyecto ML plus/
├── configs/
│   ├── inference_config.yaml       # Configuración de visión (K adaptativo, CLIP)
│   └── recommender_config.yaml     # Configuración de recomendador
│
├── data/
│   ├── raw/mm_food_100k/           # 90 GB - Dataset de imágenes (57,056 imágenes)
│   ├── processed/                  # 346 MB
│   │   ├── mm_food_metadata.csv    # Metadata con ingredientes parseados
│   │   └── scoring_training_data.csv  # Training data para XGBoost (96k samples)
│   ├── embeddings/                 # 224 MB
│   │   ├── clip_embeddings.npy     # Embeddings CLIP (57,056 x 512) - 220 MB
│   │   ├── faiss_index.bin         # Índice FAISS - 4 MB
│   │   └── image_ids.npy           # IDs de imágenes
│   └── splits/
│       ├── train_metadata.csv      # 80% train
│       ├── val_metadata.csv        # 10% val
│       └── test_metadata.csv       # 10% test
│
├── deprecated/                     # Código antiguo de clasificación
│   ├── README.md
│   ├── src/vision/                 # Modelos CNN deprecated (MobileNet, EfficientNet)
│   ├── scripts/                    # Scripts de entrenamiento deprecated
│   └── docs/
│
├── docs/
│   ├── ARCHITECTURE.md             # Arquitectura técnica detallada
│   └── QUICKSTART.md               # Guía rápida de uso
│
├── experimentos_proceso/           # Experimentos de optimización
│   ├── temp_balanced_experiments.py
│   ├── temp_advanced_sampling.py
│   └── temp_experiments/
│       ├── balanced_experiments_results.json
│       └── advanced_sampling_results.json
│
├── models/
│   ├── ingredient_scoring/
│   │   ├── xgboost_model.json      # Modelo XGBoost (ROC-AUC: 0.84)
│   │   ├── training_metrics.json
│   │   └── evaluation_results.json
│   └── recommender/
│       ├── content_model.pkl
│       ├── collaborative_model.pkl
│       └── hybrid_model.pkl
│
├── scripts/
│   ├── prepare_metadata.py         # Parsea ingredientes de JSON
│   ├── create_splits.py            # Splits estratificados 80/10/10
│   ├── generate_embeddings.py      # Genera embeddings CLIP
│   ├── build_faiss_index.py        # Construye índice FAISS
│   ├── prepare_scoring_training_data.py  # Training data con 9 features
│   ├── train_scoring_model.py      # Entrena XGBoost con hybrid sampling
│   ├── test_inference_system.py    # Tests unitarios del pipeline
│   ├── evaluate_system.py          # Evaluación completa
│   └── train_recommender.py        # Entrena sistema de recomendación
│
├── src/
│   ├── vision/
│   │   ├── retrieval.py            # ImageRetriever + FeatureEngineer
│   │   └── inference.py            # IngredientPredictor (pipeline completo)
│   ├── recommender/
│   │   ├── content_based.py
│   │   ├── collaborative.py
│   │   └── hybrid.py
│   └── preprocessing/
│       └── foodcom_processor.py
│
├── docs/
│   ├── FASE2_IMPLEMENTACION.md      # Implementación optimizada
│   ├── ORGANIZACION_PROYECTO.md     # Mapa del proyecto
│   └── RESUMEN_TRABAJO_COMPLETADO.md # Resumen ejecutivo
├── PROYECTO_ML_COMPLETO.md          # Documento consolidado (este archivo)
├── README.md                        # Documentación principal
├── requirements.txt
└── setup.py
```

---

## Features de Scoring Model (9 features)

### Features Originales (6)
1. **frequency**: Frecuencia normalizada del ingrediente en top-K
2. **avg_similarity**: Similitud promedio donde aparece
3. **top1_similarity**: Similitud del match más cercano
4. **avg_position**: Posición promedio normalizada
5. **max_similarity**: Similitud máxima donde aparece
6. **presence_ratio**: Proporción de matches donde aparece

### Features Nuevas (3) - Agregadas en optimización
7. **std_similarity**: Desviación estándar de similitudes (consistencia)
8. **global_frequency**: Frecuencia global en todo el dataset
9. **neighbor_diversity**: Diversidad de vecinos con el ingrediente

---

## Cómo Usar el Sistema

### Pipeline Completo de Visión (6 pasos)

#### 1. Preparar Metadata
```bash
python scripts/prepare_metadata.py \
    --input data/raw/mm_food_100k/metadata.json \
    --output data/processed/mm_food_metadata.csv
```

#### 2. Crear Splits Estratificados
```bash
python scripts/create_splits.py \
    --metadata data/processed/mm_food_metadata.csv \
    --output_dir data/splits
```

#### 3. Generar Embeddings CLIP
```bash
python scripts/generate_embeddings.py \
    --metadata data/processed/mm_food_metadata.csv \
    --image_dir data/raw/mm_food_100k/images \
    --output_dir data/embeddings \
    --model ViT-B-32 \
    --batch_size 64 \
    --device cuda
```

#### 4. Construir Índice FAISS
```bash
python scripts/build_faiss_index.py \
    --embeddings data/embeddings/clip_embeddings.npy \
    --output data/embeddings/faiss_index.bin
```

#### 5. Generar Training Data
```bash
python scripts/prepare_scoring_training_data.py \
    --metadata data/processed/mm_food_metadata.csv \
    --image_dir data/raw/mm_food_100k/images \
    --faiss_index data/embeddings/faiss_index.bin \
    --embeddings data/embeddings/clip_embeddings.npy \
    --output data/processed/scoring_training_data.csv \
    --min_k 10 \
    --max_k 30 \
    --threshold 0.6
```

#### 6. Entrenar Modelo (con Configuración Ganadora)
```bash
python scripts/train_scoring_model.py \
    --training_data data/processed/scoring_training_data.csv \
    --output_dir models/ingredient_scoring
```

**Output esperado**: ROC-AUC ~0.84, Average Precision ~0.64

### Inferencia

```python
from src.vision.inference import IngredientPredictor

# Cargar sistema
predictor = IngredientPredictor(config_path="configs/inference_config.yaml")

# Predecir ingredientes de una imagen
result = predictor.predict("path/to/image.jpg")

print("Ingredientes predichos:")
for ing in result['ingredients']:
    print(f"  - {ing['name']}: {ing['probability']:.2%}")
```

---

## Experimentación Realizada

### Experimentos de Balanceo de Clases

**Problema**: Dataset con 95% negativos, 5% positivos → modelo predice siempre "NO"

**Soluciones Probadas**:

| Método | ROC-AUC | Avg Precision | Samples | Observaciones |
|--------|---------|---------------|---------|---------------|
| Sin balanceo | 0.5500 | - | 96,029 | Baseline - casi aleatorio |
| Undersample 70/30 | 0.7792 | 0.5308 | 13,596 | Descarta mucha información |
| **Hybrid Oversample 1.5x** | **0.8410** | **0.6369** | **27,193** | **GANADOR** |
| Hybrid + Augmentation | 0.8058 | 0.5782 | 27,193 | Augmentation empeora |
| SMOTE | 0.7792 | 0.5308 | 13,596 | Igual que undersample |
| SMOTE + Augmentation | 0.7702 | 0.5179 | 13,596 | Peor resultado |

**Conclusión**: Hybrid oversampling (1.5x) sin data augmentation es la mejor estrategia.

### Optimización de Parámetros K

**K adaptativo**: Elbow detection en curva de similitudes

| Configuración | min_k | max_k | threshold | ROC-AUC |
|---------------|-------|-------|-----------|---------|
| Baseline | 3 | 20 | 0.70 | 0.5500 |
| Intento 1 | 5 | 40 | 0.60 | 0.5390 |
| Intento 2 | 3 | 30 | 0.40 | 0.5264 |
| **GANADOR** | **10** | **30** | **0.60** | **0.7823** |

**Insight**: Aumentar min_k a 10 genera más candidatos → menos desbalance → mejor modelo

---

## Instalación

### Dependencias

```bash
pip install -r requirements.txt
```

**Principales dependencias**:
- PyTorch 2.x + torchvision
- open_clip_torch (CLIP implementation)
- faiss-cpu (o faiss-gpu)
- xgboost >= 2.0.0
- scikit-learn
- pandas, numpy
- matplotlib, seaborn
- PyYAML

### Configuración

1. Descargar dataset MM-Food-100k
2. Configurar rutas en `configs/inference_config.yaml`
3. Ejecutar pipeline completo (pasos 1-6)

---

## Tecnologías Utilizadas

### Machine Learning
- **CLIP (ViT-B-32)**: Image embeddings (open_clip_torch)
- **FAISS**: Búsqueda de vecinos cercanos eficiente
- **XGBoost**: Clasificación binaria con Elastic Net
- **SVD**: Factorización de matrices para collaborative filtering
- **TF-IDF**: Vectorización de ingredientes y tags

### Deep Learning
- **PyTorch**: Framework principal
- **OpenCLIP**: Modelos CLIP pre-entrenados
- **torchvision**: Preprocesamiento de imágenes

### Data Processing
- **Pandas**: Manipulación de datos
- **NumPy**: Operaciones numéricas
- **scikit-learn**: Métricas y utilidades ML

---

## Próximos Pasos

### Corto Plazo
- [ ] Integrar IngredientPredictor con sistema de recomendación
- [ ] Evaluar en test set completo (actualmente solo val)
- [ ] Optimizar latencia de inferencia (batch processing)
- [ ] Implementar caching de embeddings frecuentes

### Mediano Plazo
- [ ] Feature engineering adicional (ratios, interacciones)
- [ ] Hyperparameter tuning (grid search)
- [ ] Ensemble de modelos (XGBoost + LightGBM + CatBoost)
- [ ] Threshold optimization según precision/recall objetivo

### Largo Plazo
- [ ] Fine-tuning de CLIP en MM-Food-100k
- [ ] Multi-task learning (ingredientes + platillo + nutrición)
- [ ] Deployment a producción (API + UI)
- [ ] A/B testing en usuarios reales

---

## Documentación Adicional

- [MODELO_XGBOOST.md](MODELO_XGBOOST.md): Documentación técnica completa del modelo XGBoost para Data Scientists/ML Engineers
- [docs/FASE2_IMPLEMENTACION.md](docs/FASE2_IMPLEMENTACION.md): Detalles de implementación final
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md): Arquitectura técnica completa
- [docs/QUICKSTART.md](docs/QUICKSTART.md): Guía rápida de inicio
- [docs/ORGANIZACION_PROYECTO.md](docs/ORGANIZACION_PROYECTO.md): Mapa completo del proyecto
- [deprecated/README.md](deprecated/README.md): Código antiguo y razones de migración
- [experimentos_proceso/](experimentos_proceso/): Scripts y resultados de experimentos

---

## Métricas del Sistema

### Módulo de Visión (Configuración Actual)
- **ROC-AUC**: 0.8410
- **Average Precision**: 0.6369
- **Dataset**: 57,056 imágenes, 1,431 ingredientes únicos
- **Training samples**: 96,029 (antes de sampling)
- **Balanced samples**: 27,193 (después de hybrid oversample)

### Parámetros Óptimos
```yaml
# K adaptativo
min_k: 10
max_k: 30
similarity_threshold: 0.60

# Modelo XGBoost
n_estimators: 200
max_depth: 5
learning_rate: 0.1
reg_lambda: 1.0      # L2
reg_alpha: 0.1       # L1 (Elastic Net)
subsample: 0.8
colsample_bytree: 0.8

# Sampling
oversample_factor: 1.5
target_positive_ratio: 0.3  # 70/30 neg/pos
```

---

## Contribuciones

Este proyecto fue desarrollado como parte de un trabajo de Machine Learning aplicado a sistemas de recomendación de recetas.

**Migración a Image Retrieval**: Se migró de un sistema de clasificación directa (CNNs) a un enfoque de Image Retrieval + ML Scoring, logrando:
- Mayor robustez ante variaciones visuales
- Flexibilidad (no requiere reentrenamiento para nuevos ingredientes)
- Interpretabilidad (matches que justifican predicciones)
- Mejora de +52.9% en ROC-AUC

**Optimización de Desbalanceo**: Se experimentó con múltiples técnicas de sampling, identificando hybrid oversampling como la mejor estrategia.

---

## Licencia

Este proyecto es de uso académico.

---

## Referencias

- **Dataset**: MM-Food-100K
- **CLIP**: Learning Transferable Visual Models From Natural Language Supervision (Radford et al., 2021)
- **FAISS**: Billion-scale similarity search with GPUs (Johnson et al., 2019)
- **XGBoost**: A Scalable Tree Boosting System (Chen & Guestrin, 2016)

---

**Última actualización**: Noviembre 2025
**Versión del pipeline**: 2.0 (Image Retrieval + ML Scoring)
**ROC-AUC actual**: 0.8410
