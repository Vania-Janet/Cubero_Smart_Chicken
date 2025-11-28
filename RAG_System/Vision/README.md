# Smart Budget Kitchen

Sistema inteligente de recomendación de recetas que integra análisis de ingredientes y visión computacional para sugerir platillos personalizados basados en disponibilidad y preferencias.

**ROC-AUC Actual: 0.84** | **Mejora: +52.9% vs baseline**

> **Última actualización**: Noviembre 2025
> **Pipeline optimizado**: Image Retrieval + ML Scoring con Hybrid Oversampling

## Estado Actual

| Métrica | Valor | Mejora |
|---------|-------|--------|
| ROC-AUC | 0.8410 | +52.9% vs baseline |
| Average Precision | 0.6369 | - |
| Dataset | 57,056 imágenes | 1,431 ingredientes |
| Pipeline | Image Retrieval + ML Scoring | Optimizado |

**Ver**: [docs/FASE2_IMPLEMENTACION.md](docs/FASE2_IMPLEMENTACION.md) para detalles técnicos

## Índice

- [Inicio Rápido](#inicio-rápido)
- [Resumen Ejecutivo](#resumen-ejecutivo)
- [Arquitectura del Sistema](#arquitectura-del-sistema)
- [Datasets Utilizados](#datasets-utilizados)
- [Modelos de Machine Learning](#modelos-de-machine-learning)
- [Estructura del Repositorio](#estructura-del-repositorio)
- [Instalación y Configuración](#instalación-y-configuración)
- [Uso del Sistema](#uso-del-sistema)
- [Evaluación y Métricas](#evaluación-y-métricas)
- [Documentación Adicional](#documentación-adicional)
- [Referencias](#referencias)

---

## Inicio Rápido

### Para Nuevos Desarrolladores

**¿Primera vez clonando este proyecto?** Sigue estos pasos:

#### 1. Requisitos del Sistema

**Espacio en Disco Requerido:**
- Dataset MM-Food-100k: 90 GB (57,056 imágenes)
- Embeddings CLIP: 224 MB
- Data procesada: 346 MB
- Models: 1 MB
- **TOTAL ESTIMADO: ~91 GB**

**Hardware Recomendado:**
- GPU: NVIDIA con 4GB+ VRAM (o CPU y mucha paciencia xd)
- RAM: 16GB+ (32GB ideal para training)
- CPU: 6+ cores

**IMPORTANTE**: La descarga del dataset puede tardar 18+ horas dependiendo de tu conexión y de los núcleos usados. Yo con 6 núcleos tardó 14 horas.

#### 2. Instalación Inicial

```bash
# 1. Clonar repositorio
git clone <repo-url>
cd "Proyecto ML plus"

# 2. Crear entorno virtual
python -m venv appComida
source appComida/bin/activate  # Linux/Mac
# o
appComida\Scripts\activate     # Windows

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Verificar instalación
python -c "import torch; import open_clip; import faiss; import xgboost; print('Todo instalado correctamente')"
```

#### 3. Descargar Dataset

**IMPORTANTE**: El dataset MM-Food-100k NO está incluido en el repositorio (90 GB, 57,056 imágenes). Debes descargarlo usando el script proporcionado.

```bash
# Descargar dataset completo desde Hugging Face
# ADVERTENCIA: 90 GB de descarga, puede tardar 12+ horas
python scripts/download_mm_food_images.py \
    --output_dir "data/raw/mm_food_100k"
```

Estructura final esperada:
```
data/raw/mm_food_100k/
├── images/           # 57,056 imágenes (90 GB)
└── metadata.json     # Metadata con ingredientes
```

**Nota**: El .gitignore está configurado para ignorar esta carpeta automáticamente por eso tienes que correr ese py para tener las imágenes.

#### 4. Ejecutar Pipeline Completo

```bash
# Pipeline de 6 pasos (3-7 horas en GPU, 12-29 horas en CPU -está segunda aproximación fue echa por un LLM)

# Paso 1: Preparar metadata 
python scripts/prepare_metadata.py \
    --input data/raw/mm_food_100k/metadata.json \
    --output data/processed/mm_food_metadata.csv

# Paso 2: Crear splits 
python scripts/create_splits.py \
    --metadata data/processed/mm_food_metadata.csv \
    --output_dir data/splits

# Paso 3: Generar embeddings CLIP 
python scripts/generate_embeddings.py \
    --metadata data/processed/mm_food_metadata.csv \
    --image_dir "data/raw/mm_food_100k/images" \
    --output_dir data/embeddings \
    --model ViT-B-32 \
    --batch_size 64 \
    --device cuda

# Paso 4: Construir índice FAISS 
python scripts/build_faiss_index.py \
    --embeddings data/embeddings/clip_embeddings.npy \
    --output data/embeddings/faiss_index.bin

# Paso 5: Generar training data 
python scripts/prepare_scoring_training_data.py \
    --metadata data/processed/mm_food_metadata.csv \
    --image_dir "data/raw/mm_food_100k/images" \
    --faiss_index data/embeddings/faiss_index.bin \
    --embeddings data/embeddings/clip_embeddings.npy \
    --output data/processed/scoring_training_data.csv

# Paso 6: Entrenar modelo XGBoost
python scripts/train_scoring_model.py \
    --training_data data/processed/scoring_training_data.csv \
    --output_dir models/ingredient_scoring
```

**Resultado esperado**: ROC-AUC ~0.84

#### 5. Probar el Sistema

```python
from src.vision.inference import IngredientPredictor

# Cargar sistema
predictor = IngredientPredictor(config_path="configs/inference_config.yaml")

# Predecir ingredientes
result = predictor.predict("data/raw/mm_food_100k/images/sample.jpg")

print("Ingredientes detectados:")
for ing in result['ingredients']:
    print(f"  - {ing['name']}: {ing['probability']:.2%}")
```

#### 6. Verificar Todo Funciona

```bash
# Tests unitarios 
python scripts/test_inference_system.py

# Evaluación en test set 
python scripts/evaluate_system.py
```

### Archivos Importantes

**Documentación esencial:**
- README.md (este archivo) - Documentación principal
- PROYECTO_ML_COMPLETO.md - Documento técnico consolidado
- docs/FASE2_IMPLEMENTACION.md - Detalles de optimización
- docs/QUICKSTART.md - Guía detallada paso a paso

**Configuración:**
- configs/inference_config.yaml - Parámetros del pipeline de visión
- requirements.txt - Dependencias Python

### Advertencias y Notas

**Espacio en disco:**
- El pipeline completo requiere ~100 GB
- La carpeta data/raw/mm_food_100k/ es la más pesada (90 GB, 57,056 imágenes)
- Embeddings: 224 MB
- Data procesada: 346 MB
- Asegúrate de tener suficiente espacio antes de empezar

**GPU vs CPU:**
- GPU altamente recomendado para paso 3 (embeddings) y para la descarga de imágenes
- El resto del pipeline funciona bien en CPU
- Si solo tienes CPU, considera usar un subset del dataset

**Carpetas ignoradas en Git (.gitignore):**
- appComida/ - Entorno virtual (NO versionar)
- data/raw/mm_food_100k/ - Dataset 90 GB (NO versionar - usar script de descarga)
- data/embeddings/ - Artefactos generados 224 MB (NO versionar)
- data/processed/ - Data procesada 346 MB (NO versionar)
- models/ - Modelos entrenados (NO versionar)

**NOTA IMPORTANTE**: El repositorio NO incluye los 90 GB del dataset. Los colaboradores deben ejecutar el script de descarga (paso 3) para obtener las imágenes.

---

## Resumen Ejecutivo

**Smart Budget Kitchen** es una solución integral de machine learning que ayuda a usuarios a descubrir recetas relevantes basándose en los ingredientes disponibles y fotografías de alimentos. El sistema combina técnicas avanzadas de sistemas de recomendación con visión computacional para ofrecer sugerencias personalizadas y prácticas.

### Problema a Resolver

- Los usuarios tienen ingredientes disponibles pero no saben qué cocinar
- Desperdicio de alimentos por falta de ideas de recetas
- Dificultad para identificar platillos o ingredientes desde imágenes
- Necesidad de recomendaciones que consideren popularidad y preferencias

### Solución Propuesta

Sistema multimodal que integra:

1. **Recomendador de Recetas**: Sugiere recetas basadas en ingredientes disponibles, considerando similitud de contenido, ratings de usuarios y popularidad
2. **Módulo de Visión**: Predice ingredientes desde fotografías usando Image Retrieval + ML Scoring (CLIP + FAISS + XGBoost)
3. **Integración Inteligente**: Fusiona información visual y textual para recomendaciones contextualizadas

### Características Principales

- Recomendación híbrida (content-based + collaborative filtering)
- Predicción de ingredientes por imagen con retrieval semántico y ML scoring
- K adaptativo inteligente para búsqueda de imágenes similares
- Sistema robusto ante variaciones visuales (cortes, cocciones, preparaciones)
- Sugerencias de ingredientes faltantes para compras
- Ranking por relevancia, popularidad y rating
- Sistema completamente local sin APIs externas
- Pipeline reproducible y escalable

## Arquitectura del Sistema

### Diagrama de Componentes

```
┌──────────────────────────────────────────────────────────────┐
│                    SMART BUDGET KITCHEN                       │
└──────────────────────────────────────────────────────────────┘
                              │
                ┌─────────────┴──────────────┐
                │                            │
        ┌───────▼────────┐          ┌───────▼─────────┐
        │  RECOMMENDER   │          │  VISION MODULE  │
        │    MODULE      │          │                 │
        │                │          │ CLIP + FAISS    │
        │ - Content-Based│          │ (Retrieval)     │
        │ - Collaborative│          │       ↓         │
        │ - Hybrid Fusion│          │ K Adaptativo    │
        │                │          │       ↓         │
        └───────┬────────┘          │ XGBoost Scoring │
                │                   │       ↓         │
                │                   │ Ingredientes    │
                │                   └────────┬────────┘
                │                            │
                └──────────────┬─────────────┘
                               │
                    ┌──────────▼───────────┐
                    │  INTEGRATION ENGINE  │
                    │                      │
                    │ - Score Fusion       │
                    │ - Ranking            │
                    │ - Shopping Hints     │
                    └──────────┬───────────┘
                               │
                    ┌──────────▼───────────┐
                    │   APPLICATION UI     │
                    │                      │
                    │ - Streamlit/React    │
                    │ - REST API           │
                    └──────────────────────┘
```

### Flujo de Datos

1. **Input de Usuario**: Texto (ingredientes) y/o imagen de comida
2. **Procesamiento Multimodal**:
   - Texto: vectorización TF-IDF, búsqueda de recetas similares
   - Imagen: CLIP embedding → FAISS retrieval → K adaptativo → Feature engineering → XGBoost scoring → Lista de ingredientes
3. **Fusión de Resultados**: Combinación ponderada de scores de recomendación
4. **Ranking y Presentación**: Top-K recetas ordenadas por relevancia

## Datasets Utilizados

### 1. Food.com Recipe and Interactions Dataset

Dataset completo de recetas y calificaciones de usuarios de Food.com.

**Ubicación**: `data/raw/foodcom/`

**Archivos**:
- `RAW_recipes.csv`: 231,637 recetas con ingredientes, pasos, información nutricional y tags
- `RAW_interactions.csv`: 1,132,367 interacciones usuario-receta con ratings (1-5 estrellas)
- `PP_recipes.csv`: 178,265 recetas preprocesadas
- `PP_users.csv`: 25,076 usuarios preprocesados

**Campos Clave de Recetas**:
- `id`: Identificador único de receta
- `name`: Nombre del platillo
- `ingredients`: Lista de ingredientes en formato JSON
- `tags`: Etiquetas descriptivas (ej: vegetarian, quick, mexican)
- `nutrition`: Array con [calorías, grasa%, azúcar%, sodio%, proteína%, grasa_saturada%, carbohidratos%]
- `n_steps`, `n_ingredients`: Métricas de complejidad
- `minutes`: Tiempo de preparación

**Campos Clave de Interacciones**:
- `user_id`, `recipe_id`: Identificadores
- `rating`: Calificación de 1 a 5 estrellas
- `review`: Texto de reseña
- `date`: Fecha de interacción

**Estadísticas Relevantes**:
- Rating promedio: 4.68 (sesgo positivo)
- Mediana de ingredientes por receta: 9
- Mediana de tiempo de preparación: 40 minutos
- 231,637 recetas únicas con al menos 1 rating

**Fuente**: [Kaggle - Food.com Recipes and Interactions](https://www.kaggle.com/datasets/shuyangli94/food-com-recipes-and-user-interactions)

**Consideraciones de Calidad**:
- Valores nulos mínimos (< 2.5%)
- Outliers extremos en tiempo de preparación requieren filtrado
- Distribución long-tail de ingredientes y tags
- Sesgo hacia ratings altos (78% son 5 estrellas)

### 2. MM-Food-100K Multimodal Dataset

Dataset multimodal de imágenes de alimentos con metadatos estructurados.

**Ubicación**: Descarga automática desde Hugging Face

**Tamaño**: 100,000 imágenes con anotaciones completas

**Campos Principales**:
- `image_url`: URL de la imagen del platillo
- `dish_name`: Nombre del plato (19,288 clases únicas)
- `food_type`: Categoría principal (5 clases: Homemade food, Restaurant food, Packaged food, Raw vegetables/fruits, Others)
- `ingredients`: Lista de ingredientes en formato JSON
- `nutritional_profile`: Calorías, proteínas, grasas, carbohidratos en formato JSON
- `cooking_method`: Método de cocción (2,264 variantes)
- `portion_size`: Tamaño de porción por ingrediente

**Estadísticas de Clases (food_type)**:
- Homemade food: 46.56%
- Restaurant food: 30.24%
- Packaged food: 17.65%
- Raw vegetables and fruits: 5.28%
- Others: 0.27%

**Características**:
- Imágenes accesibles vía URLs públicas
- Información nutricional completa (calorías, macronutrientes)
- Ingredientes estructurados
- Métodos de cocción diversos (frying, baking, boiling, etc.)

**Fuente**: [Hugging Face - Codatta/MM-Food-100K](https://huggingface.co/datasets/Codatta/MM-Food-100K)

**Uso en el Proyecto**:
- Base de conocimiento para Image Retrieval (57,056 imágenes indexadas)
- Embeddings CLIP para búsqueda semántica
- Ground truth de ingredientes para training del scoring model
- FAISS index para búsqueda eficiente de similitud

### 3. Datos Futuros: F-MAP Food Price Dataset

Dataset de series de tiempo de precios de alimentos (planificado para Fase 2).

**Uso Previsto**:
- Modelos de forecasting de precios
- Optimización de compras por temporada
- Sugerencias de recetas económicas
- Análisis de tendencias de mercado

## Modelos de Machine Learning

### Módulo 1: Sistema de Recomendación de Recetas

#### 1.1 Content-Based Filtering (Basado en Ingredientes)

**Descripción**: Modelo de similitud basado en representación vectorial de ingredientes y tags.

**Tipo de Modelo**: TF-IDF + Cosine Similarity

**Variables de Entrada**:
- Lista de ingredientes disponibles del usuario
- Tags opcionales (preferencias: vegetarian, quick, healthy)

**Variable Objetivo**: Score de similitud con cada receta del catálogo

**Pipeline**:
1. Preprocesamiento de ingredientes (normalización, lematización)
2. Construcción de vocabulario de ingredientes únicos
3. Vectorización con TF-IDF (matriz de recetas x ingredientes)
4. Cálculo de similitud coseno entre query del usuario y recetas
5. Filtrado por umbral de similitud y disponibilidad de ingredientes

**Hiperparámetros Principales**:
- `max_features`: 5000 términos en vocabulario
- `min_df`: 2 (mínimo 2 documentos para considerar término)
- `ngram_range`: (1, 2) para capturar combinaciones
- `similarity_threshold`: 0.3 para filtrar recetas poco relevantes

**Métricas de Evaluación**:
- Precision@K: % de recetas relevantes en top-K
- Recall@K: % de recetas relevantes recuperadas
- NDCG@K: Normalized Discounted Cumulative Gain
- Hit Rate@K: % de usuarios con al menos 1 recomendación relevante

**Justificación**: TF-IDF es eficiente, interpretable y captura bien la importancia relativa de ingredientes raros vs comunes. La similitud coseno es estándar para espacios de alta dimensionalidad.

#### 1.2 Collaborative Filtering (Basado en Usuarios)

**Descripción**: Factorización de la matriz usuario-receta para capturar patrones latentes de preferencias.

**Tipo de Modelo**: TruncatedSVD (sklearn.decomposition)

**Variables de Entrada**:
- Matriz dispersa de ratings: usuarios × recetas (formato CSR)
- Historial de ratings del usuario (si disponible)

**Variable Objetivo**: Rating predicho para pares usuario-receta no observados

**Pipeline**:
1. Construcción de matriz usuario-receta dispersa
2. Cálculo de biases (media global, bias de usuario, bias de ítem)
3. Factorización con TruncatedSVD: U (usuarios) y V (recetas) de rango k
4. Predicción: rating = global_mean + user_bias + item_bias + U × V^T
5. Clipping a rango [1, 5] y ranking por score predicho

**Hiperparámetros Principales**:
- `n_factors`: 100 componentes latentes
- `n_iter`: 20 iteraciones del algoritmo
- `random_state`: 42 (reproducibilidad)

**Métricas de Evaluación**:
- RMSE: Root Mean Squared Error en ratings predichos
- MAE: Mean Absolute Error
- Precision@K, Recall@K en conjunto de test

**Justificación**: TruncatedSVD captura interacciones latentes usuario-receta que no son evidentes con content-based. El enfoque con biases mejora la precisión de predicciones. No requiere compilación externa y es eficiente con matrices sparse.

#### 1.3 Sistema Híbrido

**Descripción**: Combinación ponderada de scores de content-based y collaborative filtering.

**Fórmula de Fusión**:
```
score_final = α × score_content + β × score_collaborative + γ × score_popularity

donde:
- score_content: similitud coseno normalizada
- score_collaborative: rating predicho normalizado
- score_popularity: log(num_ratings + 1) × rating_promedio normalizado
- α + β + γ = 1
```

**Hiperparámetros de Ponderación**:
- `α` (content): 0.5 (alta importancia a ingredientes disponibles)
- `β` (collaborative): 0.3 (patrones de usuarios)
- `γ` (popularity): 0.2 (recetas probadas y bien valoradas)

**Optimización**: Grid search sobre conjunto de validación maximizando NDCG@10

**Ventajas del Enfoque Híbrido**:
- Resuelve cold-start para nuevos usuarios (usa content-based)
- Captura preferencias personalizadas (collaborative)
- Balancea exploración-explotación (popularity bias)

### Módulo 2: Visión Computacional para Predicción de Ingredientes

#### 2.1 Image Retrieval con CLIP + FAISS

**Descripción**: Sistema de búsqueda semántica de imágenes similares usando embeddings visuales profundos.

**Modelo Base**: CLIP ViT-B/32 (Vision Transformer preentrenado)

**Pipeline de Retrieval**:
1. Generación de embeddings CLIP (512 dimensiones) para 57,056 imágenes de MM-Food-100K
2. Normalización L2 de vectores para similitud coseno
3. Construcción de índice FAISS IndexFlatIP para búsqueda eficiente
4. Query: imagen → embedding CLIP → búsqueda en FAISS → top-K imágenes similares

**Ventajas de CLIP**:
- Embeddings semánticos robustos entrenados en 400M pares imagen-texto
- Captura características visuales de alto nivel
- Zero-shot: no requiere fine-tuning específico para alimentos
- Robusto ante variaciones de iluminación, ángulo y presentación

**Complejidad**:
- Indexado: O(n) con n=57k imágenes (una sola vez)
- Búsqueda: O(d×n) con d=512 dimensiones (< 100ms por query)

#### 2.2 K Adaptativo Inteligente

**Descripción**: Selección dinámica del número de imágenes similares a considerar.

**Motivación**: Imágenes muy similares (k=3) vs ambiguas (k=20) requieren diferente cantidad de evidencia.

**Algoritmo**:
1. Filtrar por threshold de similitud (default: 0.70)
2. Detectar "codo" en curva de similitudes (gap máximo)
3. Aplicar límites min_k=3, max_k=20
4. Retornar K adaptado y top-K similitudes

**Pseudocódigo**:
```python
valid_sims = similarities[similarities > threshold]
gaps = valid_sims[:-1] - valid_sims[1:]
elbow_idx = argmax(gaps) + 1
k = max(min_k, min(elbow_idx, max_k))
```

**Hiperparámetros**:
- `similarity_threshold`: 0.70 (mínimo cosine similarity)
- `min_k`: 3 (evitar overfitting a pocas imágenes)
- `max_k`: 20 (limitar ruido de matches débiles)

**Justificación**: Enfoque heurístico eficiente que balancea evidencia y ruido. El elbow detection es técnica estándar para selección automática de parámetros.

#### 2.3 Feature Engineering para Scoring

**Descripción**: Extracción de features estadísticos por ingrediente candidato desde resultados de retrieval.

**Features Computados** (6 por ingrediente):

1. `frequency`: Número de matches donde aparece el ingrediente
2. `avg_similarity`: Similitud promedio de matches con el ingrediente
3. `top1_similarity`: Similitud del match más cercano
4. `avg_position`: Posición promedio en ranking de matches
5. `max_similarity`: Máxima similitud donde aparece
6. `presence_ratio`: Proporción de matches que contienen el ingrediente

**Intuición**:
- Alta frecuencia + alta similitud → ingrediente muy probable
- Aparición en top-1 match → fuerte señal
- Presencia en mayoría de matches → consenso robusto

**Output**: DataFrame con una fila por ingrediente candidato y 6 columnas de features

#### 2.4 Scoring Model con XGBoost

**Descripción**: Clasificador binario que predice probabilidad de que un ingrediente esté presente en la imagen query.

**Tipo de Modelo**: XGBoost Binary Classifier

**Variables de Entrada**: 6 features por ingrediente (ver sección 2.3)

**Variable Objetivo**: Label binario (1 si ingrediente en ground truth, 0 si no)

**Pipeline de Entrenamiento**:
1. Preparar training data: usar imágenes de train como queries
2. Para cada query: retrieval → candidates → features → labels
3. Entrenar XGBoost con scale_pos_weight para balancear clases
4. Validar con ROC-AUC y Average Precision
5. Guardar modelo en formato JSON

**Hiperparámetros**:
- `n_estimators`: 200 árboles
- `max_depth`: 5 (evitar overfitting)
- `learning_rate`: 0.1
- `scale_pos_weight`: ratio negatives/positives (balanceo de clases)
- `objective`: binary:logistic

**Métricas de Evaluación**:
- ROC-AUC: Área bajo curva ROC
- Average Precision: Resumen de curva precision-recall
- Precision@threshold, Recall@threshold
- Feature Importance: cuáles features son más predictivas

**Threshold de Predicción**: 0.5 default (optimizable en validación)

**Justificación**: XGBoost es eficiente con features tabulares, maneja bien desbalanceo de clases, y permite interpretabilidad via feature importance. La formulación binaria por ingrediente permite calibración independiente.

#### 2.5 Pipeline Completo de Inferencia

**Descripción**: Integración end-to-end de retrieval + scoring para predicción de ingredientes.

**Clase Principal**: `IngredientPredictor` en [src/vision/inference.py](src/vision/inference.py)

**Pipeline**:
1. Cargar imagen → convertir a RGB
2. Generar embedding CLIP
3. Búsqueda FAISS → top-50 candidatos
4. K adaptativo → ajustar a K óptimo
5. Extraer ingredientes únicos de top-K matches
6. Compute features para cada candidato
7. XGBoost scoring → probabilidades
8. Threshold → filtrar ingredientes finales
9. Ordenar por probabilidad descendente

**Output**:
```python
{
    'ingredients': ['tomato', 'onion', 'garlic', ...],  # Lista para RAG
    'probabilities': {'tomato': 0.92, 'onion': 0.87, ...},  # Saved para futuro
    'metadata': {
        'k_used': 12,
        'top1_similarity': 0.8456,
        'num_candidates': 45,
        'num_predicted': 8,
        'threshold_used': 0.5
    }
}
```

**Ventajas del Enfoque**:
- Robusto ante variaciones visuales (cortes, cocciones, preparaciones)
- No requiere reentrenamiento para nuevos ingredientes
- Interpretable: se pueden inspeccionar matches similares
- Escalable: agregar imágenes solo requiere re-indexar FAISS
- Híbrido: combina deep learning (CLIP) con ML tradicional (XGBoost)

### Módulo 3: Motor de Integración Multimodal

**Descripción**: Componente que fusiona resultados de visión y recomendación para generar sugerencias finales.

**Funcionamiento**:

1. **Modo Solo Texto**:
   - Aplicar recomendador híbrido directamente con ingredientes del usuario
   - Retornar top-K recetas rankeadas

2. **Modo Imagen + Texto**:
   - Ejecutar IngredientPredictor para extraer ingredientes de la imagen
   - Combinar ingredientes extraídos con ingredientes del usuario
   - Aplicar recomendador híbrido sobre ingredientes fusionados
   - Retornar top-K recetas con scores combinados

3. **Modo Solo Imagen**:
   - Usar solo ingredientes predichos por IngredientPredictor
   - Aplicar recomendador content-based sobre ingredientes detectados
   - Retornar top-K recetas relevantes

**Sugerencias de Compras**:
- Detectar ingredientes faltantes en recetas top-K
- Rankear por frecuencia e impacto en similitud
- Retornar lista de "ingredientes sugeridos para comprar"

## Estructura del Repositorio

```
smart-budget-kitchen/
│
├── data/
│   ├── raw/                          # Datos originales sin procesar
│   │   ├── foodcom/                  # Dataset Food.com
│   │   │   ├── RAW_recipes.csv
│   │   │   ├── RAW_interactions.csv
│   │   │   ├── PP_recipes.csv
│   │   │   └── PP_users.csv
│   │   └── mm_food_100k/             # Imágenes de MM-Food-100K
│   │       ├── images/               # 57,056 imágenes
│   │       └── metadata_labeled.csv  # Metadata con ingredientes
│   │
│   ├── processed/                    # Datos procesados listos para ML
│   │   ├── recipes_cleaned.parquet      # ~214k recetas limpias
│   │   ├── interactions_cleaned.parquet # ~595k interacciones filtradas
│   │   ├── mm_food_metadata.csv         # Metadata procesado con ingredientes parseados
│   │   ├── scoring_training_data.csv    # Training data para XGBoost
│   │   ├── tfidf_matrix.npz
│   │   └── user_item_matrix.npz
│   │
│   ├── embeddings/                   # CLIP embeddings y FAISS index
│   │   ├── clip_embeddings.npy       # [57056, 512] embeddings
│   │   ├── image_ids.npy             # IDs de imágenes
│   │   ├── faiss_index.bin           # Índice FAISS (~117 MB)
│   │   └── metadata.json             # Info de embeddings
│   │
│   └── splits/                       # Train/val/test splits estratificados
│       ├── train_metadata.csv
│       ├── val_metadata.csv
│       └── test_metadata.csv
│
├── models/                           # Modelos entrenados
│   ├── recommender/
│   │   ├── tfidf_vectorizer.pkl
│   │   ├── svd_model.pkl
│   │   └── hybrid_weights.json
│   └── ingredient_scoring/           # Scoring model para ingredientes
│       ├── xgboost_model.json        # Modelo XGBoost entrenado
│       ├── training_metrics.json     # Métricas de entrenamiento
│       ├── test_predictions.csv      # Predicciones en test set
│       └── evaluation_results.json   # Evaluación del sistema completo
│
├── notebooks/                        # Jupyter notebooks para análisis
│   ├── 01_foodcom_eda.ipynb
│   ├── 02_mm_food_100k_eda.ipynb
│   └── 03_recommender_experiments.ipynb
│
├── src/                              # Código fuente modular
│   ├── __init__.py
│   │
│   ├── preprocessing/                # Limpieza y preparación de datos
│   │   ├── __init__.py
│   │   ├── foodcom_processor.py
│   │   └── image_downloader.py
│   │
│   ├── recommender/                  # Sistema de recomendación
│   │   ├── __init__.py
│   │   ├── content_based.py
│   │   ├── collaborative.py
│   │   ├── hybrid.py
│   │   └── evaluator.py
│   │
│   ├── vision/                       # Sistema de visión (retrieval + scoring)
│   │   ├── __init__.py
│   │   ├── retrieval.py              # ImageRetriever (CLIP + FAISS) + FeatureEngineer
│   │   └── inference.py              # IngredientPredictor (pipeline completo)
│   │
│   ├── integration/                  # Motor de integración multimodal
│   │   ├── __init__.py
│   │   ├── multimodal_engine.py
│   │   ├── shopping_hints.py
│   │   └── ranker.py
│   │
│   ├── app/                          # Aplicación de usuario
│   │   ├── __init__.py
│   │   ├── streamlit_app.py
│   │   ├── api.py
│   │   └── utils.py
│   │
│   └── utils/
│       ├── __init__.py
│       ├── config.py
│       ├── logging_utils.py
│       └── metrics.py
│
├── scripts/                          # Pipeline completo de datos y entrenamiento
│   ├── prepare_metadata.py           # 1. Parsear ingredientes desde JSON
│   ├── create_splits.py              # 2. Crear splits estratificados
│   ├── generate_embeddings.py        # 3. Generar embeddings CLIP
│   ├── build_faiss_index.py          # 4. Construir índice FAISS
│   ├── prepare_scoring_training_data.py  # 5. Generar training data para XGBoost
│   ├── train_scoring_model.py        # 6. Entrenar scoring model
│   ├── test_inference_system.py      # 7. Tests de componentes
│   ├── evaluate_system.py            # 8. Evaluación completa
│   └── train_recommender.py          # Entrenar recomendador
│
├── deprecated/                       # Código obsoleto (clasificación directa)
│   ├── README.md                     # Explicación de deprecation
│   ├── src/vision/                   # Modelos de clasificación antiguos
│   ├── scripts/                      # Scripts de training antiguos
│   ├── models/vision/                # Checkpoints de clasificadores
│   ├── data/                         # Labels y metadata antiguos
│   └── docs/                         # Documentación obsoleta
│
├── configs/
│   ├── recommender_config.yaml
│   ├── inference_config.yaml         # Config del sistema de inferencia
│   └── vision_config.yaml            # Deprecated
│
├── requirements.txt
├── .gitignore
└── README.md
```

### Descripción de Módulos Principales

**data/**: Almacena datos en diferentes etapas. `raw/` contiene datos originales, `processed/` datos limpios, `embeddings/` CLIP embeddings y FAISS index, `splits/` divisiones train/val/test estratificadas.

**models/**: Modelos entrenados. `recommender/` contiene TF-IDF y SVD, `ingredient_scoring/` contiene XGBoost y métricas de evaluación.

**notebooks/**: Análisis exploratorio y experimentación interactiva.

**src/**: Código modular organizado por función. `vision/` contiene retrieval e inference, `recommender/` sistema de recomendación, `integration/` fusión multimodal.

**scripts/**: Pipeline completo numerado del 1 al 8, desde preparación de metadata hasta evaluación final.

**deprecated/**: Código obsoleto del enfoque de clasificación directa con CNNs. Ver [deprecated/README.md](deprecated/README.md) para detalles.

**configs/**: Configuraciones YAML. `inference_config.yaml` contiene paths y hiperparámetros del sistema de visión.

## Instalación y Configuración

### Requisitos del Sistema

- Python 3.9 o superior
- CUDA 11.8+ (opcional, para entrenamiento con GPU)
- 16GB RAM mínimo
- 50GB espacio en disco

### Instalación

#### Opción 1: Entorno Virtual con pip

```bash
# Clonar repositorio
git clone https://github.com/tu-usuario/smart-budget-kitchen.git
cd smart-budget-kitchen

# Crear entorno virtual
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate

# Instalar dependencias
pip install --upgrade pip
pip install -r requirements.txt

# Instalar el paquete en modo desarrollo
pip install -e .
```

#### Opción 2: Entorno Conda

```bash
# Crear entorno desde archivo
conda env create -f environment.yml
conda activate smart-budget-kitchen

# Instalar el paquete
pip install -e .
```

### Dependencias Principales

**Core ML y Data Science**:
- `numpy>=1.24.0`
- `pandas>=2.0.0`
- `scikit-learn>=1.3.0`
- `scipy>=1.11.0`

**Deep Learning**:
- `torch>=2.0.0`
- `torchvision>=0.15.0`
- `clip` (OpenAI CLIP para embeddings visuales)
- `xgboost>=2.0.0` (Scoring model para ingredientes)
- `transformers>=4.30.0` (sentence embeddings)

**Sistemas de Recomendación**:
- `scikit-learn>=1.0.0` (TruncatedSVD para collaborative filtering)
- `implicit>=0.6.0` (ALS y otras técnicas)

**Procesamiento de Imágenes y Retrieval**:
- `Pillow>=10.0.0`
- `faiss-cpu` o `faiss-gpu` (búsqueda de similitud eficiente)
- `opencv-python>=4.8.0`

**Datasets**:
- `datasets>=2.14.0` (Hugging Face datasets)

**Aplicación**:
- `streamlit>=1.25.0`
- `fastapi>=0.100.0`
- `uvicorn>=0.23.0`
- `pydantic>=2.0.0`

**Visualización**:
- `matplotlib>=3.7.0`
- `seaborn>=0.12.0`
- `plotly>=5.15.0`

**Utilidades**:
- `tqdm>=4.65.0`
- `python-dotenv>=1.0.0`
- `pyyaml>=6.0`
- `joblib>=1.3.0`

### Configuración Inicial

1. **Descargar Datos de Food.com**:

Colocar archivos en `data/raw/foodcom/`:
- `RAW_recipes.csv`
- `RAW_interactions.csv`
- `PP_recipes.csv`
- `PP_users.csv`

2. **Descargar Imágenes de MM-Food-100K**:

```bash
python scripts/download_images.py \
    --output_dir data/raw/mm_food_100k/images \
    --num_workers 8 \
    --max_images 100000
```

Este script descarga imágenes desde URLs del dataset Hugging Face con reintentos y validación.

3. **Preprocesar Datos**:

```bash
# Procesar recetas e interacciones
python -m src.preprocessing.foodcom_processor \
    --input_dir data/raw/foodcom \
    --output_dir data/processed

# Crear splits de datos
python -m src.preprocessing.data_splitter \
    --input_file data/processed/recipes_cleaned.parquet \
    --output_dir data/splits \
    --val_size 0.1 \
    --test_size 0.1 \
    --random_seed 42
```

## Uso del Sistema

### 1. Entrenar Sistema de Recomendación

```bash
python scripts/train_recommender.py \
    --recipes_path data/processed/recipes_cleaned.parquet \
    --interactions_path data/processed/interactions_cleaned.parquet \
    --output_dir models/recommender \
    --model_type hybrid \
    --n_factors 100 \
    --alpha 0.5 \
    --beta 0.3 \
    --gamma 0.2
```

**Salidas**:
- `models/recommender/tfidf_vectorizer.pkl`: Vectorizador entrenado
- `models/recommender/svd_model.pkl`: Modelo colaborativo
- `models/recommender/hybrid_weights.json`: Pesos optimizados

**Tiempo Estimado**: 30-60 minutos en CPU moderno

### 2. Pipeline de Sistema de Visión (Image Retrieval + ML Scoring)

El sistema de visión requiere ejecutar 6 pasos secuenciales:

**Paso 1: Preparar Metadata**
```bash
python scripts/prepare_metadata.py \
    --input data/raw/mm_food_100k/metadata_labeled.csv \
    --output data/processed/mm_food_metadata.csv
```
Parsea ingredientes desde JSON y calcula num_ingredients.

**Paso 2: Crear Splits Estratificados**
```bash
python scripts/create_splits.py \
    --input data/processed/mm_food_metadata.csv \
    --output_dir data/splits \
    --test_size 0.1 \
    --val_size 0.1
```
Divide 80/10/10 estratificando por número de ingredientes.

**Paso 3: Generar Embeddings CLIP**
```bash
python scripts/generate_embeddings.py \
    --metadata data/processed/mm_food_metadata.csv \
    --image_dir data/raw/mm_food_100k/images \
    --output_dir data/embeddings \
    --model ViT-B/32 \
    --batch_size 64 \
    --device cuda
```
Genera embeddings de 512 dimensiones para 57k imágenes (2-3 horas en GPU).

**Paso 4: Construir Índice FAISS**
```bash
python scripts/build_faiss_index.py \
    --embeddings data/embeddings/clip_embeddings.npy \
    --output data/embeddings/faiss_index.bin
```
Crea índice FAISS IndexFlatIP para búsqueda de similitud (15 minutos).

**Paso 5: Preparar Training Data para Scoring Model**
```bash
python scripts/prepare_scoring_training_data.py \
    --metadata data/splits/val_metadata.csv \
    --image_dir data/raw/mm_food_100k/images \
    --faiss_index data/embeddings/faiss_index.bin \
    --output data/processed/scoring_training_data.csv \
    --max_samples 5000
```
Genera features para XGBoost usando validation set (500k-1M filas).

**Paso 6: Entrenar Scoring Model**
```bash
python scripts/train_scoring_model.py \
    --training_data data/processed/scoring_training_data.csv \
    --output_dir models/ingredient_scoring \
    --n_estimators 200 \
    --max_depth 5 \
    --learning_rate 0.1
```
Entrena XGBoost classifier (10-20 minutos).

**Tiempo Total Estimado**: 3-4 horas en GPU (RTX 3050 Ti), 8-10 horas en CPU

### 3. Evaluar Modelos

#### Evaluar Recomendador

```bash
python scripts/evaluate_recommender.py \
    --model_dir models/recommender \
    --test_interactions data/processed/interactions_cleaned.parquet \
    --output_dir reports/metrics \
    --k_values 5 10 20
```

**Métricas Reportadas**:
- Precision@K, Recall@K, F1@K
- NDCG@K
- Hit Rate@K
- Coverage (% de recetas recomendadas)

#### Evaluar Sistema de Visión

**Testing de Componentes**:
```bash
python scripts/test_inference_system.py
```
Ejecuta 6 tests unitarios validando cada componente del pipeline.

**Evaluación Completa**:
```bash
python scripts/evaluate_system.py \
    --config configs/inference_config.yaml \
    --metadata data/splits/test_metadata.csv \
    --image_dir data/raw/mm_food_100k/images \
    --output models/ingredient_scoring/evaluation_results.json \
    --max_samples 1000 \
    --thresholds 0.3 0.4 0.5 0.6 0.7
```

**Métricas Reportadas**:
- Precision, Recall, F1 por threshold
- mAP (Mean Average Precision)
- Distribución de K adaptativo
- Análisis de top-1 similarities
- Mejor threshold por F1 score

### 4. Ejecutar Aplicación

#### Modo Streamlit (Prototipo Local)

```bash
streamlit run src/app/streamlit_app.py
```

Abre navegador en `http://localhost:8501`

**Funcionalidades**:
- Input de ingredientes disponibles (texto libre o multiselect)
- Upload de imagen de comida
- Visualización de top-10 recetas recomendadas
- Detalles de cada receta (ingredientes, pasos, tiempo, calorías)
- Sugerencias de ingredientes para comprar

#### Modo API REST

```bash
uvicorn src.app.api:app --host 0.0.0.0 --port 8000 --reload
```

**Endpoints**:

- `POST /recommend`: Recibe ingredientes y retorna recetas
  ```json
  {
    "ingredients": ["chicken", "tomato", "garlic"],
    "top_k": 10
  }
  ```

- `POST /recognize`: Recibe imagen y retorna platillo + ingredientes
  ```json
  {
    "image_base64": "data:image/jpeg;base64,..."
  }
  ```

- `POST /recommend_multimodal`: Recibe ingredientes + imagen
  ```json
  {
    "ingredients": ["rice", "egg"],
    "image_base64": "data:image/jpeg;base64,...",
    "top_k": 10
  }
  ```

**Documentación Interactiva**: `http://localhost:8000/docs`

### 5. Ejemplo de Uso Programático

**Predicción de Ingredientes desde Imagen**:
```python
from src.vision.inference import IngredientPredictor
from PIL import Image

# Inicializar predictor
predictor = IngredientPredictor(config_path="configs/inference_config.yaml")

# Predecir ingredientes
result = predictor.predict(
    image_path="my_food.jpg",
    threshold=0.5,
    return_probabilities=True
)

print(f"Ingredientes detectados ({len(result['ingredients'])}):")
for ing in result['ingredients']:
    prob = result['probabilities'][ing]
    print(f"  - {ing}: {prob:.3f}")

print(f"\nMetadata:")
print(f"  K usado: {result['metadata']['k_used']}")
print(f"  Top-1 similarity: {result['metadata']['top1_similarity']:.4f}")
print(f"  Candidatos evaluados: {result['metadata']['num_candidates']}")
```

**Integración con Recomendador**:
```python
from src.integration.multimodal_engine import MultimodalEngine

# Inicializar sistema completo
engine = MultimodalEngine(
    recommender_path="models/recommender",
    inference_config="configs/inference_config.yaml"
)

# Modo imagen + ingredientes
recommendations = engine.recommend_multimodal(
    ingredients=["rice", "egg"],
    image_path="my_food.jpg",
    top_k=10
)

print(f"Ingredientes detectados: {recommendations['detected_ingredients']}")
print("\nRecetas recomendadas:")
for rec in recommendations['recipes']:
    print(f"  - {rec['name']} (score: {rec['score']:.3f})")
    print(f"    Faltantes: {rec['missing_ingredients']}")
```

## Evaluación y Métricas

### Sistema de Recomendación

**Protocolo de Evaluación**:
- Validación temporal: entrenar con interacciones antes de fecha T, evaluar después
- Leave-one-out: ocultar última interacción de cada usuario

**Métricas Principales**:

| Métrica | Valor Objetivo | Descripción |
|---------|---------------|-------------|
| Precision@10 | > 0.15 | Proporción de recetas relevantes en top-10 |
| Recall@10 | > 0.08 | Proporción de recetas relevantes recuperadas |
| NDCG@10 | > 0.25 | Ganancia acumulativa descontada normalizada |
| Hit Rate@10 | > 0.40 | % usuarios con al menos 1 receta relevante en top-10 |
| Coverage | > 0.30 | % del catálogo que se recomienda |

**Baseline de Comparación**:
- Popularidad: recomendar recetas más populares
- Random: recomendaciones aleatorias
- Content-only: solo TF-IDF sin collaborative
- Collaborative-only: solo SVD sin content

### Sistema de Visión (Image Retrieval + ML Scoring)

**Protocolo de Evaluación**:
- Split 80/10/10 estratificado por número de ingredientes
- Test set: 5,706 imágenes sin overlap con train
- Evaluación por threshold (0.3, 0.4, 0.5, 0.6, 0.7)

**Métricas Principales**:

| Métrica | Valor Objetivo | Descripción |
|---------|---------------|-------------|
| Precision | > 0.60 | Proporción de ingredientes predichos correctos |
| Recall | > 0.50 | Proporción de ingredientes verdaderos detectados |
| F1-Score | > 0.55 | Media armónica de precision y recall |
| mAP | > 0.65 | Mean Average Precision (ranking quality) |

**Análisis Adicional**:
- Distribución de K adaptativo (min=3, max=20)
- Top-1 similarity statistics
- Feature importance de XGBoost
- Curva Precision-Recall por threshold

### Integración Multimodal

**Métricas de Negocio**:
- Relevancia percibida (estudio de usuarios)
- Tasa de adopción de sugerencias de compra
- Diversidad de recetas recomendadas

**A/B Testing (futuro)**:
- Sistema multimodal vs solo texto
- Diferentes valores de λ (boost por imagen)

## Trabajo Futuro

### Fase 2: Integración de Precios y Optimización Económica

**Objetivos**:
- Incorporar dataset F-MAP de precios de alimentos
- Modelos de series de tiempo para forecasting de precios
- Optimización de compras considerando presupuesto
- Sugerencias de recetas económicas por temporada

**Modelos Propuestos**:
- ARIMA/Prophet para series de tiempo de precios
- Optimización lineal para canasta de compras
- Reinforcement learning para planificación de menús semanales

### Mejoras Técnicas

**Sistema de Recomendación**:
- Deep learning para collaborative filtering (NCF, autoencoders)
- Embeddings de recetas con Graph Neural Networks
- Personalización con contextual bandits

**Visión Computacional**:
- Mejorar scoring model con deep learning (MLP, Attention)
- Object detection para ingredientes específicos (YOLO, Faster R-CNN)
- Segmentación semántica de componentes del plato
- Estimación de cantidad/porción desde imagen
- Modelos CLIP más grandes (ViT-L/14) para mejor precisión

**Multimodal**:
- Vision-Language Models para descripciones generativas
- Búsqueda imagen-texto con CLIP (ya implementado con retrieval)
- Atención cruzada entre modalidades para fusion avanzada

### Infraestructura y Despliegue

**MLOps**:
- Pipeline CI/CD con GitHub Actions
- Versionado de datos con DVC
- Tracking de experimentos con MLflow/Weights & Biases
- Monitoreo de drift de datos y modelos

**Producción**:
- Containerización con Docker
- Despliegue en Kubernetes
- Serving optimizado con TorchServe/TensorRT
- API escalable con balanceo de carga

### Aplicación Móvil

**Características**:
- Captura de foto en tiempo real
- Modo offline con modelos comprimidos
- Historial de recetas favoritas
- Integración con listas de compras
- Notificaciones de ofertas (integración con F-MAP)

**Tecnología**:
- React Native o Flutter para cross-platform
- TensorFlow Lite para inferencia móvil
- SQLite para almacenamiento local
- Sincronización con backend

### Expansión del Dataset

**Nuevas Fuentes**:
- Web scraping de sitios de recetas locales
- Contribuciones de comunidad de usuarios
- Traducción a múltiples idiomas

**Enriquecimiento**:
- Metadata de alergias e intolerancias
- Información de sustentabilidad (huella de carbono)
- Dificultad de preparación (crowdsourcing)

## Documentación Adicional

### Documentos Principales

- **[PROYECTO_ML_COMPLETO.md](PROYECTO_ML_COMPLETO.md)**: Documento consolidado con toda la información del proyecto
- **[MODELO_XGBOOST.md](MODELO_XGBOOST.md)**: Documentación técnica completa del modelo XGBoost (para Data Scientists/ML Engineers)
- **[docs/FASE2_IMPLEMENTACION.md](docs/FASE2_IMPLEMENTACION.md)**: Detalles de la implementación final optimizada (ROC-AUC 0.84)
- **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)**: Arquitectura técnica detallada del sistema
- **[docs/QUICKSTART.md](docs/QUICKSTART.md)**: Guía rápida para empezar a usar el sistema

### Documentos de Migración

- **[deprecated/README.md](deprecated/README.md)**: Código antiguo de clasificación y razones de migración
- **Experimentos**: Ver [experimentos_proceso/](experimentos_proceso/) para scripts y resultados de optimización

### Guías de Configuración

- **[configs/inference_config.yaml](configs/inference_config.yaml)**: Configuración del pipeline de visión
- **[configs/recommender_config.yaml](configs/recommender_config.yaml)**: Configuración del sistema de recomendación

## Referencias

### Datasets

- Food.com Recipes and Interactions: [Kaggle](https://www.kaggle.com/datasets/shuyangli94/food-com-recipes-and-user-interactions)
- MM-Food-100K: [Hugging Face](https://huggingface.co/datasets/Codatta/MM-Food-100K)

### Papers y Referencias Técnicas

**Image Retrieval + ML Scoring**:
- Radford, A., et al. (2021). Learning Transferable Visual Models From Natural Language Supervision. ICML.
- Johnson, J., et al. (2019). Billion-scale similarity search with GPUs. arXiv.
- Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. KDD.

**Sistemas de Recomendación**:
- Koren, Y., Bell, R., & Volinsky, C. (2009). Matrix Factorization Techniques for Recommender Systems. IEEE Computer.
- Ricci, F., Rokach, L., & Shapira, B. (2015). Recommender Systems Handbook. Springer.

**Visión Computacional en Alimentos**:
- Min, W., et al. (2023). A Survey on Food Computing. ACM Computing Surveys.

### Herramientas y Frameworks

- PyTorch: [https://pytorch.org/](https://pytorch.org/)
- Scikit-learn: [https://scikit-learn.org/](https://scikit-learn.org/)
- Hugging Face: [https://huggingface.co/](https://huggingface.co/)
- Streamlit: [https://streamlit.io/](https://streamlit.io/)

### Contacto y Contribuciones

Para preguntas, sugerencias o contribuciones:

- Issues: [GitHub Issues](https://github.com/tu-usuario/smart-budget-kitchen/issues)
- Email: tu-email@ejemplo.com
- Documentación completa: [GitHub Wiki](https://github.com/tu-usuario/smart-budget-kitchen/wiki)

### Licencia

Este proyecto está bajo la licencia MIT. Ver archivo `LICENSE` para detalles.
