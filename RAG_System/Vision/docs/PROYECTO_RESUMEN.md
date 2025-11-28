# Smart Budget Kitchen - Resumen del Proyecto

## Descripción General

Se ha diseñado y desarrollado un sistema completo de machine learning para recomendación de recetas llamado **Smart Budget Kitchen**. El sistema integra:

1. **Sistema de Recomendación Híbrido** basado en ingredientes, ratings de usuarios y popularidad
2. **Módulo de Visión Computacional** para identificar platillos e ingredientes desde imágenes
3. **Aplicación Web Interactiva** con Streamlit para usuarios finales
4. **Arquitectura Profesional** siguiendo mejores prácticas de MLOps

## Estructura del Proyecto Creada

### Documentación Principal

- **README.md**: Documentación completa del proyecto con arquitectura, datasets, modelos, instalación y uso
- **QUICKSTART.md**: Guía de inicio rápido en 5 pasos
- **PROYECTO_RESUMEN.md**: Este archivo de resumen
- **requirements.txt**: Todas las dependencias Python necesarias
- **setup.py**: Configuración para instalación como paquete

### Configuraciones

```
configs/
├── recommender_config.yaml    # Hiperparámetros del recomendador
└── vision_config.yaml          # Hiperparámetros del modelo de visión
```

### Código Fuente (src/)

#### 1. Módulo de Preprocesamiento

```
src/preprocessing/
├── __init__.py
└── foodcom_processor.py        # Procesador completo de Food.com
```

**Funcionalidades**:
- Carga de recetas e interacciones
- Limpieza de datos (valores nulos, duplicados, outliers)
- Parseo de JSON (ingredientes, tags, nutrition)
- Feature engineering (texto combinado, stats de popularidad)
- Exportación a Parquet optimizado

#### 2. Sistema de Recomendación

```
src/recommender/
├── __init__.py
├── content_based.py            # TF-IDF + similitud coseno
├── collaborative.py            # SVD collaborative filtering
└── hybrid.py                   # Fusión ponderada
```

**Características**:
- Content-based: vectorización de ingredientes y tags
- Collaborative filtering: factorización de matrices con SVD
- Híbrido: combina similitud, collaborative y popularidad
- Detección de ingredientes faltantes
- Guardado y carga de modelos

#### 3. Visión Computacional

```
src/vision/
├── __init__.py
├── dataset.py                  # PyTorch Dataset con augmentation
├── models.py                   # EfficientNetV2 con transfer learning
├── training.py                 # Trainer con early stopping
└── inference.py                # Motor de inferencia optimizado
```

**Características**:
- Clasificación multi-clase (500 platillos)
- Clasificación multi-label (200 ingredientes)
- Transfer learning desde ImageNet
- Data augmentation configurable
- Checkpointing y early stopping

#### 4. Integración Multimodal

```
src/integration/
├── __init__.py
└── multimodal_engine.py        # Motor que fusiona visión + recomendación
```

**Modos de operación**:
- Solo ingredientes (texto)
- Solo imagen
- Multimodal (imagen + ingredientes)
- Sugerencias inteligentes de compras

#### 5. Aplicación

```
src/app/
├── __init__.py
└── streamlit_app.py            # Aplicación web completa
```

**Funcionalidades**:
- Input de ingredientes (texto o lista)
- Upload de imágenes de comida
- Visualización de recomendaciones con scores
- Detalles de recetas (ingredientes, pasos, calorías, ratings)
- Sugerencias de ingredientes para comprar

#### 6. Utilidades

```
src/utils/
├── __init__.py
├── config.py                   # Gestión de configuraciones YAML
└── logging_utils.py            # Logging estructurado
```

### Scripts de Ejecución

```
scripts/
├── train_recommender.py        # Entrenar sistema de recomendación
├── train_vision_model.py       # Entrenar modelo de visión
└── download_images.py          # Descargar dataset MM-Food-100K
```

**Uso**:
```bash
# Entrenar recomendador
python scripts/train_recommender.py --recipes data/processed/recipes_cleaned.parquet --interactions data/processed/interactions_cleaned.parquet --output_dir models/recommender

# Entrenar visión
python scripts/train_vision_model.py --data_dir data/raw/mm_food_100k/images --metadata data/raw/mm_food_100k/metadata.csv --output_dir models/vision

# Descargar imágenes
python scripts/download_images.py --output_dir data/raw/mm_food_100k/images --num_workers 8
```

### Notebooks de Análisis

```
notebooks/
├── 01_foodcom_eda.ipynb        # EDA de Food.com (existente)
└── 02_mm_food_100k_eda.ipynb   # EDA de MM-Food-100K (existente)
```

### Estructura de Datos

```
data/
├── raw/
│   ├── foodcom/                # Datasets Food.com originales
│   └── mm_food_100k/           # Imágenes descargadas
├── processed/                  # Datos limpios (Parquet)
└── splits/                     # Train/val/test splits
```

### Modelos Entrenados

```
models/
├── recommender/
│   ├── tfidf_vectorizer.pkl
│   ├── svd_model.pkl
│   └── hybrid_weights.json
├── vision/
│   ├── dish_classifier_best.pth
│   └── training_history.json
└── integration/
    └── sentence_transformer/
```

### Reportes y Evaluaciones

```
reports/
├── figures/                    # Gráficos y visualizaciones
├── metrics/                    # CSVs con métricas
└── model_cards/                # Documentación de modelos
```

### Plantilla de Aplicación Móvil

```
app_template/
└── README.md                   # Guía para desarrollo móvil
```

Incluye recomendaciones para:
- React Native (recomendado)
- Flutter
- Android Nativo
- Conexión con backend
- Inferencia local con TensorFlow Lite

## Modelos de Machine Learning Implementados

### 1. Content-Based Recommender

**Algoritmo**: TF-IDF + Cosine Similarity

**Características**:
- Vectorización de ingredientes y tags
- Similitud coseno para ranking
- Detección de ingredientes faltantes
- Filtrado por umbral de similitud

**Métricas objetivo**:
- Precision@10 > 0.15
- NDCG@10 > 0.25

### 2. Collaborative Filtering

**Algoritmo**: SVD (Singular Value Decomposition)

**Características**:
- Factorización de matriz usuario-receta
- 100 factores latentes
- Predicción de ratings 1-5

**Métricas objetivo**:
- RMSE < 1.0
- MAE < 0.8

### 3. Híbrido

**Fusión**:
```
score = 0.5 × content + 0.3 × collaborative + 0.2 × popularity
```

**Ventajas**:
- Resuelve cold-start
- Personalización con collaborative
- Balance con popularidad

### 4. Clasificador de Platillos (Visión)

**Arquitectura**: EfficientNetV2-S

**Características**:
- 500 clases (platillos más frecuentes)
- Transfer learning desde ImageNet
- Fine-tuning con freeze de 80% de capas
- Data augmentation

**Métricas objetivo**:
- Top-1 Accuracy > 0.45
- Top-5 Accuracy > 0.70
- F1-Macro > 0.40

### 5. Predictor de Ingredientes (Visión)

**Arquitectura**: EfficientNetV2-S Multi-label

**Características**:
- 200 labels (ingredientes comunes)
- BCE Loss con pesos
- Threshold ajustable por clase

**Métricas objetivo**:
- Hamming Loss < 0.15
- F1-Macro > 0.40

## Datasets Utilizados

### Food.com (231,637 recetas, 1,132,367 interacciones)

**Ubicación**: `data/raw/foodcom/`

**Campos principales**:
- Recetas: nombre, ingredientes, tags, nutrition, pasos, tiempo
- Interacciones: user_id, recipe_id, rating (1-5), review, fecha

**Uso**: Sistema de recomendación

### MM-Food-100K (100,000 imágenes)

**Fuente**: Hugging Face (Codatta/MM-Food-100K)

**Campos principales**:
- image_url, dish_name, food_type, ingredients, nutritional_profile

**Uso**: Modelo de visión computacional

## Flujo de Trabajo Completo

### Fase 1: Preparación de Datos

```bash
# 1. Colocar datos Food.com en data/raw/foodcom/

# 2. Descargar imágenes MM-Food-100K
python scripts/download_images.py --output_dir data/raw/mm_food_100k/images

# 3. Procesar datos
python -m src.preprocessing.foodcom_processor \
    --recipes data/raw/foodcom/RAW_recipes.csv \
    --interactions data/raw/foodcom/RAW_interactions.csv \
    --output data/processed
```

### Fase 2: Entrenamiento de Modelos

```bash
# 1. Entrenar recomendador
python scripts/train_recommender.py \
    --recipes data/processed/recipes_cleaned.parquet \
    --interactions data/processed/interactions_cleaned.parquet \
    --output_dir models/recommender \
    --model_type hybrid

# 2. Entrenar modelo de visión (opcional)
python scripts/train_vision_model.py \
    --data_dir data/raw/mm_food_100k/images \
    --metadata data/raw/mm_food_100k/metadata.csv \
    --output_dir models/vision \
    --device cuda
```

### Fase 3: Ejecución de Aplicación

```bash
# Aplicación Streamlit
streamlit run src/app/streamlit_app.py

# O API REST
uvicorn src.app.api:app --reload
```

## Características Profesionales Implementadas

### Arquitectura

- Separación de concerns (preprocessing, models, app)
- Código modular y reutilizable
- Configuración externa (YAML)
- Logging estructurado

### Buenas Prácticas

- Docstrings completos en español
- Type hints en Python
- Manejo de errores robusto
- Guardado y carga de modelos

### MLOps

- Pipeline reproducible
- Versionado de configuraciones
- Checkpointing y early stopping
- Evaluación con métricas estándar

### Escalabilidad

- Procesamiento paralelo (threads)
- Carga eficiente con Parquet
- Inferencia batch
- Modelo modular para extensiones

## Innovaciones del Proyecto

1. **Sistema Híbrido Sofisticado**: Combina 3 enfoques diferentes con pesos optimizables

2. **Integración Multimodal**: Fusiona información visual y textual de forma inteligente

3. **Sugerencias de Compras**: Analiza ingredientes faltantes y sugiere qué comprar

4. **Boost Visual**: Las recetas similares al platillo detectado en imagen reciben score adicional

5. **Detección de Ingredientes Faltantes**: Muestra exactamente qué ingredientes necesitas comprar

6. **Arquitectura Extensible**: Preparada para:
   - Integración con dataset de precios (F-MAP)
   - Modelos de series de tiempo
   - Optimización económica
   - Aplicación móvil

## Próximos Pasos Recomendados

### Corto Plazo

1. Entrenar modelos con los datos completos
2. Evaluar performance con métricas definidas
3. Ajustar hiperparámetros basado en resultados
4. Crear visualizaciones de resultados

### Mediano Plazo

1. Integrar dataset F-MAP de precios
2. Implementar modelos de forecasting
3. Desarrollar aplicación móvil con React Native
4. Optimizar modelos para inferencia móvil (TFLite)

### Largo Plazo

1. Despliegue en producción (Docker + Kubernetes)
2. CI/CD con GitHub Actions
3. Monitoreo de drift de datos
4. A/B testing de variantes del modelo

## Archivos de Configuración Importantes

### requirements.txt

Todas las dependencias necesarias instalables con:
```bash
pip install -r requirements.txt
```

### configs/recommender_config.yaml

Hiperparámetros del sistema de recomendación:
- TF-IDF: max_features, ngram_range, min_df
- SVD: n_factors, n_epochs, learning_rate
- Híbrido: pesos alpha, beta, gamma

### configs/vision_config.yaml

Hiperparámetros del modelo de visión:
- Arquitectura: efficientnet_v2_s
- Training: batch_size, epochs, learning_rate
- Augmentation: rotations, flips, color jitter

### .gitignore

Excluye archivos grandes y sensibles:
- Datos (*.csv, *.parquet)
- Modelos entrenados (*.pth, *.pkl)
- Logs y temporales

## Recursos y Referencias

### Datasets

- [Food.com en Kaggle](https://www.kaggle.com/datasets/shuyangli94/food-com-recipes-and-user-interactions)
- [MM-Food-100K en Hugging Face](https://huggingface.co/datasets/Codatta/MM-Food-100K)

### Frameworks y Librerías

- PyTorch para deep learning
- Scikit-learn para ML clásico
- Surprise para collaborative filtering
- Streamlit para aplicación web
- Timm para modelos de visión preentrenados

### Papers Relevantes

- Matrix Factorization Techniques (Koren et al., 2009)
- EfficientNetV2 (Tan & Le, 2021)
- Recommender Systems Handbook (Ricci et al., 2015)

## Métricas de Éxito del Proyecto

### Técnicas

- Sistema de recomendación con Precision@10 > 0.15
- Modelo de visión con Top-5 Accuracy > 0.70
- Tiempo de respuesta < 2 segundos

### Negocio

- Usuarios encuentran recetas relevantes en top-10
- Reducción de desperdicio de alimentos
- Sugerencias de compra útiles y precisas

## Conclusión

Se ha desarrollado un **sistema completo y profesional** de recomendación de recetas que:

- Integra múltiples fuentes de datos (recetas, ratings, imágenes)
- Utiliza técnicas avanzadas de ML (híbrido, deep learning)
- Ofrece una interfaz amigable para usuarios
- Sigue mejores prácticas de ingeniería de software
- Está preparado para extensiones futuras

El proyecto está **listo para ser entrenado, evaluado y desplegado** siguiendo las instrucciones en QUICKSTART.md y README.md.
