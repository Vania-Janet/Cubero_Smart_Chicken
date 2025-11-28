# Smart Budget Kitchen - Presentación Técnica del Proyecto

**Sistema de Recomendación de Recetas con Visión Computacional**

---

## 📌 Resumen Ejecutivo

**Smart Budget Kitchen** es un sistema inteligente de ML que combina:
- **Sistema de recomendación híbrido** (content-based + collaborative filtering)
- **Módulo de visión computacional** (clasificación de platillos e ingredientes)
- **Integración multimodal** (texto + imagen)

**Propósito**: Recomendar recetas basadas en ingredientes disponibles y/o fotografías de comida, sin usar APIs externas (100% ML local).

---

## 🎯 Problema que Resuelve (Hacia persona individual o empresa)

1. **Usuarios tienen ingredientes pero no saben qué cocinar**
2. **Desperdicio de alimentos** por falta de ideas
3. **Identificación de platillos** desde fotografías
4. **Recomendaciones personalizadas** considerando popularidad y preferencias
5. **Tener en cuenta el precio de los productos** (Google shopping-api)


---
MAÑANA (22/11/25) EN LA NOCHE:

4 equipos! jajaja

1- Marketing y conexión (Tocar puerta a puerta) 
2- Clasificación de imágen  (Jhoshua)
3- Serie de tiempo **cdmx** (Ale Y Bruno)
4- Sistema de recomendación (Jan y Meli <3)
5- App Ios y Android (David)
--- 


## 📊 Datasets Utilizados

### 1. Food.com (Recetas + Interacciones)
- **Recetas**: 231,637 → ~200,000 después de limpieza
- **Interacciones**: 1,132,367 ratings (escala 1-5)
- **Fuente**: Kaggle - Food.com Recipes and User Interactions
- **Tamaño**: ~900 MB (CSVs crudos)

### 2. MM-Food-100K (Imágenes)
- **Imágenes**: 100,000 (o 50,000 descargadas actualmente)
- **Categorías**: 500 platillos diferentes
- **Ingredientes**: 200 ingredientes etiquetados
- **Fuente**: Hugging Face
- **Tamaño**: ~90 GB (50k imágenes)

---

## 🏗️ Arquitectura del Sistema

``` Huevo, Salchichas, papa,...    FOTO    ---->  Recetas, precio, propuestas
┌─────────────────────────────────────────────────────────┐
│              SMART BUDGET KITCHEN                       │
└─────────────────────────────────────────────────────────┘
                         │
         ┌───────────────┴───────────────┐
         │                               │
  ┌──────▼──────┐                 ┌─────▼──────┐
  │ RECOMMENDER │                 │   VISION   │
  │   MODULE    │                 │   MODULE   │
  │             │                 │            │----------- Series de tiempo
  │ Content-    │                 │ Efficient- │
  │   Based     │                 │   NetV2    │
  │ (TF-IDF)    │                 │            │
  │             │                 │ Multi-class│
  │ Collaborative│                │ Multi-label│
  │ (TruncatedSVD)│               │            │
  │             │                 │            │
  │ Hybrid      │                 │            │
  │ (Ensemble)  │                 │            │
  └──────┬──────┘                 └─────┬──────┘
         │                               │
         └───────────────┬───────────────┘
                         │
               ┌─────────▼──────────┐
               │   MULTIMODAL       │
               │   INTEGRATION      │
               │                    │
               │ - Score Fusion     │
               │ - Ranking          │
               │ - Shopping Hints   │
               └─────────┬──────────┘
                         │
               ┌─────────▼──────────┐
               │   STREAMLIT APP    │
               │   (Web Interface)  │
               └────────────────────┘
```

---

## 🔧 Componentes Técnicos Clave

### 1. Sistema de Recomendación

#### a) Content-Based Filtering
- **Algoritmo**: TF-IDF (Term Frequency - Inverse Document Frequency)
- **Input**: Ingredientes + Tags de recetas
- **Output**: Similitud coseno entre ingredientes del usuario y recetas
- **Ventaja**: No requiere historial de usuario

#### b) Collaborative Filtering
- **Algoritmo**: TruncatedSVD (Singular Value Decomposition)
- **Implementación**: `sklearn.decomposition.TruncatedSVD`
- **Input**: Matriz usuario-receta con ratings
- **Output**: Predicción de ratings (1-5) para pares usuario-receta
- **Mejoras**: Cálculo de biases (global mean, user bias, item bias)

**¿Por qué TruncatedSVD y no scikit-surprise?**
- ✅ No requiere compilación (no necesita Visual C++ Build Tools)
- ✅ Ya incluido en scikit-learn (instalación más simple)
- ✅ Manejo nativo de matrices sparse
- ✅ API compatible con el diseño original



#### c) Hybrid System
- **Fórmula**: `score = 0.5×content + 0.2×collaborative + 0.2×popularity + 0.1xPrecio`
- **Ventaja**: Combina fortalezas de ambos enfoques
- **Parámetros ajustables**: Pesos (alpha, beta, gamma)

### 2. Módulo de Visión Computacional

#### a) Clasificador de Platillos
- **Arquitectura**: EfficientNetV2-S (pretrained en ImageNet)
- **Task**: Multi-class classification (500 clases)
- **Fine-tuning**: Transfer learning con capas congeladas
- **Input**: Imagen RGB 224×224
- **Output**: Probabilidades por clase de platillo

#### b) Predictor de Ingredientes
- **Arquitectura**: EfficientNetV2-S (pretrained)
- **Task**: Multi-label classification (200 ingredientes)
- **Input**: Imagen RGB 224×224
- **Output**: Probabilidades por ingrediente

#### c) Optimizaciones
- **GPU**: Configurado para NVIDIA RTX (6 workers, batch_size 32)
- **Augmentations**: Albumentations (flips, rotations, color jitter)
- **Early stopping**: Previene overfitting

### 3. Integración Multimodal

**Modos de operación**:
1. **Solo texto**: Ingredientes → Recomendaciones
2. **Solo imagen**: Foto → Detección de ingredientes → Recomendaciones
3. **Imagen + texto**: Foto + ingredientes adicionales → Recomendaciones boosted

**Score fusion**:
```python
final_score = base_score × (1 + boost_factor)
boost_factor = similarity_to_detected_dish
```

---

## 🧹 Pipeline de Preprocesamiento (Con Detección Profesional de Outliers)

### Fase 1: Limpieza Básica
1. Eliminar duplicados
2. Filtrar recetas sin nombre/ingredientes/pasos
3. Filtrar tiempos > 48 horas (2880 minutos)
4. Parsear campos JSON (ingredientes, tags, nutrition)
5. Normalizar texto (lowercase, sin guiones)

### Fase 2: Detección de Outliers (Ensemble de 5 Métodos)

**Métodos aplicados**:
1. **IQR (Interquartile Range)** - Tukey
   - Detecta valores fuera de [Q1 - 1.5×IQR, Q3 + 1.5×IQR]
   - Robusto, no paramétrico

2. **Modified Z-Score (MAD)**
   - Usa mediana en lugar de media
   - Basado en Median Absolute Deviation
   - Más robusto ante outliers extremos

3. **Isolation Forest**
   - Ensemble de árboles de decisión
   - Detecta outliers multivariados
   - Eficiente: O(n log n)

4. **Local Outlier Factor (LOF)**
   - Basado en densidad local
   - Detecta outliers contextuales
   - Compara densidad con vecinos

5. **DBSCAN**
   - Clustering basado en densidad
   - Marca puntos de "ruido" como outliers
   - No requiere especificar número de clusters

**Estrategia de Ensemble**:
- Si **≥3 métodos** marcan un registro como outlier → **se elimina**
- Balance entre agresividad y conservación de datos

**Variables analizadas**:
- `minutes` (tiempo preparación)
- `n_ingredients` (número ingredientes)
- `n_steps` (número pasos)
- `calories` (calorías)
- Valores nutricionales: `fat_pdv`, `sugar_pdv`, `sodium_pdv`, `protein_pdv`, etc.

**Resultado esperado**:
- Eliminación de ~5-10% de recetas con datos anómalos
- Mejora significativa en calidad de datos para ML

### Fase 3: Filtrado de Interacciones
- Usuarios activos: ≥3 interacciones
- Recetas populares: ≥5 ratings
- **Propósito**: Mejorar señal para collaborative filtering

### Fase 4: Feature Engineering
- Crear campo `content_text` = ingredientes + tags
- Calcular estadísticas: `rating_mean`, `num_ratings`, `popularity_score`
- Construir vocabulario de ingredientes

### Fase 5: Almacenamiento Optimizado
- **Formato**: Parquet (compresión columnar)
- **Reducción**: ~615 MB → ~120 MB (80% menos)
- **Velocidad**: 5-6x más rápido que CSV

---

## 📁 Estructura del Repositorio

```
Proyecto ML plus/
├── data/
│   ├── raw/
│   │   ├── foodcom/              # CSVs originales (900 MB)
│   │   └── mm_food_100k/         # Imágenes (10 GB)
│   ├── processed/                # Parquets limpios (120 MB)
│   └── splits/                   # Train/val/test
│
├── src/
│   ├── preprocessing/            # Limpieza y outliers
│   │   └── foodcom_processor.py # ← 5 métodos profesionales
│   ├── recommender/              # Sistema recomendación
│   │   ├── content_based.py     # TF-IDF
│   │   ├── collaborative.py     # TruncatedSVD ← NUEVO
│   │   └── hybrid.py            # Ensemble
│   ├── vision/                   # Modelos CNN
│   │   ├── models.py            # EfficientNetV2
│   │   ├── training.py          # Loop entrenamiento
│   │   └── inference.py         # Inferencia optimizada
│   ├── integration/              # Multimodal
│   │   └── multimodal_engine.py # Fusión texto+imagen
│   ├── app/                      # Interfaces
│   │   └── streamlit_app.py     # Web UI
│   └── utils/                    # Utilidades
│
├── scripts/                      # Scripts CLI
│   ├── train_recommender.py     # Entrenar recomendador
│   ├── train_vision_model.py    # Entrenar CNN
│   └── download_images.py       # Descargar MM-Food-100K
│
├── configs/                      # Hiperparámetros
│   ├── recommender_config.yaml
│   └── vision_config.yaml
│
├── models/                       # Modelos entrenados
│   ├── recommender/
│   └── vision/
│
├── notebooks/                    # EDA
│   ├── 01_foodcom_eda.ipynb
│   └── 02_mm_food_100k_eda.ipynb
│
├── requirements.txt              # Dependencias
├── setup.py
├── README.md                     # Documentación técnica
├── QUICKSTART.md                 # Guía rápida
├── COLABORADORES.md              # Guía para colaboradores
└── PRESENTACION_EQUIPO.md        # Este archivo
```

---

## 🛠️ Stack Tecnológico

### Machine Learning
- **PyTorch**: Deep learning (visión)
- **scikit-learn**: ML clásico (TF-IDF, TruncatedSVD)
- **scipy**: Matrices sparse
- **timm**: Modelos pretrained (EfficientNet)

### Data Processing
- **pandas**: Manipulación de datos
- **numpy**: Operaciones numéricas
- **pyarrow**: Lectura/escritura Parquet

### Computer Vision
- **Pillow**: Manejo de imágenes
- **opencv-python**: Procesamiento
- **albumentations**: Data augmentation

### Application
- **Streamlit**: Web UI interactiva
- **FastAPI**: REST API (opcional)

### Utilities
- **joblib**: Serialización de modelos
- **pyyaml**: Configuraciones
- **tqdm**: Progress bars

**Versión Python**: 3.9.13
**Entorno**: `appComida` (virtual environment)

---

## 📈 Estado Actual del Proyecto

### ✅ Completado

1. **Diseño de arquitectura completa**
   - Sistema de recomendación híbrido
   - Módulo de visión
   - Integración multimodal

2. **Implementación de código**
   - 5,000+ líneas de código Python
   - Estructura modular y escalable
   - Documentación exhaustiva

3. **Sistema de preprocesamiento profesional**
   - 5 métodos de detección de outliers
   - Ensemble voting
   - Optimización de memoria (chunking)

4. **Reemplazo de scikit-surprise → TruncatedSVD**
   - Evita problemas de compilación
   - API compatible
   - Mejor integración

5. **Descarga parcial de imágenes**
   - 50,000 imágenes descargadas (~50% del dataset)

### 🔄 En Progreso

1. **Preprocesamiento de datos de Food.com**
   - Script optimizado para baja memoria
   - Carga en chunks
   - Detección de outliers ejecutándose

### ⏳ Pendiente

1. **Entrenar sistema de recomendación**
   - Content-based (TF-IDF)
   - Collaborative (TruncatedSVD)
   - Hybrid (ensemble)
   - **Tiempo estimado**: 20-40 minutos

2. **Entrenar modelo de visión** (opcional)
   - Clasificador de platillos
   - Predictor de ingredientes
   - **Tiempo estimado**: 4-6 horas con GPU RTX

3. **Desplegar aplicación Streamlit**
   - Interfaz web interactiva
   - Pruebas de usuario

---

## 🎓 Decisiones Técnicas Clave

### 1. ¿Por qué TruncatedSVD en lugar de scikit-surprise?

**Problema con scikit-surprise**:
- Requiere compilación con Microsoft Visual C++ 14.0 Build Tools
- Fallo en instalación en el entorno del proyecto

**Solución con TruncatedSVD**:
- ✅ Ya incluido en scikit-learn (no requiere instalación adicional)
- ✅ No requiere compilación
- ✅ API compatible con el diseño original
- ✅ Manejo nativo de matrices sparse con scipy
- ✅ Implementación con biases para mejorar predicciones

**Implementación**:
```python
# Predicción con biases
rating = global_mean + user_bias + item_bias + U × V^T
```

### 2. ¿Por qué 5 métodos de detección de outliers?

**Enfoque ensemble (voting)**:
- Reduce falsos positivos (outliers que en realidad son válidos)
- Captura diferentes tipos de anomalías:
  - IQR/MAD: Outliers univariados
  - Isolation Forest: Outliers multivariados
  - LOF: Outliers contextuales (locales)
  - DBSCAN: Puntos de ruido en clusters

**Balance**: Si ≥3 métodos coinciden → alta confianza de que es outlier

### 3. ¿Por qué EfficientNetV2 para visión?

- **Eficiencia**: Mejor balance accuracy/velocidad/tamaño que ResNet, VGG
- **Pretrained**: Transfer learning desde ImageNet (1.2M imágenes)
- **Escalable**: Varias versiones (S, M, L) según recursos
- **SOTA**: State-of-the-art en clasificación de imágenes

### 4. ¿Por qué Parquet en lugar de CSV?

**Ventajas de Parquet**:
- Compresión columnar: 80% menos espacio
- Lectura 5-6x más rápida
- Tipos de datos preservados
- Compresión integrada (Snappy, Gzip)
- Compatible con Spark, Dask (escalabilidad futura)

---

## 🚀 Próximos Pasos (Después del Preprocesamiento)

### Paso 1: Entrenar Sistema de Recomendación
```bash
python scripts/train_recommender.py \
    --recipes data/processed/recipes_cleaned.parquet \
    --interactions data/processed/interactions_cleaned.parquet \
    --output_dir models/recommender \
    --model_type hybrid
```

**Salida esperada**:
- `models/recommender/tfidf_vectorizer.pkl`
- `models/recommender/tfidf_matrix.pkl`
- `models/recommender/svd_model.pkl`
- `models/recommender/metadata.pkl`
- `models/recommender/hybrid_weights.json`

### Paso 2: Entrenar Modelo de Visión (Opcional)
```bash
python scripts/train_vision_model.py \
    --data_dir data/raw/mm_food_100k/images \
    --metadata data/raw/mm_food_100k/metadata.csv \
    --output_dir models/vision \
    --task dish_classification \
    --device cuda \
    --batch_size 32 \
    --num_workers 6 \
    --epochs 30
```

### Paso 3: Ejecutar Aplicación
```bash
streamlit run src/app/streamlit_app.py
```

**Interfaz en**: http://localhost:8501

---

## 📊 Métricas de Éxito

### Sistema de Recomendación
- **RMSE** (Root Mean Squared Error): < 1.0 en ratings
- **MAE** (Mean Absolute Error): < 0.8
- **Precision@10**: > 0.7 (70% de recomendaciones relevantes)
- **Coverage**: > 80% de recetas recomendables

### Modelo de Visión
- **Top-1 Accuracy**: > 70% (clasificación de platillos)
- **Top-5 Accuracy**: > 90%
- **F1-Score** (ingredientes): > 0.6

### Aplicación
- **Latencia**: < 2 segundos por recomendación
- **Escalabilidad**: 1000+ recetas procesables

---

## 🔮 Trabajo Futuro

### Corto Plazo
1. Implementar API REST con FastAPI
2. Agregar filtros (tiempo, calorías, dieta)
3. Sistema de feedback de usuarios

### Mediano Plazo
1. Reentrenamiento periódico con nuevos datos
2. A/B testing de hiperparámetros
3. Despliegue en cloud (AWS, GCP, Azure)

### Largo Plazo
1. App móvil (React Native / Flutter)
2. Integración con IoT (cámaras de cocina)
3. Generación de listas de compras automáticas
4. Modelo de lenguaje para generación de recetas

---

## 📚 Recursos y Documentación

### Documentos del Proyecto
- **README.md**: Documentación técnica completa
- **QUICKSTART.md**: Guía de inicio rápido
- **COLABORADORES.md**: Guía para nuevos desarrolladores
- **PRESENTACION_EQUIPO.md**: Este documento

### Notebooks de EDA
- `01_foodcom_eda.ipynb`: Análisis exploratorio de Food.com
- `02_mm_food_100k_eda.ipynb`: Análisis de MM-Food-100K

### Configuraciones
- `configs/recommender_config.yaml`: Hiperparámetros del recomendador
- `configs/vision_config.yaml`: Hiperparámetros de visión

---

## 💡 Preguntas Frecuentes

### ¿Puedo usar solo el recomendador sin el módulo de visión?
✅ Sí, el sistema funciona perfectamente con solo ingredientes de texto.

### ¿Necesito GPU para entrenar?
- **Recomendador**: No, CPU es suficiente
- **Visión**: GPU altamente recomendada (reduce tiempo de 3 días a 6 horas)

### ¿Qué pasa si el usuario no proporciona ingredientes?
El sistema puede recomendar recetas populares (top-rated) como fallback.

### ¿Los modelos son reproducibles?
✅ Sí, todos usan `random_state=42` para reproducibilidad.

### ¿Puedo usar mis propios datos?
✅ Sí, el pipeline es agnóstico al dominio (solo ajustar formato de entrada).

---

## 🤝 Equipo y Contacto

**Desarrollador Principal**: Roberto Jhoshua Alegre Ventura
**Entorno de Desarrollo**: Python 3.9.13, Windows
**GPU**: NVIDIA RTX (7 núcleos CUDA)
**Fecha de Inicio**: Noviembre 2025

---

**Última actualización**: 22 de Noviembre, 2025

---

*Este documento es una guía ejecutiva para presentar el proyecto "Smart Budget Kitchen" a colaboradores y stakeholders. Para detalles técnicos específicos, consultar README.md y la documentación del código.*
