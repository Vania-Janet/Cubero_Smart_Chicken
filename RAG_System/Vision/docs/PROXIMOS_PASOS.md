# Próximos Pasos - Smart Budget Kitchen

**Guía paso a paso después del preprocesamiento**

--- 

## 📍 Estado Actual

Acabas de ejecutar:
```bash
python -m src.preprocessing.foodcom_processor --recipes data/raw/foodcom/RAW_recipes.csv --interactions data/raw/foodcom/RAW_interactions.csv --output data/processed
```

**Resultado**:
- ✅ Datos cargados con optimización de memoria (chunks)
- ✅ Recetas limpiadas (~200,000 de 231,637)
- ✅ Outliers detectados y eliminados (ensemble de 5 métodos)
- ✅ Interacciones filtradas (usuarios ≥3, ratings ≥5)
- ✅ Archivos guardados en formato Parquet optimizado

**Archivos generados** en `data/processed/`:
- `recipes_cleaned.parquet` (~45-50 MB)
- `interactions_cleaned.parquet` (~65-70 MB)
- `ingredient_vocab.json` (~500 KB)

---

## 🎯 ¿QUÉ SIGUE? - Roadmap Completo

```
┌──────────────────────────────────────────────────────────┐
│  PASO 1: Preprocesamiento                    ✅ HECHO   │
└──────────────────────────────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────┐
│  PASO 2: Entrenar Recomendador (20-40 min)   ⏭️ AHORA   │
│  ├─ Content-Based (TF-IDF)                               │
│  ├─ Collaborative (TruncatedSVD)                         │
│  └─ Hybrid (Ensemble)                                    │
└──────────────────────────────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────┐
│  PASO 3: Probar Recomendador                 ⏭️ DESPUÉS  │
│  └─ Verificar predicciones, buscar errores               │
└──────────────────────────────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────                    ┐
│  PASO 4: Entrenar Visión y series de tiempo (4-6 hrs, opcional) ⏭️ OPCIONAL │
│  ├─ Clasificador de platillos                                               │
│  └─ Predictor de ingredientes                                               │
└──────────────────────────────────────────────────────────                   ┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────┐
│  PASO 5: Ejecutar App Streamlit              ⏭️ FINAL   │
│  └─ Interfaz web interactiva                            │
└──────────────────────────────────────────────────────────┘
```

---

## 📋 PASO 2: Entrenar Sistema de Recomendación

### Comando
```bash
python scripts/train_recommender.py \
    --recipes data/processed/recipes_cleaned.parquet \
    --interactions data/processed/interactions_cleaned.parquet \
    --output_dir models/recommender \
    --model_type hybrid
```

### ¿Qué hace este comando?

1. **Carga datos procesados** desde Parquet (rápido, 5-6x más que CSV)

2. **Entrena Content-Based Recommender**:
   - Vectoriza texto con TF-IDF (ingredientes + tags)
   - Crea matriz TF-IDF de ~200,000 recetas × 5,000 features
   - Calcula matriz de similitud coseno
   - **Tiempo**: ~5-10 minutos

3. **Entrena Collaborative Recommender**:
   - Construye matriz sparse usuario-receta (CSR format)
   - Calcula biases (global mean, user biases, item biases)
   - Entrena TruncatedSVD con 100 componentes latentes
   - Predice ratings 1-5 con corrección de biases
   - **Tiempo**: ~10-20 minutos

4. **Crea Hybrid Recommender**:
   - Combina scores: 0.5×content + 0.3×collaborative + 0.2×popularity
   - Guarda pesos de fusión
   - **Tiempo**: < 1 minuto

5. **Guarda modelos entrenados** en `models/recommender/`:
   - `tfidf_vectorizer.pkl`: Vectorizador entrenado
   - `tfidf_matrix.pkl`: Matriz TF-IDF precomputada
   - `recipes_metadata.pkl`: Metadatos de recetas
   - `svd_model.pkl`: Modelo TruncatedSVD
   - `metadata.pkl`: Mapeos y biases
   - `user_factors.npy`: Factores latentes de usuarios
   - `item_factors.npy`: Factores latentes de recetas
   - `hybrid_weights.json`: Pesos del sistema híbrido

### Salida esperada en consola

```
INFO - Iniciando entrenamiento de sistema de recomendación
INFO - Cargando datos...
INFO - Recetas: 200,137
INFO - Interacciones: 950,458
INFO - Entrenando modelo content-based...
INFO - Vectorizando contenido...
INFO - Matriz TF-IDF: (200137, 5000)
INFO - Calculando similitudes...
INFO - Modelo content-based guardado
INFO -
INFO - Prueba de recomendación con ['chicken', 'tomato', 'garlic', 'onion']:
name                              similarity_score  num_missing
Garlic Chicken Pasta              0.9234            0
Tomato Basil Chicken              0.8956            0
One-Pot Chicken Rice              0.8723            1
Mediterranean Chicken             0.8501            1
Easy Chicken Stir Fry             0.8234            2

INFO - Entrenando modelo colaborativo...
INFO - Dataset: 75,234 usuarios, 145,678 recetas, 950,458 interacciones
INFO - Matriz de ratings: (75234, 145678), sparsity: 0.9999
INFO - SVD entrenado: 100 factores, varianza explicada: 0.3456
INFO - Modelo colaborativo guardado
INFO - Configurando modelo híbrido...
INFO - Configuración híbrida guardada
INFO -
INFO - Modelos guardados en: models/recommender
INFO - Entrenamiento completado exitosamente!
```

### Tiempo estimado total
- **CPU moderno**: 20-30 minutos
- **CPU antiguo**: 40-60 minutos

### ¿Qué puede salir mal?

#### Error: "FileNotFoundError: recipes_cleaned.parquet"
**Solución**: Verificar que el preprocesamiento completó exitosamente
```bash
ls data/processed/
```

#### Error: "MemoryError" o "Out of memory"
**Solución**: Reducir `max_features` en `configs/recommender_config.yaml`
```yaml
content_based:
  vectorizer:
    max_features: 3000  # Reducir de 5000 a 3000
```

#### Error: "ModuleNotFoundError: No module named 'src'"
**Solución**: Reinstalar proyecto
```bash
pip install -e .
```

---

## 📋 PASO 3: Probar el Recomendador (5-10 minutos)

### Opción A: Prueba rápida en Python

Crea un archivo `test_recommender.py`:

```python
import pandas as pd
from src.recommender import ContentBasedRecommender, CollaborativeRecommender, HybridRecommender

# Cargar datos
recipes = pd.read_parquet("data/processed/recipes_cleaned.parquet")
interactions = pd.read_parquet("data/processed/interactions_cleaned.parquet")

# Cargar modelos entrenados
content_recommender = ContentBasedRecommender.load("models/recommender")

# Probar recomendación
ingredients = ["chicken", "rice", "garlic", "soy sauce"]
recommendations = content_recommender.recommend(ingredients, top_k=10)

print("\n🍳 Top 10 recomendaciones para:", ingredients)
print(recommendations[['name', 'similarity_score', 'rating_mean', 'num_missing']])
```

Ejecutar:
```bash
python test_recommender.py
```

### Opción B: Notebook interactivo

Abrir Jupyter:
```bash
jupyter notebook
```

Crear nuevo notebook y probar:
```python
# Importar
from src.recommender import ContentBasedRecommender
import pandas as pd

# Cargar modelo
recommender = ContentBasedRecommender.load("models/recommender")
recipes = pd.read_parquet("data/processed/recipes_cleaned.parquet")

# Probar con diferentes ingredientes
test_cases = [
    ["chicken", "tomato", "pasta"],
    ["beef", "potato", "carrot"],
    ["salmon", "lemon", "dill"],
    ["chocolate", "flour", "sugar", "butter"]
]

for ingredients in test_cases:
    print(f"\n{'='*60}")
    print(f"Ingredientes: {', '.join(ingredients)}")
    print(f"{'='*60}")

    recs = recommender.recommend(ingredients, top_k=5)
    print(recs[['name', 'similarity_score', 'rating_mean']])
```

### ¿Qué verificar?

✅ **Relevancia**: Las recetas recomendadas deben usar los ingredientes proporcionados
✅ **Similitud alta**: `similarity_score` > 0.7 para los top 3
✅ **Ingredientes faltantes**: `num_missing` debe ser bajo (<3)
✅ **Diversidad**: No todas deben ser del mismo tipo de platillo
✅ **Calidad**: `rating_mean` > 4.0 idealmente

---

## 📋 PASO 4 (Opcional): Entrenar Modelo de Visión

### ¿Necesito este paso?

**NO** si:
- Solo quieres recomendaciones basadas en texto (ingredientes)
- No tienes GPU (tomaría 2-3 días en CPU)
- Quieres probar el sistema rápidamente

**SÍ** si:
- Quieres clasificar platillos desde imágenes
- Quieres detectar ingredientes en fotos
- Tienes GPU NVIDIA RTX disponible

### Comando (si decides hacerlo)

```bash
# Clasificador de platillos
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

**Tiempo estimado**:
- Con GPU RTX: 4-6 horas (50k imágenes), 8-12 horas (100k imágenes)
- Con CPU: 2-3 días (NO RECOMENDADO)

**Salida**:
- `models/vision/dish_classifier_best.pth`: Modelo entrenado
- `models/vision/training_history.json`: Métricas por época
- `models/vision/class_mapping.json`: Mapeo de clases

---

## 📋 PASO 5: Ejecutar Aplicación Streamlit

### Comando
```bash
streamlit run src/app/streamlit_app.py
```

### ¿Qué pasa?

1. Streamlit inicia servidor web local
2. Abre automáticamente el navegador en `http://localhost:8501`
3. Interfaz interactiva lista para usar

### Funcionalidades de la app

#### Modo 1: Solo Ingredientes
1. Selecciona "Solo Ingredientes" en el sidebar
2. Ingresa ingredientes separados por comas: `chicken, rice, garlic`
3. Click en "Buscar Recetas"
4. Ve top 10 recomendaciones con scores

#### Modo 2: Imagen + Ingredientes (si entrenaste visión)
1. Selecciona "Imagen + Ingredientes"
2. Sube foto de comida (.jpg, .png)
3. Opcionalmente agrega ingredientes adicionales
4. El sistema detecta platillo y recomienda recetas similares

#### Modo 3: Solo Imagen (si entrenaste visión)
1. Selecciona "Solo Imagen"
2. Sube foto
3. Sistema detecta ingredientes automáticamente
4. Recomienda recetas basadas en detección

### Capturas esperadas

```
┌─────────────────────────────────────────────┐
│  🍳 Smart Budget Kitchen                    │
│                                             │
│  Modo: [Solo Ingredientes ▼]                │
│                                             │
│  Ingredientes disponibles:                  │
│  ┌───────────────────────────────────────┐  │
│  │ chicken, tomato, garlic, onion       │  │
│  └───────────────────────────────────────┘  │
│                                             │
│  [Buscar Recetas]                           │
│                                             │
│  📊 Top 10 Recomendaciones:                 │
│  ─────────────────────────────────────────  │
│  1. Garlic Chicken Pasta (⭐ 4.8, 95%)     │
│     Ingredientes faltantes: 0               │
│                                             │
│  2. Tomato Basil Chicken (⭐ 4.6, 89%)     │
│     Ingredientes faltantes: 1 (basil)       │
│                                             │
│  3. One-Pot Chicken Rice (⭐ 4.7, 87%)     │
│     Ingredientes faltantes: 1 (rice)        │
│  ...                                        │
└─────────────────────────────────────────────┘
```

---

## 🔍 Verificación de Éxito

### Checklist Final

Después de completar todos los pasos:

- [ ] **Preprocesamiento**
  - [ ] Archivos Parquet generados en `data/processed/`
  - [ ] Log muestra ~200k recetas finales
  - [ ] Outliers detectados y eliminados (5-10%)

- [ ] **Recomendador**
  - [ ] Modelos guardados en `models/recommender/`
  - [ ] Pruebas con ingredientes devuelven recetas relevantes
  - [ ] Similarity scores > 0.7 para top 3

- [ ] **Visión** (opcional)
  - [ ] Modelo guardado en `models/vision/`
  - [ ] Accuracy > 70% en test set

- [ ] **App**
  - [ ] Streamlit corre sin errores
  - [ ] Recomendaciones son relevantes
  - [ ] Interfaz es responsiva

---

## 🐛 Troubleshooting Común

### Error: "Cannot allocate memory"
**Causa**: Proceso consume demasiada RAM
**Solución**:
```bash
# Cerrar programas innecesarios
# Reducir chunk_size en foodcom_processor.py si persiste
```

### Error: "CUDA out of memory"
**Causa**: Batch size muy grande para GPU
**Solución**:
```bash
# Reducir batch_size a 16 u 8
python scripts/train_vision_model.py ... --batch_size 16
```

### Error: Recomendaciones no son relevantes
**Causa**: Modelo no entrenado correctamente
**Solución**:
```bash
# Re-entrenar con más datos o ajustar hiperparámetros
# Verificar configs/recommender_config.yaml
```

### App Streamlit muy lenta
**Causa**: Carga de modelos en cada request
**Solución**: Modelos se cachean con `@st.cache_resource`, verificar implementación

---

## 📊 Métricas a Monitorear

### Durante Entrenamiento

**Content-Based**:
- Matriz TF-IDF: debe ser sparse (~1-5% densidad)
- Vocabulario: ~5000-10000 tokens únicos

**Collaborative**:
- Sparsity: >99% es normal
- Varianza explicada: >30% con 100 factores
- RMSE: < 1.0 en test set

**Hybrid**:
- Balance de pesos: verificar que todos contribuyan

### En Producción

- Latencia de recomendación: < 2 segundos
- Coverage: > 80% de recetas recomendables
- User satisfaction: feedback positivo

---

## 🎯 Objetivos de Negocio

1. **Reducir desperdicio de alimentos**: Recomendar recetas con ingredientes disponibles
2. **Mejorar experiencia de usuario**: Recomendaciones personalizadas y relevantes
3. **Facilitar descubrimiento**: Sugerir recetas nuevas basadas en preferencias
4. **Optimizar compras**: Sugerir ingredientes faltantes para completar recetas

---

## 🚀 Mejoras Futuras

### Corto Plazo (1-2 semanas)
- [ ] Agregar filtros por tiempo de preparación
- [ ] Agregar filtros por calorías / dieta
- [ ] Sistema de feedback de usuarios
- [ ] Guardar recetas favoritas

### Mediano Plazo (1-2 meses)
- [ ] API REST con FastAPI
- [ ] Despliegue en cloud (AWS, GCP, Azure)
- [ ] A/B testing de hiperparámetros
- [ ] Reentrenamiento automático

### Largo Plazo (3-6 meses)
- [ ] App móvil (React Native / Flutter)
- [ ] Integración con IoT (cámaras de cocina)
- [ ] Generación automática de listas de compras
- [ ] Modelo de lenguaje para generación de recetas

---

## 📚 Recursos Adicionales

### Documentación
- [README.md](README.md): Documentación técnica completa
- [QUICKSTART.md](QUICKSTART.md): Guía de inicio rápido
- [COLABORADORES.md](COLABORADORES.md): Guía para colaboradores
- [PRESENTACION_EQUIPO.md](PRESENTACION_EQUIPO.md): Presentación ejecutiva

### Notebooks
- `01_foodcom_eda.ipynb`: Análisis exploratorio Food.com
- `02_mm_food_100k_eda.ipynb`: Análisis MM-Food-100K

### Configuraciones
- `configs/recommender_config.yaml`: Hiperparámetros recomendador
- `configs/vision_config.yaml`: Hiperparámetros visión

---

## ✅ Resumen de Comandos

```bash
# PASO 1: Preprocesamiento (YA HECHO ✅)
python -m src.preprocessing.foodcom_processor \
    --recipes data/raw/foodcom/RAW_recipes.csv \
    --interactions data/raw/foodcom/RAW_interactions.csv \
    --output data/processed

# PASO 2: Entrenar Recomendador (SIGUIENTE ⏭️)
python scripts/train_recommender.py \
    --recipes data/processed/recipes_cleaned.parquet \
    --interactions data/processed/interactions_cleaned.parquet \
    --output_dir models/recommender \
    --model_type hybrid

# PASO 3: Probar Recomendador
python test_recommender.py

# PASO 4: Entrenar Visión (OPCIONAL)
python scripts/train_vision_model.py \
    --data_dir data/raw/mm_food_100k/images \
    --metadata data/raw/mm_food_100k/metadata.csv \
    --output_dir models/vision \
    --task dish_classification \
    --device cuda \
    --batch_size 32 \
    --num_workers 6 \
    --epochs 30

# PASO 5: Ejecutar App
streamlit run src/app/streamlit_app.py
```

---

**¡Éxito!** 🎉

Una vez completados estos pasos, tendrás un sistema completo de recomendación de recetas funcionando.

---

**Última actualización**: 22 de Noviembre, 2025
