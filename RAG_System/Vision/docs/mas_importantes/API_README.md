# API de Detección de Ingredientes

API REST para detección automática de ingredientes desde imágenes de comida usando **CLIP + FAISS + XGBoost**.

## 🚀 Inicio Rápido

### 1. Iniciar el servidor

```bash
# Opción 1: Usando el script .bat (Windows)
start_api.bat

# Opción 2: Comando directo
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

El servidor estará disponible en:
- **API**: http://localhost:8000
- **Documentación interactiva**: http://localhost:8000/docs
- **Documentación alternativa**: http://localhost:8000/redoc

### 2. Probar la API

**Opción A: Script de prueba (Python)**

```bash
# Probar con una imagen
python test_api.py path/to/image.jpg

# Con threshold personalizado
python test_api.py path/to/image.jpg --threshold 0.6

# Solo verificar estado
python test_api.py --health
```

**Opción B: Curl**

```bash
curl -X POST "http://localhost:8000/predict" \
  -F "file=@path/to/image.jpg" \
  -F "threshold=0.5"
```

**Opción C: Interfaz web**

Abre http://localhost:8000/docs y usa la interfaz Swagger UI interactiva.

---

## 📡 Endpoints

### `GET /`
Información general de la API

**Response:**
```json
{
  "service": "Ingredient Detection API",
  "version": "2.0.0",
  "status": "running",
  "model_loaded": true,
  "endpoints": {
    "health": "GET /health",
    "predict": "POST /predict"
  }
}
```

### `GET /health`
Estado del sistema y configuración del modelo

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_info": {
    "min_k": 10,
    "max_k": 30,
    "similarity_threshold": 0.6,
    "prediction_threshold": 0.5,
    "num_global_ingredients": 1431
  }
}
```

### `POST /predict`
**Detecta ingredientes desde una imagen**

**Parámetros:**
- `file` (multipart/form-data): Imagen de comida (JPG, PNG, etc.)
- `threshold` (form): Umbral de probabilidad (0.0-1.0, default: 0.5)

**Límites:**
- Tamaño máximo: 10MB
- Tipos permitidos: `image/*`

**Response:**
```json
{
  "success": true,
  "ingredients": [
    "tomato",
    "onion",
    "garlic",
    "olive_oil",
    "salt",
    "basil"
  ],
  "num_detected": 6,
  "probabilities": {
    "tomato": 0.923,
    "onion": 0.876,
    "garlic": 0.834,
    "olive_oil": 0.712,
    "salt": 0.689,
    "basil": 0.567,
    "pepper": 0.423,
    "cheese": 0.389
  },
  "metadata": {
    "k_used": 15,
    "top1_similarity": 0.8456,
    "num_candidates": 47,
    "num_predicted": 6,
    "threshold_used": 0.5
  },
  "processing_time_ms": 234.56
}
```

**Códigos de error:**
- `400`: Archivo inválido o muy grande
- `500`: Error procesando imagen
- `503`: Modelo no cargado

---

## 🧠 Pipeline de Detección

El sistema utiliza un pipeline de **Image Retrieval + ML Scoring**:

```
1. Imagen de entrada
   ↓
2. CLIP ViT-B/32 → Embedding (512 dims)
   ↓
3. FAISS IndexFlatIP → Top-50 imágenes similares
   ↓
4. K adaptativo → Ajusta K entre 10-30 según similitudes
   ↓
5. Feature Engineering → 9 features por ingrediente candidato
   ↓
6. XGBoost Classifier → Probabilidad por ingrediente
   ↓
7. Threshold (default 0.5) → Lista final de ingredientes
```

### Features calculadas (9):
1. `frequency`: Frecuencia en top-K
2. `avg_similarity`: Similitud promedio
3. `top1_similarity`: Similitud del match más cercano
4. `avg_position`: Posición promedio normalizada
5. `max_similarity`: Similitud máxima
6. `presence_ratio`: Proporción de apariciones
7. `std_similarity`: Desviación estándar de similitudes
8. `global_frequency`: Frecuencia global en dataset
9. `neighbor_diversity`: Diversidad de vecinos

### Modelo XGBoost:
- **ROC-AUC**: 0.8410
- **Average Precision**: 0.6369
- **Training**: Hybrid oversampling 1.5x
- **Features**: 9 engineered features
- **Dataset**: MM-Food-100k (57,056 imágenes)

---

## ⚙️ Configuración

El sistema se configura mediante [configs/inference_config.yaml](configs/inference_config.yaml):

```yaml
# Rutas de archivos
faiss_index_path: "data/embeddings/faiss_index.bin"
metadata_path: "data/processed/mm_food_metadata.csv"
scoring_model_path: "models/ingredient_scoring/xgboost_model.json"
embeddings_path: "data/embeddings/clip_embeddings.npy"

# Modelo CLIP
clip_model: "ViT-B-32"
device: "cuda"  # o "cpu"

# K adaptativo (optimizado)
min_k: 10
max_k: 30
similarity_threshold: 0.60

# Threshold de predicción
prediction_threshold: 0.5
```

---

## 🔧 Troubleshooting

### Error: Modelo no cargado (503)

**Causa**: Archivos del modelo no encontrados

**Solución**:
1. Verifica que existan los archivos:
   - `data/embeddings/faiss_index.bin`
   - `data/embeddings/clip_embeddings.npy`
   - `models/ingredient_scoring/xgboost_model.json`
   - `data/processed/mm_food_metadata.csv`

2. Si faltan, ejecuta el pipeline completo:
   ```bash
   # Paso 1-6: Ver README.md principal
   python scripts/prepare_metadata.py ...
   python scripts/create_splits.py ...
   python scripts/generate_embeddings.py ...
   python scripts/build_faiss_index.py ...
   python scripts/prepare_scoring_training_data.py ...
   python scripts/train_scoring_model.py ...
   ```

### Error: CUDA out of memory

**Solución**: Cambiar a CPU en `configs/inference_config.yaml`:
```yaml
device: "cpu"
```

### Predicciones vacías

**Posibles causas**:
1. Threshold muy alto → Prueba con `threshold=0.3`
2. Imagen muy diferente al dataset → Verifica calidad de imagen
3. K adaptativo muy restrictivo → Ajusta `similarity_threshold` en config

### API lenta

**Optimizaciones**:
1. Usa GPU (`device: "cuda"`)
2. Reduce `initial_k_search` en config (default: 50)
3. Implementa caché de embeddings frecuentes

---

## 📊 Ejemplos de Uso

### Python (requests)

```python
import requests

# Predecir ingredientes
with open('pizza.jpg', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/predict',
        files={'file': f},
        data={'threshold': 0.5}
    )

result = response.json()
print(f"Ingredientes: {result['ingredients']}")
print(f"Tiempo: {result['processing_time_ms']} ms")
```

### JavaScript (fetch)

```javascript
const formData = new FormData();
formData.append('file', imageFile);
formData.append('threshold', 0.5);

const response = await fetch('http://localhost:8000/predict', {
  method: 'POST',
  body: formData
});

const result = await response.json();
console.log('Ingredientes:', result.ingredients);
```

### Curl (batch)

```bash
# Procesar múltiples imágenes
for img in *.jpg; do
  echo "Procesando $img..."
  curl -X POST "http://localhost:8000/predict" \
    -F "file=@$img" \
    -F "threshold=0.5" \
    -o "${img%.jpg}_result.json"
done
```

---

## 📈 Métricas del Sistema

- **Precisión**: ~80% (threshold=0.5)
- **Recall**: ~85% (threshold=0.5)
- **F1-Score**: ~82%
- **Latencia promedio**: 200-300ms (GPU), 800-1200ms (CPU)
- **Ingredientes únicos**: 1,431
- **Dataset**: 57,056 imágenes

---

## 🐳 Deployment (opcional)

### Docker

```dockerfile
FROM python:3.9

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

```bash
# Build
docker build -t ingredient-api .

# Run
docker run -p 8000:8000 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/models:/app/models \
  ingredient-api
```

---

## 📚 Documentación Adicional

- **Modelo XGBoost**: [MODELO_XGBOOST.md](MODELO_XGBOOST.md)
- **Proyecto completo**: [PROYECTO_ML_COMPLETO.md](PROYECTO_ML_COMPLETO.md)
- **Arquitectura**: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
- **Pipeline completo**: [README.md](README.md)

---

## 🛠️ Desarrollo

### Agregar logging

```python
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# En el endpoint
logger.info(f"Predicción para {file.filename}: {len(result['ingredients'])} ingredientes")
```

### Agregar caché

```python
from functools import lru_cache

@lru_cache(maxsize=100)
def get_embedding(image_hash):
    # Cachear embeddings
    pass
```

### Métricas de producción

```python
from prometheus_client import Counter, Histogram

predictions_total = Counter('predictions_total', 'Total predictions')
prediction_time = Histogram('prediction_seconds', 'Prediction time')
```

---

**Versión API**: 2.0.0
**Modelo**: XGBoost (ROC-AUC: 0.8410)
**Última actualización**: Noviembre 2025
