# Quickstart Guide

Guía rápida para comenzar a usar el sistema de predicción de ingredientes.

## Requisitos Previos

- Python 3.9+
- CUDA 11.8+ (recomendado para GPU)
- 16GB RAM
- 50GB espacio en disco

## Instalación

### 1. Clonar repositorio y crear entorno

```bash
git clone <repo-url>
cd smart-budget-kitchen

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Descargar datasets

**Food.com**: Descargar de [Kaggle](https://www.kaggle.com/datasets/shuyangli94/food-com-recipes-and-user-interactions) y colocar en `data/raw/foodcom/`

**MM-Food-100K**: Descargar metadata de [Hugging Face](https://huggingface.co/datasets/Codatta/MM-Food-100K) y colocar en `data/raw/mm_food_100k/`

## Pipeline Rápido (Sistema de Visión)

Ejecuta los siguientes comandos en orden:

### Paso 1: Preparar metadata (5 minutos)

```bash
python scripts/prepare_metadata.py
```

### Paso 2: Crear splits (2 minutos)

```bash
python scripts/create_splits.py
```

### Paso 3: Generar embeddings CLIP (2-3 horas GPU, 8 horas CPU)

```bash
python scripts/generate_embeddings.py --device cuda
```

### Paso 4: Construir índice FAISS (15 minutos)

```bash
python scripts/build_faiss_index.py
```

### Paso 5: Generar training data (1-2 horas)

```bash
python scripts/prepare_scoring_training_data.py --max_samples 5000
```

### Paso 6: Entrenar scoring model (15 minutos)

```bash
python scripts/train_scoring_model.py
```

**Tiempo Total**: 3-4 horas en GPU, 10-12 horas en CPU

## Testing

Verificar que el sistema funciona correctamente:

```bash
python scripts/test_inference_system.py
```

Salida esperada: ✅ 6/6 tests passed

## Uso Básico

### Python API

```python
from src.vision.inference import IngredientPredictor

predictor = IngredientPredictor("configs/inference_config.yaml")

result = predictor.predict("my_food_image.jpg")

print(f"Ingredientes: {result['ingredients']}")
print(f"K usado: {result['metadata']['k_used']}")
```

### Batch Processing

```python
image_paths = ["img1.jpg", "img2.jpg", "img3.jpg"]

results = predictor.predict_batch(image_paths)

predictor.save_predictions(results, "predictions.json")
```

## Evaluación

Evaluar en test set (30-60 minutos):

```bash
python scripts/evaluate_system.py --max_samples 1000
```

Resultados guardados en `models/ingredient_scoring/evaluation_results.json`

## Configuración Personalizada

Editar [configs/inference_config.yaml](../configs/inference_config.yaml):

```yaml
# Ajustar thresholds
similarity_threshold: 0.75  # Más restrictivo
prediction_threshold: 0.6   # Más precisión, menos recall

# Ajustar K adaptativo
min_k: 5  # Más evidencia mínima
max_k: 15  # Menos ruido

# Cambiar device
device: "cpu"  # Si no tienes GPU
```

## Troubleshooting

### Error: CUDA out of memory

**Solución**: Reducir batch_size en `generate_embeddings.py`:
```bash
python scripts/generate_embeddings.py --batch_size 32 --device cuda
```

### Error: FAISS index not found

**Solución**: Ejecutar paso 4 (build_faiss_index.py) antes de paso 5

### Error: No module named 'clip'

**Solución**: Instalar CLIP:
```bash
pip install git+https://github.com/openai/CLIP.git
```

### Warnings: Image loading errors

**Normal**: Algunas imágenes pueden estar corruptas o no descargadas. El sistema las salta automáticamente.

## Próximos Pasos

1. Revisar [README.md](../README.md) para arquitectura completa
2. Leer [docs/ARCHITECTURE.md](ARCHITECTURE.md) para detalles técnicos
3. Explorar [deprecated/README.md](../deprecated/README.md) para contexto histórico
4. Ajustar hiperparámetros en validación set
5. Integrar con sistema de recomendación

## Contacto

Para preguntas o issues, abrir ticket en GitHub Issues.
