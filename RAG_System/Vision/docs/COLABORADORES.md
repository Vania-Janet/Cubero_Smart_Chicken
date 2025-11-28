# Guía para Colaboradores - Smart Budget Kitchen

Esta guía explica cómo configurar y trabajar en el proyecto para nuevos colaboradores.

## Configuración Inicial (Solo la primera vez)

### 1. Clonar/Descargar el Proyecto

```bash
# Si usas Git
git clone <url-del-repositorio>
cd "Proyecto ML plus"

# Si descargaste un ZIP
# Descomprime y navega a la carpeta
cd "Proyecto ML plus"
```

### 2. Crear Entorno Virtual

**Importante**: El proyecto usa Python 3.9.13 con el entorno llamado `appComida`

```bash
# Crear entorno virtual
py -3.9 -m venv appComida

# Activar entorno
appComida\Scripts\activate

# Verificar Python
python --version
# Debe mostrar: Python 3.9.13
```

### 3. Instalar Dependencias

```bash
# Instalar todas las dependencias
pip install -r requirements.txt

# Instalar proyecto en modo desarrollo
pip install -e .
```

### Al Iniciar tu Sesión de Trabajo

```bash
# 1. Navegar al proyecto
cd "C:\Users\Jhoshua\Downloads\Proyecto ML plus"

# 2. Activar entorno virtual
appComida\Scripts\activate

# 3. Verificar que estás en el entorno correcto
python --version
# Debe mostrar: Python 3.9.13

# 4. Si usas Git, actualizar repositorio
git pull origin main
```

### Durante el Desarrollo

#### Ejecutar la aplicación para pruebas
```bash
streamlit run src/app/streamlit_app.py
```

#### Entrenar modelos
```bash
# Sistema de recomendación
python scripts/train_recommender.py --recipes data/processed/recipes_cleaned.parquet --interactions data/processed/interactions_cleaned.parquet --output_dir models/recommender --model_type hybrid

# Modelo de visión (con GPU)
python scripts/train_vision_model.py --data_dir data/raw/mm_food_100k/images --metadata data/raw/mm_food_100k/metadata.csv --output_dir models/vision --task dish_classification --device cuda --batch_size 32 --num_workers 6
```


### Al Finalizar tu Sesión

```bash
# Desactivar entorno virtual
deactivate
```

## Estructura del Proyecto

```
Proyecto ML plus/
├── data/                    # Datos (ignorado en Git)
│   ├── raw/                 # Datos originales
│   │   ├── foodcom/        # Dataset Food.com
│   │   └── mm_food_100k/   # Imágenes
│   ├── processed/           # Datos procesados
│   └── splits/              # Train/val/test splits
│
├── src/                     # Código fuente
│   ├── preprocessing/       # Limpieza y preprocesamiento
│   ├── recommender/         # Sistema de recomendación
│   ├── vision/              # Modelos de visión
│   ├── integration/         # Integración multimodal
│   ├── app/                 # Aplicación web
│   └── utils/               # Utilidades
│
├── scripts/                 # Scripts de entrenamiento
│   ├── train_recommender.py
│   ├── train_vision_model.py
│   └── download_images.py
│
├── configs/                 # Configuraciones YAML
│   ├── recommender_config.yaml
│   └── vision_config.yaml
│
├── models/                  # Modelos entrenados (ignorado en Git)
│   ├── recommender/
│   └── vision/
│
├── notebooks/               # Notebooks de análisis
│   ├── 01_foodcom_eda.ipynb
│   └── 02_mm_food_100k_eda.ipynb
│
├── tests/                   # Tests unitarios
│
├── requirements.txt         # Dependencias Python
├── setup.py                 # Configuración del paquete
├── README.md                # Documentación completa
├── QUICKSTART.md            # Guía de inicio rápido
└── COLABORADORES.md         # Esta guía
```

## Buenas Prácticas

### 1. Manejo de Entornos

- **Siempre activa** el entorno `appComida` antes de trabajar
- **Nunca instales paquetes** globalmente, usa el entorno virtual
- Si necesitas nuevas dependencias:
  ```bash
  pip install <paquete>
  pip freeze > requirements.txt  # Actualizar requirements
  ```

### 2. Git (Si aplica)

```bash
# Antes de hacer cambios
git pull origin main

# Crear rama para nueva feature
git checkout -b feature/mi-nueva-funcionalidad

# Hacer commits frecuentes
git add .
git commit -m "Descripción clara del cambio"

# Push a tu rama
git push origin feature/mi-nueva-funcionalidad

# Crear Pull Request en GitHub
```

### 3. Código

- **Sigue PEP 8** para estilo de Python
- **Documenta funciones** con docstrings
- **Prueba tu código** antes de commitear
- **No subas a Git**:
  - Datos grandes (`data/`)
  - Modelos entrenados (`models/`)
  - Entorno virtual (`appComida/`)
  - Archivos temporales

### 4. Configuraciones

- Modifica hiperparámetros en archivos YAML (`configs/`)
- No hardcodees valores, usa los archivos de configuración
- Ejemplo:
  ```python
  from src.utils.config import load_config
  config = load_config("configs/recommender_config.yaml")
  n_factors = config['collaborative']['n_factors']
  ```

## Comandos Útiles

### Limpiar cache y archivos temporales
```bash
# Python cache
find . -type d -name "__pycache__" -exec rm -rf {} +
find . -type f -name "*.pyc" -delete

# Jupyter checkpoints
find . -type d -name ".ipynb_checkpoints" -exec rm -rf {} +
```

### Verificar instalación
```python
# En Python
from src.recommender import ContentBasedRecommender, CollaborativeRecommender
from src.vision.models import FoodClassifier
print("✅ Importaciones correctas")
```

### Ejecutar tests
```bash
pytest tests/ -v
```

### Ver logs detallados
```bash
# En scripts de entrenamiento, los logs se muestran en consola
python scripts/train_recommender.py ... 2>&1 | tee logs/training.log
```

## GPU RTX - Configuración

Si tienes GPU RTX con 7 núcleos CUDA:

- **Usa 6 workers** en comandos paralelos (dejar 1 libre)
- **Batch size recomendado**: 32 (reducir a 16 u 8 si hay errores de memoria)
- **Device**: `--device cuda`

Ejemplo:
```bash
python scripts/train_vision_model.py \
    --device cuda \
    --batch_size 32 \
    --num_workers 6 \
    --data_dir data/raw/mm_food_100k/images \
    --output_dir models/vision \
    --task dish_classification
```

## Solución de Problemas Comunes

### Problema: "Python no encontrado"
```bash
# Verifica que Python 3.9.13 esté instalado
py -3.9 --version

# Si no está, descárgalo de python.org
```

### Problema: "No module named 'src'"
```bash
# Reinstalar proyecto
pip install -e .
```

### Problema: "CUDA out of memory"
```bash
# Reducir batch_size
python scripts/train_vision_model.py ... --batch_size 16
```

### Problema: "Paquete no se instala"
```bash
# Actualizar pip
python -m pip install --upgrade pip

# Intentar instalar de nuevo
pip install <paquete>
```

## Recursos Adicionales

- **Documentación completa**: [README.md](README.md)
- **Inicio rápido**: [QUICKSTART.md](QUICKSTART.md)
- **Notebooks de EDA**: `notebooks/`
- **Configuraciones**: `configs/`

## Contacto y Soporte

Si encuentras problemas o tienes preguntas:

1. Revisa esta guía y el README.md
2. Consulta los notebooks de EDA
3. Verifica los logs en consola
4. Contacta al equipo del proyecto

---

**Última actualización**: Noviembre 2025
