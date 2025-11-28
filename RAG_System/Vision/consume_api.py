"""
Script para consumir la API de Detección de Ingredientes
Para uso de compañeros remotos - No requiere instalación del proyecto

Uso:
    python consume_api.py imagen.jpg
    python consume_api.py imagen.jpg 0.6

Requisitos:
    pip install requests
"""

import requests
import sys
from pathlib import Path

# URL pública de la API (actualizar con tu URL de Render)
API_URL = "https://ingredient-detection-api.onrender.com"

# Para testing local, usa:
# API_URL = "http://localhost:8000"


def predict_ingredients(image_path, threshold=0.5):
    """
    Detecta ingredientes en una imagen usando la API.

    Args:
        image_path: Ruta a la imagen (JPG, PNG, etc.)
        threshold: Umbral de confianza (0.0 - 1.0)
                  Valores más altos = menos ingredientes pero más seguros
                  Valores más bajos = más ingredientes pero menos precisos
    """
    try:
        # Verificar que la imagen existe
        if not Path(image_path).exists():
            print(f"❌ Error: Imagen no encontrada: {image_path}")
            return

        # Abrir y enviar imagen
        with open(image_path, 'rb') as f:
            response = requests.post(
                f'{API_URL}/predict',
                files={'file': (Path(image_path).name, f, 'image/jpeg')},
                data={'threshold': threshold},
                timeout=30  # 30 segundos timeout (cold start puede tardar)
            )

        # Procesar respuesta
        if response.status_code == 200:
            result = response.json()

            print(f"\n{'='*60}")
            print(f"📷 Imagen: {Path(image_path).name}")
            print(f"🎯 Threshold: {threshold}")
            print(f"{'='*60}\n")

            if result['num_detected'] > 0:
                print(f"✅ Detectados {result['num_detected']} ingredientes:\n")

                # Ordenar por probabilidad (mayor a menor)
                ingredients = sorted(
                    result['probabilities'].items(),
                    key=lambda x: x[1],
                    reverse=True
                )

                for ing, prob in ingredients:
                    bar = '█' * int(prob * 20)  # Barra visual
                    print(f"  {bar:<20} {prob:>6.1%}  {ing}")

                print(f"\n⏱️  Tiempo de procesamiento: {result['processing_time_ms']:.0f}ms")

                # Metadata adicional (si disponible)
                if 'metadata' in result and result['metadata']:
                    print(f"\n📊 Metadata:")
                    for key, value in result['metadata'].items():
                        print(f"  - {key}: {value}")
            else:
                print(f"⚠️  No se detectaron ingredientes con threshold {threshold}")
                print(f"💡 Intenta con un threshold más bajo (ej: 0.3)")

            print(f"\n{'='*60}\n")

        elif response.status_code == 503:
            print(f"⏳ La API está iniciando (cold start)...")
            print(f"💡 Espera 30-60 segundos e intenta de nuevo")
        else:
            print(f"❌ Error {response.status_code}")
            print(f"📄 Respuesta: {response.text}")

    except requests.exceptions.Timeout:
        print(f"⏳ Timeout: La API tardó demasiado en responder")
        print(f"💡 Puede ser un cold start. Espera 1 minuto e intenta de nuevo")
    except requests.exceptions.ConnectionError:
        print(f"❌ Error de conexión")
        print(f"💡 Verifica que la URL de la API sea correcta:")
        print(f"   {API_URL}")
    except FileNotFoundError:
        print(f"❌ Error: Imagen no encontrada: {image_path}")
    except Exception as e:
        print(f"❌ Error inesperado: {e}")


def show_help():
    """Muestra información de uso"""
    print("""
╔══════════════════════════════════════════════════════════════╗
║        API de Detección de Ingredientes - Cliente           ║
╚══════════════════════════════════════════════════════════════╝

Uso:
    python consume_api.py <imagen> [threshold]

Argumentos:
    imagen      Ruta a la imagen de comida (JPG, PNG, etc.)
    threshold   (Opcional) Umbral de confianza (0.0 - 1.0)
                Default: 0.5

Ejemplos:
    python consume_api.py comida.jpg
    python consume_api.py comida.jpg 0.6
    python consume_api.py "C:/fotos/comida 123.jpg" 0.4

Threshold recomendado:
    0.3 - 0.4: Detecta muchos ingredientes (menos preciso)
    0.5:       Balanceado (default)
    0.6 - 0.7: Detecta menos ingredientes (más preciso)

Notas:
    - Primera petición puede tardar 30-60s (cold start)
    - Peticiones subsecuentes son más rápidas (<5s)
    - Requiere conexión a internet

URL de la API: {API_URL}
Docs interactivos: {API_URL}/docs
    """.format(API_URL=API_URL))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        show_help()
    else:
        image_path = sys.argv[1]
        threshold = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5

        # Validar threshold
        if not 0.0 <= threshold <= 1.0:
            print("❌ Error: Threshold debe estar entre 0.0 y 1.0")
            sys.exit(1)

        predict_ingredients(image_path, threshold)
