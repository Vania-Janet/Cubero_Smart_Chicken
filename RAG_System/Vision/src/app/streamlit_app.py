"""
Aplicación web de Smart Budget Kitchen con Streamlit
Interfaz para recomendación de recetas basada en ingredientes e imágenes
"""

import streamlit as st
import pandas as pd
from PIL import Image
import sys
from pathlib import Path

# Agregar src al path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.recommender import ContentBasedRecommender, HybridRecommender
from src.vision.inference import VisionInference


st.set_page_config(
    page_title="Smart Budget Kitchen",
    page_icon="🍳",
    layout="wide"
)


def load_models():
    """Carga modelos de recomendación y visión"""
    if 'content_recommender' not in st.session_state:
        with st.spinner("Cargando modelos..."):
            try:
                st.session_state.content_recommender = ContentBasedRecommender.load(
                    "models/recommender"
                )
                st.success("Modelo de recomendación cargado")
            except Exception as e:
                st.error(f"Error cargando modelo de recomendación: {e}")
                st.session_state.content_recommender = None

    if 'vision_inference' not in st.session_state:
        try:
            st.session_state.vision_inference = VisionInference(
                model_path="models/vision/dish_classifier_best.pth",
                task="dish_classification"
            )
            st.success("Modelo de visión cargado")
        except Exception as e:
            st.warning(f"Modelo de visión no disponible: {e}")
            st.session_state.vision_inference = None


def main():
    """Función principal de la aplicación"""

    st.title("🍳 Smart Budget Kitchen")
    st.markdown("""
    ### Sistema Inteligente de Recomendación de Recetas

    Encuentra las mejores recetas basadas en los ingredientes que tienes disponibles
    o sube una imagen de comida para obtener recomendaciones.
    """)

    load_models()

    st.sidebar.header("⚙️ Configuración")

    mode = st.sidebar.radio(
        "Modo de recomendación:",
        ["Solo Ingredientes", "Imagen + Ingredientes", "Solo Imagen"]
    )

    top_k = st.sidebar.slider(
        "Número de recomendaciones:",
        min_value=5,
        max_value=20,
        value=10,
        step=5
    )

    min_rating = st.sidebar.slider(
        "Rating mínimo:",
        min_value=1.0,
        max_value=5.0,
        value=3.5,
        step=0.5
    )

    st.sidebar.markdown("---")
    st.sidebar.info("""
    **Cómo usar:**
    1. Selecciona el modo de recomendación
    2. Ingresa ingredientes o sube imagen
    3. Haz clic en 'Buscar Recetas'
    4. Explora las recomendaciones
    """)

    ingredients_list = []
    uploaded_image = None

    if mode == "Solo Ingredientes":
        st.header("📝 Ingredientes Disponibles")

        ingredients_text = st.text_area(
            "Ingresa tus ingredientes (uno por línea o separados por comas):",
            height=150,
            placeholder="pollo\ntomate\najo\ncebolla"
        )

        if ingredients_text:
            if '\n' in ingredients_text:
                ingredients_list = [ing.strip() for ing in ingredients_text.split('\n') if ing.strip()]
            else:
                ingredients_list = [ing.strip() for ing in ingredients_text.split(',') if ing.strip()]

            st.write(f"**Ingredientes detectados ({len(ingredients_list)}):**")
            st.write(", ".join(ingredients_list))

    elif mode == "Imagen + Ingredientes":
        col1, col2 = st.columns(2)

        with col1:
            st.header("📸 Imagen de Comida")
            uploaded_image = st.file_uploader(
                "Sube una imagen de comida:",
                type=['jpg', 'jpeg', 'png']
            )

            if uploaded_image:
                image = Image.open(uploaded_image)
                st.image(image, caption="Imagen subida", use_column_width=True)

        with col2:
            st.header("📝 Ingredientes Adicionales")
            ingredients_text = st.text_area(
                "Ingredientes adicionales (opcional):",
                height=150,
                placeholder="arroz\nhuevo"
            )

            if ingredients_text:
                ingredients_list = [ing.strip() for ing in ingredients_text.split('\n') if ing.strip()]

    elif mode == "Solo Imagen":
        st.header("📸 Imagen de Comida")
        uploaded_image = st.file_uploader(
            "Sube una imagen de comida:",
            type=['jpg', 'jpeg', 'png']
        )

        if uploaded_image:
            image = Image.open(uploaded_image)
            st.image(image, caption="Imagen subida", use_column_width=True, width=400)

    st.markdown("---")

    if st.button("🔍 Buscar Recetas", type="primary", use_container_width=True):

        if mode == "Solo Ingredientes" and not ingredients_list:
            st.warning("Por favor ingresa al menos un ingrediente")
            return

        if mode in ["Imagen + Ingredientes", "Solo Imagen"] and uploaded_image is None:
            st.warning("Por favor sube una imagen")
            return

        if st.session_state.content_recommender is None:
            st.error("Modelo de recomendación no disponible")
            return

        detected_dish = None
        detected_ingredients = []

        if uploaded_image and st.session_state.vision_inference:
            with st.spinner("Analizando imagen..."):
                try:
                    image = Image.open(uploaded_image)
                    prediction = st.session_state.vision_inference.predict_image(
                        image,
                        top_k=5
                    )

                    if prediction['task'] == 'dish_classification':
                        detected_dish = prediction['top_prediction']
                        st.info(f"**Platillo detectado:** {detected_dish}")

                        with st.expander("Ver predicciones detalladas"):
                            for pred in prediction['predictions']:
                                st.write(f"- {pred['class']}: {pred['probability']:.2%}")

                except Exception as e:
                    st.error(f"Error procesando imagen: {e}")

        final_ingredients = ingredients_list.copy()
        if detected_ingredients:
            final_ingredients.extend(detected_ingredients)
            final_ingredients = list(set(final_ingredients))

        if not final_ingredients:
            st.warning("No se detectaron ingredientes. Por favor ingresa manualmente.")
            return

        with st.spinner("Buscando recetas..."):
            try:
                recommendations = st.session_state.content_recommender.recommend(
                    ingredients=final_ingredients,
                    top_k=top_k,
                    return_scores=True,
                    return_missing_ingredients=True
                )

                recommendations = recommendations[recommendations['rating_mean'] >= min_rating]

                if len(recommendations) == 0:
                    st.warning("No se encontraron recetas con los criterios especificados")
                    return

                st.success(f"Se encontraron {len(recommendations)} recetas!")

                st.header("🍽️ Recetas Recomendadas")

                for idx, row in recommendations.iterrows():
                    with st.expander(f"**{row['name']}** - Score: {row['similarity_score']:.3f}"):
                        col1, col2, col3 = st.columns(3)

                        with col1:
                            st.metric("Rating Promedio", f"{row['rating_mean']:.1f}/5.0")
                            st.metric("Tiempo", f"{row['minutes']} min")

                        with col2:
                            st.metric("Calorías", f"{row['calories']:.0f}")
                            st.metric("Ingredientes", row['n_ingredients'])

                        with col3:
                            st.metric("Pasos", row['n_steps'])
                            st.metric("Ratings", f"{row['num_ratings']:.0f}")

                        st.markdown("**Ingredientes:**")
                        if isinstance(row['ingredients_normalized'], list):
                            st.write(", ".join(row['ingredients_normalized']))

                        if row['missing_ingredients']:
                            st.markdown("**Ingredientes faltantes:**")
                            st.warning(", ".join(row['missing_ingredients']))

                        if 'description' in row and row['description']:
                            st.markdown("**Descripción:**")
                            st.write(row['description'][:300] + "...")

                st.markdown("---")

                missing_ingredients_freq = {}
                for _, row in recommendations.head(5).iterrows():
                    for ing in row['missing_ingredients']:
                        missing_ingredients_freq[ing] = missing_ingredients_freq.get(ing, 0) + 1

                if missing_ingredients_freq:
                    st.header("🛒 Sugerencias de Compras")
                    st.markdown("Ingredientes que aparecen frecuentemente en las top recetas:")

                    sorted_missing = sorted(
                        missing_ingredients_freq.items(),
                        key=lambda x: x[1],
                        reverse=True
                    )[:10]

                    cols = st.columns(5)
                    for idx, (ing, freq) in enumerate(sorted_missing):
                        with cols[idx % 5]:
                            st.info(f"**{ing}**\n\n{freq} recetas")

            except Exception as e:
                st.error(f"Error generando recomendaciones: {e}")
                import traceback
                st.code(traceback.format_exc())


if __name__ == "__main__":
    main()
