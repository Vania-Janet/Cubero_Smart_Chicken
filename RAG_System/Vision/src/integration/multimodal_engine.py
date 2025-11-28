"""
Motor de Integración Multimodal
Fusiona recomendador de recetas con modelo de visión computacional
"""

import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
from typing import List, Dict, Optional, Union

from ..recommender.hybrid import HybridRecommender
from ..recommender.content_based import ContentBasedRecommender
from ..vision.inference import VisionInference
from ..utils.logging_utils import setup_logger


class MultimodalEngine:
    """
    Motor principal que integra visión y recomendación

    Modos de operación:
    1. Solo texto: ingredientes -> recomendaciones
    2. Solo imagen: imagen -> ingredientes -> recomendaciones
    3. Multimodal: imagen + ingredientes -> recomendaciones fusionadas
    """

    def __init__(
        self,
        recommender_path: str,
        vision_path: Optional[str] = None,
        device: str = "cuda",
        vision_boost_factor: float = 0.3,
        logger_name: str = "multimodal_engine"
    ):
        """
        Args:
            recommender_path: Directorio con modelo de recomendación
            vision_path: Ruta opcional al modelo de visión
            device: Dispositivo para modelo de visión
            vision_boost_factor: Factor de boost por coincidencia visual
        """
        self.logger = setup_logger(logger_name)

        self.logger.info("Inicializando Motor Multimodal...")

        self.logger.info(f"Cargando recomendador desde {recommender_path}")
        self.recommender = ContentBasedRecommender.load(recommender_path)

        self.vision_inference = None
        if vision_path:
            self.logger.info(f"Cargando modelo de visión desde {vision_path}")
            try:
                self.vision_inference = VisionInference(
                    model_path=vision_path,
                    task="dish_classification",
                    device=device
                )
                self.logger.info("Modelo de visión cargado exitosamente")
            except Exception as e:
                self.logger.warning(f"No se pudo cargar modelo de visión: {e}")
                self.vision_inference = None

        self.vision_boost_factor = vision_boost_factor

        self.logger.info("Motor Multimodal listo")

    def recommend(
        self,
        ingredients: List[str],
        top_k: int = 10,
        min_rating: float = 3.0,
        **kwargs
    ) -> pd.DataFrame:
        """
        Recomienda recetas basadas solo en ingredientes

        Args:
            ingredients: Lista de ingredientes disponibles
            top_k: Número de recomendaciones
            min_rating: Rating mínimo

        Returns:
            pd.DataFrame: Recetas recomendadas
        """
        self.logger.info(f"Modo: Solo ingredientes ({len(ingredients)} ingredientes)")

        recommendations = self.recommender.recommend(
            ingredients=ingredients,
            top_k=top_k * 2,
            return_scores=True,
            return_missing_ingredients=True
        )

        recommendations = recommendations[recommendations['rating_mean'] >= min_rating]

        return recommendations.head(top_k)

    def recommend_from_image(
        self,
        image: Union[Image.Image, str, Path],
        top_k: int = 10,
        min_rating: float = 3.0,
        **kwargs
    ) -> Dict:
        """
        Recomienda recetas basadas solo en imagen

        Args:
            image: Imagen PIL o ruta
            top_k: Número de recomendaciones
            min_rating: Rating mínimo

        Returns:
            Dict con predicciones de visión y recomendaciones
        """
        if self.vision_inference is None:
            raise ValueError("Modelo de visión no disponible")

        self.logger.info("Modo: Solo imagen")

        if isinstance(image, (str, Path)):
            image = Image.open(image)

        vision_result = self.vision_inference.predict_image(image, top_k=5)

        detected_dish = vision_result.get('top_prediction', None)
        detected_ingredients = vision_result.get('detected_ingredients', [])

        self.logger.info(f"Platillo detectado: {detected_dish}")
        self.logger.info(f"Ingredientes detectados: {detected_ingredients}")

        if detected_ingredients:
            recommendations = self.recommender.recommend(
                ingredients=detected_ingredients,
                top_k=top_k * 2,
                return_scores=True,
                return_missing_ingredients=True
            )
        else:
            self.logger.warning("No se detectaron ingredientes, usando todas las recetas")
            recommendations = self.recommender.recipes_df.copy()
            recommendations['similarity_score'] = 0.0
            recommendations['missing_ingredients'] = recommendations['ingredients_normalized']
            recommendations['num_missing'] = recommendations['n_ingredients']

        if detected_dish:
            recommendations = self._boost_by_dish_name(recommendations, detected_dish)

        recommendations = recommendations[recommendations['rating_mean'] >= min_rating]

        result = {
            'vision_predictions': vision_result,
            'detected_dish': detected_dish,
            'detected_ingredients': detected_ingredients,
            'recipes': recommendations.head(top_k)
        }

        return result

    def recommend_multimodal(
        self,
        ingredients: List[str],
        image: Union[Image.Image, str, Path],
        top_k: int = 10,
        min_rating: float = 3.0,
        **kwargs
    ) -> Dict:
        """
        Recomienda recetas fusionando información de ingredientes e imagen

        Args:
            ingredients: Lista de ingredientes disponibles
            image: Imagen PIL o ruta
            top_k: Número de recomendaciones
            min_rating: Rating mínimo

        Returns:
            Dict con predicciones fusionadas y recomendaciones
        """
        if self.vision_inference is None:
            self.logger.warning("Modelo de visión no disponible, usando solo ingredientes")
            return {'recipes': self.recommend(ingredients, top_k, min_rating)}

        self.logger.info("Modo: Multimodal (imagen + ingredientes)")

        if isinstance(image, (str, Path)):
            image = Image.open(image)

        vision_result = self.vision_inference.predict_image(image, top_k=5)

        detected_dish = vision_result.get('top_prediction', None)
        detected_ingredients = vision_result.get('detected_ingredients', [])

        all_ingredients = list(set(ingredients + detected_ingredients))

        self.logger.info(f"Ingredientes del usuario: {ingredients}")
        self.logger.info(f"Ingredientes detectados: {detected_ingredients}")
        self.logger.info(f"Ingredientes fusionados: {all_ingredients}")

        recommendations = self.recommender.recommend(
            ingredients=all_ingredients,
            top_k=top_k * 2,
            return_scores=True,
            return_missing_ingredients=True
        )

        if detected_dish:
            recommendations = self._boost_by_dish_name(recommendations, detected_dish)
            recommendations = recommendations.sort_values('boosted_score', ascending=False)

        recommendations = recommendations[recommendations['rating_mean'] >= min_rating]

        shopping_hints = self._generate_shopping_hints(recommendations.head(top_k))

        result = {
            'vision_predictions': vision_result,
            'detected_dish': detected_dish,
            'detected_ingredients': detected_ingredients,
            'user_ingredients': ingredients,
            'fused_ingredients': all_ingredients,
            'recipes': recommendations.head(top_k),
            'shopping_hints': shopping_hints
        }

        return result

    def _boost_by_dish_name(
        self,
        recommendations: pd.DataFrame,
        detected_dish: str
    ) -> pd.DataFrame:
        """
        Aplica boost a recetas similares al platillo detectado

        Args:
            recommendations: DataFrame de recomendaciones
            detected_dish: Nombre del platillo detectado

        Returns:
            pd.DataFrame: Recomendaciones con boost aplicado
        """
        recommendations = recommendations.copy()

        detected_dish_lower = detected_dish.lower()

        def calculate_name_similarity(recipe_name):
            if pd.isna(recipe_name):
                return 0.0

            recipe_name_lower = recipe_name.lower()

            if detected_dish_lower == recipe_name_lower:
                return 1.0

            if detected_dish_lower in recipe_name_lower or recipe_name_lower in detected_dish_lower:
                return 0.7

            detected_words = set(detected_dish_lower.split())
            recipe_words = set(recipe_name_lower.split())

            if detected_words & recipe_words:
                overlap = len(detected_words & recipe_words)
                return overlap / max(len(detected_words), len(recipe_words))

            return 0.0

        recommendations['name_similarity'] = recommendations['name'].apply(calculate_name_similarity)

        recommendations['boosted_score'] = (
            recommendations['similarity_score'] +
            self.vision_boost_factor * recommendations['name_similarity']
        )

        return recommendations

    def _generate_shopping_hints(
        self,
        top_recipes: pd.DataFrame,
        max_hints: int = 10
    ) -> List[Dict[str, any]]:
        """
        Genera sugerencias de ingredientes para comprar

        Args:
            top_recipes: Top recetas recomendadas
            max_hints: Máximo de sugerencias

        Returns:
            List[Dict]: Ingredientes sugeridos con frecuencia
        """
        missing_freq = {}

        for _, recipe in top_recipes.iterrows():
            if 'missing_ingredients' in recipe and isinstance(recipe['missing_ingredients'], list):
                for ing in recipe['missing_ingredients']:
                    if ing not in missing_freq:
                        missing_freq[ing] = {
                            'ingredient': ing,
                            'frequency': 0,
                            'recipes': []
                        }
                    missing_freq[ing]['frequency'] += 1
                    missing_freq[ing]['recipes'].append(recipe['name'])

        hints = sorted(
            missing_freq.values(),
            key=lambda x: x['frequency'],
            reverse=True
        )[:max_hints]

        return hints

    def get_stats(self) -> Dict:
        """Retorna estadísticas del sistema"""
        stats = {
            'num_recipes': len(self.recommender.recipes_df) if self.recommender else 0,
            'vision_available': self.vision_inference is not None,
            'recommender_type': type(self.recommender).__name__
        }

        return stats


if __name__ == "__main__":
    # Ejemplo de uso
    engine = MultimodalEngine(
        recommender_path="models/recommender",
        vision_path="models/vision/dish_classifier_best.pth"
    )

    # Modo 1: Solo ingredientes
    recs = engine.recommend(
        ingredients=["chicken", "tomato", "garlic"],
        top_k=5
    )
    print("Recomendaciones (solo ingredientes):")
    print(recs[['name', 'similarity_score']].to_string(index=False))

    # Modo 2: Multimodal
    # result = engine.recommend_multimodal(
    #     ingredients=["rice", "egg"],
    #     image="path/to/food_image.jpg",
    #     top_k=5
    # )
