"""
Sistema de recomendación basado en contenido (Content-Based Filtering)
Utiliza TF-IDF sobre ingredientes y tags para calcular similitud
"""

import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix

from ..utils.logging_utils import setup_logger, LoggerContext


class ContentBasedRecommender:
    """
    Recomendador basado en similitud de contenido

    Características:
    - Vectorización TF-IDF de ingredientes y tags
    - Similitud coseno para ranking
    - Filtrado por umbral de similitud
    - Detección de ingredientes faltantes
    """

    def __init__(
        self,
        max_features: int = 5000,
        ngram_range: Tuple[int, int] = (1, 2),
        min_df: int = 2,
        max_df: float = 0.95,
        similarity_threshold: float = 0.3,
        logger_name: str = "content_recommender"
    ):
        """
        Args:
            max_features: Máximo de features en vocabulario
            ngram_range: Rango de n-gramas
            min_df: Frecuencia mínima de documento
            max_df: Frecuencia máxima de documento (proporción)
            similarity_threshold: Umbral mínimo de similitud
            logger_name: Nombre del logger
        """
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.max_df = max_df
        self.similarity_threshold = similarity_threshold

        self.vectorizer = None
        self.tfidf_matrix = None
        self.recipes_df = None
        self.recipe_id_to_idx = None
        self.idx_to_recipe_id = None

        self.logger = setup_logger(logger_name)

    def fit(self, recipes_df: pd.DataFrame, content_column: str = 'content_text') -> None:
        """
        Entrena el modelo con corpus de recetas

        Args:
            recipes_df: DataFrame con recetas
            content_column: Columna con texto de contenido
        """
        with LoggerContext(self.logger, "Entrenando modelo content-based"):
            self.recipes_df = recipes_df.copy()

            if content_column not in recipes_df.columns:
                raise ValueError(f"Columna '{content_column}' no encontrada")

            self.vectorizer = TfidfVectorizer(
                max_features=self.max_features,
                ngram_range=self.ngram_range,
                min_df=self.min_df,
                max_df=self.max_df,
                sublinear_tf=True,
                strip_accents='unicode',
                lowercase=True
            )

            self.tfidf_matrix = self.vectorizer.fit_transform(
                recipes_df[content_column].fillna('')
            )

            self.recipe_id_to_idx = {
                recipe_id: idx for idx, recipe_id in enumerate(recipes_df['id'])
            }
            self.idx_to_recipe_id = {
                idx: recipe_id for recipe_id, idx in self.recipe_id_to_idx.items()
            }

            self.logger.info(f"Recetas indexadas: {len(self.recipes_df):,}")
            self.logger.info(f"Features TF-IDF: {self.tfidf_matrix.shape[1]:,}")
            self.logger.info(f"Sparsity: {(1 - self.tfidf_matrix.nnz / (self.tfidf_matrix.shape[0] * self.tfidf_matrix.shape[1])):.2%}")

    def recommend(
        self,
        ingredients: List[str],
        top_k: int = 10,
        return_scores: bool = True,
        return_missing_ingredients: bool = True
    ) -> pd.DataFrame:
        """
        Recomienda recetas basadas en ingredientes disponibles

        Args:
            ingredients: Lista de ingredientes disponibles
            top_k: Número de recomendaciones a retornar
            return_scores: Si incluir scores de similitud
            return_missing_ingredients: Si incluir ingredientes faltantes

        Returns:
            pd.DataFrame: Recetas recomendadas con metadata
        """
        if self.vectorizer is None or self.tfidf_matrix is None:
            raise ValueError("Modelo no entrenado. Ejecute fit() primero")

        query_text = ' '.join([ing.lower().strip() for ing in ingredients])
        query_vector = self.vectorizer.transform([query_text])

        similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()

        mask = similarities >= self.similarity_threshold
        filtered_indices = np.where(mask)[0]

        if len(filtered_indices) == 0:
            self.logger.warning("No se encontraron recetas similares. Bajando umbral.")
            filtered_indices = np.arange(len(similarities))

        top_indices = filtered_indices[np.argsort(similarities[filtered_indices])[::-1][:top_k]]

        recommendations = self.recipes_df.iloc[top_indices].copy()

        if return_scores:
            recommendations['similarity_score'] = similarities[top_indices]

        if return_missing_ingredients:
            user_ingredients_set = set(ing.lower().strip() for ing in ingredients)
            recommendations['missing_ingredients'] = recommendations['ingredients_normalized'].apply(
                lambda recipe_ings: self._get_missing_ingredients(user_ingredients_set, recipe_ings)
            )
            recommendations['num_missing'] = recommendations['missing_ingredients'].apply(len)
            recommendations['ingredient_match_ratio'] = recommendations.apply(
                lambda row: (row['n_ingredients'] - row['num_missing']) / row['n_ingredients']
                if row['n_ingredients'] > 0 else 0,
                axis=1
            )

        return recommendations

    def recommend_similar_recipes(
        self,
        recipe_id: int,
        top_k: int = 10,
        return_scores: bool = True
    ) -> pd.DataFrame:
        """
        Recomienda recetas similares a una receta dada

        Args:
            recipe_id: ID de la receta de referencia
            top_k: Número de recomendaciones
            return_scores: Si incluir scores de similitud

        Returns:
            pd.DataFrame: Recetas similares
        """
        if recipe_id not in self.recipe_id_to_idx:
            raise ValueError(f"Recipe ID {recipe_id} no encontrado")

        recipe_idx = self.recipe_id_to_idx[recipe_id]
        recipe_vector = self.tfidf_matrix[recipe_idx]

        similarities = cosine_similarity(recipe_vector, self.tfidf_matrix).flatten()

        top_indices = np.argsort(similarities)[::-1][1:top_k+1]

        recommendations = self.recipes_df.iloc[top_indices].copy()

        if return_scores:
            recommendations['similarity_score'] = similarities[top_indices]

        return recommendations

    def save(self, save_dir: str) -> None:
        """Guarda modelo entrenado"""
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        joblib.dump(self.vectorizer, save_path / "tfidf_vectorizer.pkl")
        joblib.dump(self.tfidf_matrix, save_path / "tfidf_matrix.pkl")
        joblib.dump(self.recipe_id_to_idx, save_path / "recipe_id_to_idx.pkl")
        joblib.dump(self.idx_to_recipe_id, save_path / "idx_to_recipe_id.pkl")

        self.recipes_df.to_parquet(save_path / "recipes_indexed.parquet")

        self.logger.info(f"Modelo guardado en: {save_path}")

    @classmethod
    def load(cls, load_dir: str, logger_name: str = "content_recommender") -> 'ContentBasedRecommender':
        """Carga modelo desde disco"""
        load_path = Path(load_dir)

        instance = cls(logger_name=logger_name)

        instance.vectorizer = joblib.load(load_path / "tfidf_vectorizer.pkl")
        instance.tfidf_matrix = joblib.load(load_path / "tfidf_matrix.pkl")
        instance.recipe_id_to_idx = joblib.load(load_path / "recipe_id_to_idx.pkl")
        instance.idx_to_recipe_id = joblib.load(load_path / "idx_to_recipe_id.pkl")
        instance.recipes_df = pd.read_parquet(load_path / "recipes_indexed.parquet")

        instance.logger.info(f"Modelo cargado desde: {load_path}")

        return instance

    @staticmethod
    def _get_missing_ingredients(user_ingredients: set, recipe_ingredients: List[str]) -> List[str]:
        """Calcula ingredientes faltantes"""
        if not isinstance(recipe_ingredients, list):
            return []

        recipe_set = set(ing.lower().strip() for ing in recipe_ingredients)
        missing = recipe_set - user_ingredients

        return sorted(list(missing))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Entrenar recomendador content-based")
    parser.add_argument("--recipes", required=True, help="Ruta a recipes_cleaned.parquet")
    parser.add_argument("--output", required=True, help="Directorio de salida")

    args = parser.parse_args()

    recipes_df = pd.read_parquet(args.recipes)

    recommender = ContentBasedRecommender()
    recommender.fit(recipes_df)
    recommender.save(args.output)

    print("\nPrueba de recomendación:")
    test_ingredients = ["chicken", "tomato", "garlic", "onion"]
    recommendations = recommender.recommend(test_ingredients, top_k=5)
    print(recommendations[['name', 'similarity_score', 'num_missing']].to_string(index=False))
