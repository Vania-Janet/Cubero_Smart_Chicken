"""
Sistema de Recomendación Híbrido
Combina content-based, collaborative filtering y popularity scoring
"""

import numpy as np
import pandas as pd
import json
from pathlib import Path
from typing import List, Dict, Optional

from .content_based import ContentBasedRecommender
from .collaborative import CollaborativeRecommender
from ..utils.logging_utils import setup_logger


class HybridRecommender:
    """
    Recomendador híbrido con fusión ponderada

    Combina:
    - Content-based (similitud de ingredientes/tags)
    - Collaborative filtering (patrones de usuarios)
    - Popularity (ratings y número de interacciones)
    """

    def __init__(
        self,
        content_recommender: ContentBasedRecommender,
        collaborative_recommender: Optional[CollaborativeRecommender] = None,
        alpha: float = 0.5,
        beta: float = 0.3,
        gamma: float = 0.2,
        logger_name: str = "hybrid_recommender"
    ):
        """
        Args:
            content_recommender: Modelo content-based entrenado
            collaborative_recommender: Modelo collaborative opcional
            alpha: Peso para content-based score
            beta: Peso para collaborative score
            gamma: Peso para popularity score
        """
        if alpha + beta + gamma != 1.0:
            raise ValueError("Los pesos deben sumar 1.0")

        self.content_recommender = content_recommender
        self.collaborative_recommender = collaborative_recommender
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        self.logger = setup_logger(logger_name)

    def recommend(
        self,
        ingredients: List[str],
        user_id: Optional[int] = None,
        top_k: int = 10,
        min_rating: float = 3.0
    ) -> pd.DataFrame:
        """
        Genera recomendaciones híbridas

        Args:
            ingredients: Lista de ingredientes disponibles
            user_id: ID opcional del usuario para collaborative filtering
            top_k: Número de recomendaciones
            min_rating: Rating mínimo para considerar receta

        Returns:
            pd.DataFrame: Recetas rankeadas por score híbrido
        """
        content_recs = self.content_recommender.recommend(
            ingredients=ingredients,
            top_k=top_k * 3,
            return_scores=True,
            return_missing_ingredients=True
        )

        content_recs = content_recs[content_recs['rating_mean'] >= min_rating]

        scores_content = self._normalize_scores(content_recs['similarity_score'].values)

        scores_collab = np.zeros(len(content_recs))
        if self.collaborative_recommender and user_id:
            for idx, recipe_id in enumerate(content_recs['id']):
                pred = self.collaborative_recommender.predict(user_id, recipe_id)
                scores_collab[idx] = pred

            scores_collab = self._normalize_scores(scores_collab)

        scores_popularity = self._calculate_popularity_scores(content_recs)

        hybrid_scores = (
            self.alpha * scores_content +
            self.beta * scores_collab +
            self.gamma * scores_popularity
        )

        content_recs['hybrid_score'] = hybrid_scores
        content_recs['content_score'] = scores_content
        content_recs['collab_score'] = scores_collab
        content_recs['popularity_score'] = scores_popularity

        final_recs = content_recs.sort_values('hybrid_score', ascending=False).head(top_k)

        return final_recs

    def _calculate_popularity_scores(self, recipes_df: pd.DataFrame) -> np.ndarray:
        """Calcula scores de popularidad normalizados"""
        if 'num_ratings' not in recipes_df.columns or 'rating_mean' not in recipes_df.columns:
            return np.zeros(len(recipes_df))

        log_counts = np.log1p(recipes_df['num_ratings'].values)
        ratings = recipes_df['rating_mean'].values

        popularity = log_counts * (ratings / 5.0)

        return self._normalize_scores(popularity)

    @staticmethod
    def _normalize_scores(scores: np.ndarray) -> np.ndarray:
        """Normaliza scores a rango [0, 1]"""
        if len(scores) == 0:
            return scores

        min_score = scores.min()
        max_score = scores.max()

        if max_score == min_score:
            return np.ones_like(scores) * 0.5

        return (scores - min_score) / (max_score - min_score)

    def save_weights(self, output_path: str) -> None:
        """Guarda pesos de fusión"""
        weights = {
            'alpha': self.alpha,
            'beta': self.beta,
            'gamma': self.gamma
        }

        with open(output_path, 'w') as f:
            json.dump(weights, f, indent=2)

        self.logger.info(f"Pesos guardados en: {output_path}")

    @classmethod
    def load_weights(cls, weights_path: str) -> Dict[str, float]:
        """Carga pesos desde archivo"""
        with open(weights_path, 'r') as f:
            return json.load(f)
