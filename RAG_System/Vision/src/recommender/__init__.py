"""Módulo de sistema de recomendación"""

from .content_based import ContentBasedRecommender
from .collaborative import CollaborativeRecommender
from .hybrid import HybridRecommender

__all__ = ["ContentBasedRecommender", "CollaborativeRecommender", "HybridRecommender"]
