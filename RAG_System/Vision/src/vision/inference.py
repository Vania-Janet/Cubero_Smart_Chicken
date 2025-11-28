"""
Pipeline completo de inferencia para predicción de ingredientes
Integra retrieval (CLIP + FAISS) + scoring model (XGBoost)
"""

import torch
import xgboost as xgb
import pandas as pd
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Dict, List, Optional
import yaml
import json
from collections import Counter

from .retrieval import ImageRetriever, FeatureEngineer


class IngredientPredictor:
    """
    Predictor completo de ingredientes

    Pipeline:
    1. Query image → CLIP embedding
    2. FAISS search → top-K similar images
    3. K adaptativo → ajusta K según similitudes
    4. Extract candidates → lista de ingredientes
    5. Feature engineering → calcula 9 features
    6. XGBoost scoring → probabilidades
    7. Threshold → lista final de ingredientes
    """

    def __init__(self, config_path: str):
        """
        Args:
            config_path: Ruta al archivo de configuración YAML
        """
        print(f"Cargando configuración desde: {config_path}")
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Cargar embeddings para neighbor_diversity feature
        embeddings_path = self.config.get('embeddings_path', 'data/embeddings/clip_embeddings.npy')
        print(f"Cargando embeddings: {embeddings_path}")
        self.embeddings = np.load(embeddings_path)
        print(f"  Embeddings shape: {self.embeddings.shape}")

        # Calcular frecuencias globales
        print(f"Calculando frecuencias globales...")
        self.global_frequencies = self._compute_global_frequencies(
            self.config['metadata_path']
        )

        self.retriever = ImageRetriever(
            faiss_index_path=self.config['faiss_index_path'],
            metadata_path=self.config['metadata_path'],
            model_name=self.config.get('clip_model', 'ViT-B-32'),
            device=self.config.get('device', 'cuda')
        )

        print(f"Cargando scoring model: {self.config['scoring_model_path']}")
        self.scoring_model = xgb.Booster()
        self.scoring_model.load_model(self.config['scoring_model_path'])

        self.min_k = self.config.get('min_k', 3)
        self.max_k = self.config.get('max_k', 20)
        self.similarity_threshold = self.config.get('similarity_threshold', 0.70)
        self.prediction_threshold = self.config.get('prediction_threshold', 0.5)

        print("\nIngredientPredictor inicializado")
        print(f"  K adaptativo: min={self.min_k}, max={self.max_k}")
        print(f"  Similarity threshold: {self.similarity_threshold}")
        print(f"  Prediction threshold: {self.prediction_threshold}")
        print(f"  Ingredientes globales: {len(self.global_frequencies):,}")

    def _compute_global_frequencies(self, metadata_path: str) -> Dict[str, int]:
        """
        Calcula frecuencia global de cada ingrediente

        Args:
            metadata_path: Ruta al metadata

        Returns:
            Dict {ingrediente: frecuencia}
        """
        metadata_df = pd.read_csv(metadata_path)
        all_ingredients = []

        for idx, row in metadata_df.iterrows():
            ingredients_str = row.get('ingredients_list', '[]')
            try:
                if isinstance(ingredients_str, str):
                    # Primero intentar con JSON válido
                    try:
                        ingredients = json.loads(ingredients_str)
                    except json.JSONDecodeError:
                        # Si falla, usar ast.literal_eval para listas de Python
                        import ast
                        ingredients = ast.literal_eval(ingredients_str)
                elif isinstance(ingredients_str, list):
                    ingredients = ingredients_str
                else:
                    ingredients = []
            except (json.JSONDecodeError, ValueError, SyntaxError, TypeError):
                ingredients = []

            all_ingredients.extend(ingredients)

        return dict(Counter(all_ingredients))

    def predict(
        self,
        image_path: str,
        threshold: Optional[float] = None,
        return_probabilities: bool = True
    ) -> Dict:
        """
        Predice ingredientes para una imagen

        Args:
            image_path: Ruta a la imagen
            threshold: Threshold de predicción (override config)
            return_probabilities: Si retornar probabilidades

        Returns:
            Dict con keys:
            - ingredients: List[str] (para RAG)
            - probabilities: Dict[str, float] (opcional)
            - metadata: Dict con info adicional
        """
        if threshold is None:
            threshold = self.prediction_threshold

        image = Image.open(image_path).convert('RGB')

        similarities, indices = self.retriever.search(image, k=50)

        k, top_k_sims = self.retriever.get_adaptive_k(
            similarities,
            min_k=self.min_k,
            max_k=self.max_k,
            threshold=self.similarity_threshold
        )

        top_k_indices = indices[:k]

        candidates = self.retriever.extract_candidate_ingredients(top_k_indices)

        if len(candidates) == 0:
            return {
                'ingredients': [],
                'probabilities': {},
                'metadata': {
                    'k_used': k,
                    'top1_similarity': float(similarities[0]),
                    'num_candidates': 0,
                    'message': 'No candidates found'
                }
            }

        features_df = FeatureEngineer.compute_features_batch(
            ingredients=candidates,
            top_k_indices=top_k_indices,
            top_k_similarities=top_k_sims,
            ingredients_cache=self.retriever.ingredients_cache,
            global_frequencies=self.global_frequencies,
            embeddings=self.embeddings
        )

        # Usar las 9 features en orden correcto
        feature_columns = [
            'frequency', 'avg_similarity', 'top1_similarity',
            'avg_position', 'max_similarity', 'presence_ratio',
            'std_similarity', 'global_frequency', 'neighbor_diversity'
        ]

        # Crear DMatrix con feature names
        X = features_df[feature_columns].values
        dmatrix = xgb.DMatrix(X, feature_names=feature_columns)
        probabilities_array = self.scoring_model.predict(dmatrix)

        probabilities = dict(zip(candidates, probabilities_array))

        ingredients = [
            ing for ing, prob in probabilities.items()
            if prob > threshold
        ]

        ingredients_sorted = sorted(
            ingredients,
            key=lambda x: probabilities[x],
            reverse=True
        )

        result = {
            'ingredients': ingredients_sorted,
            'metadata': {
                'k_used': k,
                'top1_similarity': float(similarities[0]),
                'num_candidates': len(candidates),
                'num_predicted': len(ingredients_sorted),
                'threshold_used': threshold
            }
        }

        if return_probabilities:
            result['probabilities'] = {
                ing: float(prob)
                for ing, prob in sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
            }

        return result

    def predict_batch(
        self,
        image_paths: List[str],
        threshold: Optional[float] = None
    ) -> List[Dict]:
        """
        Predice ingredientes para un batch de imágenes

        Args:
            image_paths: Lista de rutas de imágenes
            threshold: Threshold de predicción

        Returns:
            Lista de resultados
        """
        results = []
        for image_path in image_paths:
            result = self.predict(image_path, threshold=threshold)
            result['image_path'] = image_path
            results.append(result)

        return results

    def save_predictions(
        self,
        results: List[Dict],
        output_path: str
    ):
        """
        Guarda predicciones en JSON

        Args:
            results: Lista de resultados de predict()
            output_path: Ruta para guardar JSON
        """
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"Predicciones guardadas en: {output_path}")
