"""
Sistema de Recomendación Colaborativo
Utiliza TruncatedSVD para factorización de matriz usuario-receta
"""

import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import csr_matrix
from typing import Optional, List, Dict, Tuple

from ..utils.logging_utils import setup_logger, LoggerContext


class CollaborativeRecommender:
    """
    Recomendador colaborativo con TruncatedSVD

    Implementa collaborative filtering basado en factorización de matrices
    usando sklearn.decomposition.TruncatedSVD con manejo de biases.
    """

    def __init__(
        self,
        n_factors: int = 100,
        n_iter: int = 20,
        random_state: int = 42,
        logger_name: str = "collaborative_recommender"
    ):
        """
        Args:
            n_factors: Número de factores latentes (componentes SVD)
            n_iter: Número de iteraciones para el algoritmo SVD
            random_state: Semilla aleatoria para reproducibilidad
            logger_name: Nombre del logger
        """
        self.n_factors = n_factors
        self.n_iter = n_iter
        self.random_state = random_state

        # Modelo SVD
        self.svd = TruncatedSVD(
            n_components=n_factors,
            n_iter=n_iter,
            random_state=random_state
        )

        # Mapeos y estadísticas
        self.user_id_map: Dict[int, int] = {}
        self.recipe_id_map: Dict[int, int] = {}
        self.user_id_inverse: Dict[int, int] = {}
        self.recipe_id_inverse: Dict[int, int] = {}

        # Biases para mejorar predicciones
        self.global_mean: float = 0.0
        self.user_biases: Dict[int, float] = {}
        self.item_biases: Dict[int, float] = {}

        # Matrices factorizadas
        self.user_factors: Optional[np.ndarray] = None
        self.item_factors: Optional[np.ndarray] = None

        # Flags de estado
        self.is_fitted: bool = False

        self.logger = setup_logger(logger_name)

    def fit(self, interactions_df: pd.DataFrame) -> None:
        """
        Entrena modelo con interacciones usuario-receta

        Args:
            interactions_df: DataFrame con columnas [user_id, recipe_id, rating]
        """
        with LoggerContext(self.logger, "Entrenando modelo colaborativo con TruncatedSVD"):
            # Validar columnas requeridas
            required_cols = {'user_id', 'recipe_id', 'rating'}
            if not required_cols.issubset(interactions_df.columns):
                raise ValueError(f"DataFrame debe contener columnas: {required_cols}")

            # Crear mapeos de IDs originales a índices consecutivos
            unique_users = interactions_df['user_id'].unique()
            unique_recipes = interactions_df['recipe_id'].unique()

            self.user_id_map = {uid: idx for idx, uid in enumerate(unique_users)}
            self.recipe_id_map = {rid: idx for idx, rid in enumerate(unique_recipes)}
            self.user_id_inverse = {idx: uid for uid, idx in self.user_id_map.items()}
            self.recipe_id_inverse = {idx: rid for rid, idx in self.recipe_id_map.items()}

            n_users = len(self.user_id_map)
            n_recipes = len(self.recipe_id_map)

            self.logger.info(f"Dataset: {n_users:,} usuarios, {n_recipes:,} recetas, {len(interactions_df):,} interacciones")

            # Calcular media global
            self.global_mean = interactions_df['rating'].mean()

            # Calcular biases de usuarios e ítems
            user_means = interactions_df.groupby('user_id')['rating'].mean()
            item_means = interactions_df.groupby('recipe_id')['rating'].mean()

            self.user_biases = {uid: user_means[uid] - self.global_mean for uid in unique_users}
            self.item_biases = {rid: item_means[rid] - self.global_mean for rid in unique_recipes}

            # Construir matriz sparse (usuarios x recetas)
            user_indices = interactions_df['user_id'].map(self.user_id_map).values
            recipe_indices = interactions_df['recipe_id'].map(self.recipe_id_map).values
            ratings = interactions_df['rating'].values

            rating_matrix = csr_matrix(
                (ratings, (user_indices, recipe_indices)),
                shape=(n_users, n_recipes)
            )

            self.logger.info(f"Matriz de ratings: {rating_matrix.shape}, sparsity: {1 - rating_matrix.nnz / (n_users * n_recipes):.4f}")

            # Entrenar SVD
            self.user_factors = self.svd.fit_transform(rating_matrix)
            self.item_factors = self.svd.components_.T

            self.is_fitted = True

            explained_variance = self.svd.explained_variance_ratio_.sum()
            self.logger.info(f"SVD entrenado: {self.n_factors} factores, varianza explicada: {explained_variance:.4f}")

    def predict(self, user_id: int, recipe_id: int) -> float:
        """
        Predice rating para par usuario-receta

        Args:
            user_id: ID original del usuario
            recipe_id: ID original de la receta

        Returns:
            float: Rating predicho (1-5)
        """
        if not self.is_fitted:
            raise ValueError("Modelo no entrenado. Ejecuta fit() primero.")

        # Si el usuario o receta no existen, devolver media global con biases
        if user_id not in self.user_id_map or recipe_id not in self.recipe_id_map:
            prediction = self.global_mean
            if user_id in self.user_biases:
                prediction += self.user_biases[user_id]
            if recipe_id in self.item_biases:
                prediction += self.item_biases[recipe_id]
            return np.clip(prediction, 1.0, 5.0)

        # Obtener índices internos
        user_idx = self.user_id_map[user_id]
        recipe_idx = self.recipe_id_map[recipe_id]

        # Predicción = global_mean + user_bias + item_bias + dot(user_factors, item_factors)
        base_prediction = self.global_mean
        base_prediction += self.user_biases.get(user_id, 0.0)
        base_prediction += self.item_biases.get(recipe_id, 0.0)

        # Producto punto de factores latentes
        latent_score = np.dot(self.user_factors[user_idx], self.item_factors[recipe_idx])

        prediction = base_prediction + latent_score

        # Clip a rango válido [1, 5]
        return np.clip(prediction, 1.0, 5.0)

    def recommend_for_user(
        self,
        user_id: int,
        recipe_ids: List[int],
        top_k: int = 10
    ) -> pd.DataFrame:
        """
        Recomienda recetas para un usuario específico

        Args:
            user_id: ID original del usuario
            recipe_ids: Lista de IDs de recetas candidatas
            top_k: Número de recomendaciones

        Returns:
            pd.DataFrame: Recetas rankeadas por rating predicho
                Columnas: recipe_id, predicted_rating
        """
        if not self.is_fitted:
            raise ValueError("Modelo no entrenado. Ejecuta fit() primero.")

        predictions = []

        for recipe_id in recipe_ids:
            pred_rating = self.predict(user_id, recipe_id)
            predictions.append({
                'recipe_id': recipe_id,
                'predicted_rating': pred_rating
            })

        recommendations_df = pd.DataFrame(predictions)
        recommendations_df = recommendations_df.sort_values(
            'predicted_rating',
            ascending=False
        ).head(top_k).reset_index(drop=True)

        return recommendations_df

    def get_similar_items(
        self,
        recipe_id: int,
        top_k: int = 10,
        metric: str = 'cosine'
    ) -> pd.DataFrame:
        """
        Encuentra recetas similares basadas en factores latentes

        Args:
            recipe_id: ID original de la receta
            top_k: Número de recetas similares
            metric: Métrica de similaridad ('cosine' o 'euclidean')

        Returns:
            pd.DataFrame: Recetas similares con scores
        """
        if not self.is_fitted:
            raise ValueError("Modelo no entrenado. Ejecuta fit() primero.")

        if recipe_id not in self.recipe_id_map:
            self.logger.warning(f"Recipe ID {recipe_id} no encontrado en el modelo")
            return pd.DataFrame(columns=['recipe_id', 'similarity_score'])

        recipe_idx = self.recipe_id_map[recipe_id]
        target_vector = self.item_factors[recipe_idx]

        if metric == 'cosine':
            # Similaridad coseno
            norms = np.linalg.norm(self.item_factors, axis=1)
            target_norm = np.linalg.norm(target_vector)

            similarities = np.dot(self.item_factors, target_vector) / (norms * target_norm + 1e-10)

        elif metric == 'euclidean':
            # Distancia euclidiana (invertida para que mayor sea mejor)
            distances = np.linalg.norm(self.item_factors - target_vector, axis=1)
            similarities = 1.0 / (1.0 + distances)

        else:
            raise ValueError(f"Métrica no soportada: {metric}. Usa 'cosine' o 'euclidean'")

        # Excluir la receta misma
        similarities[recipe_idx] = -np.inf

        # Top-K similares
        top_indices = np.argsort(similarities)[::-1][:top_k]

        similar_recipes = []
        for idx in top_indices:
            similar_recipes.append({
                'recipe_id': self.recipe_id_inverse[idx],
                'similarity_score': similarities[idx]
            })

        return pd.DataFrame(similar_recipes)

    def evaluate(self, test_df: pd.DataFrame) -> Dict[str, float]:
        """
        Evalúa el modelo en un conjunto de test

        Args:
            test_df: DataFrame con columnas [user_id, recipe_id, rating]

        Returns:
            Dict con métricas: rmse, mae
        """
        if not self.is_fitted:
            raise ValueError("Modelo no entrenado. Ejecuta fit() primero.")

        predictions = []
        actuals = []

        for _, row in test_df.iterrows():
            pred = self.predict(row['user_id'], row['recipe_id'])
            predictions.append(pred)
            actuals.append(row['rating'])

        predictions = np.array(predictions)
        actuals = np.array(actuals)

        rmse = np.sqrt(np.mean((predictions - actuals) ** 2))
        mae = np.mean(np.abs(predictions - actuals))

        metrics = {
            'rmse': rmse,
            'mae': mae
        }

        self.logger.info(f"Evaluación - RMSE: {rmse:.4f}, MAE: {mae:.4f}")

        return metrics

    def save(self, save_dir: str) -> None:
        """
        Guarda modelo entrenado en directorio

        Args:
            save_dir: Directorio donde guardar componentes del modelo
        """
        if not self.is_fitted:
            raise ValueError("Modelo no entrenado. Ejecuta fit() primero.")

        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        # Guardar modelo SVD
        joblib.dump(self.svd, save_path / "svd_model.pkl")

        # Guardar mapeos y estadísticas
        metadata = {
            'user_id_map': self.user_id_map,
            'recipe_id_map': self.recipe_id_map,
            'user_id_inverse': self.user_id_inverse,
            'recipe_id_inverse': self.recipe_id_inverse,
            'global_mean': self.global_mean,
            'user_biases': self.user_biases,
            'item_biases': self.item_biases,
            'n_factors': self.n_factors,
            'n_iter': self.n_iter,
            'random_state': self.random_state
        }
        joblib.dump(metadata, save_path / "metadata.pkl")

        # Guardar factores
        np.save(save_path / "user_factors.npy", self.user_factors)
        np.save(save_path / "item_factors.npy", self.item_factors)

        self.logger.info(f"Modelo guardado en: {save_dir}")

    @classmethod
    def load(cls, load_dir: str, logger_name: str = "collaborative_recommender") -> 'CollaborativeRecommender':
        """
        Carga modelo desde directorio

        Args:
            load_dir: Directorio con componentes del modelo
            logger_name: Nombre del logger

        Returns:
            CollaborativeRecommender: Instancia cargada
        """
        load_path = Path(load_dir)

        # Cargar metadata
        metadata = joblib.load(load_path / "metadata.pkl")

        # Crear instancia
        instance = cls(
            n_factors=metadata['n_factors'],
            n_iter=metadata['n_iter'],
            random_state=metadata['random_state'],
            logger_name=logger_name
        )

        # Cargar modelo SVD
        instance.svd = joblib.load(load_path / "svd_model.pkl")

        # Restaurar mapeos y estadísticas
        instance.user_id_map = metadata['user_id_map']
        instance.recipe_id_map = metadata['recipe_id_map']
        instance.user_id_inverse = metadata['user_id_inverse']
        instance.recipe_id_inverse = metadata['recipe_id_inverse']
        instance.global_mean = metadata['global_mean']
        instance.user_biases = metadata['user_biases']
        instance.item_biases = metadata['item_biases']

        # Cargar factores
        instance.user_factors = np.load(load_path / "user_factors.npy")
        instance.item_factors = np.load(load_path / "item_factors.npy")

        instance.is_fitted = True

        instance.logger.info(f"Modelo cargado desde: {load_dir}")

        return instance
