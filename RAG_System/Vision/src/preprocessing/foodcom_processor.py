"""
Procesador de datos de Food.com
Limpieza, transformación y preparación de recetas e interacciones
Incluye detección profesional de outliers con ensemble de 5 métodos
"""

import ast
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime

# Métodos de detección de outliers
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

from ..utils.logging_utils import setup_logger, LoggerContext


class FoodcomProcessor:
    """
    Clase para procesar datos del dataset Food.com

    Funcionalidades:
    - Carga de recetas e interacciones
    - Limpieza de valores nulos y outliers
    - Parseo de campos JSON (tags, ingredientes, nutrition)
    - Detección profesional de outliers con ensemble de 5 métodos
    - Feature engineering
    - Exportación a formato optimizado (Parquet)
    """

    def __init__(
        self,
        recipes_path: str,
        interactions_path: str,
        output_dir: str,
        min_interactions_per_user: int = 3,
        min_interactions_per_recipe: int = 5,
        max_minutes: int = 2880,
        logger_name: str = "foodcom_processor"
    ):
        """
        Args:
            recipes_path: Ruta al archivo RAW_recipes.csv
            interactions_path: Ruta al archivo RAW_interactions.csv
            output_dir: Directorio para guardar datos procesados
            min_interactions_per_user: Mínimo de interacciones por usuario (default: 3)
            min_interactions_per_recipe: Mínimo de interacciones por receta (default: 5)
            max_minutes: Tiempo máximo de preparación permitido (default: 2880 = 48 horas)
            logger_name: Nombre del logger
        """
        self.recipes_path = Path(recipes_path)
        self.interactions_path = Path(interactions_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.min_interactions_per_user = min_interactions_per_user
        self.min_interactions_per_recipe = min_interactions_per_recipe
        self.max_minutes = max_minutes

        self.logger = setup_logger(logger_name)

        self.recipes = None
        self.interactions = None
        self.ingredient_vocab = None

    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Carga datos desde archivos CSV con optimización de memoria

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: (recipes, interactions)
        """
        with LoggerContext(self.logger, "Cargando datos"):
            # Cargar recetas con tipos de datos optimizados
            self.logger.info("Cargando recetas...")
            self.recipes = pd.read_csv(
                self.recipes_path,
                dtype={
                    'id': 'int32',
                    'minutes': 'int32',
                    'contributor_id': 'int32',
                    'n_steps': 'int16',
                    'n_ingredients': 'int16'
                }
            )

            # Cargar interacciones en chunks para evitar memory error
            self.logger.info("Cargando interacciones en chunks...")
            chunk_size = 100000
            chunks = []

            for chunk in pd.read_csv(
                self.interactions_path,
                chunksize=chunk_size,
                dtype={
                    'user_id': 'int32',
                    'recipe_id': 'int32',
                    'rating': 'int8'
                }
            ):
                chunks.append(chunk)
                self.logger.info(f"Chunk cargado: {len(chunk):,} filas")

            self.interactions = pd.concat(chunks, ignore_index=True)
            del chunks  # Liberar memoria

            self.logger.info(f"Recetas cargadas: {len(self.recipes):,}")
            self.logger.info(f"Interacciones cargadas: {len(self.interactions):,}")

        return self.recipes, self.interactions

    def clean_recipes(self) -> pd.DataFrame:
        """
        Limpia dataset de recetas

        Pasos:
        1. Elimina duplicados por ID
        2. Maneja valores nulos
        3. Elimina recetas sin nombre, ingredientes o pasos
        4. Filtra outliers de tiempo (> 48 horas)
        5. Parsea campos JSON
        6. Extrae valores nutricionales
        7. Normaliza ingredientes y tags

        Returns:
            pd.DataFrame: Recetas limpias
        """
        with LoggerContext(self.logger, "Limpiando recetas"):
            df = self.recipes.copy()
            initial_count = len(df)

            # Eliminar duplicados
            df = df.drop_duplicates(subset=['id'], keep='first')
            self.logger.info(f"Duplicados eliminados: {initial_count - len(df)}")

            # Rellenar valores nulos
            df['name'] = df['name'].fillna('Unknown')
            df['description'] = df['description'].fillna('')

            # Filtrar recetas sin nombre válido (además de 'Unknown')
            df = df[df['name'].str.strip() != '']
            self.logger.info(f"Recetas sin nombre eliminadas")

            # Filtrar por tiempo de preparación (máximo 48 horas)
            df = df[df['minutes'] <= self.max_minutes]
            self.logger.info(f"Recetas con tiempo > {self.max_minutes} min eliminadas")

            # Filtrar recetas con 0 minutos de preparación
            df = df[df['minutes'] > 0]
            self.logger.info(f"Recetas con 0 minutos eliminadas")

            # Eliminar recetas sin ingredientes o pasos
            df = df[df['n_ingredients'] > 0]
            df = df[df['n_steps'] > 0]
            self.logger.info(f"Recetas sin ingredientes o pasos eliminadas")

            # Convertir fechas
            df['submitted'] = pd.to_datetime(df['submitted'])

            # Parsear campos JSON
            df['tags_list'] = df['tags'].apply(self._safe_parse_json)
            df['ingredients_list'] = df['ingredients'].apply(self._safe_parse_json)
            df['steps_list'] = df['steps'].apply(self._safe_parse_json)
            df['nutrition_list'] = df['nutrition'].apply(self._safe_parse_json)

            # Extraer valores nutricionales
            nutrition_cols = [
                'calories', 'total_fat_pdv', 'sugar_pdv', 'sodium_pdv',
                'protein_pdv', 'saturated_fat_pdv', 'carbohydrates_pdv'
            ]

            for idx, col in enumerate(nutrition_cols):
                df[col] = df['nutrition_list'].apply(
                    lambda x: x[idx] if isinstance(x, list) and len(x) > idx else np.nan
                )

            # Normalizar ingredientes
            df['ingredients_normalized'] = df['ingredients_list'].apply(
                self._normalize_ingredients
            )

            # Normalizar tags
            df['tags_normalized'] = df['tags_list'].apply(
                lambda tags: [t.lower().strip() for t in tags] if isinstance(tags, list) else []
            )

            df['num_tags'] = df['tags_normalized'].apply(len)

            # Crear campos de texto para búsqueda
            df['ingredients_text'] = df['ingredients_normalized'].apply(
                lambda x: ' '.join(x) if isinstance(x, list) else ''
            )

            df['tags_text'] = df['tags_normalized'].apply(
                lambda x: ' '.join(x) if isinstance(x, list) else ''
            )

            df['content_text'] = df['ingredients_text'] + ' ' + df['tags_text']

            self.logger.info(f"Recetas limpias: {len(df):,}")
            self.recipes = df

        return df

    def clean_interactions(self) -> pd.DataFrame:
        """
        Limpia dataset de interacciones

        Pasos:
        1. Elimina duplicados
        2. Filtra ratings inválidos
        3. Convierte fechas
        4. Limpia texto de reviews
        5. Filtra usuarios y recetas con pocas interacciones (≥3 usuarios, ≥5 recetas)

        Returns:
            pd.DataFrame: Interacciones limpias
        """
        with LoggerContext(self.logger, "Limpiando interacciones"):
            df = self.interactions.copy()
            initial_count = len(df)

            # Eliminar duplicados
            df = df.drop_duplicates(subset=['user_id', 'recipe_id', 'date'], keep='first')
            self.logger.info(f"Duplicados eliminados: {initial_count - len(df)}")

            # Filtrar ratings válidos (1-5)
            df = df[(df['rating'] >= 1) & (df['rating'] <= 5)]

            # Convertir fechas
            df['date'] = pd.to_datetime(df['date'])

            # Limpiar reviews
            df['review'] = df['review'].fillna('').astype(str)
            df['review_length'] = df['review'].str.len()

            # Filtrar usuarios activos (≥3 interacciones)
            user_counts = df['user_id'].value_counts()
            active_users = user_counts[user_counts >= self.min_interactions_per_user].index
            df = df[df['user_id'].isin(active_users)]

            # Filtrar recetas populares (≥5 ratings)
            recipe_counts = df['recipe_id'].value_counts()
            popular_recipes = recipe_counts[recipe_counts >= self.min_interactions_per_recipe].index
            df = df[df['recipe_id'].isin(popular_recipes)]

            self.logger.info(f"Usuarios activos (≥{self.min_interactions_per_user} interacciones): {df['user_id'].nunique():,}")
            self.logger.info(f"Recetas populares (≥{self.min_interactions_per_recipe} ratings): {df['recipe_id'].nunique():,}")
            self.logger.info(f"Interacciones finales: {len(df):,}")

            self.interactions = df

        return df

    # ========== METODOS DE DETECCION DE OUTLIERS ==========

    def _detect_outliers_iqr(self, df: pd.DataFrame, columns: List[str], multiplier: float = 1.5) -> pd.Series:
        """
        Método 1: Detección de outliers usando IQR (Interquartile Range)
        Método de Tukey - robusto y no paramétrico
        """
        outlier_mask = pd.Series(False, index=df.index)

        for col in columns:
            if col not in df.columns:
                continue

            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1

            lower_bound = Q1 - multiplier * IQR
            upper_bound = Q3 + multiplier * IQR

            col_outliers = (df[col] < lower_bound) | (df[col] > upper_bound)
            outlier_mask = outlier_mask | col_outliers

        return outlier_mask

    def _detect_outliers_modified_zscore(self, df: pd.DataFrame, columns: List[str], threshold: float = 3.5) -> pd.Series:
        """
        Método 2: Modified Z-Score basado en MAD (Median Absolute Deviation)
        Más robusto que Z-score clásico ante outliers extremos
        """
        outlier_mask = pd.Series(False, index=df.index)

        for col in columns:
            if col not in df.columns:
                continue

            median = df[col].median()
            mad = np.median(np.abs(df[col] - median))

            if mad != 0:
                modified_z_scores = 0.6745 * (df[col] - median) / mad
                col_outliers = np.abs(modified_z_scores) > threshold
            else:
                col_outliers = pd.Series(False, index=df.index)

            outlier_mask = outlier_mask | col_outliers

        return outlier_mask

    def _detect_outliers_isolation_forest(self, df: pd.DataFrame, columns: List[str], contamination: float = 0.05) -> pd.Series:
        """
        Método 3: Isolation Forest
        Ensemble basado en árboles - detecta outliers multivariados
        """
        X = df[columns].copy()

        # Manejar valores nulos
        X = X.fillna(X.median())

        # Normalizar
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Entrenar Isolation Forest
        iso_forest = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_estimators=100,
            max_samples='auto',
            n_jobs=-1
        )

        predictions = iso_forest.fit_predict(X_scaled)
        outlier_mask = pd.Series(predictions == -1, index=df.index)

        return outlier_mask

    def _detect_outliers_lof(self, df: pd.DataFrame, columns: List[str], n_neighbors: int = 20, contamination: float = 0.05) -> pd.Series:
        """
        Método 4: Local Outlier Factor (LOF)
        Basado en densidad local - detecta outliers contextuales
        """
        X = df[columns].copy()
        X = X.fillna(X.median())

        # Normalizar (importante para LOF)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Entrenar LOF
        lof = LocalOutlierFactor(
            n_neighbors=n_neighbors,
            contamination=contamination,
            metric='euclidean',
            n_jobs=-1
        )

        predictions = lof.fit_predict(X_scaled)
        outlier_mask = pd.Series(predictions == -1, index=df.index)

        return outlier_mask

    def _detect_outliers_dbscan(self, df: pd.DataFrame, columns: List[str], eps: Optional[float] = None, min_samples: int = 5) -> pd.Series:
        """
        Método 5: DBSCAN
        Clustering basado en densidad - identifica puntos de ruido
        """
        X = df[columns].copy()
        X = X.fillna(X.median())

        # Normalizar
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Encontrar eps óptimo si no se proporciona
        if eps is None:
            neighbors = NearestNeighbors(n_neighbors=5)
            neighbors.fit(X_scaled)
            distances, _ = neighbors.kneighbors(X_scaled)
            distances = np.sort(distances[:, -1], axis=0)
            eps = np.percentile(distances, 95)

        # Entrenar DBSCAN
        dbscan = DBSCAN(
            eps=eps,
            min_samples=min_samples,
            metric='euclidean',
            n_jobs=-1
        )

        labels = dbscan.fit_predict(X_scaled)
        outlier_mask = pd.Series(labels == -1, index=df.index)

        return outlier_mask

    def detect_and_remove_outliers(self, min_votes: int = 3, use_dbscan: bool = False) -> pd.DataFrame:
        """
        Detección de outliers usando ensemble de métodos profesionales

        Métodos utilizados:
        1. IQR (Interquartile Range) - Tukey
        2. Modified Z-Score (MAD)
        3. Isolation Forest
        4. Local Outlier Factor (LOF)
        5. DBSCAN (opcional, solo si use_dbscan=True - muy costoso en memoria)

        Args:
            min_votes: Mínimo de métodos que deben marcar como outlier (default: 3)
            use_dbscan: Si True, incluye DBSCAN (requiere mucha RAM, default: False)

        Returns:
            pd.DataFrame: Recetas sin outliers
        """
        num_methods = 5 if use_dbscan else 4
        with LoggerContext(self.logger, f"Detectando outliers con ensemble de {num_methods} métodos"):
            # Columnas numéricas para análisis
            numeric_cols = [
                'minutes', 'n_ingredients', 'n_steps',
                'calories', 'total_fat_pdv', 'sugar_pdv',
                'sodium_pdv', 'protein_pdv', 'saturated_fat_pdv',
                'carbohydrates_pdv'
            ]

            # Filtrar columnas disponibles
            available_cols = [col for col in numeric_cols if col in self.recipes.columns]
            self.logger.info(f"Analizando {len(available_cols)} variables numéricas: {', '.join(available_cols)}")

            # Aplicar los métodos (4 o 5 según use_dbscan)
            self.logger.info("Aplicando Método 1: IQR (Interquartile Range)")
            outliers_iqr = self._detect_outliers_iqr(self.recipes, available_cols, multiplier=1.5)

            self.logger.info("Aplicando Método 2: Modified Z-Score (MAD)")
            outliers_mad = self._detect_outliers_modified_zscore(self.recipes, available_cols, threshold=3.5)

            self.logger.info("Aplicando Método 3: Isolation Forest")
            outliers_iforest = self._detect_outliers_isolation_forest(self.recipes, available_cols, contamination=0.08)

            self.logger.info("Aplicando Método 4: Local Outlier Factor (LOF)")
            outliers_lof = self._detect_outliers_lof(self.recipes, available_cols, n_neighbors=20, contamination=0.05)

            # Crear DataFrame de resultados base
            results = pd.DataFrame({
                'IQR': outliers_iqr,
                'MAD': outliers_mad,
                'IsolationForest': outliers_iforest,
                'LOF': outliers_lof
            })

            methods_list = ['IQR', 'MAD', 'IsolationForest', 'LOF']

            # DBSCAN solo si hay suficiente RAM disponible
            if use_dbscan:
                try:
                    self.logger.info("Aplicando Método 5: DBSCAN (advertencia: muy costoso en RAM)")
                    outliers_dbscan = self._detect_outliers_dbscan(self.recipes, available_cols, min_samples=5)
                    results['DBSCAN'] = outliers_dbscan
                    methods_list.append('DBSCAN')
                except MemoryError:
                    self.logger.warning("DBSCAN omitido por falta de memoria - usando solo 4 métodos")

            # Contar votos
            results['total_votes'] = results[methods_list].sum(axis=1)
            results['is_outlier'] = results['total_votes'] >= min_votes

            # Log de estadísticas
            self.logger.info("=" * 60)
            self.logger.info("RESULTADOS ENSEMBLE OUTLIER DETECTION")
            self.logger.info("=" * 60)

            for method in methods_list:
                count = results[method].sum()
                pct = (count / len(self.recipes)) * 100
                self.logger.info(f"{method:20s}: {count:6d} outliers ({pct:5.2f}%)")

            self.logger.info("-" * 60)
            final_count = results['is_outlier'].sum()
            final_pct = (final_count / len(self.recipes)) * 100
            self.logger.info(f"{'FINAL (>=' + str(min_votes) + ' votes)':20s}: {final_count:6d} outliers ({final_pct:5.2f}%)")
            self.logger.info("=" * 60)

            # Filtrar outliers
            clean_recipes = self.recipes[~results['is_outlier']].copy()

            self.logger.info(f"Recetas originales: {len(self.recipes):,}")
            self.logger.info(f"Outliers eliminados: {final_count:,}")
            self.logger.info(f"Recetas limpias: {len(clean_recipes):,}")

            self.recipes = clean_recipes

        return clean_recipes

    # ========== FIN METODOS DE DETECCION DE OUTLIERS ==========

    def build_ingredient_vocabulary(self) -> Dict[str, int]:
        """
        Construye vocabulario de ingredientes únicos

        Returns:
            Dict[str, int]: Mapeo ingrediente -> frecuencia
        """
        with LoggerContext(self.logger, "Construyendo vocabulario de ingredientes"):
            if self.recipes is None:
                raise ValueError("Primero ejecute clean_recipes()")

            from collections import Counter

            all_ingredients = []
            for ingredients in self.recipes['ingredients_normalized']:
                if isinstance(ingredients, list):
                    all_ingredients.extend(ingredients)

            vocab = Counter(all_ingredients)
            self.ingredient_vocab = dict(vocab.most_common())

            self.logger.info(f"Ingredientes únicos: {len(self.ingredient_vocab):,}")

        return self.ingredient_vocab

    def create_recipe_stats(self) -> pd.DataFrame:
        """
        Crea estadísticas agregadas de recetas

        Returns:
            pd.DataFrame: Recetas con stats de interacciones
        """
        with LoggerContext(self.logger, "Creando estadísticas de recetas"):
            if self.recipes is None or self.interactions is None:
                raise ValueError("Primero ejecute clean_recipes() y clean_interactions()")

            recipe_stats = self.interactions.groupby('recipe_id').agg({
                'rating': ['mean', 'std', 'count'],
                'user_id': 'nunique',
                'date': ['min', 'max']
            }).reset_index()

            recipe_stats.columns = [
                'id', 'rating_mean', 'rating_std', 'num_ratings',
                'num_unique_users', 'first_interaction', 'last_interaction'
            ]

            recipe_stats['rating_std'] = recipe_stats['rating_std'].fillna(0)

            recipe_stats['popularity_score'] = (
                np.log1p(recipe_stats['num_ratings']) * recipe_stats['rating_mean']
            )

            df_with_stats = self.recipes.merge(recipe_stats, on='id', how='left')

            df_with_stats['rating_mean'] = df_with_stats['rating_mean'].fillna(0)
            df_with_stats['num_ratings'] = df_with_stats['num_ratings'].fillna(0)
            df_with_stats['popularity_score'] = df_with_stats['popularity_score'].fillna(0)

            self.logger.info(f"Recetas con estadísticas: {len(df_with_stats):,}")
            self.recipes = df_with_stats

        return df_with_stats

    def save_processed_data(self) -> None:
        """Guarda datos procesados en formato Parquet optimizado"""
        with LoggerContext(self.logger, "Guardando datos procesados"):
            recipes_path = self.output_dir / "recipes_cleaned.parquet"
            self.recipes.to_parquet(recipes_path, index=False, engine='pyarrow')
            self.logger.info(f"Recetas guardadas en: {recipes_path}")

            interactions_path = self.output_dir / "interactions_cleaned.parquet"
            self.interactions.to_parquet(interactions_path, index=False, engine='pyarrow')
            self.logger.info(f"Interacciones guardadas en: {interactions_path}")

            if self.ingredient_vocab:
                vocab_path = self.output_dir / "ingredient_vocab.json"
                with open(vocab_path, 'w', encoding='utf-8') as f:
                    json.dump(self.ingredient_vocab, f, indent=2, ensure_ascii=False)
                self.logger.info(f"Vocabulario guardado en: {vocab_path}")

    def process_all(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Ejecuta pipeline completo de procesamiento

        Pipeline:
        1. Cargar datos
        2. Limpiar recetas (filtros básicos)
        3. Detectar y remover outliers (ensemble de 5 métodos)
        4. Limpiar interacciones
        5. Construir vocabulario de ingredientes
        6. Crear estadísticas de recetas
        7. Guardar datos procesados

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: (recipes, interactions) procesados
        """
        self.logger.info("=" * 60)
        self.logger.info("Iniciando pipeline de procesamiento de Food.com")
        self.logger.info("=" * 60)

        # 1. Cargar datos
        self.load_data()

        # 2. Limpiar recetas (filtros básicos)
        self.clean_recipes()

        # 3. Detectar y remover outliers (ensemble)
        self.detect_and_remove_outliers(min_votes=3)

        # 4. Limpiar interacciones
        self.clean_interactions()

        # 5. Construir vocabulario
        self.build_ingredient_vocabulary()

        # 6. Crear estadísticas
        self.create_recipe_stats()

        # 7. Guardar
        self.save_processed_data()

        self.logger.info("=" * 60)
        self.logger.info("Pipeline completado exitosamente")
        self.logger.info("=" * 60)

        return self.recipes, self.interactions

    @staticmethod
    def _safe_parse_json(value):
        """Parsea JSON de forma segura"""
        if pd.isna(value):
            return []

        if isinstance(value, list):
            return value

        try:
            return ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return []

    @staticmethod
    def _normalize_ingredients(ingredients):
        """Normaliza lista de ingredientes"""
        if not isinstance(ingredients, list):
            return []

        normalized = []
        for ing in ingredients:
            if isinstance(ing, str):
                ing_clean = ing.lower().strip().replace('-', ' ')
                if ing_clean:
                    normalized.append(ing_clean)

        return normalized


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Procesar datos de Food.com con detección profesional de outliers")
    parser.add_argument("--recipes", required=True, help="Ruta a RAW_recipes.csv")
    parser.add_argument("--interactions", required=True, help="Ruta a RAW_interactions.csv")
    parser.add_argument("--output", required=True, help="Directorio de salida")

    args = parser.parse_args()

    processor = FoodcomProcessor(
        recipes_path=args.recipes,
        interactions_path=args.interactions,
        output_dir=args.output
    )

    processor.process_all()
