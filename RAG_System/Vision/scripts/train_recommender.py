"""
Script para entrenar sistema de recomendación
"""

import argparse
import pandas as pd
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from src.recommender import ContentBasedRecommender, CollaborativeRecommender, HybridRecommender
from src.utils.config import load_config
from src.utils.logging_utils import setup_logger


def main():
    parser = argparse.ArgumentParser(description="Entrenar sistema de recomendación")
    parser.add_argument(
        "--recipes",
        required=True,
        help="Ruta a recipes_cleaned.parquet"
    )
    parser.add_argument(
        "--interactions",
        required=True,
        help="Ruta a interactions_cleaned.parquet"
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directorio de salida para modelos"
    )
    parser.add_argument(
        "--config",
        default="configs/recommender_config.yaml",
        help="Archivo de configuración"
    )
    parser.add_argument(
        "--model_type",
        choices=["content", "collaborative", "hybrid"],
        default="hybrid",
        help="Tipo de modelo a entrenar"
    )

    args = parser.parse_args()

    logger = setup_logger("train_recommender")
    logger.info("Iniciando entrenamiento de sistema de recomendación")

    config = load_config(args.config)

    logger.info(f"Cargando datos...")
    recipes_df = pd.read_parquet(args.recipes)
    interactions_df = pd.read_parquet(args.interactions)

    logger.info(f"Recetas: {len(recipes_df):,}")
    logger.info(f"Interacciones: {len(interactions_df):,}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.model_type in ["content", "hybrid"]:
        logger.info("Entrenando modelo content-based...")

        content_config = config['content_based']

        content_recommender = ContentBasedRecommender(
            max_features=content_config['vectorizer']['max_features'],
            ngram_range=tuple(content_config['vectorizer']['ngram_range']),
            min_df=content_config['vectorizer']['min_df'],
            max_df=content_config['vectorizer']['max_df'],
            similarity_threshold=content_config['similarity']['threshold']
        )

        content_recommender.fit(recipes_df)
        content_recommender.save(output_dir)

        logger.info("Modelo content-based guardado")

        test_ingredients = ["chicken", "tomato", "garlic", "onion"]
        recommendations = content_recommender.recommend(test_ingredients, top_k=5)
        logger.info(f"\nPrueba de recomendación con {test_ingredients}:")
        logger.info(f"\n{recommendations[['name', 'similarity_score', 'num_missing']].to_string(index=False)}")

    if args.model_type in ["collaborative", "hybrid"]:
        logger.info("Entrenando modelo colaborativo...")

        collab_config = config['collaborative']

        collaborative_recommender = CollaborativeRecommender(
            n_factors=collab_config['n_factors'],
            n_iter=collab_config.get('n_iter', collab_config.get('n_epochs', 20)),
            random_state=collab_config['random_state']
        )

        collaborative_recommender.fit(interactions_df)
        collaborative_recommender.save(str(output_dir))

        logger.info("Modelo colaborativo guardado")

    if args.model_type == "hybrid":
        logger.info("Configurando modelo híbrido...")

        hybrid_config = config['hybrid']

        hybrid_recommender = HybridRecommender(
            content_recommender=content_recommender,
            collaborative_recommender=collaborative_recommender,
            alpha=hybrid_config['weights']['content'],
            beta=hybrid_config['weights']['collaborative'],
            gamma=hybrid_config['weights']['popularity']
        )

        hybrid_recommender.save_weights(output_dir / "hybrid_weights.json")

        logger.info("Configuración híbrida guardada")

    logger.info(f"\nModelos guardados en: {output_dir}")
    logger.info("Entrenamiento completado exitosamente!")


if __name__ == "__main__":
    main()
