"""
Runs the complete pipeline: train ViT model, train baselines, compare models,
run inference and generate Grad-CAM visualizations.

Usage:
    python -m model.main \
        --credentials config/credentials.yml \
        --datamodule_config config/training_datamodule.yml \
        --litmodule_config config/training_litmodules.yml \
        --trainer_config config/training_trainer.yml \
        --run_baselines \
        --run_gradcam \
        --output_dir output/
"""

import argparse
import os
import logging
import torch
import numpy as np
from ruamel.yaml import YAML
from pytorch_lightning import seed_everything
from sklearn.model_selection import train_test_split

from model.config_loaders import (
    load_datamodule_from_config,
    load_litmodule_from_config,
    load_trainer_from_config,
)
from model.pipeline import (
    load_all_embeddings,
    extract_ml_features,
    train_ml_baselines,
    train_all_dl_baselines,
    generate_comparison_table,
    plot_confusion_matrices,
    run_inference,
    plot_prediction_distribution,
    generate_gradcam_per_category,
    generate_gradcam_samples,
    CATEGORY_NAMES,
)


seed_everything(42, workers=True)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_credentials(path: str) -> dict:
    """Load credentials from the YAML file in the config folder"""
    with open(path, 'r') as f:
        yaml = YAML(typ='safe')
        creds = yaml.load(f)
    logger.info(f"Credentials loaded from {path}")
    return creds


def build_argparser():
    parser = argparse.ArgumentParser(
        description="Mangrove Project — Full Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train ViT only
  python -m model.main --credentials config/credentials.yml \\
      --datamodule_config config/training_datamodule.yml \\
      --litmodule_config config/training_litmodules.yml \\
      --trainer_config config/training_trainer.yml --train

  # Full pipeline with baselines and Grad-CAM
  python -m model.main --credentials config/credentials.yml \\
      --datamodule_config config/training_datamodule.yml \\
      --litmodule_config config/training_litmodules.yml \\
      --trainer_config config/training_trainer.yml \\
      --train --run_baselines --run_gradcam

  # Inference + Grad-CAM only (no training)
  python -m model.main --credentials config/credentials.yml \\
      --datamodule_config config/training_datamodule.yml \\
      --litmodule_config config/training_litmodules.yml \\
      --trainer_config config/training_trainer.yml \\
      --run_inference --run_gradcam
        """
    )

    # Config files
    parser.add_argument("--credentials", default="config/credentials.yml",
                        help="Path to credentials YAML (GEE key, GMW shapefile path)")
    parser.add_argument("--datamodule_config", required=True,
                        help="Path to DataModule config YAML")
    parser.add_argument("--litmodule_config", required=True,
                        help="Path to LitModule config YAML")
    parser.add_argument("--trainer_config", required=True,
                        help="Path to Trainer config YAML")

    # Pipeline stages
    parser.add_argument("--train", action="store_true",
                        help="Train the main ViT model via PyTorch Lightning")
    parser.add_argument("--test", action="store_true",
                        help="Test the main model on the test set")
    parser.add_argument("--run_baselines", action="store_true",
                        help="Train and evaluate baseline models (RF, XGBoost, FCN, CNN, ViT)")
    parser.add_argument("--run_inference", action="store_true",
                        help="Run inference on all embedding tiles")
    parser.add_argument("--run_gradcam", action="store_true",
                        help="Generate Grad-CAM visualizations per category")

    # Options
    parser.add_argument("--output_dir", default="output/",
                        help="Base output directory (default: output/)")
    parser.add_argument("--baseline_epochs", type=int, default=20,
                        help="Number of epochs for baseline DL training (default: 20)")
    parser.add_argument("--gradcam_samples", type=int, default=8,
                        help="Number of random samples for Grad-CAM (default: 8)")

    return parser


def main():
    parser = build_argparser()
    args = parser.parse_args()

    # 0. Setup

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Device: {device}")

    os.makedirs(os.path.join(args.output_dir, 'figures'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'figures', 'gradcam'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'predictions'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'checkpoints'), exist_ok=True)

    # Load credentials
    credentials = load_credentials(args.credentials)
    logger.info(f"GEE project: {credentials.get('gee_project_key', 'N/A')}")
    logger.info(f"GMW shapefile: {credentials.get('gmw_shapefile_path', 'N/A')}")


    # 1. Load DataModule and setup


    logger.info("=" * 60)
    logger.info("STAGE 1: Loading dataset")
    logger.info("=" * 60)

    datamodule = load_datamodule_from_config(args.datamodule_config)
    datamodule.setup()
    logger.info(f"Training set: {len(datamodule.train_dataset)} samples")
    logger.info(f"Validation set: {len(datamodule.val_dataset)} samples")
    logger.info(f"Test set: {len(datamodule.test_dataset)} samples")

    # Read embedding_dir from datamodule config for inference/gradcam
    with open(args.datamodule_config, 'r') as f:
        yaml = YAML(typ='safe')
        dm_config = yaml.load(f)
    embedding_dir = os.path.expanduser(dm_config['path'])

    # 2. Train ViT via PyTorch Lightning

    if args.train:
        logger.info("=" * 60)
        logger.info("STAGE 2: Training main model (PyTorch Lightning)")
        logger.info("=" * 60)

        litmodule = load_litmodule_from_config(args.litmodule_config)
        logger.info(f"Model: {litmodule.net.__class__.__name__} ({litmodule.net.num_params():,} params)")

        trainer = load_trainer_from_config(args.trainer_config)
        logger.info(f"Max epochs: {trainer.max_epochs}")

        trainer.fit(litmodule,
                    datamodule.train_dataloader(),
                    datamodule.val_dataloader())

        logger.info("Training complete. Checkpoint saved.")


    # 3. Test main model


    if args.test:
        logger.info("=" * 60)
        logger.info("STAGE 3: Testing main model")
        logger.info("=" * 60)

        if not args.train:
            litmodule = load_litmodule_from_config(args.litmodule_config)
        trainer = load_trainer_from_config(args.trainer_config)
        trainer.test(litmodule, datamodule.test_dataloader())


    # 4. Baseline comparison


    if args.run_baselines:
        logger.info("=" * 60)
        logger.info("STAGE 4: Baseline model comparison")
        logger.info("=" * 60)

        embeddings_list, labels, _ = load_all_embeddings(embedding_dir)

        if len(embeddings_list) == 0:
            logger.error("No embeddings found. Skipping baselines.")
        else:
            # ML features
            X_ml = extract_ml_features(embeddings_list)
            X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
                X_ml, labels, np.arange(len(labels)),
                test_size=0.2, random_state=42, stratify=labels)
            logger.info(f"Train: {len(X_train)}, Test: {len(X_test)}")

            # ML baselines
            ml_results = train_ml_baselines(X_train, y_train, X_test, y_test)

            # DL baselines
            train_embs = [embeddings_list[i] for i in idx_train]
            test_embs = [embeddings_list[i] for i in idx_test]
            dl_results = train_all_dl_baselines(
                train_embs, y_train, test_embs, y_test,
                num_classes=len(CATEGORY_NAMES),
                epochs=args.baseline_epochs, device=device)

            # Merge results
            all_results = {**ml_results, **dl_results}

            # Comparison table
            table = generate_comparison_table(
                all_results,
                output_path=os.path.join(args.output_dir, 'predictions', 'baseline_comparison.csv'))
            logger.info(f"\n{table.to_string(index=False)}")

            # Confusion matrices
            plot_confusion_matrices(
                all_results, y_test,
                output_path=os.path.join(args.output_dir, 'figures', 'confusion_matrices.png'))


    # 5. Inference


    if args.run_inference:
        logger.info("=" * 60)
        logger.info("STAGE 5: Running inference on all tiles")
        logger.info("=" * 60)

        # Load model for inference
        litmodule = load_litmodule_from_config(args.litmodule_config)
        model = litmodule.net.to(device).eval()

        predictions_df = run_inference(model, embedding_dir, device=device)
        csv_path = os.path.join(args.output_dir, 'predictions', 'predictions.csv')
        predictions_df.to_csv(csv_path, index=False)
        logger.info(f"Saved {len(predictions_df)} predictions to {csv_path}")

        plot_prediction_distribution(
            predictions_df,
            output_path=os.path.join(args.output_dir, 'figures', 'prediction_distribution.png'))


    # 6. Grad-CAM


    if args.run_gradcam:
        logger.info("=" * 60)
        logger.info("STAGE 6: Generating Grad-CAM visualizations")
        logger.info("=" * 60)

        # Load model for Grad-CAM
        litmodule = load_litmodule_from_config(args.litmodule_config)
        model = litmodule.net.to(device).eval()

        # Read model class name from config
        with open(args.litmodule_config, 'r') as f:
            yaml = YAML(typ='safe')
            lit_config = yaml.load(f)
        model_class = lit_config.get('model_options', {}).get('model_class', 'ViTClassifier')

        # Per-category Grad-CAM
        logger.info("Generating per-category Grad-CAM...")
        generate_gradcam_per_category(
            model, embedding_dir, device,
            output_dir=os.path.join(args.output_dir, 'figures', 'gradcam'),
            model_class=model_class)

        # Sample Grad-CAM
        logger.info(f"Generating Grad-CAM for {args.gradcam_samples} random samples...")
        generate_gradcam_samples(
            model, embedding_dir, device,
            output_path=os.path.join(args.output_dir, 'figures', 'gradcam', 'gradcam_samples.png'),
            num_samples=args.gradcam_samples,
            model_class=model_class)


    # Done

    logger.info("Pipeline complete")
    logger.info(f"All outputs saved to: {args.output_dir}")


if __name__ == '__main__':
    main()
