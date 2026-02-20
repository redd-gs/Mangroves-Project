"""
End-to-end pipeline functions for the Mangrove classification project.

Provides functions for:
- Loading embeddings and labels from .npz files
- Training and evaluating ML baselines (Random Forest, XGBoost)
- Training and evaluating DL models (FCN, CNN, ViT)
- Running inference on embedding tiles
- Generating Grad-CAM visualizations per category
"""

import os
import re
import glob
import logging
import numpy as np
import torch
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from model.model_architectures import Classifier, CNNClassifier, ViTClassifier
from model.training_module import LitModule
from model.grad_cam import GradCAM

logger = logging.getLogger(__name__)

CATEGORY_NAMES = ['1-20%', '21-40%', '41-60%', '61-80%', '81-100%']


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def parse_label(filename: str):
    """Extract 0-indexed label from filename like cat2_id50.0_cov0.30.npz."""
    m = re.match(r'cat(\d+)_id[\d.]+_cov[\d.]+\.npz', filename)
    if m:
        return int(m.group(1)) - 1
    return None


def load_all_embeddings(embedding_dir: str):
    """
    Load all .npz embeddings and their labels from a directory.

    Returns:
        embeddings_list: list of (64, H, W) numpy arrays
        labels: numpy array of 0-indexed labels
        npz_files: list of file paths
    """
    npz_files = sorted(glob.glob(os.path.join(embedding_dir, '*.npz')))
    logger.info(f"Found {len(npz_files)} .npz files in {embedding_dir}")

    embeddings_list = []
    labels_list = []
    valid_files = []

    for fpath in tqdm(npz_files, desc='Loading embeddings'):
        label = parse_label(Path(fpath).name)
        if label is None:
            continue
        data = np.load(fpath)
        key = 'embeddings' if 'embeddings' in data else list(data.keys())[0]
        emb = data[key].astype(np.float32)
        embeddings_list.append(emb)
        labels_list.append(label)
        valid_files.append(fpath)

    labels = np.array(labels_list)
    logger.info(f"Loaded {len(labels)} samples. Distribution:")
    for c, name in enumerate(CATEGORY_NAMES):
        logger.info(f"  {name}: {(labels == c).sum()}")

    return embeddings_list, labels, valid_files


# ---------------------------------------------------------------------------
# ML Baselines (Random Forest, XGBoost)
# ---------------------------------------------------------------------------

def extract_ml_features(embeddings_list: list) -> np.ndarray:
    """Compute mean + std over spatial dims for each band → (N, 128) feature matrix."""
    return np.array([
        np.concatenate([emb.mean(axis=(1, 2)), emb.std(axis=(1, 2))])
        for emb in embeddings_list
    ])


def train_ml_baselines(X_train, y_train, X_test, y_test):
    """
    Train Random Forest and XGBoost on pre-extracted features.

    Returns:
        dict of {model_name: {'accuracy': float, 'f1': float, 'predictions': array}}
    """
    results = {}

    # Random Forest
    logger.info("Training Random Forest...")
    rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_preds = rf.predict(X_test)
    results['Random Forest'] = {
        'type': 'ML',
        'accuracy': accuracy_score(y_test, rf_preds),
        'f1': f1_score(y_test, rf_preds, average='macro'),
        'predictions': rf_preds,
    }
    logger.info(f"  Random Forest — Acc: {results['Random Forest']['accuracy']:.4f}  F1: {results['Random Forest']['f1']:.4f}")

    # XGBoost
    try:
        from xgboost import XGBClassifier
        logger.info("Training XGBoost...")
        xgb = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                            use_label_encoder=False, eval_metric='mlogloss',
                            random_state=42, n_jobs=-1)
        xgb.fit(X_train, y_train)
        xgb_preds = xgb.predict(X_test)
        results['XGBoost'] = {
            'type': 'ML',
            'accuracy': accuracy_score(y_test, xgb_preds),
            'f1': f1_score(y_test, xgb_preds, average='macro'),
            'predictions': xgb_preds,
        }
        logger.info(f"  XGBoost — Acc: {results['XGBoost']['accuracy']:.4f}  F1: {results['XGBoost']['f1']:.4f}")
    except ImportError:
        logger.warning("XGBoost not installed. Skipping. Install with: pip install xgboost")

    return results


# ---------------------------------------------------------------------------
# DL Training (simple loop, no Lightning — for baselines + quick comparison)
# ---------------------------------------------------------------------------

def train_dl_model(model, train_embs, train_labels, test_embs, test_labels,
                   epochs=20, batch_size=8, lr=1e-3, device='cpu'):
    """
    Train a DL model with a simple loop and evaluate on test set.

    Returns:
        (accuracy, f1, predictions_array)
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    n = len(train_embs)
    for epoch in range(epochs):
        model.train()
        perm = np.random.permutation(n)
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            x = torch.stack([torch.from_numpy(train_embs[i]) for i in idx]).to(device)
            y = torch.tensor([train_labels[i] for i in idx], dtype=torch.long).to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()

    model.eval()
    all_preds = []
    with torch.no_grad():
        for start in range(0, len(test_embs), batch_size):
            x = torch.stack([
                torch.from_numpy(test_embs[i])
                for i in range(start, min(start + batch_size, len(test_embs)))
            ]).to(device)
            all_preds.extend(model(x).argmax(dim=1).cpu().numpy())

    preds = np.array(all_preds)
    acc = accuracy_score(test_labels, preds)
    f1 = f1_score(test_labels, preds, average='macro')
    return acc, f1, preds


def train_all_dl_baselines(train_embs, train_labels, test_embs, test_labels,
                           num_classes=5, epochs=20, batch_size=8, lr=1e-3, device='cpu'):
    """
    Train FCN, CNN, and ViT baselines.

    Returns:
        dict of {model_name: {'accuracy', 'f1', 'predictions', 'type'}}
    """
    results = {}
    H, W = train_embs[0].shape[1], train_embs[0].shape[2]

    models_to_train = [
        ('FCN', Classifier(in_channels=64, out_features=num_classes), epochs),
        ('CNN', CNNClassifier(in_channels=64, out_features=num_classes), epochs),
        ('ViT', ViTClassifier(in_channels=64, out_features=num_classes,
                              img_size=H, patch_size=16, embed_dim=128,
                              depth=4, num_heads=4), epochs + 10),
    ]

    for name, model, ep in models_to_train:
        logger.info(f"Training {name}...")
        acc, f1, preds = train_dl_model(
            model, train_embs, train_labels, test_embs, test_labels,
            epochs=ep, batch_size=batch_size, lr=lr, device=device)
        results[name] = {
            'type': 'DL',
            'accuracy': acc,
            'f1': f1,
            'predictions': preds,
        }
        logger.info(f"  {name} — Acc: {acc:.4f}  F1: {f1:.4f}")

    return results


# ---------------------------------------------------------------------------
# Comparison outputs
# ---------------------------------------------------------------------------

def generate_comparison_table(results: dict, output_path: str = None) -> pd.DataFrame:
    """Generate a comparison table from results dict. Optionally save to CSV."""
    rows = []
    for name, r in results.items():
        rows.append({
            'Model': name,
            'Type': r['type'],
            'Accuracy': round(r['accuracy'], 4),
            'Macro-F1': round(r['f1'], 4),
        })
    df = pd.DataFrame(rows).sort_values('Macro-F1', ascending=False).reset_index(drop=True)

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.info(f"Comparison table saved to {output_path}")

    return df


def plot_confusion_matrices(results: dict, y_test, output_path: str):
    """Plot confusion matrices for all models and save to file."""
    models_with_preds = [(name, r['predictions']) for name, r in results.items()
                         if r['predictions'] is not None]

    n = len(models_with_preds)
    if n == 0:
        return

    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]

    for ax, (name, preds) in zip(axes, models_with_preds):
        cm = confusion_matrix(y_test, preds)
        disp = ConfusionMatrixDisplay(cm, display_labels=CATEGORY_NAMES)
        disp.plot(ax=ax, cmap='Blues', values_format='d', colorbar=False)
        ax.set_title(name)

    plt.suptitle('Confusion Matrices — Model Comparison', fontsize=14)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Confusion matrices saved to {output_path}")


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def run_inference(model, embedding_dir: str, device: str = 'cpu') -> pd.DataFrame:
    """
    Run inference on all .npz files in a directory.

    Returns:
        DataFrame with columns: file, predicted_class, predicted_label, confidence, prob_*
    """
    npz_files = sorted(glob.glob(os.path.join(embedding_dir, '*.npz')))
    logger.info(f"Running inference on {len(npz_files)} tiles...")

    model = model.to(device).eval()
    results = []

    with torch.no_grad():
        for fpath in tqdm(npz_files, desc='Inference'):
            data = np.load(fpath)
            key = 'embeddings' if 'embeddings' in data else list(data.keys())[0]
            emb = torch.from_numpy(data[key].astype(np.float32)).unsqueeze(0).to(device)

            logits = model(emb)
            probs = torch.softmax(logits, dim=1)[0]
            pred_class = probs.argmax().item()
            confidence = probs[pred_class].item()

            results.append({
                'file': Path(fpath).name,
                'predicted_class': pred_class,
                'predicted_label': CATEGORY_NAMES[pred_class],
                'confidence': round(confidence, 4),
                **{f'prob_{name}': round(probs[i].item(), 4)
                   for i, name in enumerate(CATEGORY_NAMES)}
            })

    return pd.DataFrame(results)


def plot_prediction_distribution(df: pd.DataFrame, output_path: str):
    """Plot predicted class distribution and confidence histogram."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    df['predicted_label'].value_counts().sort_index().plot.bar(
        ax=axes[0], color='teal', edgecolor='k')
    axes[0].set_title('Predicted Coverage Distribution')
    axes[0].set_ylabel('Count')

    axes[1].hist(df['confidence'], bins=20, color='coral', edgecolor='k')
    axes[1].set_title('Prediction Confidence')
    axes[1].set_xlabel('Confidence')
    axes[1].set_ylabel('Count')

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Prediction distribution saved to {output_path}")


# ---------------------------------------------------------------------------
# Grad-CAM
# ---------------------------------------------------------------------------

def embeddings_to_pseudo_rgb(emb: np.ndarray) -> np.ndarray:
    """Convert (64, H, W) embedding to (H, W, 3) pseudo-RGB with percentile stretching."""
    rgb = emb[:3].transpose(1, 2, 0).copy()
    for c in range(3):
        lo, hi = np.percentile(rgb[:, :, c], [2, 98])
        rgb[:, :, c] = np.clip((rgb[:, :, c] - lo) / (hi - lo + 1e-8), 0, 1)
    return rgb


def overlay_heatmap(rgb: np.ndarray, heatmap: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Overlay a Grad-CAM heatmap on a pseudo-RGB image."""
    cmap = plt.cm.jet
    heatmap_color = cmap(heatmap)[:, :, :3]
    blended = (1 - alpha) * rgb + alpha * heatmap_color
    return np.clip(blended, 0, 1)


def generate_gradcam_per_category(model, embedding_dir: str, device: str,
                                  output_dir: str, model_class: str = 'ViTClassifier'):
    """
    Generate Grad-CAM visualizations: one sample per category, showing all class heatmaps.
    Saves one figure per category to output_dir.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Select target layer
    if model_class == 'ViTClassifier':
        target_layer = model.blocks[-1].norm1
    elif model_class == 'CNNClassifier':
        target_layer = model.features[-3]
    else:
        target_layer = model.net[0]

    gradcam = GradCAM(model.to(device).eval(), target_layer)
    npz_files = sorted(glob.glob(os.path.join(embedding_dir, '*.npz')))

    # Group files by category
    category_files = {c: [] for c in range(5)}
    for fpath in npz_files:
        label = parse_label(Path(fpath).name)
        if label is not None and label < 5:
            category_files[label].append(fpath)

    num_classes = len(CATEGORY_NAMES)

    for cat_idx in range(5):
        if not category_files[cat_idx]:
            continue

        fpath = category_files[cat_idx][0]
        data = np.load(fpath)
        key = 'embeddings' if 'embeddings' in data else list(data.keys())[0]
        emb_np = data[key].astype(np.float32)
        emb_tensor = torch.from_numpy(emb_np).unsqueeze(0).to(device)
        rgb = embeddings_to_pseudo_rgb(emb_np)

        fig, axes = plt.subplots(1, num_classes + 1, figsize=(4 * (num_classes + 1), 4))
        axes[0].imshow(rgb)
        axes[0].set_title('Pseudo-RGB')
        axes[0].axis('off')

        for c in range(num_classes):
            heatmap = gradcam(emb_tensor, target_class=c,
                              img_size=(emb_np.shape[1], emb_np.shape[2]))
            overlay = overlay_heatmap(rgb, heatmap, alpha=0.5)
            axes[c + 1].imshow(overlay)
            axes[c + 1].set_title(f'Class {CATEGORY_NAMES[c]}')
            axes[c + 1].axis('off')

        plt.suptitle(f'Grad-CAM — Category {CATEGORY_NAMES[cat_idx]} — {Path(fpath).stem}', fontsize=13)
        plt.tight_layout()
        out_path = os.path.join(output_dir, f'gradcam_category_{cat_idx + 1}.png')
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"  Saved {out_path}")

    gradcam.remove_hooks()


def generate_gradcam_samples(model, embedding_dir: str, device: str,
                             output_path: str, num_samples: int = 8,
                             model_class: str = 'ViTClassifier'):
    """
    Generate Grad-CAM for N random samples: pseudo-RGB + heatmap + overlay.
    Saves a single multi-row figure.
    """
    if model_class == 'ViTClassifier':
        target_layer = model.blocks[-1].norm1
    elif model_class == 'CNNClassifier':
        target_layer = model.features[-3]
    else:
        target_layer = model.net[0]

    gradcam = GradCAM(model.to(device).eval(), target_layer)
    npz_files = sorted(glob.glob(os.path.join(embedding_dir, '*.npz')))[:num_samples]

    fig, axes = plt.subplots(len(npz_files), 3, figsize=(15, 5 * len(npz_files)))
    if len(npz_files) == 1:
        axes = axes[np.newaxis, :]

    for i, fpath in enumerate(npz_files):
        data = np.load(fpath)
        key = 'embeddings' if 'embeddings' in data else list(data.keys())[0]
        emb_np = data[key].astype(np.float32)
        emb_tensor = torch.from_numpy(emb_np).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(emb_tensor)
            probs = torch.softmax(logits, dim=1)[0]
            pred_class = probs.argmax().item()
            confidence = probs[pred_class].item()

        heatmap = gradcam(emb_tensor, target_class=pred_class,
                          img_size=(emb_np.shape[1], emb_np.shape[2]))
        rgb = embeddings_to_pseudo_rgb(emb_np)
        overlay = overlay_heatmap(rgb, heatmap, alpha=0.5)

        axes[i, 0].imshow(rgb)
        axes[i, 0].set_title(f'Pseudo-RGB — {Path(fpath).stem}')
        axes[i, 0].axis('off')

        im = axes[i, 1].imshow(heatmap, cmap='jet', vmin=0, vmax=1)
        axes[i, 1].set_title(f'Grad-CAM — Pred: {CATEGORY_NAMES[pred_class]} ({confidence:.1%})')
        axes[i, 1].axis('off')
        plt.colorbar(im, ax=axes[i, 1], fraction=0.046)

        axes[i, 2].imshow(overlay)
        axes[i, 2].set_title('Overlay')
        axes[i, 2].axis('off')

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    gradcam.remove_hooks()
    logger.info(f"Grad-CAM samples saved to {output_path}")
