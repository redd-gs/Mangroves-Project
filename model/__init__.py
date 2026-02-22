"""
Core Python package for mangrove mapping with AlphaEarth embeddings.

Modules:
    constants : Project-wide constants (resolution, bands, etc.)

    Geospatial & Data Acquisition:
    geo_utilities : Geographic region / bounding-box utilities
    math_utilities : Haversine distance, geodesic circles, etc.
    gee_collection : Google Earth Engine interface for satellite embeddings
    embedding_manager : Embedding download, save and load

    Data Loading & Preprocessing:
    dataset : PyTorch Dataset and DataModule classes
    data_transforms : Data augmentation transforms

    Models & Architectures:
    model_architectures : Classifier, CNNClassifier and ViTClassifier

    Training:
    training_module : PyTorch Lightning module wrapper
    config_loaders : YAML configuration loaders
    train : Main training entry point

    Interpretability:
    grad_cam : Grad-CAM visualization for CNN and ViT models
"""
