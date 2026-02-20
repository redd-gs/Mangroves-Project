# Input Data Guide

This directory contains input data for the Mangroves classification pipeline.

## Directory Structure

- **raw_data/**: External data you must provide (GMW shapefile, GEE credentials)
- **metadata/**: Intermediate metadata files generated during processing
- **dataset/**: AlphaEarth embeddings for model training (244×244×64 tensors)
- **templates/**: Example file formats to help you prepare your data

## Required External Data

1. **GMW v3 2020 Shapefile** → Place in `raw_data/gmw_v3_2020/`
2. **Google Earth Engine Project Key** → Place in `raw_data/gee_credentials/`

See subdirectory READMEs for detailed instructions.

## Data Flow

1. User provides: GMW shapefile + GEE credentials
2. Pipeline generates: metadata CSV files (in metadata/)
3. Pipeline downloads: AlphaEarth embeddings → `dataset/*.npz`
4. Model trains on: `dataset/*.npz` files

## Important Notes

- The model's training data is in `dataset/` (244×244×64 embeddings, ~354 MB)
- Files in `metadata/` are automatically generated during processing
- Never commit GEE credentials to git (protected by .gitignore)
