# Hybrid_Conditional_VAE

This folder contains a hybrid conditional VAE for generating `64 x 64 x 64` voxel grids from text, with an optional refinement path that also uses three 2D orthographic slice images.

The README is based on:

- `Cond_VAE/train_hybrid_vae.py`
- `Cond_VAE/inference_hybrid_vae.py`

## Main files

- `train_hybrid_vae.py`: training script and model definition
- `inference_hybrid_vae.py`: inference script for text-only or text-plus-slices generation

## What the model uses

- 3D voxel input: `(1, 64, 64, 64)` during training
- 2D slice input: three grayscale JPG slices resized to `256 x 256`
- Text input: `SentenceTransformer("all-MiniLM-L6-v2")` embeddings
- Latent dimension: `128`

At training time, the encoder uses voxel + slices + text.  
At text-only inference time, the model samples from a learned text prior and decodes directly to a voxel grid.

## Dataset paths used by `train_hybrid_vae.py`

These paths are currently hardcoded in the script:

- `data/datasets/captions_labeled.csv`
- `data/datasets/geometries_2000x64x64x64.npy`
- `data/datasets/2d_Slices/`

Expected slice naming format:

- `{sample_id}_0.jpg`
- `{sample_id}_1.jpg`
- `{sample_id}_2.jpg`

The CSV must contain either a `caption` column or a `Captions` column, and the training split logic expects a `label` column.

## Install dependencies

From the repository root:

```bash
pip install torch numpy pandas scikit-learn pillow sentence-transformers
```

Optional packages used by inference output export:

```bash
pip install matplotlib scikit-image trimesh
```

Notes:

- `sentence-transformers` may download the `all-MiniLM-L6-v2` model the first time it runs.
- GPU is used automatically if CUDA is available.

## Train the model

Run from the repository root:

```bash
python Cond_VAE/train_hybrid_vae.py
```

Checkpoints are written to:

```text
hybrid_vae_checkpoints/
```

The training script saves:

- `hybrid_vae_checkpoints/latest.pt`
- `hybrid_vae_checkpoints/best.pt`
- `hybrid_vae_checkpoints/final.pt`

## Perform inference

The inference script supports two modes:

1. `text`: generate from caption only
2. `image`: generate from caption plus three slice images

```

### Text-only inference

```bash
python Cond_VAE/inference_hybrid_vae.py \
  --mode text \
  --caption "Microstructure featuring octet lattices in an ordered design." \
  --checkpoint hybrid_vae_checkpoints/best.pt \
  --output_dir ./hybrid_outputs/text_only_run \
  --text_strategy best_of_k \
  --text_candidate_pool 12 \
  --temperature 0.20 \
  --occupancy_target 0.16 \
  --occupancy_tolerance 0.08 \
  --threshold 0.5
```

Useful text-mode arguments:

- `--text_strategy`: `best_of_k`, `sample`, or `mean`
- `--text_candidate_pool`: number of candidates to rank when using `best_of_k`
- `--temperature`: lower values are usually more conservative
- `--occupancy_target`: preferred occupancy ratio for candidate selection
- `--threshold`: voxel threshold used for occupancy stats and mesh extraction

### Text + slice image inference

```bash
python Cond_VAE/inference_hybrid_vae.py \
  --mode image \
  --caption "Microstructure featuring octet lattices in an ordered design." \
  --slice_0 data/datasets/2d_Slices/1550_0.jpg \
  --slice_1 data/datasets/2d_Slices/1550_1.jpg \
  --slice_2 data/datasets/2d_Slices/1550_2.jpg \
  --checkpoint hybrid_vae_checkpoints/best.pt \
  --output_dir ./hybrid_outputs/text_image_run \
  --posterior_mix 0.6 \
  --refinement_steps 3 \
  --voxel_update_mix 0.8 \
  --init_voxel_mode prior_mean \
  --temperature 0.15 \
  --threshold 0.5
```

Useful image-mode arguments:

- `--slice_0`, `--slice_1`, `--slice_2`: required slice images
- `--posterior_mix`: blend between text prior and slice-refined posterior
- `--refinement_steps`: number of refinement passes
- `--voxel_update_mix`: how strongly each pass updates the current voxel
- `--init_voxel_mode`: `prior_mean`, `prior_sample`, or `zeros`
- `--init_voxel_threshold`: optional binarization of the initial prior voxel

### Generate multiple samples

```bash
python Cond_VAE/inference_hybrid_vae.py \
  --mode text \
  --caption "Porous lattice with repeating diagonal struts." \
  --checkpoint hybrid_vae_checkpoints/best.pt \
  --output_dir ./hybrid_outputs/multi_sample_run \
  --n_samples 5
```

## Inference outputs

For each generated sample, the script saves:

- `voxel.npy`: raw voxel probabilities with shape `(64, 64, 64)`
- `slices.png`: orthographic middle-slice visualization
- `generated_0_sampleX.png`: 3D preview image
- `generated_0_sampleX.glb`: mesh export from marching cubes, if optional deps are installed

If `--n_samples > 1`, each sample is written into its own subdirectory inside `output_dir`.

## Common tips

- If the exported mesh is empty, try lowering `--threshold`.
- If text-only results are too noisy, reduce `--temperature`.
- If text-only shapes are too sparse or too dense, tune `--occupancy_target` and `--occupancy_tolerance`.
- If image-mode refinement is too weak, increase `--posterior_mix` or `--refinement_steps`.

