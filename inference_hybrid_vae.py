
import argparse
import sys
import os
import textwrap
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
from skimage import measure
import trimesh

TRAIN_SCRIPT = Path(__file__).resolve().parent / "Cond_VAE" / "train_hybrid_vae.py"
if not TRAIN_SCRIPT.exists():
    # Fallback: try same directory
    TRAIN_SCRIPT = Path(__file__).resolve().parent / "train_hybrid_vae.py"
if not TRAIN_SCRIPT.exists():
    print(f"ERROR: Cannot find train_hybrid_vae.py. Tried:\n"
          f"  {Path(__file__).resolve().parent / 'Cond_VAE' / 'train_hybrid_vae.py'}\n"
          f"  {Path(__file__).resolve().parent / 'train_hybrid_vae.py'}\n"
          f"Set TRAIN_SCRIPT path manually in inference.py if needed.")
    sys.exit(1)

import importlib.util
spec = importlib.util.spec_from_file_location("train_hybrid_vae", TRAIN_SCRIPT)
train_module = importlib.util.module_from_spec(spec) # type: ignore
spec.loader.exec_module(train_module) # type: ignore

ConditionalVAE = train_module.ConditionalVAE
LATENT_DIM     = train_module.LATENT_DIM
TEXT_DIM       = train_module.TEXT_DIM
SLICE_SIZE     = train_module.SLICE_SIZE
DEFAULT_THRESHOLD = 0.5

from sentence_transformers import SentenceTransformer


# ── Visualization / mesh export helpers ──────────────────────────────────────
def downsample_mask(mask: np.ndarray, factor: int, min_fraction: float) -> np.ndarray:
    if factor <= 1:
        return mask

    shape = mask.shape
    trimmed = mask[
        : shape[0] - shape[0] % factor,
        : shape[1] - shape[1] % factor,
        : shape[2] - shape[2] % factor,
    ]
    pooled = trimmed.reshape(
        trimmed.shape[0] // factor,
        factor,
        trimmed.shape[1] // factor,
        factor,
        trimmed.shape[2] // factor,
        factor,
    )
    return pooled.mean(axis=(1, 3, 5)) >= min_fraction


def build_render_mask(voxel: np.ndarray, threshold: float) -> np.ndarray:
    mask = voxel > threshold
    fill_ratio = float(mask.mean())

    if fill_ratio > 0.6:
        adaptive_threshold = max(threshold, float(np.quantile(voxel, 0.7)))
        mask = voxel > adaptive_threshold
        fill_ratio = float(mask.mean())

    if fill_ratio > 0.5:
        return downsample_mask(mask, factor=4, min_fraction=0.35)
    if fill_ratio > 0.25:
        return downsample_mask(mask, factor=2, min_fraction=0.5)
    return mask


def save_voxel_preview(
    voxel: np.ndarray,
    output_path: Path,
    caption: str,
    sample_index: int,
    threshold: float = DEFAULT_THRESHOLD,
) -> None:
    """Save a 3D preview PNG similar to the legacy generated_*_sample*.png outputs."""
    if plt is None:
        print("  matplotlib not installed — skipping preview PNG export. "
              "Install with: pip install matplotlib")
        return

    voxel_mask = build_render_mask(voxel, threshold)
    wrapped_caption = "\n".join(textwrap.wrap(str(caption), width=55))
    title_lines = max(1, len(wrapped_caption.splitlines()))
    fig_height = 6 + 0.35 * (title_lines - 1)
    fig = plt.figure(figsize=(6, fig_height))
    fig.text(0.5, 0.99, wrapped_caption, ha="center", va="top", fontsize=10)

    ax = fig.add_subplot(111, projection="3d")
    if voxel_mask.any():
        ax.voxels(voxel_mask, facecolors="C2", edgecolor=None)
    else:
        ax.text2D(0.5, 0.5, "No occupied voxels above threshold", ha="center", va="center")

    ax.set_title(f"Generated {sample_index + 1}")
    ax.set_axis_off()
    ax.view_init(elev=24, azim=42)
    top_margin = 0.88 - 0.03 * (title_lines - 1)
    fig.subplots_adjust(top=max(0.68, top_margin))
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  Preview saved → {output_path}")


def voxel_to_mesh(voxel: np.ndarray, threshold: float = DEFAULT_THRESHOLD):
    if measure is None:
        print("  scikit-image not installed — skipping mesh export. "
              "Install with: pip install scikit-image")
        return None, None

    if voxel.max() <= threshold:
        print(f"  Warning: voxel max={voxel.max():.3f} is below threshold={threshold}. "
              f"Mesh may be empty. Try lowering --threshold.")
        return None, None

    try:
        verts, faces, _normals, _values = measure.marching_cubes(voxel, level=threshold)
    except Exception as e:
        print(f"  Marching cubes failed: {e}")
        return None, None
    return verts, faces


def save_glb_mesh(voxel: np.ndarray, threshold: float, output_path: Path) -> None:
    verts, faces = voxel_to_mesh(voxel, threshold=threshold)
    if verts is None or faces is None:
        return

    if trimesh is None:
        print("  trimesh not installed — skipping .glb export. "
              "Install with: pip install trimesh")
        return

    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    try:
        mesh.export(output_path)
    except Exception as e:
        print(f"  Failed to save .glb: {e}")
        return

    print(f"  Mesh saved    → {output_path}  "
          f"({len(verts)} verts, {len(faces)} faces)")


def save_slice_views(voxel: np.ndarray, output_path: Path):
    """Save three orthographic slice views (XY, XZ, YZ) as a PNG."""
    mid = voxel.shape[0] // 2

    if plt is not None:
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        slice_views = (
            (voxel[mid, :, :], "XY Slice"),
            (voxel[:, mid, :], "XZ Slice"),
            (voxel[:, :, mid], "YZ Slice"),
        )

        for axis, (image, title) in zip(axes, slice_views):
            axis.imshow(image, cmap="gray", vmin=0, vmax=1)
            axis.set_title(title)
            axis.axis("off")

        fig.tight_layout()
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"  Slice views  → {output_path}")
        return

    xy = voxel[mid, :, :]
    xz = voxel[:, mid, :]
    yz = voxel[:, :, mid]

    def to_uint8(arr):
        arr = (arr * 255).clip(0, 255).astype(np.uint8)
        return Image.fromarray(arr)

    imgs = [to_uint8(xy), to_uint8(xz), to_uint8(yz)]
    total_w = sum(im.width for im in imgs) + 2 * 4
    canvas = Image.new("L", (total_w, imgs[0].height), color=20)
    x = 0
    for im in imgs:
        canvas.paste(im, (x, 0))
        x += im.width + 4
    canvas.save(output_path)
    print(f"  Slice views  → {output_path}")


def load_slice_images(paths: list[str], size: int = SLICE_SIZE) -> torch.Tensor:
    """
    Load 3 axis slice JPEGs → (1, 3, H, W) float32 tensor.
    paths = [axis0_path, axis1_path, axis2_path]
    """
    channels = []
    for p in paths:
        with Image.open(p) as img:
            img = img.convert("L").resize((size, size), Image.Resampling.LANCZOS)
            channels.append(
                torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0)
            )
    return torch.stack(channels, dim=0).unsqueeze(0)   # (1, 3, H, W)


def encode_text(caption: str) -> torch.Tensor:
    """Encode a caption string → (1, 384) float32 tensor."""
    model = SentenceTransformer("all-MiniLM-L6-v2")
    emb   = model.encode([caption], batch_size=1, convert_to_numpy=True)
    return torch.tensor(emb, dtype=torch.float32)       # (1, 384)


def sample_gaussian(mu: torch.Tensor, logvar: torch.Tensor, temperature: float) -> torch.Tensor:
    """Sample from a diagonal Gaussian, or return the mean when temperature <= 0."""
    if temperature <= 0:
        return mu
    std = torch.exp(0.5 * logvar) * temperature
    return mu + torch.randn_like(std) * std


# ── Generation functions ──────────────────────────────────────────────────────

def score_voxel_candidate(
    voxel: np.ndarray,
    threshold: float = DEFAULT_THRESHOLD,
    occupancy_target: float = 0.18,
    occupancy_tolerance: float = 0.10,
) -> tuple[float, dict[str, float]]:
    """
    Heuristic score for text-only sampling.

    We prefer candidates whose occupancy is neither too sparse nor too dense,
    and whose probabilities are confident rather than hovering around 0.5.
    """
    occupancy = float((voxel > threshold).mean())
    confidence = float(np.mean(np.abs(voxel - 0.5)))
    mid_band = float(((voxel > 0.35) & (voxel < 0.65)).mean())
    peak = float(voxel.max())

    tol = max(occupancy_tolerance, 1e-6)
    occupancy_penalty = abs(occupancy - occupancy_target) / tol
    score = confidence + 0.10 * peak - 0.35 * mid_band - occupancy_penalty

    return score, {
        "occupancy": occupancy,
        "confidence": confidence,
        "mid_band": mid_band,
        "peak": peak,
    }


@torch.no_grad()
def generate_text_only(
    model: ConditionalVAE,
    text_emb: torch.Tensor,
    n_samples: int,
    device: str,
    temperature: float = 0.7, # changed from 1.0
    strategy: str = "best_of_k",
    candidate_pool: int = 8,
    occupancy_target: float = 0.18,
    occupancy_tolerance: float = 0.10,
    threshold: float = DEFAULT_THRESHOLD,
) -> list[np.ndarray]:
    """
    Sample z ~ TextPrior(text), decode → voxel.
    Returns a list of n_samples (64,64,64) float32 numpy arrays.
    """
    model.eval()
    t_single = text_emb.to(device)                  # (1, 384)

    if strategy == "mean":
        t = t_single.expand(n_samples, -1)
        mu_p, _logvar_p = model.prior(t)
        recon = model.decoder(mu_p, t)
        return [recon[i, 0].cpu().numpy() for i in range(n_samples)]

    if strategy == "sample":
        t = t_single.expand(n_samples, -1)
        mu_p, logvar_p = model.prior(t)
        z = sample_gaussian(mu_p, logvar_p, temperature)
        recon = model.decoder(z, t)
        return [recon[i, 0].cpu().numpy() for i in range(n_samples)]

    outputs: list[np.ndarray] = []
    pool = max(1, int(candidate_pool))

    for sample_idx in range(n_samples):
        t = t_single.expand(pool, -1)
        mu_p, logvar_p = model.prior(t)
        z = sample_gaussian(mu_p, logvar_p, temperature)

        recon = model.decoder(z, t)

        best_score = None
        best_voxel = None
        best_stats = None
        for candidate_idx in range(pool):
            voxel = recon[candidate_idx, 0].cpu().numpy()
            score, stats = score_voxel_candidate(
                voxel=voxel,
                threshold=threshold,
                occupancy_target=occupancy_target,
                occupancy_tolerance=occupancy_tolerance,
            )
            if best_score is None or score > best_score:
                best_score = score
                best_voxel = voxel
                best_stats = stats

        assert best_voxel is not None
        outputs.append(best_voxel)
        print(
            f"  Text sample {sample_idx:03d}: picked best of {pool} "
            f"(score={best_score:.4f}, occ={best_stats['occupancy']*100:.1f}%, "
            f"conf={best_stats['confidence']:.3f})"
        )

    return outputs


@torch.no_grad()
def generate_text_and_image(
    model: ConditionalVAE,
    text_emb: torch.Tensor,
    slice_tensor: torch.Tensor,
    n_samples: int,
    device: str,
    temperature: float = 0.7,
    posterior_mix: float = 0.5,
    init_voxel_mode: str = "prior_mean",
    init_voxel_threshold: float | None = None,
    refinement_steps: int = 2,
    voxel_update_mix: float = 0.7,
) -> list[np.ndarray]:
    
    model.eval()
    t = text_emb.to(device).expand(n_samples, -1)           # (N, 384)
    s = slice_tensor.to(device).expand(n_samples, -1, -1, -1)  # (N, 3, H, W)

    mu_p, logvar_p = model.prior(t)

    if init_voxel_mode == "zeros":
        init_voxel = torch.zeros(n_samples, 1, 64, 64, 64, device=device)
    else:
        z_init = mu_p
        if init_voxel_mode == "prior_sample":
            z_init = sample_gaussian(mu_p, logvar_p, temperature=1.0)
        init_voxel = model.decoder(z_init, t)
        if init_voxel_threshold is not None:
            init_voxel = (init_voxel > init_voxel_threshold).float()

    mix = float(np.clip(posterior_mix, 0.0, 1.0))
    voxel_mix = float(np.clip(voxel_update_mix, 0.0, 1.0))
    steps = max(1, int(refinement_steps))

    current_voxel = init_voxel
    mu, logvar = mu_p, logvar_p

    for _ in range(steps):
        mu_q, logvar_q = model.encoder(current_voxel, s, t)
        mu = (1.0 - mix) * mu_p + mix * mu_q
        logvar = (1.0 - mix) * logvar_p + mix * logvar_q

        refined_voxel = model.decoder(mu, t)
        current_voxel = voxel_mix * refined_voxel + (1.0 - voxel_mix) * current_voxel

    z = sample_gaussian(mu, logvar, temperature)           # (N, 128)

    recon = model.decoder(z, t)                            # (N, 1, 64, 64, 64)
    recon = voxel_mix * recon + (1.0 - voxel_mix) * current_voxel
    return [recon[i, 0].cpu().numpy() for i in range(n_samples)]


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Generate 3D voxels from the Hybrid Conditional VAE.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument(
        "--mode", choices=["text", "image"], required=True,
        help="Generation mode:\n"
             "  text  — use TextPrior branch (text only, no images)\n"
             "  image — use full encoder with 3 axis slice images + text"
    )
    p.add_argument(
        "--caption", type=str, required=True,
        help='Text description, e.g. "a small red chair with four legs"'
    )
    p.add_argument(
        "--checkpoint", type=str,
        default="hybrid_vae_checkpoints/best.pt",
        help="Path to model checkpoint (default: hybrid_vae_checkpoints/best.pt)"
    )
    p.add_argument(
        "--output_dir", type=str, default="./inference_outputs",
        help="Directory to save outputs (default: ./inference_outputs)"
    )
    p.add_argument(
        "--slice_0", type=str, default=None,
        help="[image mode] Path to axis-0 (front) slice image"
    )
    p.add_argument(
        "--slice_1", type=str, default=None,
        help="[image mode] Path to axis-1 (side) slice image"
    )
    p.add_argument(
        "--slice_2", type=str, default=None,
        help="[image mode] Path to axis-2 (top) slice image"
    )
    p.add_argument(
        "--n_samples", type=int, default=1,
        help="Number of samples to generate (default: 1). Each sample is "
             "stochastic — useful for exploring the posterior."
    )
    p.add_argument(
        "--threshold", type=float, default=DEFAULT_THRESHOLD,
        help=f"Occupancy threshold for marching cubes mesh (default: {DEFAULT_THRESHOLD}). "
             "Lower this if the mesh is empty."
    )
    p.add_argument(
        "--temperature", type=float, default=0.7,
        help="Sampling temperature (default: 0.7). "
             "< 1.0 = denser, more conservative shapes. > 1.0 = more diverse."
    )
    p.add_argument(
        "--text_strategy", choices=["best_of_k", "sample", "mean"],
        default="best_of_k",
        help="Text mode only: latent sampling strategy. 'best_of_k' samples a "
             "small pool and keeps the most plausible candidate (default)."
    )
    p.add_argument(
        "--text_candidate_pool", type=int, default=8,
        help="Text mode only: number of candidates to sample before selecting "
             "the best one when using --text_strategy best_of_k (default: 8)."
    )
    p.add_argument(
        "--occupancy_target", type=float, default=0.18,
        help="Text mode only: preferred occupancy ratio used when ranking "
             "best-of-k candidates (default: 0.18)."
    )
    p.add_argument(
        "--occupancy_tolerance", type=float, default=0.10,
        help="Text mode only: how far occupancy may drift from the target "
             "before candidates are penalized (default: 0.10)."
    )
    p.add_argument(
        "--posterior_mix", type=float, default=0.5,
        help="Image mode only: blend factor between text prior and the "
             "slice-refined posterior. 0.0 = text prior only, 1.0 = full "
             "refinement (default: 0.5)."
    )
    p.add_argument(
        "--init_voxel_mode", choices=["prior_mean", "prior_sample", "zeros"],
        default="prior_mean",
        help="Image mode only: initial voxel fed into the encoder before slice "
             "refinement. 'prior_mean' is the most stable default."
    )
    p.add_argument(
        "--init_voxel_threshold", type=float, default=None,
        help="Image mode only: optional threshold used to binarize the initial "
             "prior voxel before encoder refinement. Omit for a soft voxel "
             "input (default: soft/no threshold)."
    )
    p.add_argument(
        "--refinement_steps", type=int, default=2,
        help="Image mode only: number of slice-guided encoder/decoder "
             "refinement passes (default: 2)."
    )
    p.add_argument(
        "--voxel_update_mix", type=float, default=0.7,
        help="Image mode only: how strongly each refinement pass replaces the "
             "current voxel estimate. 1.0 = fully replace, 0.0 = keep the "
             "previous estimate (default: 0.7)."
    )
    p.add_argument(
        "--device", type=str, default=None,
        help="Device override: 'cpu' or 'cuda' (default: auto-detect)"
    )
    p.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for reproducibility (default: no seed)"
    )
    return p.parse_args()


def main():
    args = parse_args()

    # ── Seed ──
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        print(f"Seed: {args.seed}")

    # ── Device ──
    if args.device:
        device = args.device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # ── Validate image-mode args ──
    if args.mode == "image":
        missing = [
            f"--slice_{i}" for i, p in enumerate([args.slice_0, args.slice_1, args.slice_2])
            if p is None
        ]
        if missing:
            print(f"ERROR: image mode requires {', '.join(missing)}")
            sys.exit(1)
        for i, p in enumerate([args.slice_0, args.slice_1, args.slice_2]):
            if not Path(p).exists():
                print(f"ERROR: slice_{i} not found: {p}")
                sys.exit(1)

    # ── Load checkpoint ──
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        print(f"ERROR: Checkpoint not found: {ckpt_path}")
        sys.exit(1)

    print(f"Loading checkpoint: {ckpt_path}")
    model = ConditionalVAE()
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model loaded ({total_params:,} parameters)")

    # ── Encode text ──
    print(f"Encoding caption: \"{args.caption}\"")
    text_emb = encode_text(args.caption)              # (1, 384)

    # ── Generate ──
    print(f"Generating {args.n_samples} sample(s) in [{args.mode}] mode ...")

    if args.mode == "text":
        print(f"  Text refine : strategy={args.text_strategy}  "
              f"pool={args.text_candidate_pool}  "
              f"occ_target={args.occupancy_target:.2f}  "
              f"occ_tol={args.occupancy_tolerance:.2f}  "
              f"temperature={args.temperature:.2f}")
        voxels = generate_text_only(
            model=model,
            text_emb=text_emb,
            n_samples=args.n_samples,
            device=device,
            temperature=args.temperature,
            strategy=args.text_strategy,
            candidate_pool=args.text_candidate_pool,
            occupancy_target=args.occupancy_target,
            occupancy_tolerance=args.occupancy_tolerance,
            threshold=args.threshold,
        )
    else:
        slices = load_slice_images([args.slice_0, args.slice_1, args.slice_2])
        init_thresh_text = (
            "soft"
            if args.init_voxel_threshold is None
            else f"{args.init_voxel_threshold:.2f}"
        )
        print(f"  Slice tensor: {tuple(slices.shape)}  "
              f"range [{slices.min():.2f}, {slices.max():.2f}]")
        print(f"  Image refine: init={args.init_voxel_mode}  "
              f"init_threshold={init_thresh_text}  "
              f"posterior_mix={args.posterior_mix:.2f}  "
              f"steps={args.refinement_steps}  "
              f"voxel_update_mix={args.voxel_update_mix:.2f}  "
              f"temperature={args.temperature:.2f}")
        voxels = generate_text_and_image(
            model=model,
            text_emb=text_emb,
            slice_tensor=slices,
            n_samples=args.n_samples,
            device=device,
            temperature=args.temperature,
            posterior_mix=args.posterior_mix,
            init_voxel_mode=args.init_voxel_mode,
            init_voxel_threshold=args.init_voxel_threshold,
            refinement_steps=args.refinement_steps,
            voxel_update_mix=args.voxel_update_mix,
        )

    # ── Save outputs ──
    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    for i, voxel in enumerate(voxels):
        # Subdirectory per sample when n_samples > 1
        out_dir = out_root / f"sample_{i:03d}" if args.n_samples > 1 else out_root
        out_dir.mkdir(parents=True, exist_ok=True)

        # Voxel statistics
        print(f"\nSample {i:03d}:")
        print(f"  Voxel range   : [{voxel.min():.4f}, {voxel.max():.4f}]")
        print(f"  Occupancy     : {(voxel > args.threshold).mean()*100:.1f}% "
              f"of voxels above threshold {args.threshold}")

        # Save .npy
        npy_path = out_dir / "voxel.npy"
        np.save(npy_path, voxel)
        print(f"  Voxel saved   → {npy_path}")

        # Save orthographic slice views
        slices_path = out_dir / "slices.png"
        save_slice_views(voxel, slices_path)

        # Save 3D preview similar to legacy inference outputs
        preview_path = out_dir / f"generated_0_sample{i}.png"
        save_voxel_preview(
            voxel=voxel,
            output_path=preview_path,
            caption=args.caption,
            sample_index=i,
            threshold=args.threshold,
        )

        # Save GLB mesh
        glb_path = out_dir / f"generated_0_sample{i}.glb"
        save_glb_mesh(voxel, threshold=args.threshold, output_path=glb_path)

    print(f"\nDone. All outputs in: {out_root.resolve()}")


if __name__ == "__main__":
    main()
