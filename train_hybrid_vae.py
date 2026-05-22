# ===================== train_hybrid_vae.py =====================

import os
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sentence_transformers import SentenceTransformer

# ===================== CONFIG =====================
CAPTION_PATH = "/mnt/home2/home/shiva_gp/txt23d_pytorch/data/datasets/captions_labeled.csv"
VOXEL_PATH   = "/mnt/home2/home/shiva_gp/txt23d_pytorch/data/datasets/geometries_2000x64x64x64.npy"
SLICE_DIR    = "/mnt/home2/home/shiva_gp/txt23d_pytorch/data/datasets/2d_Slices"

LATENT_DIM = 128
TEXT_DIM   = 384
SLICE_SIZE = 256
BATCH_SIZE = 16
EPOCHS     = 200

DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
ROOT           = Path(__file__).resolve().parents[1]
CHECKPOINT_DIR = ROOT / "hybrid_vae_checkpoints"

KL_WARMUP_EPOCHS = 30
PRIOR_MATCH_WARMUP_EPOCHS = 30
PRIOR_MATCH_WEIGHT = 0.10
FREE_BITS_NATS = 1.0   
# ===================== DATASET =====================
class Dataset3D(Dataset):
    """
    Each sample: voxel (1,64,64,64) + 3 axis slices (3,256,256) + text emb.
    """
    def __init__(self, voxels, text_embs, sample_ids, slice_dir, augment=False):
        self.voxels     = voxels
        self.text       = text_embs
        self.sample_ids = np.asarray(sample_ids)
        self.slice_dir  = Path(slice_dir)
        self.augment    = augment

    def __len__(self):
        return len(self.voxels)

    def _load_slices(self, sample_id):
        channels = []
        for axis in range(3):
            img_path = self.slice_dir / f"{sample_id}_{axis}.jpg"
            with Image.open(img_path) as img:
                img = img.convert("L").resize(
                    (SLICE_SIZE, SLICE_SIZE), Image.Resampling.LANCZOS
                )
                channels.append(
                    torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0)
                )
        return torch.stack(channels, dim=0)   # (3, H, W)

    def __getitem__(self, idx):
        v = torch.tensor(self.voxels[idx], dtype=torch.float32).unsqueeze(0)
        s = self._load_slices(int(self.sample_ids[idx]))
        t = torch.tensor(self.text[idx], dtype=torch.float32)

        if self.augment:
            for axis in [1, 2, 3]:   # (1,D,H,W): spatial = 1,2,3
                if torch.rand(1).item() > 0.5:
                    v = torch.flip(v, dims=[axis])

        return v, s, t


# ===================== SLICE ENCODER =====================
class SliceEncoder(nn.Module):
    """
    5 strided Conv2d layers reduce 256x256 → 8x8 before global pooling.
    Channels: 3 → 32 → 64 → 128 → 128 → 128.
    Output: 128-dim vector per sample.
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            # 256 → 128
            nn.Conv2d(3,   32,  4, stride=2, padding=1), nn.BatchNorm2d(32),  nn.ReLU(),
            # 128 → 64
            nn.Conv2d(32,  64,  4, stride=2, padding=1), nn.BatchNorm2d(64),  nn.ReLU(),
            # 64 → 32
            nn.Conv2d(64,  128, 4, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            # 32 → 16
            nn.Conv2d(128, 128, 4, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            # 16 → 8
            nn.Conv2d(128, 128, 4, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            # 8×8 → 1×1
            nn.AdaptiveAvgPool2d(1),
        )

    def forward(self, x):
        return self.net(x).view(x.size(0), -1)   # (B, 128)


# ===================== TEXT PRIOR =====================
class TextPrior(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(TEXT_DIM, 256),
            nn.ReLU(),
            nn.Linear(256, LATENT_DIM * 2)
        )

    def forward(self, text):
        out = self.net(text)
        mu, logvar = out.chunk(2, dim=1)
        logvar = torch.clamp(logvar, -4, 4)
        return mu, logvar


# ===================== BUILDING BLOCKS =====================
class ResBlock3D(nn.Module):
    def __init__(self, in_c, out_c, dropout=0.0):
        super().__init__()
        self.conv1 = nn.Conv3d(in_c, out_c, 3, padding=1)
        self.bn1   = nn.BatchNorm3d(out_c)
        self.conv2 = nn.Conv3d(out_c, out_c, 3, padding=1)
        self.bn2   = nn.BatchNorm3d(out_c)
        self.act   = nn.ReLU()
        self.skip  = nn.Conv3d(in_c, out_c, 1) if in_c != out_c else nn.Identity()
        self.drop  = nn.Dropout3d(p=dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        h = self.act(self.bn1(self.conv1(x)))
        h = self.drop(self.bn2(self.conv2(h)))
        return self.act(h + self.skip(x))


# ===================== ENCODER =====================
class Encoder(nn.Module):
    """
    Fuses three modalities: 3D voxel, 2D slices, text embedding.

    Concatenated (128 + 128 + 384 = 640) → FC(640, 512) → mu, logvar.
    """
    def __init__(self):
        super().__init__()
        self.voxel_conv = nn.Sequential(
            nn.Conv3d(1,   32,  4, 2, 1), nn.BatchNorm3d(32),  nn.ReLU(),
            nn.Conv3d(32,  64,  4, 2, 1), nn.BatchNorm3d(64),  nn.ReLU(),
            nn.Conv3d(64,  128, 4, 2, 1), nn.BatchNorm3d(128), nn.ReLU(),
            nn.Conv3d(128, 128, 4, 2, 1), nn.BatchNorm3d(128), nn.ReLU(),
            nn.Conv3d(128, 128, 4, 2, 1), nn.BatchNorm3d(128), nn.ReLU(),
            nn.AdaptiveAvgPool3d(1),
        )
        self.slice_enc   = SliceEncoder()
        self.slice_drop  = nn.Dropout(p=0.3)   # elementwise mask on slice features

        # 128 (voxel) + 128 (slice) + 384 (text) = 640
        self.fc      = nn.Linear(128 + 128 + TEXT_DIM, 512)
        self.fc_mu     = nn.Linear(512, LATENT_DIM)
        self.fc_logvar = nn.Linear(512, LATENT_DIM)

    def forward(self, voxel, slices, text):
        if self.training:
            voxel = voxel + 0.02 * torch.randn_like(voxel)

        v = self.voxel_conv(voxel).view(voxel.size(0), -1)   # (B, 128)
        s = self.slice_enc(slices)                             # (B, 128)

        if self.training:
            s = self.slice_drop(s)  

        combined = torch.cat([v, s, text], dim=1)              # (B, 640)
        h      = F.relu(self.fc(combined))
        mu     = self.fc_mu(h)
        logvar = torch.clamp(self.fc_logvar(h), -4, 4)
        return mu, logvar


# ===================== FILM LAYER =====================
class FiLM(nn.Module):
    """
    Feature-wise Linear Modulation with identity initialisation.
    gamma=1, beta=0 at init — decoder starts as a plain VAE and FiLM
    gradually learns to steer features based on text conditioning.
    """
    def __init__(self, text_dim, num_channels):
        super().__init__()
        self.fc = nn.Linear(text_dim, num_channels * 2)
        nn.init.zeros_(self.fc.weight)
        nn.init.constant_(self.fc.bias[:num_channels], 1.0)   # gamma = 1
        nn.init.zeros_(self.fc.bias[num_channels:])            # beta  = 0

    def forward(self, x, text):
        params = self.fc(text)
        gamma, beta = params.chunk(2, dim=1)
        gamma = gamma.view(gamma.size(0), -1, 1, 1, 1)
        beta  = beta.view(beta.size(0),   -1, 1, 1, 1)
        return gamma * x + beta


# ===================== DECODER =====================
class Decoder(nn.Module):
    """
    FiLM conditioning restored at 3 resolutions (2³, 4³, 8³).
    """
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(LATENT_DIM, 128 * 2 * 2 * 2)

        self.up1 = ResBlock3D(128, 64,  dropout=0.2)
        self.up2 = ResBlock3D(64,  32,  dropout=0.2)
        self.up3 = ResBlock3D(32,  16,  dropout=0.2)
        self.up4 = ResBlock3D(16,  16,  dropout=0.1)
        self.up5 = ResBlock3D(16,  8,   dropout=0.0)

        self.to_voxel = nn.Conv3d(8, 1, 1)

        self.film1 = FiLM(TEXT_DIM, 128)   # at 2³
        self.film2 = FiLM(TEXT_DIM, 64)    # at 4³
        self.film3 = FiLM(TEXT_DIM, 32)    # at 8³

    def forward(self, z, text):
        x = self.fc(z).view(-1, 128, 2, 2, 2)
        x = self.film1(x, text)

        x = F.interpolate(x, scale_factor=2)
        x = self.up1(x)
        x = self.film2(x, text)

        x = F.interpolate(x, scale_factor=2)
        x = self.up2(x)
        x = self.film3(x, text)

        x = F.interpolate(x, scale_factor=2)
        x = self.up3(x)

        x = F.interpolate(x, scale_factor=2)
        x = self.up4(x)

        x = F.interpolate(x, scale_factor=2)
        x = self.up5(x)

        return torch.sigmoid(self.to_voxel(x))


# ===================== MODEL =====================
class ConditionalVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder   = Encoder()
        self.decoder   = Decoder()
        self.prior     = TextPrior()   # trained against a detached posterior target
        self.text_proj = nn.Sequential(
            nn.Linear(TEXT_DIM, LATENT_DIM),
            nn.ReLU(),
            nn.Linear(LATENT_DIM, LATENT_DIM),
        )

    def reparam(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def forward(self, voxel, slices, text):
        mu_q, logvar_q = self.encoder(voxel, slices, text)
        mu_p, logvar_p = self.prior(text)
        z              = self.reparam(mu_q, logvar_q)
        recon          = self.decoder(z, text)
        text_latent    = self.text_proj(text)
        return recon, mu_q, logvar_q, z, text_latent, mu_p, logvar_p

    @torch.no_grad()
    def generate(self, text):
        """
        Inference: sample z from the text prior, decode to voxel.
        No voxel or slice input required.
        """
        self.eval()
        mu_p, logvar_p = self.prior(text)
        z              = self.reparam(mu_p, logvar_p)
        return self.decoder(z, text)


# ===================== LOSS =====================
def dice_loss(pred, target, eps=1e-6):
    pred   = pred.view(pred.size(0), -1)
    target = target.view(target.size(0), -1)
    intersection = (pred * target).sum(dim=1)
    dice = (2. * intersection + eps) / (pred.sum(dim=1) + target.sum(dim=1) + eps)
    return 1 - dice.mean()


def info_nce_loss(z, text_latent, temperature=0.10):
    """
    Symmetric InfoNCE contrastive alignment loss.
    temperature=0.10 suits batch=16 (15 in-batch negatives).
    """
    z_norm = F.normalize(z, dim=1)
    t_norm = F.normalize(text_latent, dim=1)
    logits = torch.matmul(z_norm, t_norm.T) / temperature
    labels = torch.arange(logits.size(0), device=logits.device)
    return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2.0


def gaussian_kl(mu_a, logvar_a, mu_b, logvar_b):
    """
    KL[N(mu_a, sigma_a^2) || N(mu_b, sigma_b^2)] averaged over batch and dim.
    """
    var_a = logvar_a.exp()
    var_b = logvar_b.exp()
    kl = logvar_b - logvar_a + (var_a + (mu_a - mu_b).pow(2)) / var_b - 1.0
    return 0.5 * kl.mean()


def loss_fn(recon, x, mu_q, logvar_q, z, text_latent, mu_p, logvar_p, epoch):
    # Reconstruction: BCE + Dice (equal weight).
    bce        = F.binary_cross_entropy(recon, x)
    dice       = dice_loss(recon, x)
    recon_loss = 0.5 * bce + 0.5 * dice

    kl = -0.5 * torch.mean(1 + logvar_q - mu_q.pow(2) - logvar_q.exp())

    if FREE_BITS_NATS > 0:
        kl_loss = (kl - FREE_BITS_NATS).clamp(min=0)
    else:
        kl_loss = kl

    kl_weight = min(1.0, epoch / KL_WARMUP_EPOCHS)

    align = info_nce_loss(z, text_latent)

    prior_match = gaussian_kl(
        mu_q.detach(), logvar_q.detach(),
        mu_p, logvar_p,
    )
    prior_weight = PRIOR_MATCH_WEIGHT * min(1.0, epoch / PRIOR_MATCH_WARMUP_EPOCHS)

    total = recon_loss + kl_weight * kl_loss + 0.3 * align + prior_weight * prior_match
    return total, recon_loss, kl, kl_loss, align, prior_match


# ===================== HELPERS =====================
def resolve_caption_column(df):
    for candidate in ("caption", "Captions"):
        if candidate in df.columns:
            return candidate
    raise KeyError(
        f"Expected 'caption' or 'Captions' column, found {list(df.columns)}"
    )


def verify_slices(slice_dir, num_samples, n_check=10):
    """Spot-check that slice images exist and are the right format."""
    import random
    slice_dir = Path(slice_dir)
    missing   = []
    samples   = random.sample(range(num_samples), min(n_check, num_samples))
    for sid in samples:
        for axis in range(3):
            p = slice_dir / f"{sid}_{axis}.jpg"
            if not p.exists():
                missing.append(str(p))
    if missing:
        raise FileNotFoundError(
            f"Missing slice files (showing first 5): {missing[:5]}"
        )
    # Check one image size
    with Image.open(slice_dir / f"0_0.jpg") as img:
        print(f"  Slice native size : {img.size}  mode={img.mode}")
        print(f"  Slice resized to  : ({SLICE_SIZE}, {SLICE_SIZE})")


# ===================== MAIN =====================
def main():
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    df     = pd.read_csv(CAPTION_PATH)
    # Voxels are bool dtype — cast directly to float32 (no /max() needed)
    voxels = np.load(VOXEL_PATH).astype(np.float32)

    if len(voxels) != len(df):
        raise ValueError(
            f"Voxel count {len(voxels)} != caption count {len(df)}"
        )

    caption_col = resolve_caption_column(df)

    print("Verifying slices...")
    verify_slices(SLICE_DIR, len(df))

    print("Encoding text...")
    text_model = SentenceTransformer('all-MiniLM-L6-v2')
    text_embs  = text_model.encode(df[caption_col].tolist(), batch_size=32)

    indices = np.arange(len(df))
    train_idx, temp_idx = train_test_split(
        indices, test_size=0.2, stratify=df['label'], random_state=42
    )
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.25,
        stratify=df['label'].iloc[temp_idx], random_state=42
    )

    train_ds = Dataset3D(
        voxels[train_idx], text_embs[train_idx],
        sample_ids=train_idx, slice_dir=SLICE_DIR, augment=True,
    )
    val_ds = Dataset3D(
        voxels[val_idx], text_embs[val_idx],
        sample_ids=val_idx, slice_dir=SLICE_DIR, augment=False,
    )

    train_dl = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=4, pin_memory=True,
    )
    val_dl = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=4, pin_memory=True,
    )

    model = ConditionalVAE().to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,}")

    opt = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=EPOCHS, eta_min=1e-6
    )

    print("Starting training...")
    print(f"KL warmup epochs: {KL_WARMUP_EPOCHS}")
    print(f"Prior-match warmup epochs: {PRIOR_MATCH_WARMUP_EPOCHS}")
    print(f"Prior-match weight: {PRIOR_MATCH_WEIGHT:.2f}")
    print(f"Free-bits floor: {FREE_BITS_NATS:.2f} nats")
    best_val_loss    = float('inf')
    patience         = 10
    patience_counter = 0

    for epoch in range(EPOCHS):
        # ── TRAIN ──
        model.train()
        tr_total = tr_recon = tr_kl = tr_kl_eff = tr_align = tr_prior = 0.0

        for v, s, t in train_dl:
            v, s, t = v.to(DEVICE), s.to(DEVICE), t.to(DEVICE)

            recon, mu_q, logvar_q, z, text_latent, mu_p, logvar_p = model(v, s, t)
            loss, recon_l, kl_l, kl_eff_l, align_l, prior_l = loss_fn(
                recon, v, mu_q, logvar_q, z, text_latent, mu_p, logvar_p, epoch
            )

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()

            tr_total += loss.item();    tr_recon += recon_l.item()
            tr_kl    += kl_l.item();   tr_kl_eff += kl_eff_l.item()
            tr_align += align_l.item()
            tr_prior += prior_l.item()

        n = len(train_dl)
        tr_total /= n;  tr_recon /= n;  tr_kl /= n;  tr_kl_eff /= n
        tr_align /= n;  tr_prior /= n

        scheduler.step()

        # ── VALIDATION ──
        model.eval()
        va_total = va_recon = va_kl = va_kl_eff = va_align = va_prior = 0.0

        with torch.no_grad():
            for v, s, t in val_dl:
                v, s, t = v.to(DEVICE), s.to(DEVICE), t.to(DEVICE)
                recon, mu_q, logvar_q, z, text_latent, mu_p, logvar_p = model(v, s, t)
                loss, recon_l, kl_l, kl_eff_l, align_l, prior_l = loss_fn(
                    recon, v, mu_q, logvar_q, z, text_latent, mu_p, logvar_p, epoch
                )
                va_total += loss.item();    va_recon += recon_l.item()
                va_kl    += kl_l.item();   va_kl_eff += kl_eff_l.item()
                va_align += align_l.item()
                va_prior += prior_l.item()

        n = len(val_dl)
        va_total /= n;  va_recon /= n;  va_kl /= n;  va_kl_eff /= n
        va_align /= n;  va_prior /= n

        kl_w       = min(1.0, epoch / KL_WARMUP_EPOCHS)
        prior_w    = PRIOR_MATCH_WEIGHT * min(1.0, epoch / PRIOR_MATCH_WARMUP_EPOCHS)
        current_lr = opt.param_groups[0]['lr']

        print(
            f"Epoch {epoch:03d} | kl_w={kl_w:.2f} | prior_w={prior_w:.2f} | lr={current_lr:.2e} | "
            f"Train: Total {tr_total:.4f} | Recon {tr_recon:.4f} | "
            f"KL_raw {tr_kl:.4f} | KL_eff {tr_kl_eff:.4f} | Align {tr_align:.4f} | Prior {tr_prior:.4f} || "
            f"Val: Total {va_total:.4f} | Recon {va_recon:.4f} | "
            f"KL_raw {va_kl:.4f} | KL_eff {va_kl_eff:.4f} | Align {va_align:.4f} | Prior {va_prior:.4f}"
        )

        torch.save(model.state_dict(), CHECKPOINT_DIR / "latest.pt")

        if va_total < best_val_loss:
            best_val_loss = va_total
            torch.save(model.state_dict(), CHECKPOINT_DIR / "best.pt")
            print(f"  ✓ Best model saved (val_loss={best_val_loss:.4f})")
            if epoch >= KL_WARMUP_EPOCHS:
                patience_counter = 0
        else:
            if epoch >= KL_WARMUP_EPOCHS:
                patience_counter += 1
                if patience_counter >= patience:
                    print("Early stopping triggered.")
                    break

    torch.save(model.state_dict(), CHECKPOINT_DIR / "final.pt")
    print(f"Training complete. Checkpoints saved to {CHECKPOINT_DIR}")


if __name__ == "__main__":
    main()
