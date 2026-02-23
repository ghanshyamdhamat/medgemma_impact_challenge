# # Fine-tune MedSigLIP on BraTS MRI Slices with LoRA
# 
# This notebook fine-tunes [MedSigLIP](https://huggingface.co/google/medsiglip-448) for binary
# **Tumor vs No Tumor** classification on axial FLAIR MRI slices using
# [LoRA](https://arxiv.org/abs/2106.09685) (Low-Rank Adaptation).
# 
# **Pipeline:**
# 1. Load 3D FLAIR `.nii` volumes and corresponding segmentation masks
# 2. Extract 2D axial slices, filter empty slices, label as Tumor / No Tumor
# 3. Optionally add healthy (tumor-free) volumes for better class balance
# 4. Apply LoRA adapters to MedSigLIP's attention layers
# 5. Train with BF16 mixed-precision, gradient accumulation, and gradient checkpointing
# 6. Evaluate on a held-out test set via zero-shot contrastive classification
# 
# **Requirements:** GPU with ≥16 GB VRAM (tested on RTX 4090 24 GB)

# ## 1. Setup & Authentication


# Install dependencies (uncomment on first run)
# !pip install --upgrade --quiet nibabel peft accelerate transformers scikit-learn matplotlib


import os
import sys

if "google.colab" in sys.modules and not os.environ.get("VERTEX_PRODUCT"):
    from google.colab import userdata
    os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")
elif os.environ.get("VERTEX_PRODUCT") == "COLAB_ENTERPRISE":
    os.environ["HF_HOME"] = "/content/hf"
else:
    from huggingface_hub import get_token
    if get_token() is None:
        from huggingface_hub import notebook_login
        notebook_login()


from pathlib import Path
import random

import numpy as np
import nibabel as nib
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader, ConcatDataset, random_split
from transformers import AutoProcessor, AutoModel
from peft import LoraConfig, get_peft_model, PeftModel
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt

print("All imports ready.")


# ## 2. Configuration
# 
# Update the paths below to point to your data directories.


# ── Paths ──
BRATS_DATA_ROOT = Path("./BraTS2020_Training_Data")       # BraTS volumes with flair + seg
HEALTHY_DATA_ROOT = Path("./MedgemmaHealthy")             # Healthy FLAIR volumes (no tumors)
BEST_MODEL_DIR = Path("./medsiglip-lora-brats-best")      # Best checkpoint saved during training
FINAL_MODEL_DIR = Path("./medsiglip-lora-brats-final")    # Final model after training

# ── Model ──
MODEL_ID = "google/medsiglip-448"

# ── Dataset ──
MAX_TUMOR_VOLUMES = 400       # Max BraTS volumes to use (indices 101+ for train)
MAX_HEALTHY_VOLUMES = 50      # Max healthy volumes to use
TRAIN_TUMOR_START_IDX = 101   # BraTS volumes [0:101) reserved for testing
SLICE_AXIS = 2                # Axial slices
MIN_NONZERO_PIXELS = 1000     # Minimum non-zero pixels to keep a slice

# ── Training ──
BATCH_SIZE = 10
GRAD_ACCUM_STEPS = 8          # Effective batch = BATCH_SIZE * GRAD_ACCUM_STEPS
NUM_EPOCHS = 20
LR = 1e-4
MAX_GRAD_NORM = 2.0
VAL_FRACTION = 0.1
SEED = 42

# ── Labels ──
LABEL_TUMOR = "Flair slice with Tumor"
LABEL_NO_TUMOR = "Flair slice with No Tumor"
CANDIDATE_LABELS = [LABEL_TUMOR, LABEL_NO_TUMOR]

# ── Reproducibility ──
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")

# ## 3. Dataset Classes
# 
# Two dataset classes handle BraTS (tumor + segmentation) and healthy (no tumor) volumes.
# Both:
# - Load 3D NIfTI volumes and extract 2D axial slices
# - Filter out near-empty slices via `slice_select()`
# - Rebuild filtered volumes for efficient `__getitem__` access


def _normalize_slice(slice_2d: np.ndarray) -> np.ndarray:
    """Percentile-based normalization to [0, 1]."""
    arr = slice_2d.astype(np.float32)
    vmin, vmax = np.percentile(arr, [1, 99])
    arr = np.clip(arr, vmin, vmax)
    if vmax - vmin < 1e-6:
        return np.zeros_like(arr)
    return (arr - vmin) / (vmax - vmin)


def slice_select(slice_2d: np.ndarray, threshold: int = MIN_NONZERO_PIXELS) -> bool:
    """Return True if the slice has enough non-zero content."""
    return int(np.count_nonzero(slice_2d)) > threshold


def _slice_to_pil(slice_2d: np.ndarray) -> Image.Image:
    """Normalize a 2D array and convert to RGB PIL image."""
    normed = _normalize_slice(slice_2d)
    uint8 = (normed * 255).astype(np.uint8)
    return Image.fromarray(uint8).convert("RGB")


class FlairSliceDataset(Dataset):
    """Dataset from BraTS FLAIR + SEG volume pairs.

    Labels each axial slice as Tumor / No Tumor based on the
    corresponding segmentation mask.
    """

    def __init__(self, volume_pairs, axis=2):
        self.axis = axis
        self.volumes = []
        self.index = []  # (volume_idx, slice_idx_in_filtered_vol, label)

        for v_idx, (flair_path, seg_path) in enumerate(volume_pairs):
            flair_data = np.asanyarray(nib.load(str(flair_path)).dataobj)
            seg_data = np.asanyarray(nib.load(str(seg_path)).dataobj)
            assert flair_data.shape == seg_data.shape, (
                f"Shape mismatch: {flair_path} {flair_data.shape} vs {seg_path} {seg_data.shape}"
            )

            valid_slices = []
            for s in range(flair_data.shape[self.axis]):
                flair_2d = np.take(flair_data, s, axis=self.axis)
                if not slice_select(flair_2d):
                    continue
                valid_slices.append(flair_2d)
                seg_2d = np.take(seg_data, s, axis=self.axis)
                label = LABEL_TUMOR if np.any(seg_2d > 0) else LABEL_NO_TUMOR
                self.index.append((v_idx, len(valid_slices) - 1, label))

            filtered_vol = np.stack(valid_slices, axis=self.axis) if valid_slices else np.empty(0)
            self.volumes.append(filtered_vol)

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        v_idx, s_idx, label = self.index[idx]
        slice_2d = np.take(self.volumes[v_idx], s_idx, axis=self.axis)
        return _slice_to_pil(slice_2d), label


class HealthySliceDataset(Dataset):
    """Dataset from healthy FLAIR volumes (all slices labelled No Tumor)."""

    def __init__(self, flair_paths, axis=2):
        self.axis = axis
        self.volumes = []
        self.index = []

        for v_idx, flair_path in enumerate(flair_paths):
            flair_data = np.asanyarray(nib.load(str(flair_path)).dataobj)

            valid_slices = []
            for s in range(flair_data.shape[self.axis]):
                flair_2d = np.take(flair_data, s, axis=self.axis)
                if not slice_select(flair_2d):
                    continue
                valid_slices.append(flair_2d)
                self.index.append((v_idx, len(valid_slices) - 1, LABEL_NO_TUMOR))

            filtered_vol = np.stack(valid_slices, axis=self.axis) if valid_slices else np.empty(0)
            self.volumes.append(filtered_vol)

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        v_idx, s_idx, label = self.index[idx]
        slice_2d = np.take(self.volumes[v_idx], s_idx, axis=self.axis)
        return _slice_to_pil(slice_2d), label

# ## 4. Discover Volumes & Build Datasets


# ── BraTS tumor volumes (train split: indices 101+) ──
all_brats_flair = sorted(BRATS_DATA_ROOT.glob("*BraTS20_Training_*/*flair.nii"))
train_flair = all_brats_flair[TRAIN_TUMOR_START_IDX:MAX_TUMOR_VOLUMES]

tumor_pairs = []
for fp in train_flair:
    sp = Path(str(fp).replace("_flair.nii", "_seg.nii"))
    if sp.exists():
        tumor_pairs.append((fp, sp))

print(f"BraTS tumor volume pairs (train): {len(tumor_pairs)}")

# ── Healthy volumes ──
healthy_flair = sorted(HEALTHY_DATA_ROOT.glob("*FLAIR*.nii*"))[:MAX_HEALTHY_VOLUMES]
print(f"Healthy FLAIR volumes: {len(healthy_flair)}")


print("Loading BraTS tumor slices...")
tumor_dataset = FlairSliceDataset(tumor_pairs, axis=SLICE_AXIS)
print(f"  → {len(tumor_dataset)} slices")

print("Loading healthy slices...")
healthy_dataset = HealthySliceDataset(healthy_flair, axis=SLICE_AXIS)
print(f"  → {len(healthy_dataset)} slices")

# Combine into a single dataset
combined_dataset = ConcatDataset([tumor_dataset, healthy_dataset])
print(f"Combined dataset: {len(combined_dataset)} slices")


# Verify a sample
sample_img, sample_label = combined_dataset[0]
print(f"Sample label: {sample_label}")
print(f"Image size: {sample_img.size}, mode: {sample_img.mode}")


# ## 5. Train / Validation Split & DataLoaders


val_size = max(1, int(len(combined_dataset) * VAL_FRACTION))
train_size = len(combined_dataset) - val_size
train_ds, val_ds = random_split(
    combined_dataset, [train_size, val_size],
    generator=torch.Generator().manual_seed(SEED),
)
print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")


# Load processor (needed by collate_fn)
processor = AutoProcessor.from_pretrained(MODEL_ID)


def collate_fn(batch):
    """Collate (image, label) pairs into model-ready tensors."""
    images, labels = zip(*batch)
    inputs = processor(
        text=list(labels),
        images=list(images),
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    inputs["return_loss"] = True
    return inputs


train_loader = DataLoader(
    train_ds, batch_size=BATCH_SIZE, shuffle=True,
    num_workers=2, pin_memory=True, collate_fn=collate_fn,
)
val_loader = DataLoader(
    val_ds, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=2, pin_memory=True, collate_fn=collate_fn,
)
print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")


# ## 6. Load MedSigLIP + Apply LoRA


model = AutoModel.from_pretrained(MODEL_ID)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
    # NOTE: Do NOT set task_type — PeftModelForFeatureExtraction's forward
    # conflicts with SigLIP's multi-modal forward signature.
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()


# ## 7. Training Loop
# 
# - **BF16 mixed-precision** via `torch.autocast`
# - **Gradient checkpointing** to reduce activation memory
# - **Gradient accumulation** for larger effective batch sizes
# - **Gradient clipping** to stabilise training
# - **Best-model checkpointing** based on validation loss


model.to(DEVICE)

# Enable gradient checkpointing (trades compute for memory)
model.base_model.model.vision_model.encoder.gradient_checkpointing = True
model.base_model.model.text_model.encoder.gradient_checkpointing = True

optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

BEST_MODEL_DIR.mkdir(parents=True, exist_ok=True)
best_val_loss = float("inf")

for epoch in range(1, NUM_EPOCHS + 1):
    # ── Training ──
    model.train()
    running_loss = 0.0
    optimizer.zero_grad(set_to_none=True)

    for step, batch in enumerate(train_loader, start=1):
        batch = {k: v.to(DEVICE) if hasattr(v, "to") else v for k, v in batch.items()}

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            outputs = model(**batch)
            loss = outputs.loss / GRAD_ACCUM_STEPS

        loss.backward()

        if step % GRAD_ACCUM_STEPS == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=MAX_GRAD_NORM)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        running_loss += loss.item() * GRAD_ACCUM_STEPS

        if step % 50 == 0:
            print(f"  Epoch {epoch} | Step {step} | Loss {running_loss / step:.4f}")

    # Flush remaining accumulated gradients
    if step % GRAD_ACCUM_STEPS != 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=MAX_GRAD_NORM)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    # ── Validation ──
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to(DEVICE) if hasattr(v, "to") else v for k, v in batch.items()}
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = model(**batch)
            val_loss += outputs.loss.item()
    val_loss /= max(1, len(val_loader))

    print(f"Epoch {epoch} | Train Loss {running_loss / len(train_loader):.4f} | Val Loss {val_loss:.4f}")

    # Save best checkpoint
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        model.save_pretrained(BEST_MODEL_DIR)
        processor.save_pretrained(BEST_MODEL_DIR)
        print(f"  ✓ Best model saved (val_loss={val_loss:.4f})")

print(f"\nTraining complete. Best validation loss: {best_val_loss:.4f}")


# Save final model (last epoch)
FINAL_MODEL_DIR.mkdir(parents=True, exist_ok=True)
model.save_pretrained(FINAL_MODEL_DIR)
processor.save_pretrained(FINAL_MODEL_DIR)
print(f"Final model saved to {FINAL_MODEL_DIR}")

# ## 8. Load Fine-tuned Model for Inference

# Load best checkpoint, merge LoRA into base weights for faster inference
base_model = AutoModel.from_pretrained(MODEL_ID)
ft_model = PeftModel.from_pretrained(base_model, str(BEST_MODEL_DIR))
ft_model = ft_model.merge_and_unload()
ft_model = ft_model.to(DEVICE).eval()

ft_processor = AutoProcessor.from_pretrained(str(BEST_MODEL_DIR))
print("Fine-tuned model loaded and LoRA weights merged.")




