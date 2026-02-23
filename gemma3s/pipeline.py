"""
MedSigLIP Tumor Detection Pipeline
───────────────────────────────────
Loads a LoRA-finetuned MedSigLIP model, processes NIfTI FLAIR volumes
slice-by-slice, predicts tumor presence, and updates patient JSON records.
"""

import json
import os
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from PIL import Image
from huggingface_hub import login
from natsort import natsorted
from peft import PeftModel
from transformers import AutoModel, AutoProcessor

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
# Root of the project (MedSAM2)
# pipeline.py is in gemma3s/, so root is parent.parent
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Where the LoRA checkpoint is stored
OUTPUT_DIR = PROJECT_ROOT / "gemma3s/gemma_checkpoints/medsiglip-lora-brats-flair"

# Where patient data lives
BASE_DATA_DIR = PROJECT_ROOT / "common_data"
CONFIG_FILE = BASE_DATA_DIR / "comman_format.json"

# Base model ID (still needed to load the base architecture, but weights come from local mostly)
# Note: The base model usually needs to be downloaded or cached. 
# If "google/medsiglip-448" is not local, it will try to download. 
# Assuming internet access or cached model.
MODEL_ID = "google/medsiglip-448"

CANDIDATE_LABELS = ["FLAIR slice with TUMOR", "FLAIR slice with no TUMOR"]
CONFIDENCE_THRESHOLD = 0.25
TUMOR_SLICE_CUTOFF = 4
VALID_SLICE_THRESHOLD = 3000

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ──────────────────────────────────────────────
# Model Loading
# ──────────────────────────────────────────────
def load_model():
    """Load base MedSigLIP, apply LoRA weights, merge, and return (model, processor)."""
    # The base model is gated, so we need to login even if using local LoRA weights
    token = os.getenv("HF_ACCESS_TOKEN")
    if token:
        login(token=token)
    else:
        print("Warning: HF_ACCESS_TOKEN not found. Model load might fail if not cached.")

    print(f"Loading base model: {MODEL_ID}")
    base_model = AutoModel.from_pretrained(MODEL_ID)
    
    print(f"Loading LoRA from: {OUTPUT_DIR}")
    model = PeftModel.from_pretrained(base_model, str(OUTPUT_DIR), device_map="auto")
    model = model.merge_and_unload()
    model = model.to(DEVICE).eval()

    processor = AutoProcessor.from_pretrained(str(OUTPUT_DIR))
    print("Fine-tuned model loaded and merged.")
    return model, processor


# ──────────────────────────────────────────────
# Image Utilities
# ──────────────────────────────────────────────
def normalize_to_uint8(img: np.ndarray) -> np.ndarray:
    """Normalize a 2-D array to 0–255 uint8."""
    img = img.astype(np.float32)
    if img.max() > img.min():
        img = (img - img.min()) / (img.max() - img.min())
    else:
        img = np.zeros_like(img)
    return (img * 255).astype(np.uint8)


def slice_to_pil(slice_2d: np.ndarray) -> Image.Image:
    """Convert a single 2-D numpy slice to an RGB PIL Image."""
    return Image.fromarray(normalize_to_uint8(slice_2d)).convert("RGB")


def find_valid_slice_range(img_data: np.ndarray, threshold: int = 1000):
    """Return (first, last) axial indices where non-zero pixel count > threshold."""
    nz_counts = [np.count_nonzero(img_data[:, :, i]) for i in range(img_data.shape[2])]
    valid_slices = [i for i, c in enumerate(nz_counts) if c > threshold]
    if not valid_slices:
        return None, None
    return valid_slices[0], valid_slices[-1]


def labels_from_seg(nii_path: str) -> list[int]:
    """Return per-axial-slice binary labels from a segmentation NIfTI."""
    vol = nib.load(nii_path).get_fdata()
    return [
        1 if np.any(np.abs(vol[:, :, z]) > 0) else 0
        for z in range(vol.shape[2])
    ]


# ──────────────────────────────────────────────
# Prediction
# ──────────────────────────────────────────────
def predict_volume_tumor(model, processor, volume_path):
    """
    Run zero-shot classification on every valid axial slice of a FLAIR volume.

    Returns
    -------
    predictions : list[int]   – 1 = tumor, 0 = normal (per valid slice)
    slice_results : dict[int, float] – {axial_index: confidence_score}
    first_slice : int – first valid slice index (for coordinate mapping)
    """
    vol = nib.load(str(volume_path)).get_fdata()
    first, last = find_valid_slice_range(vol, VALID_SLICE_THRESHOLD)

    if first is None or last is None:
        print(f"No valid slices in {volume_path}")
        return [], {}, 0

    predictions = []
    slice_results = {}

    for z in range(first, last + 1):
        slice_png = slice_to_pil(vol[:, :, z])

        inputs = processor(
            text=CANDIDATE_LABELS,
            images=[slice_png],
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).to(DEVICE)

        with torch.no_grad():
            outputs = model(**inputs)

        # logits = outputs.logits_per_image
        # #take softmax
        # probs = torch.nn.functional.softmax(logits, dim=-1)
        # # confidence = (logits[0][0] - logits[0][1]).item()
        # confidence = max(probs[0][0],probs[0][1])

        # slice_results[z] = confidence
        # predictions.append(1 if confidence == probs[0][0] else 0)
        logits = outputs.logits_per_image
        probs = torch.softmax(logits, dim=-1)   # shape [1, 2]

        pred_idx = torch.argmax(probs, dim=-1).item()   # 0=tumor, 1=no tumor
        confidence = probs[0, pred_idx].item()

        slice_results[z] = confidence
        predictions.append(1 if pred_idx == 0 else 0)

    return predictions, slice_results, first


def important_slice(predictions: list[int], slice_offset: int = 0):
    """
    Find the mid-index of the largest contiguous tumor interval.

    Args
    ----
    predictions : list[int]
        Binary predictions (1=tumor, 0=normal) for consecutive slices
    slice_offset : int
        Offset to add to returned index (first valid slice index from volume)

    Returns
    -------
    mid_index : int
        Actual axial slice index in the volume
    has_tumor : bool
    """
    intervals = []
    start = None

    for i, pred in enumerate(predictions):
        if pred == 1 and start is None:
            start = i
        elif pred == 0 and start is not None:
            intervals.append((start, i - 1))
            start = None

    # Close interval if it runs to the end
    if start is not None:
        intervals.append((start, len(predictions) - 1))

    has_tumor = False

    if not intervals:
        print("No predicted tumor interval found.")
        mid_index_relative = len(predictions) // 2
        return mid_index_relative + slice_offset, False

    largest = max(intervals, key=lambda x: x[1] - x[0])
    length = largest[1] - largest[0] + 1

    if length >= TUMOR_SLICE_CUTOFF:
        has_tumor = True
        mid_index_relative = (largest[0] + largest[1]) // 2
    else:
        if largest[0] <= len(predictions) // 2 <= largest[1]:
            mid_index_relative = largest[1] + 1
        else:
            mid_index_relative = len(predictions) // 2

    # Add offset to convert from predictions-list index to actual volume slice index
    mid_index_actual = mid_index_relative + slice_offset

    print(f"Largest interval: start={largest[0]}, end={largest[1]}, length={length}")
    print(f"Mid index (relative): {mid_index_relative}, (actual): {mid_index_actual}")

    return mid_index_actual, has_tumor


# ──────────────────────────────────────────────
# Patient Pipeline
# ──────────────────────────────────────────────
def load_pending_patients(config_path: str):
    """Load patients with reviewed_by_radio == false from the master JSON (list format)."""
    with open(config_path, "r") as f:
        config_data = json.load(f)

    # config_data is expected to be a list of dicts
    if isinstance(config_data, dict) and "patients" in config_data:
         # Fallback for old format just in case
         pending = [p for p in config_data["patients"] if not p.get("reviewed_by_radio", False)]
    elif isinstance(config_data, list):
        pending = [p for p in config_data if not p.get("reviewed_by_radio", False)]
    else:
        print("Error: Unknown config format")
        return [], config_data

    return pending, config_data


def process_patient(patient_entry: dict, config_data: dict, config_path: str,
                    model, processor):
    """Process a single patient: predict, pick mid-slice, update JSONs."""
    pid = patient_entry["pid"]
    pid_dir = BASE_DATA_DIR / str(pid)
    print(f"Processing patient: {pid}  path: {pid_dir}")

    try:
        # Load patient-specific JSON
        pid_json_files = list(pid_dir.glob(f"patient_results.json"))
        if not pid_json_files:
            print(f"Error: No patient JSON found for {pid}")
            # Create one if missing? For now, skip or maybe creating it is better.
            # But per requirements, we just process. Let's return None if critical file missing.
            return None
        
        pid_json_path = pid_json_files[0]
        try:
            with open(pid_json_path, "r") as f:
                if os.stat(pid_json_path).st_size == 0:
                    pid_data = {}
                else:
                    pid_data = json.load(f)
        except json.JSONDecodeError:
            print(f"Warning: JSON decode error for {pid_json_path}. initializing empty dict.")
            pid_data = {}

        # Find the latest session or all sessions? 
        # The snippet used sorted(pid_data.keys())[-1], implying last session.
        # We will stick to that logic for now.
        if not pid_data:
            # If empty, try to infer session from directory structure
            session_dirs = sorted([d.name for d in (pid_dir / "mri_scans").iterdir() if d.is_dir()])
            if session_dirs:
                # Initialize for found sessions
                for sess in session_dirs:
                    pid_data[sess] = {}
            else:
                print(f"Error: Empty patient JSON for {pid} and no sessions found on disk")
                return None
            
        sess_id = sorted(pid_data.keys())[-1]
        
        # Locate FLAIR NIfTI
        # Path: common_data/pid_XXX/mri_scans/sess_XX/*flair*.nii*
        flair_files = list((pid_dir / "mri_scans"/sess_id).rglob("*flair*.nii*")) + list((pid_dir / "mri_scans"/sess_id).rglob("*t2f*.nii*")) + list((pid_dir / "mri_scans"/sess_id).rglob("*FLAIR*.nii*"))
        if not flair_files:
            print(f"Error: No FLAIR file found for {pid} session {sess_id}")
            return None

        mri_nii_path = flair_files[0]
        print(f"Using FLAIR: {mri_nii_path}")

        # Run prediction
        predictions, slice_results, first_slice = predict_volume_tumor(model, processor, mri_nii_path)
        mid_index, has_tumor = important_slice(predictions, slice_offset=first_slice)

        # confidence_score = slice_results.get(mid_index, 0.0)
        
        # Calculate average confidence of all non-tumor slices
        non_tumor_confidences = []
        for i, pred in enumerate(predictions):
            if pred == 0:  # non-tumor slice
                actual_slice_idx = first_slice + i
                if actual_slice_idx in slice_results:
                    non_tumor_confidences.append(slice_results[actual_slice_idx])
        
        # Use average of non-tumor slice confidences, or fallback to mid_index confidence
        if non_tumor_confidences:
            confidence_score = sum(non_tumor_confidences) / len(non_tumor_confidences)
        else:
            confidence_score = slice_results.get(mid_index, 0.0)
        
        # Old method: single slice confidence
        # confidence_score = slice_results.get(mid_index, 0.0)

        gemma_remark = "Tumor detected on FLAIR" if has_tumor else "No tumor detected on FLAIR"
        timestamp = str(np.datetime64("now"))

        print(f"[RESULTS] mid_index={mid_index}, has_tumor={has_tumor}, confidence={confidence_score:.3f}")
        print(f"[RESULTS] remark={gemma_remark}")

        # Update patient JSON
        if sess_id in pid_data:
            print(f"[UPDATE] Updating patient_results.json for session {sess_id}")
            pid_data[sess_id].update({
                "mid_idx": mid_index,
                "tumor": has_tumor, # Lowercase key
                "conf_score": float(confidence_score),
                "modality_processed": "flair",
                "gemma_hard_coded_remark": gemma_remark,
                "processed_timestamp": timestamp,
            })
            print(f"[UPDATE] Session data after update: {pid_data[sess_id]}")
        else:
            print(f"Warning: Session {sess_id} not in pid_data keys from json.")

        with open(pid_json_path, "w") as f:
            json.dump(pid_data, f, indent=4)
        print(f"[SAVED] patient_results.json written to {pid_json_path}")

        # Update master config entry
        # Finding the entry in the list again to be safe (though 'patient_entry' is a ref)
        # We need to update the entry in 'config_data' which is a list.
        
        print(f"[UPDATE] Updating comman_format.json entry for {pid}")
        patient_entry.update({
            "mid_idx": mid_index,
            "tumor": has_tumor, # Lowercase key
            "conf_score": float(confidence_score),
            "modality_processed": "flair",
            "gemma_hard_coded_remark": gemma_remark,
            # "reviewed_by_radio": False, # It was already false to get here. 
            # Requirement: "Run ... and update comman_format.json in gemma_hard_coded_remark."
            # It doesn't explicitly say set reviewed_by_radio to True, but usually pipeline implies doing the work.
            # But per logic, if we leave it false, it picks it up again?
            # Let's keep reviewed_by_radio as False or maybe we should set it to True aka 'Processed by AI'? 
            # Actually 'reviewed_by_radio' likely means Radiologist. So AI processing leaves it False.
            "processed_timestamp": timestamp,
        })
        print(f"[UPDATE] Patient entry after update: mid_idx={patient_entry.get('mid_idx')}, tumor={patient_entry.get('tumor')}")
        
        # Write back the full list
        with open(config_path, "w") as f:
            json.dump(config_data, f, indent=4)
        print(f"[SAVED] comman_format.json written to {config_path}")

        print(f"Finished patient: {pid}")
        print("=" * 100)
        return patient_entry
    
    except Exception as e:
        print(f"[ERROR] Exception processing patient {pid}: {e}")
        import traceback
        traceback.print_exc()
        return None


def process_all_pending_patients(config_path: str, model, processor):
    """Load pending patients and process each one."""
    pending, config_data = load_pending_patients(config_path)
    print(f"Found {len(pending)} patients pending review")

    for entry in pending:
        process_patient(entry, config_data, config_path, model, processor)


# ──────────────────────────────────────────────
# Entry Point
# ──────────────────────────────────────────────
def main():
    import argparse
    parser = argparse.ArgumentParser(description="MedSigLIP Tumor Detection Pipeline")
    parser.add_argument("--pid", type=str, help="Process a specific patient ID (e.g., pid_001). If not provided, processes all pending patients.")
    args = parser.parse_args()

    model, processor = load_model()

    if args.pid:
        print(f"Processing specific patient: {args.pid}")
        # Construct a dummy entry to pass to process_patient, or fetch it from config
        # process_patient needs the entry from config_data to update it.
        
        # Load config to find the patient entry
        with open(CONFIG_FILE, "r") as f:
            config_data = json.load(f)
            
        target_entry = None
        for entry in config_data:
            if entry["pid"] == args.pid:
                target_entry = entry
                break
        
        if not target_entry:
            print(f"Error: Patient {args.pid} not found in {CONFIG_FILE}")
            # Optional: Allow processing even if not in config? 
            # For now, let's assume the upload process added the entry first.
            return
            
        process_patient(target_entry, config_data, str(CONFIG_FILE), model, processor)
            
    else:
        # Option A: process patients from master JSON
        print(f"Starting pipeline on config: {CONFIG_FILE}")
        process_all_pending_patients(str(CONFIG_FILE), model, processor)

    # Option B: batch-test on BraTS volumes (Commented out)
    # root = Path(
    #     "/home/pratyakshtandon/Documents/MedGem-Hackathon/biomedia_medgemma/BraTS2020_Training_Data"
    # )
    # flair_files = natsorted(root.rglob("*flair*.nii"))[:100]
    # print(f"Found {len(flair_files)} FLAIR volumes")

    # for flair_path in flair_files:
    #     predictions, slice_results = predict_volume_tumor(model, processor, flair_path)
    #     mid_index, has_tumor = important_slice(predictions)
    #     confidence = slice_results.get(mid_index, 0.0)
    #     print(f"{flair_path.name}: tumor={has_tumor}, mid={mid_index}, conf={confidence:.3f}")


if __name__ == "__main__":
    main()