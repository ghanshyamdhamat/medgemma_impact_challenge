"""
Gradio app for interactive medical video segmentation using MedSAM2.
Please use gradio==3.38.0
"""

import datetime
import gc
from glob import glob
import hashlib
import json
import math
import multiprocessing as mp
import platform
import os
from os.path import basename, splitext, dirname, exists, join
import threading
import time
import traceback
import tempfile
os.environ["TORCH_CUDNN_SDPA_ENABLED"] = "1"
import shutil
import ffmpeg
from moviepy import ImageSequenceClip
import zipfile
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.build_sam import build_sam2_video_predictor
import cv2
import nibabel as nib
from omegaconf import OmegaConf
from simplify_report import MedGemmaSimplify

# Register custom resolvers to avoid omegaconf errors with certain configs
for resolver_name, resolver_func in [
    ("times", lambda x, y: x * y),
    ("divide", lambda x, y: x / y),
    ("plus", lambda x, y: x + y),
    ("minus", lambda x, y: x - y)
]:
    if not OmegaConf.has_resolver(resolver_name):
        try:
            OmegaConf.register_new_resolver(resolver_name, resolver_func)
        except Exception as e:
            print(f"Warning: Failed to register '{resolver_name}' resolver: {e}")


user_processes = {}
PROCESS_TIMEOUT = datetime.timedelta(minutes=15)

# Path to common_data directory containing patient folders
COMMON_DATA_PATH = "/mnt/bb586fde-943d-4653-af27-224147bfba7e/Medgemma/MedSAM2/common_data"

# Gradio temp directory for file downloads (Gradio can always serve from here)
GRADIO_TEMP_DIR = "/tmp/gradio_downloads"
os.makedirs(GRADIO_TEMP_DIR, exist_ok=True)


def copy_file_for_gradio_download(src_path):
    """Copy a file to Gradio's temp directory for reliable downloads.
    
    Gradio 3.x has issues serving files from arbitrary paths even with allowed_paths.
    Copying to /tmp ensures the file is accessible for download.
    
    Args:
        src_path: Source file path
    
    Returns:
        str: Path to the copied file in temp directory, or None if failed
    """
    if not src_path or not os.path.exists(src_path):
        return None
    try:
        filename = os.path.basename(src_path)
        # Add timestamp to avoid collisions
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        dest_filename = f"{timestamp}_{filename}"
        dest_path = os.path.join(GRADIO_TEMP_DIR, dest_filename)
        shutil.copy2(src_path, dest_path)
        print(f"[DEBUG] Copied {src_path} to {dest_path} for Gradio download")
        return dest_path
    except Exception as e:
        print(f"[ERROR] Failed to copy file for Gradio download: {e}")
        return None


# Lazy-loaded MedGemma instance (loads model in bf16 on first use)
_medgemma_instance = None

def get_medgemma_instance():
    """Get or create the MedGemma model instance. Loads in bf16 on first call."""
    global _medgemma_instance
    if _medgemma_instance is None:
        print("[INFO] Loading MedGemma model (bf16)... This may take a moment.")
        hf_token = os.getenv("HF_TOKEN")
        _medgemma_instance = MedGemmaSimplify(hf_token=hf_token)
        print("[INFO] MedGemma model loaded successfully.")
    return _medgemma_instance

def get_patient_list():
    """Get list of all patient folders in common_data directory."""
    if not exists(COMMON_DATA_PATH):
        return []
    patients = []
    for item in os.listdir(COMMON_DATA_PATH):
        patient_path = join(COMMON_DATA_PATH, item)
        if os.path.isdir(patient_path):
            patients.append(item)
    return sorted(patients)

def get_patient_sessions(patient_id):
    """Get list of all sessions for a patient."""
    patient_path = join(COMMON_DATA_PATH, patient_id)
    mri_scans_path = join(patient_path, "mri_scans")
    
    if not exists(mri_scans_path):
        return []
    
    sessions = []
    for item in os.listdir(mri_scans_path):
        session_path = join(mri_scans_path, item)
        if os.path.isdir(session_path):
            sessions.append(item)
    return sorted(sessions)

def get_most_recent_session(patient_id):
    """Get the most recent session for a patient."""
    sessions = get_patient_sessions(patient_id)
    if not sessions:
        return None
    # Assuming sessions are named in a sortable way (e.g., sess_0, sess_1, etc.)
    # The last one when sorted should be the most recent
    return sessions[-1]

def get_latest_session_from_patient_results(patient_id):
    """Get the latest session from patient_results.json file.
    
    Args:
        patient_id: Patient ID
    
    Returns:
        str: Latest session ID or None if not found
    """
    try:
        patient_folder = join(COMMON_DATA_PATH, patient_id)
        patient_results_file = join(patient_folder, "patient_results.json")
        
        if not exists(patient_results_file):
            print(f"[INFO] patient_results.json not found for {patient_id}, using filesystem sessions")
            return get_most_recent_session(patient_id)
        
        import json
        with open(patient_results_file, 'r') as f:
            patient_data = json.load(f)
        
        if not patient_data:
            print(f"[INFO] Empty patient_results.json for {patient_id}, using filesystem sessions")
            return get_most_recent_session(patient_id)
        
        # Get all session keys and sort them to find the latest
        session_keys = [key for key in patient_data.keys() if key.startswith('sess')]
        if not session_keys:
            print(f"[INFO] No sessions found in patient_results.json for {patient_id}")
            return get_most_recent_session(patient_id)
        
        # Sort sessions (assuming format like sess_01, sess_02, etc.)
        session_keys.sort()
        latest_session = session_keys[-1]
        
        print(f"[INFO] Found latest session from patient_results.json: {latest_session}")
        return latest_session
        
    except Exception as e:
        print(f"[ERROR] Error reading patient_results.json for {patient_id}: {e}")
        import traceback
        traceback.print_exc()
        # Fallback to filesystem-based session detection
        return get_most_recent_session(patient_id)

def get_latest_session_from_patient_results(patient_id):
    """Get the latest session from patient_results.json file.
    
    Args:
        patient_id: Patient ID
    
    Returns:
        str: Latest session ID or None if not found
    """
    try:
        patient_folder = join(COMMON_DATA_PATH, patient_id)
        patient_results_file = join(patient_folder, "patient_results.json")
        
        if not exists(patient_results_file):
            print(f"[INFO] patient_results.json not found for {patient_id}, using filesystem sessions")
            return get_most_recent_session(patient_id)
        
        import json
        with open(patient_results_file, 'r') as f:
            patient_data = json.load(f)
        
        if not patient_data:
            print(f"[INFO] Empty patient_results.json for {patient_id}, using filesystem sessions")
            return get_most_recent_session(patient_id)
        
        # Get all session keys and sort them to find the latest
        session_keys = [key for key in patient_data.keys() if key.startswith('sess')]
        if not session_keys:
            print(f"[INFO] No sessions found in patient_results.json for {patient_id}")
            return get_most_recent_session(patient_id)
        
        # Sort sessions (assuming format like sess_01, sess_02, etc.)
        session_keys.sort()
        latest_session = session_keys[-1]
        
        print(f"[INFO] Found latest session from patient_results.json: {latest_session}")
        return latest_session
        
    except Exception as e:
        print(f"[ERROR] Error reading patient_results.json for {patient_id}: {e}")
        import traceback
        traceback.print_exc()
        # Fallback to filesystem-based session detection
        return get_most_recent_session(patient_id)

def find_flair_scan(patient_id, session_id):
    """Find the FLAIR scan in a given session.
    
    Returns:
        str: Name of the FLAIR file, or None if not found
    """
    scans = get_session_scans(patient_id, session_id)
    
    # Look for files containing 'flair' (case-insensitive)
    for scan in scans:
        if 'flair' in scan.lower():
            return scan
    
    # If no FLAIR found, return None
    return None

def get_session_scans(patient_id, session_id):
    """Get list of all scans (nii.gz files) in a session."""
    session_path = join(COMMON_DATA_PATH, patient_id, "mri_scans", session_id)
    
    print(f"[DEBUG] get_session_scans: Checking path: {session_path}")
    
    if not exists(session_path):
        print(f"[DEBUG] get_session_scans: Path does not exist!")
        return []
    
    scans = []
    for item in os.listdir(session_path):
        if item.endswith('.nii.gz') or item.endswith('.nii'):
            scans.append(item)
    
    print(f"[DEBUG] get_session_scans: Found {len(scans)} scans: {scans}")
    return sorted(scans)

def get_patient_json_files(patient_id):
    """Get list of JSON annotation files for a patient."""
    patient_path = join(COMMON_DATA_PATH, patient_id)
    json_path = join(patient_path, "json")
    
    if not exists(json_path):
        print(f"[DEBUG] get_patient_json_files: JSON path does not exist for patient {patient_id}")
        return []
    
    json_files = []
    for item in os.listdir(json_path):
        if item.endswith('.json'):
            json_files.append(item)
    return sorted(json_files)

def load_nifti_slice(nifti_path, slice_idx=None, normalize=True):
    """Load a NIfTI file and extract a specific slice or middle slice."""
    try:
        # Load the NIfTI file
        img = nib.load(nifti_path)
        data = img.get_fdata()
        
        # Handle 4D data (e.g., time series)
        if data.ndim == 4:
            data = data[..., 0]  # Take first time point
        
        # Determine slice index
        if slice_idx is None:
            slice_idx = data.shape[2] // 2  # Middle slice
        else:
            slice_idx = min(slice_idx, data.shape[2] - 1)
        
        # Extract the slice
        slice_data = data[:, :, slice_idx]
        
        # Normalize to 0-255 range
        if normalize:
            slice_min = np.percentile(slice_data, 2)
            slice_max = np.percentile(slice_data, 98)
            slice_data = np.clip(slice_data, slice_min, slice_max)
            slice_data = ((slice_data - slice_min) / (slice_max - slice_min + 1e-5) * 255).astype(np.uint8)
        
        # Convert to RGB for display
        slice_rgb = cv2.cvtColor(slice_data, cv2.COLOR_GRAY2RGB)
        return slice_rgb, slice_idx, data.shape[2]
    except Exception as e:
        print(f"Error loading NIfTI file: {e}")
        return None, None, None

def load_annotation_from_json(patient_id, scan_name):
    """Load annotation from JSON file for a specific scan."""
    patient_path = join(COMMON_DATA_PATH, patient_id)
    json_path = join(patient_path, "json")
    
    # Try to find matching JSON file
    if not exists(json_path):
        return None, None, None
    
    try:
        # First, try to load all annotation files and find matching entries
        for json_file in os.listdir(json_path):
            if json_file.endswith('.json'):
                json_full_path = join(json_path, json_file)
                with open(json_full_path, 'r') as f:
                    annotations = json.load(f)
                
                # Check if any key in annotations matches the scan name (without extension)
                scan_base = splitext(scan_name)[0]
                for key in annotations.keys():
                    if key.lower() in scan_base.lower() or scan_base.lower() in key.lower():
                        annotation_data = annotations[key]
                        return annotation_data, json_file, key
        
        return None, None, None
    except Exception as e:
        print(f"Error loading annotation: {e}")
        return None, None, None

def get_nifti_files_in_session(patient_id, session_id):
    """Get all NIfTI files in a session with their full paths."""
    session_path = join(COMMON_DATA_PATH, patient_id, "mri_scans", session_id)
    
    if not exists(session_path):
        return {}
    
    nifti_files = {}
    for item in os.listdir(session_path):
        if item.endswith('.nii.gz') or item.endswith('.nii'):
            file_path = join(session_path, item)
            nifti_files[item] = file_path
    
    return nifti_files


def get_slice_id_for_patient_session(patient_id, session_id):
    """Get slice index for a given patient/session from patient's JSON folder.

    Reads JSON files from COMMON_DATA_PATH/patient_id/json/ with structure:
    {
        "sess_01": {"slice_id": 133},
        "sess_02": {"slice_id": 150}
    }
    
    Or can also handle scan-specific structure:
    {
        "scan_name": {"slice_id": 133, "session": "sess_01"}
    }
    """
    patient_path = join(COMMON_DATA_PATH, patient_id)
    json_path = patient_path
    print(f"[DEBUG] get_slice_id_for_patient_session: Looking for annotations in: {json_path}")
    print(f"[DEBUG] get_slice_id_for_patient_session: Looking for patient_id={patient_id}, session_id={session_id}")
    
    if not exists(json_path):
        print(f"[DEBUG] get_slice_id_for_patient_session: JSON folder does not exist")
        return None
    
    try:
        # Try to find slice_id in any JSON file in the patient's json folder
        json_files = [f for f in os.listdir(json_path) if f.endswith('.json')]
        print(f"[DEBUG] get_slice_id_for_patient_session: Found {len(json_files)} JSON files: {json_files}")
        
        for json_file in json_files:
            json_full_path = join(json_path, json_file)
            print(f"[DEBUG] get_slice_id_for_patient_session: Reading {json_file}")
            
            with open(json_full_path, 'r') as f:
                data = json.load(f)
            
            print(f"[DEBUG] get_slice_id_for_patient_session: Loaded data from {json_file}: {data}")
            
            # Try to find session_id directly in the JSON
            if session_id in data:
                slice_id = data[session_id].get("mid_idx", None)
                if slice_id is not None:
                    print(f"[DEBUG] get_slice_id_for_patient_session: Found slice_id={slice_id} for session={session_id} in {json_file}")
                    return int(slice_id)
            
            # Also check if any key contains session info
            for key, value in data.items():
                if isinstance(value, dict):
                    # Check if this entry is for our session
                    if value.get("session") == session_id or key == session_id:
                        slice_id = value.get("mid_idx", None)
                        if slice_id is not None:
                            print(f"[DEBUG] get_slice_id_for_patient_session: Found slice_id={slice_id} for session={session_id} in {json_file} under key={key}")
                            return int(slice_id)
        
        print(f"[DEBUG] get_slice_id_for_patient_session: No slice_id found for session={session_id}")
        return None
        
    except Exception as e:
        print(f"[ERROR] Error reading slice id from {json_path}: {e}")
        import traceback
        traceback.print_exc()
        return None


def nifti_to_video(patient_id, session_id, scan_name):
    """Convert full NIfTI volume into an MP4 video.

    - Returns (video_path, target_slice_idx).
    - target_slice_idx comes from common_data/anotation.json (or middle slice).
    """
    nifti_path = get_scan_full_path(patient_id, session_id, scan_name)
    if not exists(nifti_path):
        raise FileNotFoundError(f"NIfTI file not found: {nifti_path}")

    # Load volume
    try:
        img = nib.load(nifti_path)
        data = img.get_fdata()
        # Handle 4D data (time series) - take first time point
        if data.ndim == 4:
            data = data[..., 0]
    except Exception as e:
        raise RuntimeError(f"Failed to load NIfTI: {e}")

    total_slices = data.shape[2]
    
    # Check annotation for target slice
    target_slice_idx = get_slice_id_for_patient_session(patient_id, session_id)
    if target_slice_idx is None:
        target_slice_idx = total_slices // 2
    else:
        # Clamp to valid range
        target_slice_idx = max(0, min(target_slice_idx, total_slices - 1))

    os.makedirs("/tmp/nifti_videos", exist_ok=True)
    base = scan_name
    if base.endswith(".nii.gz"):
        base = base[:-7]
    elif base.endswith(".nii"):
        base = base[:-4]

    video_path = os.path.join(
        "/tmp/nifti_videos", 
        f"{patient_id}_{session_id}_{base}.mp4"
    )

    # Process slices to frames
    frames = []
    # Using global normalization usually looks better for video, but per-slice ensures visibility.
    # We'll stick to per-slice normalization as in load_nifti_slice for robustness.
    for i in range(total_slices):
        slice_data = data[:, :, i]
        slice_min = np.percentile(slice_data, 2)
        slice_max = np.percentile(slice_data, 98)
        slice_data = np.clip(slice_data, slice_min, slice_max)
        denom = slice_max - slice_min + 1e-5
        slice_norm = ((slice_data - slice_min) / denom * 255).astype(np.uint8)
        # Convert to RGB
        slice_rgb = cv2.cvtColor(slice_norm, cv2.COLOR_GRAY2RGB)
        frames.append(slice_rgb)

    # Write video
    # Use 24 fps as default for smoother scrubbing, or less if slices are few.
    # 24 is standard.
    clip = ImageSequenceClip(frames, fps=24)
    clip.write_videofile(video_path, codec="libx264", fps=24)

    print(f"Created NIfTI video {video_path} with {total_slices} frames.")
    return video_path, target_slice_idx


def get_scan_full_path(patient_id, session_id, scan_name):
    """Get the full path to a scan file."""
    return join(COMMON_DATA_PATH, patient_id, "mri_scans", session_id, scan_name)

def get_patient_data_structure():
    """Get complete hierarchical structure of all patients, sessions, and scans."""
    data = {}
    patients = get_patient_list()
    
    for patient_id in patients:
        data[patient_id] = {
            "json_files": get_patient_json_files(patient_id),
            "sessions": {}
        }
        sessions = get_patient_sessions(patient_id)
        for session_id in sessions:
            scans = get_session_scans(patient_id, session_id)
            data[patient_id]["sessions"][session_id] = scans
    
    return data

def load_patient_summary_data():
    """Load patient summary data from med_gemma_sample_data directory.
    
    Returns:
        list: List of patient dictionaries with summary information
    """
    sample_data_path = "/mnt/bb586fde-943d-4653-af27-224147bfba7e/Medgemma/MedSAM2/common_data"
    common_format_file = join(sample_data_path, "comman_format.json")
    
    if not exists(common_format_file):
        print(f"[ERROR] Common format file not found: {common_format_file}")
        return []
    
    patient_data = []
    
    try:
        with open(common_format_file, 'r') as f:
            data_list = json.load(f)
        
        # The JSON is now an array of patient objects
        for patient_entry in data_list:
            patient_id = patient_entry.get('pid', '')
            
            # Get modalities from the actual patient directory if it exists
            modalities = []
            patient_dir = join(sample_data_path, patient_id)
            if exists(patient_dir):
                # Try to get modalities from patient_results.json if available
                results_file = join(patient_dir, 'patient_results.json')
                if exists(results_file):
                    try:
                        with open(results_file, 'r') as rf:
                            results_data = json.load(rf)
                            if 'sess_0' in results_data:
                                modalities = results_data['sess_0'].get('mod', [])
                    except Exception as e:
                        print(f"[WARNING] Could not read modalities from {results_file}: {e}")
            
            patient_info = {
                'patient_id': patient_id,
                'tumor': patient_entry.get('tumor', False),
                'conf_score': patient_entry.get('conf_score', 0.0),
                'reviewed_by_radio': patient_entry.get('reviewed_by_radio', False),
                'remark': patient_entry.get('gemma_hard_coded_remark', 'N/A'),
                'modalities': ', '.join(modalities) if modalities else 'N/A',
                'mid_idx': 'N/A'
            }
            patient_data.append(patient_info)
            
    except Exception as e:
        print(f"Error loading patient data from {common_format_file}: {e}")
        import traceback
        traceback.print_exc()
    
    return patient_data

def sort_patient_data(patient_data):
    """Sort patient data according to specified rules:
    1. Not reviewed by radiologist come first
    2. Reviewed by radiologist come second
    3. Within reviewed: tumor patients before normal patients
    
    Args:
        patient_data: List of patient dictionaries
    
    Returns:
        list: Sorted patient data
    """
    def sort_key(patient):
        reviewed = patient.get('reviewed_by_radio', False)
        has_tumor = patient.get('tumor', False)
        
        # Create sort tuple: (reviewed, not has_tumor)
        # This will put not reviewed first (False < True)
        # Within reviewed, tumor (False for not has_tumor) before normal (True for not has_tumor)
        return (reviewed, not has_tumor if reviewed else False)
    
    return sorted(patient_data, key=sort_key)

def create_patient_table_html(patient_data):
    """Create HTML table for patient summary display.
    
    Args:
        patient_data: List of patient dictionaries
    
    Returns:
        str: HTML string for the table
    """
    if not patient_data:
        return "<p>No patient data available.</p>"
    
    html = """
    <style>
        .patient-table {
            width: 100%;
            border-collapse: collapse;
            font-family: Arial, sans-serif;
            margin: 20px 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            table-layout: fixed;
        }
        .patient-table thead {
            background-color: #4CAF50;
            color: white;
        }
        .patient-table th {
            padding: 8px;
            text-align: left;
            font-weight: bold;
            border: 1px solid #ddd;
            font-size: 0.9em;
        }
        .patient-table td {
            padding: 6px 8px;
            border: 1px solid #ddd;
            color: #ffffff;
            font-size: 0.85em;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }
        .patient-table tbody tr.reviewed-row {
            background-color: #1b5e20;
        }
        .patient-table tbody tr.not-reviewed-row {
            background-color: #b71c1c;
        }
        .patient-table tbody tr.reviewed-row:hover {
            background-color: #2e7d32;
        }
        .patient-table tbody tr.not-reviewed-row:hover {
            background-color: #c62828;
        }
        .patient-table th:nth-child(1),
        .patient-table td:nth-child(1) {
            width: 4%;
        }
        .patient-table th:nth-child(2),
        .patient-table td:nth-child(2) {
            width: 10%;
        }
        .patient-table th:nth-child(3),
        .patient-table td:nth-child(3) {
            width: 13%;
        }
        .patient-table th:nth-child(4),
        .patient-table td:nth-child(4) {
            width: 9%;
        }
        .patient-table th:nth-child(5),
        .patient-table td:nth-child(5) {
            width: 9%;
        }
        .patient-table th:nth-child(6),
        .patient-table td:nth-child(6) {
            width: 40%;
        }
        .patient-table th:nth-child(7),
        .patient-table td:nth-child(7) {
            width: 15%;
        }
        .tumor-yes {
            color: #ff6b6b;
            font-weight: bold;
        }
        .tumor-no {
            color: #51cf66;
            font-weight: bold;
        }
        .reviewed-yes {
            color: #a5d6a7;
        }
        .reviewed-no {
            color: #ef9a9a;
        }
        .conf-score {
            font-weight: bold;
        }
        .row-number {
            color: #e0e0e0;
            font-size: 0.85em;
        }
    </style>
    <table class="patient-table">
        <thead>
            <tr>
                <th>#</th>
                <th>Patient ID</th>
                <th>Tumor Status</th>
                <th>Conf Score</th>
                <th>Reviewed</th>
                <th>Remark</th>
                <th>Modalities</th>
            </tr>
        </thead>
        <tbody>
    """
    
    for idx, patient in enumerate(patient_data):
        tumor_class = "tumor-yes" if patient['tumor'] else "tumor-no"
        tumor_text = "Tumor Present" if patient['tumor'] else "Normal"
        
        reviewed_class = "reviewed-yes" if patient['reviewed_by_radio'] else "reviewed-no"
        reviewed_text = "Yes" if patient['reviewed_by_radio'] else "No"
        
        # Add row class based on reviewed status
        row_class = "reviewed-row" if patient['reviewed_by_radio'] else "not-reviewed-row"
        
        conf_score = patient['conf_score']
        
        html += f"""
            <tr class="{row_class}">
                <td class="row-number">{idx + 1}</td>
                <td><strong>{patient['patient_id']}</strong></td>
                <td class="{tumor_class}">{tumor_text}</td>
                <td class="conf-score">{conf_score:.2f}</td>
                <td class="{reviewed_class}">{reviewed_text}</td>
                <td title="{patient['remark']}">{patient['remark']}</td>
                <td title="{patient['modalities']}">{patient['modalities']}</td>
            </tr>
        """
    
    html += """
        </tbody>
    </table>
    """
    
    return html

def reset(seg_tracker):
    if seg_tracker is not None:
        predictor, inference_state, image_predictor = seg_tracker
        predictor.reset_state(inference_state)
        del predictor
        del inference_state
        del image_predictor
        del seg_tracker
        gc.collect()
        torch.cuda.empty_cache()
    return None, ({}, {}), None, None, 0, None, None, None, 0, 0, 

def extract_video_info(input_video):
    if input_video is None:
        return 4, 4, None, None, None, None, None
    cap = cv2.VideoCapture(input_video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return fps, total_frames, None, None, None, None, None

def get_meta_from_video(session_id, input_video, scale_slider, config_path, checkpoint_path, target_slice=None):
    output_dir = f'/tmp/output_frames/{session_id}'
    output_masks_dir = f'/tmp/output_masks/{session_id}'
    output_combined_dir = f'/tmp/output_combined/{session_id}'
    clear_folder(output_dir)
    clear_folder(output_masks_dir)
    clear_folder(output_combined_dir)
    if input_video is None:
        return None, ({}, {}), None, None, (4, 1, 4), None, None, None, 0, 0
    cap = cv2.VideoCapture(input_video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    frame_interval = max(1, int(fps // scale_slider))
    print(f"frame_interval: {frame_interval}")
    try:
        ffmpeg.input(input_video, hwaccel='cuda').output(
            os.path.join(output_dir, '%07d.jpg'), q=2, start_number=0, 
            vf=rf'select=not(mod(n\,{frame_interval}))', vsync='vfr'
        ).run()
    except:
        print(f"ffmpeg cuda err")
        ffmpeg.input(input_video).output(
            os.path.join(output_dir, '%07d.jpg'), q=2, start_number=0, 
            vf=rf'select=not(mod(n\,{frame_interval}))', vsync='vfr'
        ).run()

    # Determine which frame to show first.
    # If target_slice is provided, we try to show that frame.
    # Note: ffmpeg extraction with 'select' and 'vsync=vfr' might skip frames or renumber them.
    # The output files are named sequentially 0000000.jpg, 0000001.jpg etc.
    # If frame_interval is 1, then 0000000.jpg is frame 0, 0000133.jpg is frame 133.
    target_file_idx = 0
    if target_slice is not None:
        target_file_idx = target_slice // frame_interval
    
    first_frame_path = os.path.join(output_dir, f'{target_file_idx:07d}.jpg')
    if not os.path.exists(first_frame_path):
        print(f"Frame {first_frame_path} not found, falling back to 0")
        first_frame_path = os.path.join(output_dir, '0000000.jpg')
    
    first_frame = cv2.imread(first_frame_path)
    if first_frame is None: 
         # Extremely unlikely unless video failed entirely
         first_frame = np.zeros((256, 256, 3), dtype=np.uint8)

    first_frame_rgb = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)

    torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
   
    predictor = build_sam2_video_predictor(config_path, checkpoint_path, device="cuda")
    sam2_model = build_sam2(config_path, checkpoint_path, device="cuda")
    image_predictor = SAM2ImagePredictor(sam2_model)
    inference_state = predictor.init_state(video_path=output_dir)
    predictor.reset_state(inference_state)
    return (predictor, inference_state, image_predictor), ({}, {}), first_frame_rgb, first_frame_rgb, (fps, frame_interval, total_frames), None, None, None, 0, 0

def mask2bbox(mask):
    if len(np.where(mask > 0)[0]) == 0:
        print(f'not mask')
        return np.array([0, 0, 0, 0]).astype(np.int64), False
    x_ = np.sum(mask, axis=0)
    y_ = np.sum(mask, axis=1)
    x0 = np.min(np.nonzero(x_)[0])
    x1 = np.max(np.nonzero(x_)[0])
    y0 = np.min(np.nonzero(y_)[0])
    y1 = np.max(np.nonzero(y_)[0])
    return np.array([x0, y0, x1, y1]).astype(np.int64), True

def sam_stroke(session_id, seg_tracker, drawing_board, last_draw, frame_num, ann_obj_id):
    predictor, inference_state, image_predictor = seg_tracker
    image_path = f'/tmp/output_frames/{session_id}/{frame_num:07d}.jpg'
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    display_image = drawing_board["image"]
    image_predictor.set_image(image)
    input_mask = drawing_board["mask"]
    input_mask[input_mask != 0] = 255
    if last_draw is not None:
        diff_mask = cv2.absdiff(input_mask, last_draw)
        input_mask = diff_mask
    bbox, hasMask = mask2bbox(input_mask[:, :, 0]) 
    if not hasMask :
        return seg_tracker, display_image, display_image, None
    masks, scores, logits = image_predictor.predict( point_coords=None, point_labels=None, box=bbox[None, :], multimask_output=False,)
    mask = masks > 0.0
    masked_frame = show_mask(mask, display_image, ann_obj_id)
    masked_with_rect = draw_rect(masked_frame, bbox, ann_obj_id)
    frame_idx, object_ids, masks = predictor.add_new_mask(inference_state, frame_idx=frame_num, obj_id=ann_obj_id, mask=mask[0])
    last_draw = drawing_board["mask"]
    return seg_tracker, masked_with_rect, masked_with_rect, last_draw

def draw_rect(image, bbox, obj_id):
    cmap = plt.get_cmap("tab10")
    color = np.array(cmap(obj_id)[:3])
    rgb_color = tuple(map(int, (color[:3] * 255).astype(np.uint8)))
    inv_color = tuple(map(int, (255 - color[:3] * 255).astype(np.uint8)))
    x0, y0, x1, y1 = bbox
    image_with_rect = cv2.rectangle(image.copy(), (x0, y0), (x1, y1), rgb_color, thickness=2)
    return image_with_rect

def sam_click(session_id, seg_tracker, frame_num, point_mode, click_stack, ann_obj_id, point):
    points_dict, labels_dict = click_stack
    predictor, inference_state, image_predictor = seg_tracker
    ann_frame_idx = frame_num  # the frame index we interact with
    print(f'ann_frame_idx: {ann_frame_idx}')
    if point_mode == "Positive":
        label = np.array([1], np.int32)
    else:
        label = np.array([0], np.int32)

    if ann_frame_idx not in points_dict:
        points_dict[ann_frame_idx] = {}
    if ann_frame_idx not in labels_dict:
        labels_dict[ann_frame_idx] = {}

    if ann_obj_id not in points_dict[ann_frame_idx]:
        points_dict[ann_frame_idx][ann_obj_id] = np.empty((0, 2), dtype=np.float32)
    if ann_obj_id not in labels_dict[ann_frame_idx]:
        labels_dict[ann_frame_idx][ann_obj_id] = np.empty((0,), dtype=np.int32)

    points_dict[ann_frame_idx][ann_obj_id] = np.append(points_dict[ann_frame_idx][ann_obj_id], point, axis=0)
    labels_dict[ann_frame_idx][ann_obj_id] = np.append(labels_dict[ann_frame_idx][ann_obj_id], label, axis=0)

    click_stack = (points_dict, labels_dict)

    frame_idx, out_obj_ids, out_mask_logits = predictor.add_new_points(
        inference_state=inference_state,
        frame_idx=ann_frame_idx,
        obj_id=ann_obj_id,
        points=points_dict[ann_frame_idx][ann_obj_id],
        labels=labels_dict[ann_frame_idx][ann_obj_id],
    )

    image_path = f'/tmp/output_frames/{session_id}/{ann_frame_idx:07d}.jpg'
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    masked_frame = image.copy()
    for i, obj_id in enumerate(out_obj_ids):
        mask = (out_mask_logits[i] > 0.0).cpu().numpy()
        masked_frame = show_mask(mask, image=masked_frame, obj_id=obj_id)
    masked_frame_with_markers = draw_markers(masked_frame, points_dict[ann_frame_idx], labels_dict[ann_frame_idx])

    return seg_tracker, masked_frame_with_markers, masked_frame_with_markers, click_stack

def draw_markers(image, points_dict, labels_dict):
    cmap = plt.get_cmap("tab10")
    image_h, image_w = image.shape[:2]
    marker_size = max(1, int(min(image_h, image_w) * 0.05))

    for obj_id in points_dict:
        color = np.array(cmap(obj_id)[:3])
        rgb_color = tuple(map(int, (color[:3] * 255).astype(np.uint8)))
        inv_color = tuple(map(int, (255 - color[:3] * 255).astype(np.uint8)))
        for point, label in zip(points_dict[obj_id], labels_dict[obj_id]):
            x, y = int(point[0]), int(point[1])
            if label == 1:
                cv2.drawMarker(image, (x, y), inv_color, markerType=cv2.MARKER_CROSS, markerSize=marker_size, thickness=2)
            else:
                cv2.drawMarker(image, (x, y), inv_color, markerType=cv2.MARKER_TILTED_CROSS, markerSize=int(marker_size / np.sqrt(2)), thickness=2)
    
    return image

def show_mask(mask, image=None, obj_id=None):
    cmap = plt.get_cmap("tab10")
    cmap_idx = 0 if obj_id is None else obj_id
    color = np.array([*cmap(cmap_idx)[:3], 0.6])
    
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    mask_image = (mask_image * 255).astype(np.uint8)
    if image is not None:
        image_h, image_w = image.shape[:2]
        if (image_h, image_w) != (h, w):
            raise ValueError(f"Image dimensions ({image_h}, {image_w}) and mask dimensions ({h}, {w}) do not match")
        colored_mask = np.zeros_like(image, dtype=np.uint8)
        for c in range(3):
            colored_mask[..., c] = mask_image[..., c]
        alpha_mask = mask_image[..., 3] / 255.0
        for c in range(3):
            image[..., c] = np.where(alpha_mask > 0, (1 - alpha_mask) * image[..., c] + alpha_mask * colored_mask[..., c], image[..., c])
        return image
    return mask_image

def show_res_by_slider(session_id, frame_per, click_stack):
    image_path = f'/tmp/output_frames/{session_id}'
    output_combined_dir = f'/tmp/output_combined/{session_id}'
    
    combined_frames = sorted([os.path.join(output_combined_dir, img_name) for img_name in os.listdir(output_combined_dir)])
    if combined_frames:
        output_masked_frame_path = combined_frames
    else:
        original_frames = sorted([os.path.join(image_path, img_name) for img_name in os.listdir(image_path)])
        output_masked_frame_path = original_frames
       
    total_frames_num = len(output_masked_frame_path)
    if total_frames_num == 0:
        print("No output results found")
        return None, None, 0
    else:
        frame_num = math.floor(total_frames_num * frame_per)
        if frame_num >= total_frames_num:
            frame_num = total_frames_num - 1
        chosen_frame_path = output_masked_frame_path[frame_num]
        print(f"{chosen_frame_path}")
        chosen_frame_show = cv2.imread(chosen_frame_path)
        chosen_frame_show = cv2.cvtColor(chosen_frame_show, cv2.COLOR_BGR2RGB)
        points_dict, labels_dict = click_stack
        if frame_num in points_dict and frame_num in labels_dict:
            chosen_frame_show = draw_markers(chosen_frame_show, points_dict[frame_num], labels_dict[frame_num])
        return chosen_frame_show, chosen_frame_show, frame_num

def clear_folder(folder_path):
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
    os.makedirs(folder_path)

def zip_folder(folder_path, output_zip_path):
    with zipfile.ZipFile(output_zip_path, 'w', zipfile.ZIP_STORED) as zipf:
        for root, _, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                zipf.write(file_path, os.path.relpath(file_path, folder_path))

def convert_masks_to_nifti(session_id, original_nifti_path, obj_id=0, patient_id=None, session_id_selected=None):
    """Convert segmented masks back to NIfTI format and calculate volume.
    
    Args:
        session_id: User session ID
        original_nifti_path: Path to the original NIfTI file
        obj_id: Object ID to convert (default 0)
        patient_id: Patient ID for saving to patient folder
        session_id_selected: Session ID for saving to patient folder
    
    Returns:
        tuple: (output_nifti_path, volume_ml, voxel_count, report_path, patient_seg_path)
    """
    output_masks_dir = f'/tmp/output_masks/{session_id}'
    output_files_dir = f'/tmp/output_files/{session_id}'
    
    if not os.path.exists(output_masks_dir):
        raise FileNotFoundError(f"No masks found for session {session_id}")
    
    # Load original NIfTI to get header, affine, and dimensions
    try:
        original_img = nib.load(original_nifti_path)
        original_data = original_img.get_fdata()
        
        # Handle 4D data (time series) - take first time point
        if original_data.ndim == 4:
            original_data = original_data[..., 0]
        
        affine = original_img.affine
        header = original_img.header.copy()
        
        # Get voxel spacing (in mm)
        voxel_spacing = header.get_zooms()[:3]  # (x, y, z) spacing in mm
        print(f"[DEBUG] Voxel spacing: {voxel_spacing} mm")
        
        # Calculate voxel volume in mm³
        voxel_volume_mm3 = np.prod(voxel_spacing)
        print(f"[DEBUG] Voxel volume: {voxel_volume_mm3} mm³")
        
    except Exception as e:
        raise RuntimeError(f"Failed to load original NIfTI: {e}")
    
    # Get all mask files for the specified object ID
    mask_files = sorted([
        f for f in os.listdir(output_masks_dir) 
        if f.startswith(f'{obj_id}_') and f.endswith('.png')
    ])
    
    if not mask_files:
        raise FileNotFoundError(f"No masks found for object ID {obj_id}")
    
    print(f"[DEBUG] Found {len(mask_files)} mask files for object {obj_id}")
    
    # Initialize 3D volume with same shape as original
    segmentation_volume = np.zeros(original_data.shape, dtype=np.uint8)
    
    # Load each mask and place it in the correct slice
    for mask_file in mask_files:
        # Extract frame index from filename: {obj_id}_{frame_idx:07d}.png
        frame_idx_str = mask_file.split('_')[1].split('.')[0]
        frame_idx = int(frame_idx_str)
        
        # Load mask
        mask_path = os.path.join(output_masks_dir, mask_file)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        if mask is None:
            print(f"[WARNING] Failed to load mask: {mask_path}")
            continue
        
        # Resize mask to match original slice dimensions if needed
        if mask.shape != original_data.shape[:2]:
            mask = cv2.resize(mask, (original_data.shape[1], original_data.shape[0]), 
                            interpolation=cv2.INTER_NEAREST)
        
        # Binarize mask (threshold at 127)
        mask_binary = (mask > 127).astype(np.uint8)
        
        # Place mask in the corresponding slice
        if frame_idx < segmentation_volume.shape[2]:
            segmentation_volume[:, :, frame_idx] = mask_binary
    
    # Count segmented voxels
    voxel_count = np.sum(segmentation_volume > 0)
    
    # Calculate volume in mm³ and convert to mL (1 mL = 1000 mm³)
    volume_mm3 = voxel_count * voxel_volume_mm3
    volume_ml = volume_mm3 / 1000.0
    
    print(f"[INFO] Segmented voxel count: {voxel_count}")
    print(f"[INFO] Total volume: {volume_mm3:.2f} mm³ = {volume_ml:.2f} mL")
    
    # Save as NIfTI
    output_nifti_path = os.path.join(output_files_dir, f'segmentation_obj{obj_id}.nii.gz')
    segmentation_img = nib.Nifti1Image(segmentation_volume, affine, header)
    nib.save(segmentation_img, output_nifti_path)
    
    print(f"[INFO] Saved segmentation to: {output_nifti_path}")
    
    # Also save to patient folder if patient_id and session_id_selected are provided
    patient_seg_path = None
    if patient_id and session_id_selected:
        patient_folder = join(COMMON_DATA_PATH, patient_id)
        os.makedirs(patient_folder, exist_ok=True)
        patient_seg_path = join(patient_folder, f"{patient_id}_{session_id_selected}_seg.nii.gz")
        nib.save(segmentation_img, patient_seg_path)
        print(f"[INFO] Saved segmentation to patient folder: {patient_seg_path}")
    
    # Also save volume report as text file
    report_path = os.path.join(output_files_dir, f'volume_report_obj{obj_id}.txt')
    with open(report_path, 'w') as f:
        f.write(f"Segmentation Volume Report\n")
        f.write(f"=========================\n\n")
        f.write(f"Object ID: {obj_id}\n")
        f.write(f"Original NIfTI: {original_nifti_path}\n")
        f.write(f"Voxel spacing: {voxel_spacing[0]:.4f} x {voxel_spacing[1]:.4f} x {voxel_spacing[2]:.4f} mm\n")
        f.write(f"Voxel volume: {voxel_volume_mm3:.4f} mm³\n")
        f.write(f"Segmented voxel count: {voxel_count}\n")
        f.write(f"Total volume: {volume_mm3:.2f} mm³\n")
        f.write(f"Total volume: {volume_ml:.2f} mL\n")
    
    print(f"[INFO] Saved volume report to: {report_path}")
    
    return output_nifti_path, volume_ml, voxel_count, report_path, patient_seg_path

def run_parcellation_analysis(patient_id, session_id_selected, segmentation_path):
    """Run parcellation analysis using parcellation_by_registration.py.
    
    Args:
        patient_id: Patient ID
        session_id_selected: Session ID
        segmentation_path: Path to segmentation NIfTI file
    
    Returns:
        tuple: (parcellation_result_string, parcellation_output_path) or (None, None) if error
    """
    try:
        # Find T1 scan in the session
        session_path = join(COMMON_DATA_PATH, patient_id, "mri_scans", session_id_selected)
        if not exists(session_path):
            print(f"[ERROR] Session path not found: {session_path}")
            return None, None
        
        # Look for T1 scan (case-insensitive)
        t1_scan = None
        for scan_file in os.listdir(session_path):
            if 't1' in scan_file.lower() and (scan_file.endswith('.nii.gz') or scan_file.endswith('.nii')):
                t1_scan = join(session_path, scan_file)
                break
        
        if not t1_scan:
            print(f"[WARNING] No T1 scan found in session {session_id_selected}, trying to use FLAIR or first available scan")
            # Try FLAIR as fallback
            for scan_file in os.listdir(session_path):
                if 'flair' in scan_file.lower() and (scan_file.endswith('.nii.gz') or scan_file.endswith('.nii')):
                    t1_scan = join(session_path, scan_file)
                    break
            
            # If still no scan, use first .nii.gz file
            if not t1_scan:
                nii_files = [f for f in os.listdir(session_path) if f.endswith('.nii.gz') or f.endswith('.nii')]
                if nii_files:
                    t1_scan = join(session_path, nii_files[0])
        
        if not t1_scan:
            print(f"[ERROR] No scan found for parcellation")
            return None, None
        
        print(f"[INFO] Using scan for parcellation: {t1_scan}")
        
        # Set up paths for parcellation script
        parcellation_script = "/mnt/bb586fde-943d-4653-af27-224147bfba7e/Medgemma/MedSAM2/parcellations/parcellation_by_registration.py"
        mni_t1_path = "/mnt/bb586fde-943d-4653-af27-224147bfba7e/Medgemma/MedSAM2/parcellations/MNI152_T1_1mm_Brain.nii.gz"
        mni_parcellation_path = "/mnt/bb586fde-943d-4653-af27-224147bfba7e/Medgemma/MedSAM2/parcellations/aparc.DKTatlas+aseg.deep.mgz"
        lut_path = "/mnt/bb586fde-943d-4653-af27-224147bfba7e/Medgemma/MedSAM2/parcellations/FreeSurferColorLUT.txt"
        
        if not exists(parcellation_script):
            print(f"[ERROR] Parcellation script not found: {parcellation_script}")
            return None, None
        
        # Run parcellation script
        import subprocess
        cmd = [
            "python",
            parcellation_script,
            t1_scan,
            segmentation_path,
            mni_t1_path,
            mni_parcellation_path,
            lut_path
        ]
        
        print(f"[INFO] Running parcellation: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        
        if result.returncode != 0:
            print(f"[ERROR] Parcellation script failed with return code {result.returncode}")
            print(f"[ERROR] stderr: {result.stderr}")
            return None, None
        
        # Parse output - looking for the line with tumor volume and regions
        output_lines = result.stdout.strip().split('\n')
        parcellation_result = None
        for line in output_lines:
            if 'ml tumor extending into' in line:
                parcellation_result = line.strip()
                break
        
        print(f"[INFO] Parcellation result: {parcellation_result}")
        
        # Save parcellation output to patient folder
        parcellation_output_path = None
        if parcellation_result:
            patient_folder = join(COMMON_DATA_PATH, patient_id)
            os.makedirs(patient_folder, exist_ok=True)
            parcellation_output_path = join(patient_folder, f"{patient_id}_{session_id_selected}_parcellation.txt")
            with open(parcellation_output_path, 'w') as f:
                f.write(f"Parcellation Analysis Report\n")
                f.write(f"===========================\n\n")
                f.write(f"Patient ID: {patient_id}\n")
                f.write(f"Session ID: {session_id_selected}\n")
                f.write(f"Segmentation: {segmentation_path}\n")
                f.write(f"T1 Scan: {t1_scan}\n\n")
                f.write(f"Result:\n{parcellation_result}\n\n")
                f.write(f"Full Output:\n")
                f.write(result.stdout)
            print(f"[INFO] Saved parcellation output to: {parcellation_output_path}")
        
        return parcellation_result, parcellation_output_path
        
    except Exception as e:
        print(f"[ERROR] Failed to run parcellation: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def update_patient_results_json(patient_id, session_id_selected, parcellation_result, segmentation_path, parcellation_path):
    """Update patient's patient_results.json file with session-specific parcellation data.
    
    Args:
        patient_id: Patient ID
        session_id_selected: Session ID
        parcellation_result: Parcellation result string
        segmentation_path: Path to segmentation file
        parcellation_path: Path to parcellation output file
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        patient_folder = join(COMMON_DATA_PATH, patient_id)
        patient_results_file = join(patient_folder, "patient_results.json")
        
        import json
        
        # Load or create patient_results.json
        if exists(patient_results_file):
            with open(patient_results_file, 'r') as f:
                patient_data = json.load(f)
        else:
            patient_data = {}
        
        # Update session data
        session_key = session_id_selected.replace('_', '_')  # Keep original format
        if session_key not in patient_data:
            patient_data[session_key] = {}
        
        # Update with parcellation results
        patient_data[session_key]["manual report from SAM"] = parcellation_result
        patient_data[session_key]["SAM segmentation file path"] = segmentation_path
        patient_data[session_key]["parcellation file path"] = parcellation_path if parcellation_path else "N/A"
        
        # Save updated data
        with open(patient_results_file, 'w') as f:
            json.dump(patient_data, f, indent=4)
        
        print(f"[INFO] Updated patient_results.json for {patient_id}, session {session_id_selected}")
        return True
        
    except Exception as e:
        print(f"[ERROR] Failed to update patient_results.json: {e}")
        import traceback
        traceback.print_exc()
        return False

def update_patient_json_with_parcellation(patient_id, parcellation_result):
    """Update patient's JSON file with parcellation result.
    
    Args:
        patient_id: Patient ID
        parcellation_result: Parcellation result string
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        patient_path = join(COMMON_DATA_PATH, patient_id)
        json_path = patient_path
        if not exists(json_path):
            print(f"[ERROR] JSON folder not found: {json_path}")
            return False
        
        # Find patient's JSON file in med_gemma_sample_data
        sample_data_path = COMMON_DATA_PATH
        common_format_file = join(sample_data_path, "comman_format.json")
        
        if not exists(common_format_file):
            print(f"[ERROR] Common format JSON not found: {common_format_file}")
            return False
        
        # Load and update JSON
        import json
        with open(common_format_file, 'r') as f:
            data = json.load(f)
        
        # Find the patient entry (using 'pid' field from comman_format.json)
        updated = False
        for entry in data:
            if entry.get("pid") == patient_id:
                entry["manual report from SAM"] = parcellation_result
                updated = True
                print(f"[INFO] Updated patient {patient_id} with parcellation result in comman_format.json")
                break
        
        if not updated:
            print(f"[WARNING] Patient {patient_id} not found in comman_format.json, adding new entry")
            # Create new entry if patient not found
            data.append({
                "pid": patient_id,
                "manual report from SAM": parcellation_result
            })
        
        # Save updated JSON
        with open(common_format_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"[INFO] Successfully updated JSON file")
        return True
        
    except Exception as e:
        print(f"[ERROR] Failed to update patient JSON: {e}")
        import traceback
        traceback.print_exc()
        return False

def update_patient_results_json_with_medgemma(patient_id, session_id_selected, clinical_report, patient_report):
    """Update patient's session JSON file with MedGemma results.
    
    Args:
        patient_id: Patient ID
        session_id_selected: Session ID
        clinical_report: MedGemma clinical/radiology report string
        patient_report: MedGemma patient-friendly report string
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        patient_folder = join(COMMON_DATA_PATH, patient_id)
        patient_results_file = join(patient_folder, "patient_results.json")
        
        if not exists(patient_folder):
            print(f"[ERROR] Patient folder not found: {patient_folder}")
            return False
            
        # Load or create JSON
        if exists(patient_results_file):
            with open(patient_results_file, 'r') as f:
                patient_data = json.load(f)
        else:
            patient_data = {}
            
        # Ensure session key exists
        session_key = session_id_selected
        if session_key not in patient_data:
            patient_data[session_key] = {}
            
        # Update fields - store clinical and patient reports separately
        patient_data[session_key]["gemma radiology report"] = clinical_report
        patient_data[session_key]["gemma patient report"] = patient_report
        
        # Save updated data
        with open(patient_results_file, 'w') as f:
            json.dump(patient_data, f, indent=4)
        
        print(f"[INFO] Updated patient_results.json with both MedGemma reports for {patient_id}, session {session_id_selected}")
        return True
        
    except Exception as e:
        print(f"[ERROR] Failed to update patient_results.json for MedGemma: {e}")
        import traceback
        traceback.print_exc()
        return False

def update_comman_format_json_with_medgemma(patient_id, clinical_report, patient_report):
    """Update comman_format.json with both MedGemma reports.
    
    Args:
        patient_id: Patient ID
        clinical_report: MedGemma clinical/radiology report string
        patient_report: MedGemma patient-friendly report string
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Find patient's JSON file in med_gemma_sample_data
        sample_data_path = COMMON_DATA_PATH
        common_format_file = join(sample_data_path, "comman_format.json")
        
        if not exists(common_format_file):
            print(f"[ERROR] Common format JSON not found: {common_format_file}")
            return False
        
        # Load and update JSON
        import json
        with open(common_format_file, 'r') as f:
            data = json.load(f)
        
        # Find the patient entry (using 'pid' field from comman_format.json)
        updated = False
        for entry in data:
            if entry.get("pid") == patient_id:
                entry["gemma radiology report"] = clinical_report
                entry["gemma patient report"] = patient_report
                updated = True
                print(f"[INFO] Updated patient {patient_id} with both MedGemma reports in comman_format.json")
                break
        
        if not updated:
            print(f"[WARNING] Patient {patient_id} not found in comman_format.json, skipping MedGemma update for summary")
        
        # Save updated JSON
        with open(common_format_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"[INFO] Successfully updated comman_format.json with MedGemma reports")
        return True
        
    except Exception as e:
        print(f"[ERROR] Failed to update comman_format.json for MedGemma: {e}")
        import traceback
        traceback.print_exc()
        return False


def generate_medgemma_reports(patient_id, session_id_selected, scan_name, segmentation_path, parcellation_result=None):
    """Generate clinical and patient-friendly reports using MedGemma.
    
    Args:
        patient_id: Patient ID
        session_id_selected: Session ID
        scan_name: Name of the scan file
        segmentation_path: Path to segmentation NIfTI file
        parcellation_result: Optional parcellation result string for context
    
    Returns:
        tuple: (clinical_report, patient_report, clinical_docx_path, patient_docx_path)
    """
    try:
        # Get MRI volume path
        mr_volume_path = get_scan_full_path(patient_id, session_id_selected, scan_name)
        if not exists(mr_volume_path):
            raise FileNotFoundError(f"MRI volume not found: {mr_volume_path}")
        
        if not exists(segmentation_path):
            raise FileNotFoundError(f"Segmentation file not found: {segmentation_path}")
        
        # Determine slice index from annotations
        slice_index = get_slice_id_for_patient_session(patient_id, session_id_selected)
        if slice_index is None:
            # Fallback to middle slice
            img = nib.load(mr_volume_path)
            data = img.get_fdata()
            if data.ndim == 4:
                data = data[..., 0]
            slice_index = data.shape[2] // 2
            print(f"[INFO] No annotation slice found, using middle slice: {slice_index}")
        
        # Build manual report context from parcellation result
        if parcellation_result:
            manual_report_context = parcellation_result
        else:
            manual_report_context = "No parcellation data available. Please describe findings based on imaging alone."
        
        # Output paths for DOCX files
        patient_folder = join(COMMON_DATA_PATH, patient_id)
        os.makedirs(patient_folder, exist_ok=True)
        clinical_docx_path = join(patient_folder, f"{patient_id}_{session_id_selected}_Clinical_MRI_Report.docx")
        patient_docx_path = join(patient_folder, f"{patient_id}_{session_id_selected}_Patient_Friendly_Report.docx")
        
        # Load MedGemma and generate reports
        simplifier = get_medgemma_instance()
        
        print(f"[INFO] Generating MedGemma reports for patient {patient_id}, session {session_id_selected}")
        print(f"[INFO] MRI volume: {mr_volume_path}")
        print(f"[INFO] Segmentation: {segmentation_path}")
        print(f"[INFO] Slice index: {slice_index}")
        
        clinical_report, patient_report = simplifier.generate_reports(
            mr_volume_path=mr_volume_path,
            slice_index=slice_index,
            segmentation_mask_path=segmentation_path,
            manual_report_context=manual_report_context,
            clinical_docx=clinical_docx_path,
            patient_docx=patient_docx_path
        )
        
        print(f"[INFO] MedGemma reports generated successfully.")
        print(f"[INFO] Clinical report saved to: {clinical_docx_path}")
        print(f"[INFO] Patient report saved to: {patient_docx_path}")
        
        return clinical_report, patient_report, clinical_docx_path, patient_docx_path
        
    except Exception as e:
        print(f"[ERROR] Failed to generate MedGemma reports: {e}")
        traceback.print_exc()
        return None, None, None, None

def tracking_objects(session_id, seg_tracker, frame_num, input_video):
    output_dir = f'/tmp/output_frames/{session_id}'
    output_masks_dir = f'/tmp/output_masks/{session_id}'
    output_combined_dir = f'/tmp/output_combined/{session_id}'
    output_files_dir = f'/tmp/output_files/{session_id}'
    output_video_path = f'{output_files_dir}/output_video.mp4'
    output_zip_path = f'{output_files_dir}/output_masks.zip'
    clear_folder(output_masks_dir)
    clear_folder(output_combined_dir)
    clear_folder(output_files_dir)
    video_segments = {}
    predictor, inference_state, image_predictor = seg_tracker
    
    print(f"[DEBUG] Starting bidirectional tracking from frame {frame_num}")
    
    # Propagate forward from the annotated frame
    print(f"[DEBUG] Propagating forward...")
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state, start_frame_idx=frame_num):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }
    
    # Propagate backward from the annotated frame
    print(f"[DEBUG] Propagating backward...")
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state, start_frame_idx=frame_num, reverse=True):
        # Only add if not already processed (avoid overwriting the annotated frame)
        if out_frame_idx not in video_segments:
            video_segments[out_frame_idx] = {
                out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                for i, out_obj_id in enumerate(out_obj_ids)
            }
    
    print(f"[DEBUG] Bidirectional tracking complete. Processed {len(video_segments)} frames.")
    frame_files = sorted([f for f in os.listdir(output_dir) if f.endswith('.jpg')])
    # for frame_idx in sorted(video_segments.keys()):
    for frame_file in frame_files:
        frame_idx = int(os.path.splitext(frame_file)[0])
        frame_path = os.path.join(output_dir, frame_file)
        image = cv2.imread(frame_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        masked_frame = image.copy()
        if frame_idx in video_segments:
            for obj_id, mask in video_segments[frame_idx].items():
                masked_frame = show_mask(mask, image=masked_frame, obj_id=obj_id)
                mask_output_path = os.path.join(output_masks_dir, f'{obj_id}_{frame_idx:07d}.png')
                # Save binary mask (not colored visualization)
                # Convert boolean mask to uint8: True->255, False->0
                binary_mask = (mask.squeeze() * 255).astype(np.uint8)
                cv2.imwrite(mask_output_path, binary_mask)
        combined_output_path = os.path.join(output_combined_dir, f'{frame_idx:07d}.png')
        combined_image_bgr = cv2.cvtColor(masked_frame, cv2.COLOR_RGB2BGR)
        cv2.imwrite(combined_output_path, combined_image_bgr)
        if frame_idx == frame_num:
            final_masked_frame = masked_frame

    cap = cv2.VideoCapture(input_video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    # output_frames = int(total_frames * scale_slider)
    output_frames = len([name for name in os.listdir(output_combined_dir) if os.path.isfile(os.path.join(output_combined_dir, name)) and name.endswith('.png')])
    out_fps = fps * output_frames / total_frames

    # ffmpeg.input(os.path.join(output_combined_dir, '%07d.png'), framerate=out_fps).output(output_video_path, vcodec='h264_nvenc', pix_fmt='yuv420p').run()

    # fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    # out = cv2.VideoWriter(output_video_path, fourcc, out_fps, (frame_width, frame_height))
    # for i in range(output_frames):
    #     frame_path = os.path.join(output_combined_dir, f'{i:07d}.png')
    #     frame = cv2.imread(frame_path)
    #     out.write(frame)
    # out.release()

    image_files = [os.path.join(output_combined_dir, f'{i:07d}.png') for i in range(output_frames)]
    clip = ImageSequenceClip(image_files, fps=out_fps)
    clip.write_videofile(output_video_path, codec="libx264", fps=out_fps)

    zip_folder(output_masks_dir, output_zip_path)
    print("done")
    return final_masked_frame, final_masked_frame, output_video_path, output_video_path, output_zip_path, ({}, {})

def increment_ann_obj_id(max_obj_id):
    max_obj_id += 1
    ann_obj_id = max_obj_id
    return ann_obj_id, max_obj_id

def update_current_id(ann_obj_id):
    return ann_obj_id

def drawing_board_get_input_first_frame(input_first_frame):
    return input_first_frame

def process_video(queue, result_queue, session_id):
    seg_tracker = None
    click_stack = ({}, {})
    frame_num = int(0)
    ann_obj_id = int(0)
    last_draw = None
    while True:
        task = queue.get() 
        if task["command"] == "exit":
            print(f"Process for {session_id} exiting.")
            break
        elif task["command"] == "extract_video_info":
            input_video = task["input_video"]
            fps, total_frames, input_first_frame, drawing_board, output_video, output_mp4, output_mask = extract_video_info(input_video)
            result_queue.put({"fps": fps, "total_frames": total_frames, "input_first_frame": input_first_frame, "drawing_board": drawing_board, "output_video": output_video, "output_mp4": output_mp4, "output_mask": output_mask})
        elif task["command"] == "load_nifti_for_segmentation":
            nifti_path = task["nifti_path"]
            slice_idx = task.get("slice_idx", None)
            annotation_data = task.get("annotation_data", None)
            
            slice_image, actual_slice_idx, total_slices = load_nifti_slice(nifti_path, slice_idx, normalize=True)
            
            # Draw annotations on the slice if available
            if slice_image is not None and annotation_data:
                slice_display = slice_image.copy()
                
                # Draw points if available
                if "points" in annotation_data and "labels" in annotation_data:
                    points = annotation_data["points"]
                    labels = annotation_data["labels"]
                    
                    for point, label in zip(points, labels):
                        x, y = int(point[0]), int(point[1])
                        color = (0, 255, 0) if label == 1 else (0, 0, 255)  # Green for positive, red for negative
                        cv2.circle(slice_display, (x, y), 5, color, -1)
                        cv2.circle(slice_display, (x, y), 8, color, 2)
                
                # Draw bounding box if available
                if "bbox" in annotation_data:
                    bbox = annotation_data["bbox"]  # [x0, y0, x1, y1]
                    cv2.rectangle(slice_display, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 0, 0), 2)
            else:
                slice_display = slice_image
            
            result_queue.put({
                "slice_image": slice_display,
                "actual_slice_idx": actual_slice_idx,
                "total_slices": total_slices,
                "nifti_path": nifti_path
            })
        elif task["command"] == "get_meta_from_video":
            input_video = task["input_video"]
            scale_slider = task["scale_slider"]
            config_path = task["config_path"]
            checkpoint_path = task["checkpoint_path"]
            target_slice = task.get("target_slice", None)
            seg_tracker, click_stack, input_first_frame, drawing_board, frame_per, output_video, output_mp4, output_mask, ann_obj_id, max_obj_id = get_meta_from_video(session_id, input_video, scale_slider, config_path, checkpoint_path, target_slice)
            result_queue.put({"input_first_frame": input_first_frame, "drawing_board": drawing_board, "frame_per": frame_per, "output_video": output_video, "output_mp4": output_mp4, "output_mask": output_mask, "ann_obj_id": ann_obj_id, "max_obj_id": max_obj_id})
        elif task["command"] == "sam_stroke":
            if seg_tracker is None:
                print("[ERROR] sam_stroke called but model not initialized (seg_tracker is None). Run preprocessing first.")
                result_queue.put({"input_first_frame": None, "drawing_board": None, "last_draw": None})
                continue
            drawing_board = task["drawing_board"]
            last_draw = task["last_draw"]
            frame_num = task["frame_num"]
            ann_obj_id = task["ann_obj_id"]
            seg_tracker, input_first_frame, drawing_board, last_draw = sam_stroke(session_id, seg_tracker, drawing_board, last_draw, frame_num, ann_obj_id)
            result_queue.put({"input_first_frame": input_first_frame, "drawing_board": drawing_board, "last_draw": last_draw})
        elif task["command"] == "sam_click":
            if seg_tracker is None:
                print("[ERROR] sam_click called but model not initialized (seg_tracker is None). Run preprocessing first.")
                result_queue.put({"input_first_frame": None, "drawing_board": None, "last_draw": None})
                continue
            frame_num = task["frame_num"]
            point_mode = task["point_mode"]
            click_stack = task["click_stack"]
            ann_obj_id = task["ann_obj_id"]
            point = task["point"]
            seg_tracker, input_first_frame, drawing_board, last_draw = sam_click(session_id, seg_tracker, frame_num, point_mode, click_stack, ann_obj_id, point)
            result_queue.put({"input_first_frame": input_first_frame, "drawing_board": drawing_board, "last_draw": last_draw})
        elif task["command"] == "increment_ann_obj_id":
            max_obj_id = task["max_obj_id"]
            ann_obj_id, max_obj_id = increment_ann_obj_id(max_obj_id)
            result_queue.put({"ann_obj_id": ann_obj_id, "max_obj_id": max_obj_id})
        elif task["command"] == "update_current_id":
            ann_obj_id = task["ann_obj_id"]
            ann_obj_id = update_current_id(ann_obj_id)
            result_queue.put({"ann_obj_id": ann_obj_id})
        elif task["command"] == "drawing_board_get_input_first_frame":
            input_first_frame = task["input_first_frame"]
            input_first_frame = drawing_board_get_input_first_frame(input_first_frame)
            result_queue.put({"input_first_frame": input_first_frame})
        elif task["command"] == "reset":
            seg_tracker, click_stack, input_first_frame, drawing_board, frame_per, output_video, output_mp4, output_mask, ann_obj_id, max_obj_id = reset(seg_tracker)
            result_queue.put({"click_stack": click_stack, "input_first_frame": input_first_frame, "drawing_board": drawing_board, "frame_per": frame_per, "output_video": output_video, "output_mp4": output_mp4, "output_mask": output_mask, "ann_obj_id": ann_obj_id, "max_obj_id": max_obj_id})
        elif task["command"] == "show_res_by_slider":
            frame_per = task["frame_per"]
            click_stack = task["click_stack"]
            input_first_frame, drawing_board, frame_num = show_res_by_slider(session_id, frame_per, click_stack)
            result_queue.put({"input_first_frame": input_first_frame, "drawing_board": drawing_board, "frame_num": frame_num})
        elif task["command"] == "tracking_objects":
            if seg_tracker is None:
                print("[ERROR] tracking_objects called but model not initialized (seg_tracker is None). Run preprocessing first.")
                result_queue.put({"input_first_frame": None, "drawing_board": None, "output_video": None, "output_mp4": None, "output_mask": None, "click_stack": ({}, {})})
                continue
            frame_num = task["frame_num"]
            input_video = task["input_video"]
            input_first_frame, drawing_board, output_video, output_mp4, output_mask, click_stack = tracking_objects(session_id, seg_tracker, frame_num, input_video)
            result_queue.put({"input_first_frame": input_first_frame, "drawing_board": drawing_board, "output_video": output_video, "output_mp4": output_mp4, "output_mask": output_mask, "click_stack": click_stack})
        elif task["command"] == "convert_to_nifti":
            original_nifti_path = task["original_nifti_path"]
            obj_id = task.get("obj_id", 0)
            patient_id = task.get("patient_id", None)
            session_id_selected = task.get("session_id_selected", None)
            try:
                output_nifti_path, volume_ml, voxel_count, report_path, patient_seg_path = convert_masks_to_nifti(
                    session_id, original_nifti_path, obj_id, patient_id, session_id_selected
                )
                result_queue.put({
                    "success": True, 
                    "output_nifti_path": output_nifti_path, 
                    "volume_ml": volume_ml, 
                    "voxel_count": voxel_count, 
                    "report_path": report_path,
                    "patient_seg_path": patient_seg_path
                })
            except Exception as e:
                print(f"Error converting to NIfTI: {e}")
                traceback.print_exc()
                result_queue.put({"success": False, "error": str(e)})
        else:
            print(f"Unknown command {task['command']} for {session_id}")
            result_queue.put("Unknown command")

def start_process(session_id):
    if session_id not in user_processes:
        queue = mp.Queue()
        result_queue = mp.Queue()
        process = mp.Process(target=process_video, args=(queue, result_queue, session_id))
        process.start()
        user_processes[session_id] = {
            "process": process,
            "queue": queue,
            "result_queue": result_queue,
            "last_active": datetime.datetime.now()
        }
    else:
        user_processes[session_id]["last_active"] = datetime.datetime.now()
    return user_processes[session_id]["queue"]

# def clean_up_processes(session_id, init_clean = False):
#     now = datetime.datetime.now()
#     to_remove = []
#     for s_id, process_info in user_processes.items():
#         if (now - process_info["last_active"] > PROCESS_TIMEOUT) or (s_id == session_id and init_clean):
#             process_info["queue"].put({"command": "exit"})
#             process_info["process"].terminate()
#             process_info["process"].join()
#             to_remove.append(s_id)
#     for s_id in to_remove:
#         del user_processes[s_id]
#         print(f"Cleaned up process for session {s_id}.")
        
def monitor_and_cleanup_processes():
    while True:
        now = datetime.datetime.now()
        to_remove = []
        for session_id, process_info in user_processes.items():
            if now - process_info["last_active"] > PROCESS_TIMEOUT:
                process_info["queue"].put({"command": "exit"})
                process_info["process"].terminate()
                process_info["process"].join()
                to_remove.append(session_id)
        for session_id in to_remove:
            del user_processes[session_id]
            print(f"Automatically cleaned up process for session {session_id}.")
        time.sleep(10)

def seg_track_app():
    # Only supports gradio==3.38.0
    import gradio as gr
    
    def extract_session_id_from_request(request: gr.Request):
        session_id = hashlib.sha256(f'{request.client.host}:{request.client.port}'.encode('utf-8')).hexdigest()
        # cookies = request.kwargs["headers"].get('cookie', '')
        # session_id = None
        # if '_gid=' in cookies:
        #     session_id = cookies.split('_gid=')[1].split(';')[0]
        # else:
        #     session_id = str(uuid.uuid4())
        print(f"session_id {session_id}")
        return session_id

    def handle_extract_video_info(session_id, input_video, skip_flag, current_slider_state):
        # If the pipeline already preprocessed the video, skip this handler
        # to avoid resetting the slider/frame back to 0
        if skip_flag:
            print("[DEBUG] handle_extract_video_info: Skipping (pipeline already handled preprocessing)")
            return gr.Slider.update(), gr.Slider.update(), current_slider_state, gr.Image.update(), gr.Image.update(), gr.Video.update(), gr.File.update(), gr.File.update(), False
        
        # clean_up_processes(session_id, init_clean=True)
        if input_video == None:
            return 0, 0, {
            "minimum": 0.0,
            "maximum": 100,
            "step": 0.01,
            "value": 0.0,
        }, None, None, None, None, None, False
        queue = start_process(session_id)
        result_queue = user_processes[session_id]["result_queue"]
        queue.put({"command": "extract_video_info", "input_video": input_video})
        result = result_queue.get()
        fps = result.get("fps")
        total_frames = result.get("total_frames")
        input_first_frame = result.get("input_first_frame")
        drawing_board = result.get("drawing_board")
        output_video = result.get("output_video")
        output_mp4 = result.get("output_mp4")
        output_mask = result.get("output_mask")
        scale_slider = gr.Slider.update(minimum=1.0,
                                    maximum=fps,
                                    step=1.0,
                                    value=fps,)
        frame_per = gr.Slider.update(minimum= 0.0,
                                maximum= total_frames / fps,
                                step=1.0/fps,
                                value=0.0,)
        slider_state = {
            "minimum": 0.0,
            "maximum": total_frames / fps,
            "step": 1.0/fps,
            "value": 0.0,
        }
        return scale_slider, frame_per, slider_state, input_first_frame, drawing_board, output_video, output_mp4, output_mask, False

    def handle_get_meta_from_video(session_id, input_video, scale_slider, config_path, checkpoint_path, patient_info=None):
        # Validate that a video is loaded
        if input_video is None:
            print("[ERROR] handle_get_meta_from_video: No video loaded. Please click 'Use selected scan (NIfTI) as video' first.")
            # Return safe default values to prevent errors
            return (
                None,  # input_first_frame
                None,  # drawing_board
                gr.Slider.update(minimum=0.0, maximum=100, step=0.01, value=0.0),  # frame_per
                {"minimum": 0.0, "maximum": 100, "step": 0.01, "value": 0.0},  # slider_state
                None,  # output_video
                None,  # output_mp4
                None,  # output_mask
                0,     # ann_obj_id
                0,     # max_obj_id
                gr.Slider.update(maximum=0, value=0),  # obj_id_slider
                0,     # frame_num
                "⚠️ **Error:** No video loaded. Please click 'Use selected scan (NIfTI) as video' button first to load a scan."  # current_frame_display
            )
        
        # Paths are now passed directly
        # config_path = config_file_map[selected_config]
        # checkpoint_path = checkpoint_file_map[selected_checkpoint]
        
        print(f"[DEBUG] handle_get_meta_from_video: patient_info={patient_info}")
        print(f"[DEBUG] Using config: {config_path}")
        print(f"[DEBUG] Using checkpoint: {checkpoint_path}")
        
        target_slice = None
        if patient_info and isinstance(patient_info, dict):
            pid = patient_info.get("patient_id")
            sid = patient_info.get("session_id")
            print(f"[DEBUG] handle_get_meta_from_video: Extracted pid={pid}, sid={sid}")
            if pid and sid:
                # Retrieve slice id from json
                target_slice = get_slice_id_for_patient_session(pid, sid)
                print(f"[DEBUG] handle_get_meta_from_video: target_slice from JSON={target_slice}")

        # clean_up_processes(session_id)
        queue = start_process(session_id)
        result_queue = user_processes[session_id]["result_queue"]
        queue.put({
            "command": "get_meta_from_video",
            "input_video": input_video,
            "scale_slider": scale_slider,
            "config_path": config_path,
            "checkpoint_path": checkpoint_path,
            "target_slice": target_slice
        })
        result = result_queue.get()
        input_first_frame = result.get("input_first_frame")
        drawing_board = result.get("drawing_board")
        (fps, frame_interval, total_frames) = result.get("frame_per")
        output_video = result.get("output_video")
        output_mp4 = result.get("output_mp4")
        output_mask = result.get("output_mask")
        ann_obj_id = result.get("ann_obj_id")
        max_obj_id = result.get("max_obj_id")
        
        initial_time = 0.0
        initial_frame_num = 0
        if target_slice is not None:
             initial_time = target_slice / fps
             # Calculate the frame number that corresponds to this slice
             initial_frame_num = target_slice // frame_interval

        frame_per = gr.Slider.update(minimum= 0.0,
                                maximum= total_frames / fps,
                                step=frame_interval / fps / 2,
                                value=initial_time,)
        slider_state = {
            "minimum": 0.0,
            "maximum": total_frames / fps,
            "step": frame_interval/fps / 2 ,
            "value": initial_time,
        }
        obj_id_slider = gr.Slider.update(
                                    maximum=max_obj_id, 
                                    value=ann_obj_id
                                )
        
        # Create frame display text
        frame_display_text = f"**Current Frame:** {initial_frame_num}"
        if target_slice is not None:
            frame_display_text += f" | **Target Slice ID (from JSON):** {target_slice}"
            print(f"[DEBUG] handle_get_meta_from_video: Created frame_display_text with slice ID: {frame_display_text}")
        else:
            frame_display_text += " | **Slice ID:** N/A (no annotation found)"
            print(f"[DEBUG] handle_get_meta_from_video: No target_slice found, showing N/A")
        
        print(f"[DEBUG] handle_get_meta_from_video: Returning frame_display_text={frame_display_text}")
        return input_first_frame, drawing_board, frame_per, slider_state, output_video, output_mp4, output_mask, ann_obj_id, max_obj_id, obj_id_slider, initial_frame_num, frame_display_text

    def handle_sam_stroke(session_id, drawing_board, last_draw, frame_num, ann_obj_id):
        # clean_up_processes(session_id)
        queue = start_process(session_id)
        result_queue = user_processes[session_id]["result_queue"]
        queue.put({"command": "sam_stroke", "drawing_board": drawing_board, "last_draw": last_draw, "frame_num": frame_num, "ann_obj_id": ann_obj_id})
        result = result_queue.get()
        input_first_frame = result.get("input_first_frame")
        drawing_board = result.get("drawing_board")
        last_draw = result.get("last_draw")
        return input_first_frame, drawing_board, last_draw

    def handle_sam_click(session_id, frame_num, point_mode, click_stack, ann_obj_id, evt: gr.SelectData):
        # clean_up_processes(session_id)
        queue = start_process(session_id)
        result_queue = user_processes[session_id]["result_queue"]
        point = np.array([[evt.index[0], evt.index[1]]], dtype=np.float32)
        queue.put({"command": "sam_click", "frame_num": frame_num, "point_mode": point_mode, "click_stack": click_stack, "ann_obj_id": ann_obj_id, "point": point})
        result = result_queue.get()
        input_first_frame = result.get("input_first_frame")
        drawing_board = result.get("drawing_board")
        last_draw = result.get("last_draw")
        return input_first_frame, drawing_board, last_draw

    def handle_increment_ann_obj_id(session_id, max_obj_id):
        # clean_up_processes(session_id)
        queue = start_process(session_id)
        result_queue = user_processes[session_id]["result_queue"]
        queue.put({"command": "increment_ann_obj_id", "max_obj_id": max_obj_id})
        result = result_queue.get()
        ann_obj_id = result.get("ann_obj_id")
        max_obj_id = result.get("max_obj_id")
        obj_id_slider = gr.Slider.update(maximum=max_obj_id, value=ann_obj_id)
        return ann_obj_id, max_obj_id, obj_id_slider

    def handle_update_current_id(session_id, ann_obj_id):
        # clean_up_processes(session_id)
        queue = start_process(session_id)
        result_queue = user_processes[session_id]["result_queue"]
        queue.put({"command": "update_current_id", "ann_obj_id": ann_obj_id})
        result = result_queue.get()
        ann_obj_id = result.get("ann_obj_id")
        return ann_obj_id

    def handle_drawing_board_get_input_first_frame(session_id, input_first_frame):
        # clean_up_processes(session_id)
        queue = start_process(session_id)
        result_queue = user_processes[session_id]["result_queue"]
        queue.put({"command": "drawing_board_get_input_first_frame", "input_first_frame": input_first_frame})
        result = result_queue.get()
        input_first_frame = result.get("input_first_frame")
        return input_first_frame

    def handle_reset(session_id):
        # clean_up_processes(session_id)
        queue = start_process(session_id)
        result_queue = user_processes[session_id]["result_queue"]
        queue.put({"command": "reset"})
        result = result_queue.get()
        click_stack = result.get("click_stack")
        input_first_frame = result.get("input_first_frame")
        drawing_board = result.get("drawing_board")
        slider_state = {
            "minimum": 0.0,
            "maximum": 100,
            "step": 0.01,
            "value": 0.0,
        }
        output_video = result.get("output_video")
        output_mp4 = result.get("output_mp4")
        output_mask = result.get("output_mask")
        ann_obj_id = result.get("ann_obj_id")
        max_obj_id = result.get("max_obj_id")
        obj_id_slider = gr.Slider.update(
                            maximum=max_obj_id, 
                            value=ann_obj_id)
        return click_stack, input_first_frame, drawing_board, frame_per, slider_state, output_video, output_mp4, output_mask, ann_obj_id, max_obj_id, obj_id_slider

    def handle_show_res_by_slider(session_id, frame_per, slider_state, click_stack):
        # clean_up_processes(session_id)
        queue = start_process(session_id)
        frame_per = frame_per/slider_state["maximum"]
        result_queue = user_processes[session_id]["result_queue"]
        queue.put({"command": "show_res_by_slider", "frame_per": frame_per, "click_stack": click_stack})
        result = result_queue.get()
        input_first_frame = result.get("input_first_frame")
        drawing_board = result.get("drawing_board")
        frame_num = result.get("frame_num")
        
        # Update frame display
        frame_display_text = f"**Current Frame:** {frame_num} | **Slice ID:** Frame {frame_num} (use slider to navigate)"
        
        return input_first_frame, drawing_board, frame_num, frame_display_text

    def handle_tracking_objects(session_id, frame_num, input_video):
        # clean_up_processes(session_id)
        queue = start_process(session_id)
        result_queue = user_processes[session_id]["result_queue"]
        queue.put({"command": "tracking_objects", "frame_num": frame_num, "input_video": input_video})
        result = result_queue.get()
        input_first_frame = result.get("input_first_frame")
        drawing_board = result.get("drawing_board")
        output_video = result.get("output_video")
        output_mp4 = result.get("output_mp4")
        output_mask = result.get("output_mask")
        click_stack = result.get("click_stack")
        return input_first_frame, drawing_board, output_video, output_mp4, output_mask, click_stack
    
    def load_existing_reports(patient_info):
        """Load existing analysis reports from patient_results.json and comman_format.json.
        
        Returns: (nifti_status, clinical_report, patient_report)
        """
        if not (patient_info and isinstance(patient_info, dict)):
            return "*No patient selected.*", "*No report available.*", "*No report available.*"
        
        patient_id = patient_info.get("patient_id")
        session_id_sel = patient_info.get("session_id")
        
        if not patient_id:
            return "*No patient selected.*", "*No report available.*", "*No report available.*"
        
        nifti_status_text = ""
        clinical_report_text = "*No report available yet.*"
        patient_report_text = "*No report available yet.*"
        
        # 1. Try to read from patient_results.json (session-specific, most detailed)
        try:
            patient_folder = join(COMMON_DATA_PATH, patient_id)
            results_file = join(patient_folder, "patient_results.json")
            if exists(results_file):
                with open(results_file, 'r') as f:
                    results_data = json.load(f)
                
                session_data = results_data.get(session_id_sel, {}) if session_id_sel else {}
                
                # Quantitative report (manual report from SAM)
                sam_report = session_data.get("manual report from SAM", "")
                if sam_report:
                    nifti_status_text = f"### 📊 Existing Analysis (from records)\n\n"
                    nifti_status_text += f"**Patient:** {patient_id} | **Session:** {session_id_sel}\n\n"
                    nifti_status_text += f"**Quantitative Report:** {sam_report}\n\n"
                    
                    # Add parcellation file info if available
                    parcellation_path = session_data.get("parcellation file path", "")
                    seg_path = session_data.get("SAM segmentation file path", "")
                    if seg_path:
                        nifti_status_text += f"**Segmentation File:** `{seg_path}`\n"
                    if parcellation_path:
                        nifti_status_text += f"**Parcellation File:** `{parcellation_path}`\n"
                
                # MedGemma clinical / radiology report
                gemma_clinical = session_data.get("gemma radiology report", "")
                if gemma_clinical:
                    clinical_report_text = gemma_clinical
                
                # MedGemma patient report
                gemma_patient = session_data.get("gemma patient report", "")
                if gemma_patient:
                    patient_report_text = gemma_patient
                    
                print(f"[INFO] Loaded existing reports from patient_results.json for {patient_id}/{session_id_sel}")
        except Exception as e:
            print(f"[WARNING] Could not read patient_results.json: {e}")
        
        # 2. Fall back to comman_format.json for patient-level data if session data was empty
        if clinical_report_text == "*No report available yet.*":
            try:
                comman_file = join(COMMON_DATA_PATH, "comman_format.json")
                if exists(comman_file):
                    with open(comman_file, 'r') as f:
                        comman_data = json.load(f)
                    for entry in comman_data:
                        if entry.get("pid") == patient_id:
                            gemma_report = entry.get("gemma patient report", "")
                            if gemma_report:
                                clinical_report_text = gemma_report
                            
                            sam_report_comman = entry.get("manual report from SAM", "")
                            if sam_report_comman and not nifti_status_text:
                                nifti_status_text = f"### 📊 Existing Analysis (from records)\n\n"
                                nifti_status_text += f"**Patient:** {patient_id}\n\n"
                                nifti_status_text += f"**Quantitative Report:** {sam_report_comman}\n"
                            break
                    print(f"[INFO] Loaded existing reports from comman_format.json for {patient_id}")
            except Exception as e:
                print(f"[WARNING] Could not read comman_format.json: {e}")
        
        if not nifti_status_text:
            nifti_status_text = "*No existing analysis found. Complete segmentation and tracking, then run analysis.*"
        
        return nifti_status_text, clinical_report_text, patient_report_text

    def handle_convert_to_nifti(session_id, patient_info, ann_obj_id):
        """Convert segmented masks to NIfTI format, calculate volume, and run parcellation."""
        try:
            if not (patient_info and isinstance(patient_info, dict)):
                return None, "❌ Error: No scan selected. Please select a scan first.", None
            
            patient_id = patient_info.get("patient_id")
            session_id_selected = patient_info.get("session_id")
            scan_name = patient_info.get("scan_name")
            
            if not (patient_id and session_id_selected and scan_name):
                return None, "❌ Error: Incomplete scan information.", None
            
            original_nifti_path = get_scan_full_path(patient_id, session_id_selected, scan_name)
            
            if not os.path.exists(original_nifti_path):
                return None, f"❌ Error: Original NIfTI file not found: {original_nifti_path}", None
            
            queue = start_process(session_id)
            result_queue = user_processes[session_id]["result_queue"]
            queue.put({
                "command": "convert_to_nifti",
                "original_nifti_path": original_nifti_path,
                "obj_id": ann_obj_id,
                "patient_id": patient_id,
                "session_id_selected": session_id_selected
            })
            result = result_queue.get()
            
            if result.get("success"):
                output_nifti_path = result.get("output_nifti_path")
                volume_ml = result.get("volume_ml")
                voxel_count = result.get("voxel_count")
                report_path = result.get("report_path")
                patient_seg_path = result.get("patient_seg_path")
                
                # Build the comprehensive report
                status_msg = f"# Segmentation and Volume Analysis Report\n\n"
                status_msg += f"## Patient Information\n"
                status_msg += f"- **Patient ID:** {patient_id}\n"
                status_msg += f"- **Session ID:** {session_id_selected}\n"
                status_msg += f"- **Scan:** {scan_name}\n\n"
                
                status_msg += f"## Segmentation Results\n"
                status_msg += f"- **Object ID:** {ann_obj_id}\n"
                status_msg += f"- **Segmented Voxels:** {voxel_count:,}\n"
                status_msg += f"- **Total Volume:** {volume_ml:.2f} mL\n\n"
                
                # Read and display volume report
                if report_path and os.path.exists(report_path):
                    try:
                        with open(report_path, 'r') as f:
                            volume_report_content = f.read()
                        status_msg += f"## Detailed Volume Report\n```\n{volume_report_content}\n```\n\n"
                    except Exception as e:
                        print(f"[ERROR] Failed to read volume report: {e}")
                        status_msg += f"⚠️ Could not read volume report from `{report_path}`\n\n"
                
                status_msg += f"## File Locations\n"
                status_msg += f"- **Segmentation (temp):** `{output_nifti_path}`\n"
                if patient_seg_path:
                    status_msg += f"- **Segmentation (patient folder):** `{patient_seg_path}`\n"
                status_msg += f"- **Volume Report:** `{report_path}`\n\n"
                
                # Run parcellation analysis if segmentation was saved to patient folder
                if patient_seg_path and os.path.exists(patient_seg_path):
                    status_msg += "---\n\n"
                    status_msg += "🔄 **Running parcellation analysis...**\n\n"
                    parcellation_result, parcellation_path = run_parcellation_analysis(patient_id, session_id_selected, patient_seg_path)
                    
                    if parcellation_result:
                        status_msg += f"## Parcellation Analysis Results\n\n"
                        status_msg += f"**Summary:** {parcellation_result}\n\n"
                        
                        # Read and display parcellation report
                        if parcellation_path and os.path.exists(parcellation_path):
                            try:
                                with open(parcellation_path, 'r') as f:
                                    parcellation_report_content = f.read()
                                status_msg += f"### Detailed Parcellation Report\n```\n{parcellation_report_content}\n```\n\n"
                            except Exception as e:
                                print(f"[ERROR] Failed to read parcellation report: {e}")
                                status_msg += f"⚠️ Could not read parcellation report from `{parcellation_path}`\n\n"
                        
                        # Update patient_results.json (session-specific)
                        if update_patient_results_json(patient_id, session_id_selected, parcellation_result, patient_seg_path, parcellation_path):
                            status_msg += "✅ **Updated patient_results.json**\n"
                        else:
                            status_msg += "⚠️ **Warning: Failed to update patient_results.json**\n"
                        
                        # Update comman_format.json (patient-level)
                        if update_patient_json_with_parcellation(patient_id, parcellation_result):
                            status_msg += "✅ **Updated comman_format.json**\n"
                        else:
                            status_msg += "⚠️ **Warning: Failed to update comman_format.json**\n"
                        
                        if parcellation_path:
                            status_msg += f"\n**Parcellation Report File:** `{parcellation_path}`\n"
                    else:
                        status_msg += "## Parcellation Analysis\n\n"
                        status_msg += "⚠️ **Warning: Parcellation analysis failed**\n"
                
                return output_nifti_path, status_msg, report_path
            else:
                error_msg = result.get("error", "Unknown error")
                return None, f"❌ **Conversion Failed:** {error_msg}", None
        except Exception as e:
            print(f"[CRITICAL ERROR] Exception in handle_convert_to_nifti: {e}")
            traceback.print_exc()
            return None, f"❌ **Error during conversion:** {str(e)}", None
    
    def handle_use_selected_nifti_as_video(session_id_val, patient_info):
        """Convert selected NIfTI scan into a video and load it.

        This reuses the existing video pipeline: after this runs, the
        `input_video` component points to an MP4 containing all slices
        of the NIfTI volume.
        """
        if not (patient_info and isinstance(patient_info, dict)):
            return gr.Video.update(value=None)
        
        print(f"[DEBUG] handle_use_selected_nifti_as_video: Received patient_info={patient_info}")

        patient_id = patient_info.get("patient_id")
        session_id_selected = patient_info.get("session_id")
        scan_name = patient_info.get("scan_name")

        if not (patient_id and scan_name):
            return gr.Video.update(value=None)
        
        # Get the latest session from patient_results.json
        if patient_id:
            latest_session = get_latest_session_from_patient_results(patient_id)
            if latest_session:
                session_id_selected = latest_session
                print(f"[INFO] Using latest session: {session_id_selected} for patient {patient_id}")
            elif not session_id_selected:
                print(f"[ERROR] No session found for patient {patient_id}")
                return gr.Video.update(value=None)

        try:
            video_path, target_slice = nifti_to_video(
                patient_id, session_id_selected, scan_name
            )
        except Exception as e:
            print(f"Failed to create NIfTI video for {patient_id}/{session_id_selected}: {e}")
            return gr.Video.update(value=None)

        return gr.Video.update(value=video_path)

    ##########################################################
    ######################  Front-end ########################
    ##########################################################
    css = """
    #input_output_video video {
        max-height: 550px;
        max-width: 100%;
        height: auto;
    }
    """

    if platform.system() == "Windows":
        config_path = os.path.abspath(os.environ.get("CONFIG_PATH", "sam2/configs/"))
        checkpoint_path = os.environ.get("CHECKPOINT_PATH", "checkpoints/")

        config_files = glob(os.path.join(config_path, "*.yaml"))
        config_files.sort(key=lambda x: '_t.' not in basename(x))

        checkpoint_files = glob(os.path.join(checkpoint_path, "*.pt"))
        checkpoint_files.sort(key=lambda x: 'tiny' not in basename(x))

        medsam_checkpoints = glob("checkpoints/*.pt")
    else:
        # Resolve paths relative to the project root (parent of gemma3s/)
        # NOTE: Hydra requires absolute paths to start with '//' to be treated as filesystem-absolute
        _project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        config_path = "/" + os.environ.get("CONFIG_PATH", os.path.join(_project_root, "sam2", "configs"))
        checkpoint_path = os.environ.get("CHECKPOINT_PATH", os.path.join(_project_root, "checkpoints"))

        config_files = glob(os.path.join(config_path, "*.yaml"))
        config_files.sort(key=lambda x: '_t.' not in basename(x))

        checkpoint_files = glob(os.path.join(checkpoint_path, "*.pt"))
        checkpoint_files.sort(key=lambda x: 'tiny' not in basename(x))

        medsam_checkpoints = glob(os.path.join(_project_root, "checkpoints", "*.pt"))

    config_display = [splitext(basename(f))[0] for f in config_files]
    medsam_display = [
        f"{os.path.basename(dirname(dirname(path)))} / {splitext(basename(path))[0]}"
        for path in medsam_checkpoints
    ]
    checkpoint_display = [
        splitext(basename(f))[0] for f in checkpoint_files
    ] + medsam_display
    checkpoint_files.extend(medsam_checkpoints)

    config_file_map = dict(zip(config_display, config_files))
    checkpoint_file_map = dict(zip(checkpoint_display, checkpoint_files))

    # CSS for patient selection page
    css = css + """
    .patient-card {
        border: 1px solid #ddd;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
        background: #f9f9f9;
    }
    .session-container {
        margin-left: 20px;
        padding: 10px;
        border-left: 3px solid #4CAF50;
    }
    .scan-button {
        margin: 5px;
    }
    """

    app = gr.Blocks(css=css)
    with app:
        session_id = gr.State()
        selected_scan_path = gr.State(value=None)
        selected_patient_info = gr.State(value={})
        skip_video_change = gr.State(value=False)
        
        app.load(extract_session_id_from_request, None, session_id)
        
        gr.Markdown(
            '''
            <div style="text-align:center; margin-bottom:20px;">
                <span style="font-size:3em; font-weight:bold;">MedSAM2: Segment Anything in 3D Medical Images and Videos</span>
            </div>
            '''
        )
        
        # Main tabs for navigation
        with gr.Tabs() as main_tabs:
            # ==================== PATIENT SELECTION TAB ====================
            with gr.Tab("📋 Patient Overview", id=0) as patient_tab:
                gr.Markdown("""
                ### Patient Database Overview
                Review the patient list below, then use the dropdown to select a patient for segmentation.  
                Patients are sorted by: **Not Reviewed → Reviewed (Tumor → Normal)**
                """)
                
                # Patient summary table
                patient_summary_table = gr.HTML(label="Patient Summary")
                
                gr.Markdown("---")
                gr.Markdown("### Select Patient for Segmentation")
                
                with gr.Row():
                    # Visible dropdown for patient selection
                    patient_selection_dropdown = gr.Dropdown(
                        label="Select Patient (will load most recent session's FLAIR scan)",
                        choices=[],
                        interactive=True,
                        allow_custom_value=False,
                        scale=3
                    )
                    go_to_segmentation_btn = gr.Button(
                        "🔬 Load & Go to Segmentation →",
                        variant="primary",
                        scale=1
                    )
                
                refresh_table_btn = gr.Button("🔄 Refresh Patient Table", variant="secondary")
                
                # Hidden state to store patient data list
                patient_data_state = gr.State(value=[])
                
                # Hidden - kept for compatibility but not used
                selected_patient_row = gr.Number(value=-1, visible=False)
                
                # Hidden components for maintaining compatibility
                patient_dropdown = gr.Dropdown(
                    label="Select Patient",
                    choices=[],
                    interactive=True,
                    allow_custom_value=False,
                    visible=False
                )
                session_dropdown = gr.Dropdown(
                    label="Select Session",
                    choices=[],
                    interactive=True,
                    allow_custom_value=False,
                    visible=False
                )
                scan_dropdown = gr.Dropdown(
                    label="Select Scan",
                    choices=[],
                    interactive=True,
                    allow_custom_value=False,
                    visible=False
                )
                
                selected_scan_display = gr.Textbox(
                    label="Selected Scan Path",
                    interactive=False,
                    placeholder="No scan selected",
                    visible=False
                )
                # Legacy button - hidden, not used
                _legacy_go_to_segmentation_btn = gr.Button(
                    "Go to Segmentation",
                    variant="primary",
                    interactive=False,
                    visible=False
                )
                
                # JSON annotation files display
                json_files_display = gr.Textbox(
                    label="Available Annotation Files (JSON)",
                    interactive=False,
                    lines=2,
                    placeholder="Select a patient to see annotation files",
                    visible=False
                )
            
            # ==================== SEGMENTATION TAB ====================
            with gr.Tab("🔬 Segmentation", id=1) as segmentation_tab:
                # Back to patient selection button
                back_to_patients_btn = gr.Button("← Back to Patient Selection", variant="secondary")
                
                # Display currently selected scan info
                current_scan_info = gr.Markdown("**No scan selected.** Please go to Patient Selection to choose a scan.")
                
                gr.Markdown(
                    '''
                    <div style="text-align:center; margin-bottom:20px;">
                        <a href="https://github.com/bowang-lab/MedSAM/tree/MedSAM2">
                            <img src="https://badges.aleen42.com/src/github.svg" alt="GitHub" style="display:inline-block; margin-right:10px;">
                        </a>
                        <a href="https://arxiv.org/abs/2408.03322">
                            <img src="https://img.shields.io/badge/arXiv-2408.03322-green?style=plastic" alt="Paper" style="display:inline-block; margin-right:10px;">
                        </a>
                        <a href="https://github.com/bowang-lab/MedSAMSlicer/tree/MedSAM2">
                            <img src="https://img.shields.io/badge/3D-Slicer-Plugin" alt="3D Slicer Plugin" style="display:inline-block; margin-right:10px;">
                        </a>
                    </div>
                    <div style="text-align:left; margin-bottom:20px;">
                        This API supports using box (generated by scribble) and point prompts for medical image and video segmentation.
                    </div>
                    <div style="margin-bottom:20px;">
                        <ol style="list-style:none; padding-left:0;">
                            <li>1. Upload video file or load NIfTI scan from patient data</li>
                            <li>2. Select model size, downsample frame rate and run <b>Preprocess</b> (for video) or <b>Load NIfTI Slice</b> (for medical images)</li>
                            <li>3. Use <b>Stroke to Box Prompt</b> to draw box on the image or <b>Point Prompt</b> to click on the image.</li>
                            <li>&nbsp;&nbsp;&nbsp;Note: The bounding rectangle of the stroke should be able to cover the segmentation target.</li>
                            <li>4. Click <b>Segment</b> to get the segmentation result</li>
                            <li>5. Click <b>Add New Object</b> to add new object</li>
                            <li>6. Click <b>Start Tracking</b> to track objects (for video)</li>
                            <li>7. Click <b>Reset</b> to reset the app</li>
                            <li>8. Download the results</li>
                        </ol>
                    </div>
                    <div style="text-align:left; line-height:1.8;">
                        If you find these tools useful, please consider citing the following papers:
                    </div>
                    <div style="text-align:left; line-height:1.8;">
                        Ravi, N., Gabeur, V., Hu, Y.T., Hu, R., Ryali, C., Ma, T., Khedr, H., Rädle, R., Rolland, C., Gustafson, L., Mintun, E., Pan, J., Alwala, K.V., Carion, N., Wu, C.Y., Girshick, R., Dollár, P., Feichtenhofer, C.: SAM 2: Segment Anything in Images and Videos. ICLR 2025
                    </div>            
                    <div style="text-align:left; line-height:1.8;">
                        Ma, J.*, Yang, Z.*, Kim, S., Chen, B., Baharoon, M., Fallahpour, A, Asakereh, R., Lyu, H., Wang, B.: MedSAM2: Segment Anything in Medical Images and Videos. arXiv preprint (2025)
                    </div> 
                    '''
                )

                click_stack = gr.State(({}, {}))
                frame_num = gr.State(value=(int(0)))
                ann_obj_id = gr.State(value=(int(0)))
                max_obj_id = gr.State(value=(int(0)))
                last_draw = gr.State(None)
                slider_state = gr.State(value={
                    "minimum": 0.0,
                    "maximum": 100,
                    "step": 0.01,
                    "value": 0.0,
                })

                with gr.Row():
                    with gr.Column(scale=0.5):
                        # Video input (either uploaded or auto-generated from NIfTI)
                        input_video = gr.Video(
                            label='Input video',
                            elem_id="input_output_video",
                        )
                        with gr.Row():
                            scale_slider = gr.Slider(
                                label="Downsample Frame Rate (fps)",
                                minimum=0.0,
                                maximum=1.0,
                                step=0.25,
                                value=1.0,
                                interactive=True,
                            )
                            
                            # Hidden components for compatibility if needed (or just removed)
                            # config_dropdown, checkpoint_dropdown removed locally
                            # preprocess_button removed completely
                        
                        # use_nifti_button removed completely

                        with gr.Row():
                            tab_stroke = gr.Tab(label="Stroke to Box Prompt")
                            with tab_stroke:
                                drawing_board = gr.Image(label='Drawing Board', tool="sketch", brush_radius=10, interactive=True)
                                with gr.Row():
                                    seg_acc_stroke = gr.Button(value="Segment", interactive=True)
                                    
                            tab_click = gr.Tab(label="Point Prompt")
                            with tab_click:
                                input_first_frame = gr.Image(label='Segment result / Input image', interactive=True, height=550)
                                with gr.Row():
                                    point_mode = gr.Radio(
                                                choices=["Positive",  "Negative"],
                                                value="Positive",
                                                label="Point Prompt",
                                                interactive=True)
                                    
                        # Current frame/slice display
                        current_frame_display = gr.Markdown(
                            value="**Current Frame:** 0 | **Slice ID:** N/A",
                            label="Frame Information"
                        )
                        
                        with gr.Row():
                            with gr.Column():
                                frame_per = gr.Slider(
                                    label = "Time (seconds)",
                                    minimum= 0.0,
                                    maximum= 100.0,
                                    step=0.01,
                                    value=0.0,
                                )
                                with gr.Row():
                                    with gr.Column():
                                        obj_id_slider = gr.Slider(
                                            minimum=0, 
                                            maximum=0, 
                                            step=1, 
                                            interactive=True,
                                            label="Current Object ID"
                                        )
                                    with gr.Column():
                                        new_object_button = gr.Button(
                                            value="Add New Object", 
                                            interactive=True
                                        )
                                track_for_video = gr.Button(
                                    value="Start Tracking",
                                        interactive=True,
                                        )
                                reset_button = gr.Button(
                                    value="Reset",
                                    interactive=True, visible=False,
                                )
                        

                    with gr.Column(scale=0.5):
                        output_video = gr.Video(label='Visualize Results', elem_id="input_output_video")
                        output_mp4 = gr.File(label="Predicted video")
                        output_mask = gr.File(label="Predicted masks")


                gr.Markdown(
                    '''
                    <div style="text-align:center; margin-top: 20px;">
                        The authors of this work highly appreciate Meta AI for making SAM2 publicly available to the community. 
                        The interface was built on <a href="https://github.com/z-x-yang/Segment-and-Track-Anything/blob/main/tutorial/tutorial%20for%20WebUI-1.0-Version.md" target="_blank">SegTracker</a>, which is also an amazing tool for video segmentation tracking. 
                        <a href="https://docs.google.com/document/d/1idDBV0faOjdjVs-iAHr0uSrw_9_ZzLGrUI2FEdK-lso/edit?usp=sharing" target="_blank">Data source</a>
                    </div>
                        '''
                )

            # ==================== ANALYSIS & REPORTING TAB ====================
            with gr.Tab("📊 Analysis & Reporting", id=2) as analysis_tab:
                gr.Markdown("""
                ### 📦 Analysis & Reporting Dashboard
                Run quantitative volume analysis and generate AI-powered clinical reports.  
                **Note:** Complete segmentation and tracking on the Segmentation tab before using these tools.
                """)
                
                with gr.Row():
                    # ---- Left Column: Quantitative Analysis ----
                    with gr.Column(scale=1):
                        gr.Markdown("### 📊 Quantitative Analysis")
                        gr.Markdown("Convert segmentation masks to NIfTI format and calculate volume statistics.")
                        convert_to_nifti_btn = gr.Button("🔄 Convert & Calculate", variant="primary")
                        
                        gr.Markdown("---")
                        gr.Markdown("#### Analysis Report")
                        nifti_status = gr.Markdown("*No analysis run yet. Click the button above after completing segmentation.*")
                        
                        with gr.Row():
                            output_nifti_file = gr.File(label="Segmentation NIfTI", interactive=False, file_count="single")
                            output_report_file = gr.File(label="Volume Report", interactive=False, file_count="single")

                    # ---- Right Column: Qualitative Analysis (MedGemma) ----
                    with gr.Column(scale=1):
                        gr.Markdown("### 🧠 Qualitative Analysis (MedGemma)")
                        gr.Markdown("Generate AI-powered clinical and patient-friendly reports.")
                        generate_reports_btn = gr.Button("✨ Generate AI Reports", variant="primary")
                        medgemma_status = gr.Markdown("")
                        
                        gr.Markdown("---")
                        gr.Markdown("#### 📋 Clinical Report")
                        clinical_report_display = gr.Markdown("*No report generated yet. Click 'Generate AI Reports' above.*")
                        
                        gr.Markdown("---")
                        gr.Markdown("#### 💬 Patient-Friendly Report")
                        patient_report_display = gr.Markdown("*No report generated yet.*")
                        
                        gr.Markdown("---")
                        with gr.Row():
                            clinical_docx_file = gr.File(label="Clinical DOCX", interactive=False, file_count="single")
                            patient_docx_file = gr.File(label="Patient DOCX", interactive=False, file_count="single")

        ##########################################################
        ######################  back-end #########################
        ##########################################################

        # Patient selection handlers
        def load_patient_summary_table():
            """Load and display patient summary table and populate dropdown."""
            patient_data = load_patient_summary_data()
            sorted_data = sort_patient_data(patient_data)
            table_html = create_patient_table_html(sorted_data)
            
            # Create dropdown choices with row number for reference
            dropdown_choices = []
            for idx, p in enumerate(sorted_data):
                tumor_status = "🔴 Tumor" if p.get('tumor', False) else "🟢 Normal"
                reviewed = "✓ Reviewed" if p.get('reviewed_by_radio', False) else "⏳ Not Reviewed"
                label = f"{idx+1}. {p['patient_id']} - {tumor_status} | {reviewed}"
                dropdown_choices.append(label)
            
            # Return table, patient data, and dropdown update
            return table_html, sorted_data, gr.Dropdown.update(choices=dropdown_choices, value=None)
        
        def handle_patient_selection_from_dropdown(selected_value, patient_data_list):
            """Handle patient selection from dropdown.
            
            Args:
                selected_value: Selected dropdown value (format: "1. patient_id - status")
                patient_data_list: List of patient dictionaries
            
            Returns:
                Updates for navigation to segmentation tab with loaded patient scan
            """
            if not selected_value or not patient_data_list:
                return gr.Tabs.update(), "❌ Please select a patient from the dropdown first.", {}
            
            # Extract row index from dropdown value (format: "1. patient_id - ...")
            try:
                row_num = int(selected_value.split('.')[0].strip()) - 1  # Convert to 0-based index
            except (ValueError, IndexError):
                return gr.Tabs.update(), "❌ Error: Could not parse patient selection", {}
            
            if row_num < 0 or row_num >= len(patient_data_list):
                return gr.Tabs.update(), "❌ Error: Patient index out of range", {}
            
            patient = patient_data_list[row_num]
            patient_id = patient['patient_id']
            
            print(f"[DEBUG] Selected patient: {patient_id}")
            
            # Get most recent session
            session_id = get_most_recent_session(patient_id)
            if not session_id:
                return gr.Tabs.update(), f"❌ Error: No sessions found for patient {patient_id}", {}
            
            print(f"[DEBUG] Most recent session: {session_id}")
            
            # Find FLAIR scan
            scan_name = find_flair_scan(patient_id, session_id)
            if not scan_name:
                return gr.Tabs.update(), f"❌ Error: No FLAIR scan found for patient {patient_id}, session {session_id}", {}
            
            print(f"[DEBUG] Found FLAIR scan: {scan_name}")
            
            # Get full path
            scan_path = get_scan_full_path(patient_id, session_id, scan_name)
            
            if not exists(scan_path):
                return gr.Tabs.update(), f"❌ Error: Scan file not found at {scan_path}", {}
            
            # Create patient info dict
            patient_info = {
                "patient_id": patient_id,
                "session_id": session_id,
                "scan_name": scan_name
            }
            
            # Get slice ID from JSON
            slice_id = get_slice_id_for_patient_session(patient_id, session_id)
            slice_id_display = str(slice_id) if slice_id is not None else "N/A"
            
            # Create info text
            info_text = f"""### 📊 Current Scan Information
**Patient ID:** {patient_id}  
**Session:** {session_id}  
**Scan:** {scan_name}  
**Slice ID (from JSON):** {slice_id_display}  
**Tumor Status:** {'Tumor Present' if patient.get('tumor', False) else 'Normal'}  
**Confidence Score:** {patient.get('conf_score', 0.0):.2f}  
**Reviewed by Radiologist:** {'Yes' if patient.get('reviewed_by_radio', False) else 'No'}  

**Full Path:** `{scan_path}`
"""
            
            print(f"[DEBUG] Patient selection successful. Navigating to segmentation tab.")
            
            # Return: navigate to segmentation tab (tab 1), update scan info, store patient info
            return gr.Tabs.update(selected=1), info_text, patient_info
        
        def on_patient_select(patient_id):
            """When a patient is selected, load their sessions."""
            if not patient_id:
                return gr.Dropdown.update(choices=[], value=None), gr.Dropdown.update(choices=[], value=None), "", ""
            
            sessions = get_patient_sessions(patient_id)
            json_files = get_patient_json_files(patient_id)
            json_display = ", ".join(json_files) if json_files else "No annotation files found"
            
            print(f"[DEBUG] on_patient_select: patient={patient_id}, found {len(sessions)} sessions: {sessions}")
            
            # Don't auto-select session - force user to explicitly select to ensure clean state
            return (
                gr.Dropdown.update(choices=sessions, value=None),
                gr.Dropdown.update(choices=[], value=None),
                "",
                json_display
            )
        
        def on_session_select(patient_id, session_id):
            """When a session is selected, load its scans."""
            if not patient_id or not session_id:
                return gr.Dropdown.update(choices=[], value=None), ""
            
            scans = get_session_scans(patient_id, session_id)
            print(f"[DEBUG] on_session_select: patient={patient_id}, session={session_id}, found {len(scans)} scans: {scans}")
            return gr.Dropdown.update(choices=scans, value=scans[0] if scans else None), ""
        
        def on_scan_select(patient_id, session_id, scan_name):
            """When a scan is selected, show the full path."""
            if not patient_id or not session_id or not scan_name:
                return "", gr.Button.update(interactive=False), None
            
            scan_path = get_scan_full_path(patient_id, session_id, scan_name)
            return scan_path, gr.Button.update(interactive=True), scan_path
        
        def navigate_to_segmentation(patient_id, session_id, scan_name, scan_path):
            """Navigate to segmentation tab with selected scan info."""
            if not scan_path:
                return gr.Tabs.update(selected=1), "**No scan selected.** Please go back and select a scan.", {}

            info = {
                "patient_id": patient_id,
                "session_id": session_id,
                "scan_name": scan_name,
                "scan_path": scan_path,
            }
            info_text = (
                f"**Selected Scan:** Patient `{patient_id}` → Session `{session_id}` → `{scan_name}`"\
                f"\n\n**Full Path:** `{scan_path}`"
            )
            return gr.Tabs.update(selected=1), info_text, info
        
        def navigate_to_patients():
            """Navigate back to patient selection tab and reset dropdown."""
            return gr.Tabs.update(selected=0), gr.Dropdown.update(value=None)

        def auto_load_patient_pipeline(selected_value, patient_data_list, scale_slider_val, gradio_session_id):
            """
            Automated pipeline:
            1. Select Patient -> Get Info & Path
            2. Convert NIfTI to Video
            3. Preprocess Video (Load Config/Checkpoint)
            """
            # 1. Select Patient
            tabs_update, info_text, patient_info = handle_patient_selection_from_dropdown(selected_value, patient_data_list)
            
            # Default empty returns for pipeline failure
            empty_meta = (
                None, None, gr.Slider.update(), 
                {"minimum": 0.0, "maximum": 100, "step": 0.01, "value": 0.0}, 
                None, None, None, 0, 0, gr.Slider.update(), 0, ""
            )
            
            # Check for error in step 1
            if isinstance(info_text, str) and info_text.startswith("❌"):
                return (
                    tabs_update, info_text, patient_info, 
                    False,  # skip_video_change
                    gr.Video.update(value=None),
                    *empty_meta
                )

            # Extract correct session_id from the new patient info
            # logic in handle_patient_selection_from_dropdown sets this
            updated_session_id = patient_info.get("session_id")
            
            # 2. Convert NIfTI to Video
            video_update = handle_use_selected_nifti_as_video(updated_session_id, patient_info)
            
            video_path = None
            if isinstance(video_update, dict) and "value" in video_update:
                video_path = video_update["value"]
            
            if not video_path:
                 return (
                    tabs_update, info_text + "\n\n❌ **Error:** Failed to convert NIfTI to video.", patient_info, 
                    False,  # skip_video_change
                    video_update,
                    *empty_meta
                )

            # 3. Preprocess Video
            # Hardcoded paths as requested
            # IMPORTANT: Hydra requires absolute paths to start with '//' (double slash) to be treated as filesystem-absolute
            # Reference: Original code at line 2151-2153
            if platform.system() == "Windows":
                checkpoint_path = os.path.abspath("checkpoints/MedSAM2_latest.pt")
                # For Windows, check if we need special handling
                _project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
                config_path = "/" + os.path.join(_project_root, "sam2", "configs", "sam2.1_hiera_t512.yaml")
            else:
                _project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
                checkpoint_path = os.path.join(_project_root, "checkpoints", "MedSAM2_latest.pt")
                # Prepend '/' to make Hydra treat this as filesystem-absolute
                config_path = "/" + os.path.join(_project_root, "sam2", "configs", "sam2.1_hiera_t512.yaml")
            
            # Ensure checkpoint exists
            if not os.path.exists(checkpoint_path):
                 print(f"[ERROR] Checkpoint not found at {checkpoint_path}")

            # For NIfTI-sourced videos, each frame = one slice.
            # The video is created at 24fps, so we must set scale_slider = 24
            # to ensure frame_interval = max(1, int(fps // scale_slider)) = 1,
            # i.e. every single slice/frame is extracted and used.
            nifti_scale_slider = 24.0

            meta_results = handle_get_meta_from_video(
                gradio_session_id, video_path, nifti_scale_slider, config_path, checkpoint_path, patient_info
            )
            
            return (
                tabs_update, info_text, patient_info, 
                True,   # skip_video_change - prevent input_video.change from resetting slider
                video_update,
                *meta_results
            )

        # Connect patient selection events
        refresh_table_btn.click(
            fn=load_patient_summary_table,
            inputs=[],
            outputs=[patient_summary_table, patient_data_state, patient_selection_dropdown]
        )
        
        # Connect the dropdown change event to the full automation pipeline
        patient_selection_dropdown.change(
            fn=auto_load_patient_pipeline,
            inputs=[patient_selection_dropdown, patient_data_state, scale_slider, session_id],
            outputs=[
                main_tabs, current_scan_info, selected_patient_info, # Step 1
                skip_video_change,  # Flag to prevent input_video.change from resetting
                input_video, # Step 2
                # Step 3 (Meta results)
                input_first_frame, drawing_board, frame_per, slider_state, output_video, output_mp4, output_mask, ann_obj_id, max_obj_id, obj_id_slider, frame_num, current_frame_display
            ]
        ).then(
            fn=load_existing_reports,
            inputs=[selected_patient_info],
            outputs=[nifti_status, clinical_report_display, patient_report_display]
        )
        
        # Legacy/Other connections
        back_to_patients_btn.click(
            fn=navigate_to_patients,
            inputs=[],
            outputs=[main_tabs, patient_selection_dropdown]
        )
        
        # Load patient summary table on app start
        app.load(
            fn=load_patient_summary_table,
            inputs=[],
            outputs=[patient_summary_table, patient_data_state, patient_selection_dropdown]
        )

        frame_per.release(
            fn=handle_show_res_by_slider, 
            inputs=[
                session_id, frame_per, slider_state, click_stack
                ], 
            outputs=[
                input_first_frame, drawing_board, frame_num, current_frame_display
            ]
        )

        # Interactively modify the mask acc click
        input_first_frame.select(
            fn=handle_sam_click,
            inputs=[
                session_id, frame_num, point_mode, click_stack, ann_obj_id
            ],
            outputs=[
                input_first_frame, drawing_board, click_stack
            ]
        )

        # Track object in video
        tracking_event = track_for_video.click(
            fn=handle_tracking_objects,
            inputs=[
                session_id,
                frame_num,
                input_video,
            ],
            outputs=[
                input_first_frame,
                drawing_board,
                output_video,
                output_mp4,
                output_mask,
                click_stack
            ], queue=False, every=15
        )

        reset_button.click(
            fn=handle_reset,
            inputs=[session_id],
            outputs=[
                click_stack, input_first_frame, drawing_board, frame_per, slider_state, output_video, output_mp4, output_mask, ann_obj_id, max_obj_id, obj_id_slider
            ]
        )

        new_object_button.click(
            fn=handle_increment_ann_obj_id, 
            inputs=[ session_id, max_obj_id ], 
            outputs=[ ann_obj_id, max_obj_id, obj_id_slider ]
        )

        obj_id_slider.change(
            fn=handle_update_current_id, 
            inputs=[session_id, obj_id_slider], 
            outputs={ann_obj_id}
        )

        tab_stroke.select(
            fn=handle_drawing_board_get_input_first_frame,
            inputs=[session_id, input_first_frame],
            outputs=[drawing_board,],
        )

        seg_acc_stroke.click(
            fn=handle_sam_stroke,
            inputs=[
                session_id, drawing_board, last_draw, frame_num, ann_obj_id
            ],
            outputs=[
                input_first_frame, drawing_board, last_draw
            ]
        )

        input_video.change(
            fn=handle_extract_video_info,
            inputs=[session_id, input_video, skip_video_change, slider_state],
            outputs=[scale_slider, frame_per, slider_state, input_first_frame, drawing_board, output_video, output_mp4, output_mask, skip_video_change], queue=False, every=15
        )
        
        # use_nifti_button.click removed
        
        # MedGemma AI report generation handler
        def handle_generate_medgemma_reports(patient_info, progress=gr.Progress()):
            """Generate clinical and patient-friendly reports using MedGemma."""
            def safe_progress(val, desc=""):
                try:
                    progress(val, desc=desc)
                except Exception:
                    pass  # Progress not available in this context
            try:
                safe_progress(0, "Initializing...")
                if not (patient_info and isinstance(patient_info, dict)):
                    return "❌ **Error:** No scan selected. Please select a patient and complete segmentation first.", \
                           "*No report generated.*", "*No report generated.*", None, None
                
                patient_id = patient_info.get("patient_id")
                session_id_selected = patient_info.get("session_id")
                scan_name = patient_info.get("scan_name")
                
                if not (patient_id and session_id_selected and scan_name):
                    return "❌ **Error:** Incomplete scan information.", \
                           "*No report generated.*", "*No report generated.*", None, None
                
                # Look for the segmentation file in patient folder
                patient_folder = join(COMMON_DATA_PATH, patient_id)
                segmentation_path = join(patient_folder, f"{patient_id}_{session_id_selected}_seg.nii.gz")
                
                if not os.path.exists(segmentation_path):
                    return (
                        "❌ **Error:** Segmentation file not found. "
                        "Please complete segmentation and click 'Convert to NIfTI & Calculate Volume' first.\n\n"
                        f"Expected file: `{segmentation_path}`",
                        "*No report generated.*", "*No report generated.*", None, None
                    )
                
                # Try to get parcellation result from patient_results.json
                progress(0.1, desc="Reading context...")
                parcellation_result = None
                try:
                    patient_results_file = join(patient_folder, "patient_results.json")
                    if exists(patient_results_file):
                        with open(patient_results_file, 'r') as f:
                            results_data = json.load(f)
                        session_data = results_data.get(session_id_selected, {})
                        parcellation_result = session_data.get("manual report from SAM", None)
                        print(f"[INFO] Found parcellation context: {parcellation_result}")
                except Exception as e:
                    print(f"[WARNING] Could not read parcellation result: {e}")
                
                # Generate reports
                safe_progress(0.2, desc="Loading Model (may take 60s)...")
                status_msg = "🔄 **Generating reports with MedGemma...**\n\n"
                
                safe_progress(0.4, desc="Generating Reports...")
                try:
                    clinical_report, patient_report, clinical_docx_path, patient_docx_path = generate_medgemma_reports(
                        patient_id, session_id_selected, scan_name, segmentation_path, parcellation_result
                    )
                except Exception as gen_error:
                    print(f"[ERROR] generate_medgemma_reports failed: {gen_error}")
                    traceback.print_exc()
                    return (
                        f"❌ **Error:** Failed to generate MedGemma reports.\n\n**Details:** {str(gen_error)}",
                        "*Report generation failed.*", "*Report generation failed.*", 
                        None, None
                    )
                
                safe_progress(0.9, desc="Finalizing...")
                
                if clinical_report is None or patient_report is None:
                    return (
                        "❌ **Error:** Failed to generate MedGemma reports. Check the console for details.",
                        "*Report generation failed.*", "*Report generation failed.*", 
                        None, None
                    )
                
                # Ensure reports are strings
                clinical_report_str = str(clinical_report) if clinical_report else "*No clinical report generated.*"
                patient_report_str = str(patient_report) if patient_report else "*No patient report generated.*"
                
                # Update JSON files with both reports
                safe_progress(0.95, desc="Updating records...")
                json_status = ""
                
                # Update patient_results.json (session level)
                if update_patient_results_json_with_medgemma(patient_id, session_id_selected, clinical_report_str, patient_report_str):
                    json_status += "✅ **Updated patient_results.json**\n"
                else:
                    json_status += "⚠️ **Warning: Failed to update patient_results.json**\n"

                # Update comman_format.json (patient level)
                if update_comman_format_json_with_medgemma(patient_id, clinical_report_str, patient_report_str):
                    json_status += "✅ **Updated comman_format.json**\n"
                else:
                    json_status += "⚠️ **Warning: Failed to update comman_format.json**\n"
                
                status_msg = f"✅ **MedGemma reports generated successfully!**\n\n"
                status_msg += f"- **Patient:** {patient_id}\n"
                status_msg += f"- **Session:** {session_id_selected}\n"
                
                if clinical_docx_path and os.path.exists(clinical_docx_path):
                    status_msg += f"- **Clinical Report:** `{clinical_docx_path}`\n"
                if patient_docx_path and os.path.exists(patient_docx_path):
                    status_msg += f"- **Patient Report:** `{patient_docx_path}`\n"
                
                status_msg += f"\n**Record Updates:**\n{json_status}"
                
                safe_progress(1.0, "Done!")
                
                # Debug: log exactly what we're returning
                print(f"[DEBUG] Return status_msg length: {len(status_msg)}")
                print(f"[DEBUG] Return clinical_report length: {len(clinical_report_str)}")
                print(f"[DEBUG] Return patient_report length: {len(patient_report_str)}")
                print(f"[DEBUG] Return clinical_docx_path: {clinical_docx_path}")
                print(f"[DEBUG] Return patient_docx_path: {patient_docx_path}")
                print(f"[DEBUG] clinical_docx exists: {os.path.exists(clinical_docx_path) if clinical_docx_path else 'None'}")
                print(f"[DEBUG] patient_docx exists: {os.path.exists(patient_docx_path) if patient_docx_path else 'None'}")
                
                # Copy files to Gradio temp directory for reliable downloads
                clinical_file_result = copy_file_for_gradio_download(clinical_docx_path)
                patient_file_result = copy_file_for_gradio_download(patient_docx_path)
                
                print(f"[DEBUG] About to return from handle_generate_medgemma_reports")
                print(f"[DEBUG] clinical_file_result: {clinical_file_result}")
                print(f"[DEBUG] patient_file_result: {patient_file_result}")
                return (
                    status_msg,
                    clinical_report_str,
                    patient_report_str,
                    clinical_file_result,
                    patient_file_result
                )

            except Exception as e:
                print(f"[CRITICAL ERROR] Exception in handle_generate_medgemma_reports: {e}")
                traceback.print_exc()
                return (
                    f"❌ **Critical Error:** {str(e)}",
                    "*Error occurred.*", "*Error occurred.*", 
                    None, None
                )

        # Button to convert segmentation masks back to NIfTI
        convert_to_nifti_btn.click(
            fn=handle_convert_to_nifti,
            inputs=[session_id, selected_patient_info, ann_obj_id],
            outputs=[output_nifti_file, nifti_status, output_report_file]
        )
        



        # Wrapper for auto-chain: regular function (not generator) for .then() chain compatibility
        def auto_generate_medgemma_reports(patient_info):
            """Auto-chain wrapper for handle_generate_medgemma_reports without progress bar.
            
            NOTE: This MUST be a regular function (not a generator) for .then() chains to work
            properly in Gradio 3.38.0. Generator functions with yield break the chain.
            """
            try:
                if not (patient_info and isinstance(patient_info, dict)):
                    return ("⏭️ Skipped AI report generation (no patient selected).",
                            "*No report generated.*", "*No report generated.*", None, None)
                
                patient_id = patient_info.get("patient_id")
                session_id_selected = patient_info.get("session_id")
                scan_name = patient_info.get("scan_name")
                
                if not (patient_id and session_id_selected and scan_name):
                    return ("⏭️ Skipped AI report generation (incomplete scan info).",
                            "*No report generated.*", "*No report generated.*", None, None)
                
                # Look for the segmentation file
                patient_folder = join(COMMON_DATA_PATH, patient_id)
                segmentation_path = join(patient_folder, f"{patient_id}_{session_id_selected}_seg.nii.gz")
                
                if not os.path.exists(segmentation_path):
                    return (
                        f"⏭️ Skipped AI report generation (no segmentation file found).\n\n"
                        f"Expected: `{segmentation_path}`",
                        "*No report generated.*", "*No report generated.*", None, None
                    )

                # Get parcellation context
                parcellation_result = None
                try:
                    patient_results_file = join(patient_folder, "patient_results.json")
                    if exists(patient_results_file):
                        with open(patient_results_file, 'r') as f:
                            results_data = json.load(f)
                        session_data = results_data.get(session_id_selected, {})
                        parcellation_result = session_data.get("manual report from SAM", None)
                except Exception as e:
                    print(f"[WARNING] Could not read parcellation result: {e}")
                
                # Generate reports
                print(f"[AUTO-CHAIN] Generating MedGemma reports for {patient_id}/{session_id_selected}...")
                try:
                    clinical_report, patient_report, clinical_docx_path, patient_docx_path = generate_medgemma_reports(
                        patient_id, session_id_selected, scan_name, segmentation_path, parcellation_result
                    )
                except Exception as gen_error:
                    print(f"[AUTO-CHAIN ERROR] generate_medgemma_reports failed: {gen_error}")
                    traceback.print_exc()
                    return (
                        f"❌ **Error:** Failed to generate MedGemma reports.\n\n**Details:** {str(gen_error)}",
                        "*Report generation failed.*", "*Report generation failed.*", 
                        None, None
                    )
                
                print(f"[AUTO-CHAIN DEBUG] Reports returned - clinical: {clinical_report is not None}, patient: {patient_report is not None}")
                
                if clinical_report is None or patient_report is None:
                    return (
                        "❌ **Error:** Failed to generate MedGemma reports. Check console for details.",
                        "*Report generation failed.*", "*Report generation failed.*", 
                        None, None
                    )
                
                # Ensure reports are strings
                clinical_report_str = str(clinical_report) if clinical_report else "*No clinical report generated.*"
                patient_report_str = str(patient_report) if patient_report else "*No patient report generated.*"
                
                print(f"[AUTO-CHAIN DEBUG] Clinical report length: {len(clinical_report_str)}")
                print(f"[AUTO-CHAIN DEBUG] Patient report length: {len(patient_report_str)}")
                
                # Update JSON files with both reports
                json_status = ""
                try:
                    if update_patient_results_json_with_medgemma(patient_id, session_id_selected, clinical_report_str, patient_report_str):
                        json_status += "✅ Updated patient_results.json\n"
                    if update_comman_format_json_with_medgemma(patient_id, clinical_report_str, patient_report_str):
                        json_status += "✅ Updated comman_format.json\n"
                except Exception as json_error:
                    print(f"[AUTO-CHAIN WARNING] JSON update failed: {json_error}")
                    json_status += "⚠️ Warning: Failed to update JSON files\n"
                
                status_msg = f"✅ **MedGemma reports generated automatically!**\n\n"
                status_msg += f"- **Patient:** {patient_id}\n"
                status_msg += f"- **Session:** {session_id_selected}\n"
                
                if clinical_docx_path and os.path.exists(clinical_docx_path):
                    status_msg += f"- **Clinical Report:** `{clinical_docx_path}`\n"
                if patient_docx_path and os.path.exists(patient_docx_path):
                    status_msg += f"- **Patient Report:** `{patient_docx_path}`\n"
                
                status_msg += f"\n**Records:** {json_status}"
                
                # Copy files to Gradio temp directory for reliable downloads
                clinical_file_result = copy_file_for_gradio_download(clinical_docx_path)
                patient_file_result = copy_file_for_gradio_download(patient_docx_path)

                print(f"[AUTO-CHAIN DEBUG] Returning final results...")
                print(f"[AUTO-CHAIN DEBUG] clinical_file_result: {clinical_file_result}")
                print(f"[AUTO-CHAIN DEBUG] patient_file_result: {patient_file_result}")
                return (status_msg, clinical_report_str, patient_report_str, clinical_file_result, patient_file_result)
                
            except Exception as e:
                print(f"[AUTO-CHAIN CRITICAL ERROR] Unexpected exception in auto_generate_medgemma_reports: {e}")
                traceback.print_exc()
                return (
                    f"❌ **Critical Error:** {str(e)}", 
                    "*Error occurred.*", "*Error occurred.*", 
                    None, None
                )
        
        # Auto-trigger analysis chain after tracking completes
        # (defined here because handle_generate_medgemma_reports must be defined first)
        tracking_event.then(
            fn=handle_convert_to_nifti,
            inputs=[session_id, selected_patient_info, ann_obj_id],
            outputs=[output_nifti_file, nifti_status, output_report_file]
        ).then(
            fn=auto_generate_medgemma_reports,
            inputs=[selected_patient_info],
            outputs=[medgemma_status, clinical_report_display, patient_report_display, clinical_docx_file, patient_docx_file]
        ).then(
            fn=lambda: gr.Tabs.update(selected=2),
            inputs=[],
            outputs=[main_tabs]
        )

        # Wired manually for testing the generator fix
        generate_reports_btn.click(
            fn=auto_generate_medgemma_reports,
            inputs=[selected_patient_info],
            outputs=[medgemma_status, clinical_report_display, patient_report_display, clinical_docx_file, patient_docx_file]
        )
        
    app.queue(concurrency_count=1)
    app.launch(debug=True, enable_queue=True, share=False, server_name="0.0.0.0", server_port=18862,
               allowed_paths=[COMMON_DATA_PATH, GRADIO_TEMP_DIR])
    # app.launch(debug=True, enable_queue=True, share=True)

if __name__ == "__main__":
    mp.set_start_method("spawn")
    monitor_thread = threading.Thread(target=monitor_and_cleanup_processes)
    monitor_thread.daemon = True
    monitor_thread.start()
    seg_track_app()