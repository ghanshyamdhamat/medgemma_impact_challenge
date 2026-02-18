from glob import glob
from tqdm import tqdm
import os
from os.path import join, basename
import re
import matplotlib.pyplot as plt
from collections import OrderedDict
import pandas as pd
import numpy as np
import argparse
import json

from PIL import Image
import SimpleITK as sitk
import torch
import torch.multiprocessing as mp
from sam2.build_sam import build_sam2_video_predictor_npz
from skimage import measure, morphology

torch.set_float32_matmul_precision('high')
torch.manual_seed(2024)
torch.cuda.manual_seed(2024)
np.random.seed(2024)

parser = argparse.ArgumentParser()

parser.add_argument(
    '--checkpoint',
    type=str,
    default="checkpoints/MedSAM2_latest.pt",
    help='checkpoint path',
)
parser.add_argument(
    '--cfg',
    type=str,
    default="configs/sam2.1_hiera_t512.yaml",
    help='model config',
)

parser.add_argument(
    '-i',
    '--imgs_path',
    type=str,
    default="MRI_data/images",
    help='MRI images path',
)
parser.add_argument(
    '-a',
    '--annotations_json',
    type=str,
    required=True,
    help='path to JSON file with bounding boxes and point prompts',
)
parser.add_argument(
    '--gts_path',
    default=None,
    help='ground truth annotations path',
)
parser.add_argument(
    '-o',
    '--pred_save_dir',
    type=str,
    default="./MRI_results",
    help='path to save segmentation results',
)
parser.add_argument(
    '--propagate_with_box',
    default=True,
    action='store_true',
    help='whether to propagate with box prompt',
)
parser.add_argument(
    '--mri_window_level',
    type=float,
    default=None,
    help='MRI window level (optional)',
)
parser.add_argument(
    '--mri_window_width',
    type=float,
    default=None,
    help='MRI window width (optional)',
)

args = parser.parse_args()
checkpoint = args.checkpoint
model_cfg = args.cfg
imgs_path = args.imgs_path
annotations_json = args.annotations_json
gts_path = args.gts_path
pred_save_dir = args.pred_save_dir
os.makedirs(pred_save_dir, exist_ok=True)
propagate_with_box = args.propagate_with_box
mri_window_level = args.mri_window_level
mri_window_width = args.mri_window_width

# Load annotations from JSON
with open(annotations_json, 'r') as f:
    annotations = json.load(f)

def getLargestCC(segmentation):
    labels = measure.label(segmentation)
    largestCC = labels == np.argmax(np.bincount(labels.flat)[1:])+1
    return largestCC

def dice_multi_class(preds, targets):
    smooth = 1.0
    assert preds.shape == targets.shape
    labels = np.unique(targets)[1:]
    dices = []
    for label in labels:
        pred = preds == label
        target = targets == label
        intersection = (pred * target).sum()
        dices.append((2.0 * intersection + smooth) / (pred.sum() + target.sum() + smooth))
    return np.mean(dices)

def show_mask(mask, ax, mask_color=None, alpha=0.5):
    """show mask on the image"""
    if mask_color is not None:
        color = np.concatenate([mask_color, np.array([alpha])], axis=0)
    else:
        color = np.array([251/255, 252/255, 30/255, alpha])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_box(box, ax, edgecolor='blue'):
    """show bounding box on the image"""
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor=edgecolor, facecolor=(0,0,0,0), lw=2))

def preprocess_mri(image_data, window_level=None, window_width=None):
    """
    Preprocess MRI data with optional windowing.
    For MRI, normalization is typically done per-volume.
    """
    if window_level is not None and window_width is not None:
        lower_bound = window_level - window_width / 2
        upper_bound = window_level + window_width / 2
        image_data = np.clip(image_data, lower_bound, upper_bound)
    
    # Normalize to [0, 255]
    img_min = np.min(image_data)
    img_max = np.max(image_data)
    if img_max > img_min:
        image_data = (image_data - img_min) / (img_max - img_min) * 255.0
    else:
        image_data = np.zeros_like(image_data)
    
    return np.uint8(image_data)

def resize_grayscale_to_rgb_and_resize(array, image_size):
    """Resize a 3D grayscale array to RGB and then resize it."""
    d, h, w = array.shape
    resized_array = np.zeros((d, 3, image_size, image_size))
    
    for i in range(d):
        img_pil = Image.fromarray(array[i].astype(np.uint8))
        img_rgb = img_pil.convert("RGB")
        img_resized = img_rgb.resize((image_size, image_size))
        img_array = np.array(img_resized).transpose(2, 0, 1)
        resized_array[i] = img_array
    
    return resized_array

def mask2D_to_bbox(gt2D, max_shift=20):
    y_indices, x_indices = np.where(gt2D > 0)
    x_min, x_max = np.min(x_indices), np.max(x_indices)
    y_min, y_max = np.min(y_indices), np.max(y_indices)
    H, W = gt2D.shape
    bbox_shift = np.random.randint(0, max_shift + 1, 1)[0]
    x_min = max(0, x_min - bbox_shift)
    x_max = min(W-1, x_max + bbox_shift)
    y_min = max(0, y_min - bbox_shift)
    y_max = min(H-1, y_max + bbox_shift)
    boxes = np.array([x_min, y_min, x_max, y_max])
    return boxes

def mask3D_to_bbox(gt3D, max_shift=20):
    z_indices, y_indices, x_indices = np.where(gt3D > 0)
    x_min, x_max = np.min(x_indices), np.max(x_indices)
    y_min, y_max = np.min(y_indices), np.max(y_indices)
    z_min, z_max = np.min(z_indices), np.max(z_indices)
    D, H, W = gt3D.shape
    bbox_shift = np.random.randint(0, max_shift + 1, 1)[0]
    x_min = max(0, x_min - bbox_shift)
    x_max = min(W-1, x_max + bbox_shift)
    y_min = max(0, y_min - bbox_shift)
    y_max = min(H-1, y_max + bbox_shift)
    z_min = max(0, z_min)
    z_max = min(D-1, z_max)
    boxes3d = np.array([x_min, y_min, z_min, x_max, y_max, z_max])
    return boxes3d

# Add at the top with other imports
def dice_score_binary(pred, target):
    """Calculate binary Dice score"""
    smooth = 1.0
    pred_binary = (pred > 0).astype(np.uint8)
    target_binary = (target > 0).astype(np.uint8)
    intersection = (pred_binary * target_binary).sum()
    dice = (2.0 * intersection + smooth) / (pred_binary.sum() + target_binary.sum() + smooth)
    return dice

# Initialize predictor
predictor = build_sam2_video_predictor_npz(model_cfg, checkpoint)

# Get list of MRI files
mri_fnames = sorted(os.listdir(imgs_path))
mri_fnames = [i for i in mri_fnames if i.endswith('T1.nii.gz')]
gt_fnames = [i for i in sorted(os.listdir(gts_path)) if i.endswith('Lesion.nii.gz')] if gts_path is not None else []
# mri_fnames = [i for i in mri_fnames if not i.startswith('._')]
print(f'Processing {len(mri_fnames)} MRI files')

assert len(mri_fnames)==len(gt_fnames) or gts_path is None, f'Number of MRI files ({len(mri_fnames)}) and GT files ({len(gt_fnames)}) do not match!'

# Update seg_info to include dice scores
seg_info = OrderedDict()
seg_info['mri_name'] = []
seg_info['mid_slice_index'] = []
seg_info['dice_score'] = []  # Add this line

for mri_fname in tqdm(mri_fnames):
    # Load MRI image
    mri_image = sitk.ReadImage(join(imgs_path, mri_fname))
    mri_image_data = sitk.GetArrayFromImage(mri_image)
    
    # Get annotations for this file
    file_key = mri_fname.split('.nii.gz')[0]
    if file_key not in annotations:
        print(f'Warning: No annotations found for {mri_fname}, skipping...')
        continue
    
    file_annotation = annotations[file_key]
    
    # Preprocess MRI data
    mri_image_data_pre = preprocess_mri(mri_image_data, mri_window_level, mri_window_width)
    
    segs_3D = np.zeros(mri_image_data_pre.shape, dtype=np.uint8)
    
    # Get middle slice as reference from annotation or use default
    if 'mid_slice_idx' in file_annotation:
        mid_slice_idx = file_annotation['mid_slice_idx']
    else:
        mid_slice_idx = mri_image_data_pre.shape[0] // 2
    
    mid_slice_img = mri_image_data_pre[mid_slice_idx, :, :]
    
    assert np.max(mri_image_data_pre) < 256, f'input data should be in range [0, 255], but got {np.unique(mri_image_data_pre)}'
    
    video_height = mid_slice_img.shape[0]
    video_width = mid_slice_img.shape[1]
    
    # Resize and normalize
    img_resized = resize_grayscale_to_rgb_and_resize(mri_image_data_pre, 512)
    img_resized = img_resized / 255.0
    img_resized = torch.from_numpy(img_resized).cuda()
    
    img_mean = (0.485, 0.456, 0.406)
    img_std = (0.229, 0.224, 0.225)
    img_mean = torch.tensor(img_mean, dtype=torch.float32)[:, None, None].cuda()
    img_std = torch.tensor(img_std, dtype=torch.float32)[:, None, None].cuda()
    img_resized -= img_mean
    img_resized /= img_std
    
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        inference_state = predictor.init_state(img_resized, video_height, video_width)
        
        # Get bbox from annotation
        bbox = None
        if 'bbox' in file_annotation:
            bbox = np.array(file_annotation['bbox'], dtype=np.float32)
        
        # Get points from annotation if available
        points_data = None
        labels_data = None
        if 'points' in file_annotation:
            points_data = np.array(file_annotation['points'], dtype=np.float32)
            labels_data = np.array(file_annotation.get('labels', [1] * len(file_annotation['points'])), dtype=np.int32)
        
        # Determine what prompts to use
        has_box = bbox is not None
        has_points = points_data is not None
        
        # Use both prompts if available, otherwise use what's available
        if has_box and has_points and not propagate_with_box:
            # Use both box and points
            _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=mid_slice_idx,
                obj_id=1,
                box=bbox,
                points=points_data,
                labels=labels_data,
            )
        elif has_box:
            # Use only box
            _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=mid_slice_idx,
                obj_id=1,
                box=bbox,
            )
        elif has_points:
            # Use only points
            _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=mid_slice_idx,
                obj_id=1,
                points=points_data,
                labels=labels_data,
            )
        else:
            # Fallback: generate box from center region
            print(f'Warning: No prompts in annotation for {mri_fname}, using center region')
            h, w = mid_slice_img.shape
            bbox = np.array([w//4, h//4, 3*w//4, 3*h//4])
            _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=mid_slice_idx,
                obj_id=1,
                box=bbox,
            )
        
        # Forward propagation
        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
            segs_3D[out_frame_idx, (out_mask_logits[0] > 0.0).cpu().numpy()[0]] = 1
        
        # Reset and backward propagation
        predictor.reset_state(inference_state)
        inference_state = predictor.init_state(img_resized, video_height, video_width)
        
        # Repeat prompts for backward propagation
        if has_box and has_points and not propagate_with_box:
            _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=mid_slice_idx,
                obj_id=1,
                box=bbox,
                points=points_data,
                labels=labels_data,
            )
        elif has_box:
            _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=mid_slice_idx,
                obj_id=1,
                box=bbox,
            )
        elif has_points:
            _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=mid_slice_idx,
                obj_id=1,
                points=points_data,
                labels=labels_data,
            )
        else:
            _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=mid_slice_idx,
                obj_id=1,
                box=bbox,
            )
        
        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state, reverse=True):
            segs_3D[out_frame_idx, (out_mask_logits[0] > 0.0).cpu().numpy()[0]] = 1
        
        predictor.reset_state(inference_state)
    
    # Post-processing
    if np.max(segs_3D) > 0:
        segs_3D = getLargestCC(segs_3D)
        segs_3D = np.uint8(segs_3D)
    
    # Calculate Dice score if ground truth is available
    dice = None
    if gts_path is not None:
        try:
            print(f'Looking for GT for {gt_fnames}')    
            gt_path = join(gts_path, gt_fnames[mri_fnames.index(mri_fname)])
            print(f'GT path: {gt_path}')
            
            if os.path.exists(gt_path):
                gt_image = sitk.ReadImage(gt_path)
                gt_data = sitk.GetArrayFromImage(gt_image)
                
                # Ensure GT has same shape as prediction
                if gt_data.shape == segs_3D.shape:
                    # Convert GT to binary if it's not already
                    gt_binary = (gt_data > 0).astype(np.uint8)
                    
                    # Calculate binary Dice score
                    dice = dice_score_binary(segs_3D, gt_binary)
                    print(f'{mri_fname}: Dice score = {dice:.4f}')
                else:
                    print(f'Warning: GT shape {gt_data.shape} does not match prediction shape {segs_3D.shape}')
            else:
                print(f'Warning: Ground truth not found for {mri_fname}')
        except Exception as e:
            print(f'Error loading ground truth for {mri_fname}: {e}')
    
    # Save results
    sitk_image = sitk.GetImageFromArray(mri_image_data_pre)
    sitk_image.CopyInformation(mri_image)
    sitk_mask = sitk.GetImageFromArray(segs_3D)
    sitk_mask.CopyInformation(mri_image)
    
    save_seg_name = mri_fname.split('.nii.gz')[0] + '_mask.nii.gz'
    sitk.WriteImage(sitk_image, os.path.join(pred_save_dir, mri_fname.replace('.nii.gz', '_img.nii.gz')))
    sitk.WriteImage(sitk_mask, os.path.join(pred_save_dir, save_seg_name))
    
    seg_info['mri_name'].append(save_seg_name)
    seg_info['mid_slice_index'].append(mid_slice_idx)
    seg_info['dice_score'].append(dice if dice is not None else np.nan)

# Save info dataframe
seg_info_df = pd.DataFrame(seg_info)
seg_info_df.to_csv(join(pred_save_dir, 'mri_seg_dwi_info.csv'), index=False)

# Print summary statistics if Dice scores were calculated
if gts_path is not None and 'dice_score' in seg_info:
    valid_dice_scores = [d for d in seg_info['dice_score'] if not np.isnan(d)]
    if valid_dice_scores:
        print(f'\n=== Dice Score Summary ===')
        print(f'Mean Dice: {np.mean(valid_dice_scores):.4f}')
        print(f'Std Dice: {np.std(valid_dice_scores):.4f}')
        print(f'Median Dice: {np.median(valid_dice_scores):.4f}')
        print(f'Min Dice: {np.min(valid_dice_scores):.4f}')
        print(f'Max Dice: {np.max(valid_dice_scores):.4f}')
        print(f'Number of cases evaluated: {len(valid_dice_scores)}')



