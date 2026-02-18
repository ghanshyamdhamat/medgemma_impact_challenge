# NIfTI Integration Guide for MedSAM2

## Overview
The updated `app_medgemma.py` now includes full support for loading and segmenting medical images from NIfTI files (.nii.gz) stored in the patient database. This complements the existing video segmentation functionality.

## New Features Added

### 1. NIfTI Loading and Conversion
- **Function**: `load_nifti_slice(nifti_path, slice_idx=None, normalize=True)`
  - Loads NIfTI medical image files
  - Extracts specified slice (or middle slice if not specified)
  - Normalizes intensity values to 0-255 range using percentile-based normalization
  - Returns RGB image for display

### 2. Annotation Loading from JSON
- **Function**: `load_annotation_from_json(patient_id, scan_name)`
  - Loads annotation metadata from JSON files in patient directories
  - Matches scan names to annotation entries
  - Returns point prompts and other annotation data

- **Supported Annotation Format**:
  ```json
  {
    "scan_name": {
      "mid_slice_idx": 109,
      "points": [[75, 63]],
      "labels": [1],
      "bbox": [x0, y0, x1, y1]  // optional
    }
  }
  ```

### 3. Patient Session Data Structure
- **Function**: `get_nifti_files_in_session(patient_id, session_id)`
  - Retrieves all NIfTI files from a specific patient session
  - Returns dictionary mapping filenames to full paths

## User Interface Changes

### Segmentation Tab - New NIfTI Section
The segmentation tab now includes a new "📁 NIfTI Medical Image" tab alongside the existing video input:

```
┌─────────────────────────────────────────┐
│ 📁 NIfTI Medical Image │ 📹 Video Input  │
├─────────────────────────────────────────┤
│ Select NIfTI Scan: [dropdown] [Load]    │
│ Slice Index: [slider]                   │
│ [Annotation info display]               │
└─────────────────────────────────────────┘
```

### Workflow
1. **Patient Selection Tab**:
   - Select Patient → Session → Scan
   - Click "Go to Segmentation"
   - All NIfTI files in that session are loaded

2. **Segmentation Tab**:
   - **NIfTI Section**:
     - Select scan from dropdown
     - Click "Load NIfTI Slice" to load the image
     - Use slider to navigate through slices
     - View annotations and slice information
   
   - **Prompting Section** (shared for both video and NIfTI):
     - Stroke to Box Prompt: Draw bounding box
     - Point Prompt: Click to add/remove points
     - Add New Object: Create multiple object tracking
   
   - **Output Section**:
     - View segmentation results
     - Download masks

## Technical Implementation

### NIfTI Data Handling
```python
# Load NIfTI file
img = nib.load(nifti_path)
data = img.get_fdata()

# Handle 4D data (time series)
if data.ndim == 4:
    data = data[..., 0]

# Extract slice and normalize
slice_data = data[:, :, slice_idx]
```

### Annotation Visualization
When a slice is loaded, annotations are automatically drawn:
- **Green circles**: Positive points (to include in segmentation)
- **Red circles**: Negative points (to exclude from segmentation)
- **Blue rectangles**: Bounding boxes

### Integration with Segmentation
- NIfTI slices are converted to the same format as video frames
- All segmentation prompts (box and point) work identically
- Results can be displayed and downloaded in the same way

## File Structure Requirements

```
common_data/
├── patient_id/
│   ├── mri_scans/
│   │   └── session_id/
│   │       ├── scan_name.nii.gz
│   │       └── other_scans.nii.gz
│   └── json/
│       ├── annotation_file.json
│       └── other_annotations.json
```

## JSON Annotation Format

### Example annotation_json.json:
```json
{
  "BraTS-GLI-00005-100": {
    "mid_slice_idx": 109,
    "points": [[75, 63]],
    "labels": [1]
  },
  "sub-10_space-T1w_desc-dwi_flirt": {
    "mid_slice_idx": 274,
    "points": [[105, 162]],
    "labels": [1],
    "bbox": [50, 40, 150, 180]
  }
}
```

## Dependencies
- **nibabel**: NIfTI file I/O (already installed)
- **numpy**: Array operations
- **opencv-python (cv2)**: Image processing
- **torch, gradio**: For segmentation and UI

## Performance Notes

- Large 3D volumes are handled by loading individual slices on demand
- Normalization uses percentile-based approach for better visualization
- Multiprocessing ensures UI remains responsive during NIfTI loading
- Memory is efficiently managed through garbage collection

## Error Handling

- Missing NIfTI files: Displays error message to user
- Missing annotation files: Loads image without annotations
- 4D data: Automatically takes first time point
- Empty dimensions: Gracefully falls back to middle slice

## Future Enhancements

Potential additions:
- 3D volume rendering
- Slice navigation through trackpad/mouse wheel
- Batch processing multiple scans
- 3D segmentation export
- Volume measurement and statistics
- DICOM file support
