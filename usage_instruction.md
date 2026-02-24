# GEMMA3S Usage Instructions

> **Note**: GEMMA3S is built on the [MedSAM2](https://github.com/bowang-lab/MedSAM2) segmentation framework. The core segmentation capabilities are powered by MedSAM2, with extensions for report generation and patient management.

## Table of Contents

1. [Getting Started](#getting-started)
2. [Uploading MRI Scans](#uploading-mri-scans)
3. [Patient Selection](#patient-selection)
4. [Interactive Segmentation](#interactive-segmentation)
5. [Tracking and Propagation](#tracking-and-propagation)
6. [Analysis and Reporting](#analysis-and-reporting)
7. [Advanced Features](#advanced-features)
8. [Tips and Best Practices](#tips-and-best-practices)
9. [Troubleshooting Guide](#troubleshooting-guide)

---

## Getting Started

### Download Model Checkpoints

Before launching the application you need the MedSAM2 and related checkpoints in the `checkpoints/` folder.
The repository includes a helper script `download.sh` at the project root that downloads commonly used checkpoints into `checkpoints/`.

From the repository root run:

```bash
chmod +x download.sh
./download.sh
```

This will download MedSAM2 checkpoints and place them in `checkpoints/`. If you prefer, you can manually place checkpoint files there instead.

### Launching the Application

1. **Start the Server**

   ```bash
   cd /path/to/MedSAM2/gemma3s
   python app_gemma3s.py
   ```
2. **Access the Interface**

   - Open your web browser
   - Navigate to: `http://localhost:18863`
   - You should see the GEMMA3S interface with the title "GEMMA3S: Spot, Segment & Simplify"
3. **Verify System Status**

   - Check the console for model loading messages
   - Ensure no CUDA errors appear
   - Wait for "Running on..." message indicating the server is ready

---

## Uploading MRI Scans

### For New Patients

1. **Navigate to Upload Tab**

   - Click on the "📤 Upload Scans" tab at the top
2. **Select Patient Mode**

   - Keep "New Patient" selected (default)
   - Note the auto-generated Patient ID (e.g., `pid_007`)
3. **Upload Files**

   - Click "Upload MRI Scan (NIfTI)" button
   - Select one or more `.nii.gz` or `.nii` files from your computer
   - Supported modalities: FLAIR, T1, T2, DWI, etc.
   - Multiple files will be stored in the same session
4. **BRATS Dataset Option** (if applicable)

   - If uploading BRATS dataset scans, check the "BRATS Dataset" toggle
   - This enables mirroring corrections for known BRATS data issues
5. **Start Upload & Analysis**

   - Click "🚀 Upload & Analyze" button
   - The system will:
     - Create patient folder structure
     - Save uploaded files
     - Update patient database (comman_format.json)
     - Run automated analysis pipeline
6. **Monitor Progress**

   - Watch the "Status & Pipeline Logs" textbox
   - Look for:
     - ✅ Upload successful message
     - Pipeline execution status
     - Any error messages (in red)

### For Existing Patients (Adding New Session)

1. **Switch Patient Mode**

   - Select "Existing Patient" radio button
2. **Select Patient**

   - Choose patient from the dropdown list
   - Dropdown shows all existing patient IDs
3. **Upload and Analyze**

   - Follow steps 3-5 from "New Patients" section
   - A new session (e.g., `sess_02`) will be created automatically

---

## Patient Selection

### Viewing Patient Overview

1. **Navigate to Patient Overview Tab**

   - Click "📋 Patient Overview" tab
2. **Understanding the Patient Table**

   **Color Coding:**

   - 🔴 **Red (Dark)**: Tumor detected, Not reviewed by radiologist
   - 🟠 **Orange**: Healthy/Normal, Not reviewed
   - 🟡 **Yellow**: Tumor detected, Reviewed by radiologist
   - 🟢 **Green**: Healthy/Normal, Reviewed

   **Table Columns:**

   - **#**: Row number
   - **Patient ID**: Unique identifier (e.g., pid_001)
   - **Tumor Status**: "Tumor Present" or "Healthy/Normal"
   - **Conf Score**: AI confidence score (0.0-1.0)
   - **Reviewed**: Radiologist review status
   - **Remark**: Additional notes or AI-generated comments
   - **Modalities**: Available scan types (FLAIR, T1, T2, etc.)
3. **Refresh Table**

   - Click "🔄 Refresh Patient Table" to update with latest data

### Selecting Patient for Segmentation

1. **Choose Patient**

   - Use the dropdown under "Select Patient for Segmentation"
   - The dropdown shows all available patients with their status
   - Format: "1. pid_001 - 🔴 Tumor | ⏳ Not Reviewed"
2. **Load Patient Data**

   - Click "🔬 Load & Segment →" button
   - The system will:
     - Load the most recent session for the patient
     - Find the FLAIR scan automatically
     - Prepare it for segmentation
     - Switch to the Segmentation tab

---

## Interactive Segmentation

### Initial Setup

When you arrive at the Segmentation tab after loading a patient:

1. **Verify Loaded Patient**

   - Check the blue info box showing:
     - Patient ID
     - Session ID
     - Scan filename
2. **Select Model Configuration** (Optional)

   - **SAM Config**: Model architecture variant

     - Recommended: `sam2.1_hiera_small` (balanced)
     - `sam2.1_hiera_tiny` for faster, lower memory
     - `sam2.1_hiera_base_plus` for best accuracy
   - **Checkpoint**: Pre-trained model weights

     - Recommended: `MedSAM2_latest` (general purpose)
     - `MedSAM2_CTLesion` for CT scans
     - `MedSAM2_MRI_LiverLesion` for liver MRI
3. **Load Video from NIfTI**

   - Click "🎬 Use selected scan (NIfTI) as video" button
   - Wait for processing (~10-30 seconds depending on volume size)
   - The system converts 3D volume to video format for annotation
4. **Verify Initialization**

   - Check "Current Frame" display shows target slice (from annotation JSON if available)
   - Canvas should display the initial frame
   - Frame slider should be enabled

### Annotation Methods

You have two annotation methods available:

#### Method 1: Stroke-Based Annotation (Recommended for Lesions)

1. **Select Annotation Tool**

   - Ensure you're on the "Stroke-Based Annotation" section
2. **Draw on Canvas**

   - Use your mouse to draw/paint over the lesion area
   - The drawing editor allows you to:
     - **Draw**: Click and drag to paint
     - **Erase**: Use eraser tool if available
     - **Zoom**: Zoom in for precision
3. **Generate Mask**

   - After drawing, click "🎨 Segment with stroke" button
   - SAM2 will generate a refined mask based on your stroke
   - The mask appears as a colored overlay
4. **Review Result**

   - Check if the segmentation covers the lesion accurately
   - If not satisfied, redraw and click segment again

#### Method 2: Click-Based Annotation (Recommended for Well-Defined Objects)

1. **Select Point Mode**

   - Choose "Positive" for points inside the object
   - Choose "Negative" for points outside/background
2. **Click on Image**

   - Click directly on the canvas where you want to add points
   - Positive points (green) indicate "include this region"
   - Negative points (red) indicate "exclude this region"
3. **Refine with Multiple Points**

   - Add multiple positive points on different parts of the lesion
   - Add negative points on nearby areas to exclude
   - Each click updates the segmentation automatically

### Managing Multiple Objects

1. **Object ID Slider**

   - Use the slider to create/select different objects
   - Default: Object 0
   - Increment to create new objects (e.g., Object 1 for second lesion)
2. **Current Object ID**

   - Displays which object you're currently annotating
   - All annotations apply to the current object only
3. **Increment Object**

   - Click "➕ Increment Object ID" to create a new object
   - Previous objects are preserved

### Frame Navigation

1. **Frame Slider**

   - Drag slider to navigate through video frames
   - Each position corresponds to a slice in the 3D volume
2. **Reannotate Different Frames**

   - Move slider to another frame
   - Add annotations on that frame
   - Useful for refining tracking or annotating multiple slices

---

## Tracking and Propagation

### Running Automatic Tracking

1. **Ensure Initial Annotation Exists**

   - At least one frame should have a segmentation mask
   - Typically the target frame shown initially
2. **Start Tracking**

   - Click "Start Tracking" button
   - The system will:
     - Propagate mask forward (to subsequent frames)
     - Propagate mask backward (to previous frames)
     - Generate masks for all frames in the video
3. **Monitor Progress**

   - Console shows debug messages for tracking progress
   - "Bidirectional tracking complete" indicates success
4. **Review Results**

   - Use frame slider to navigate through volume
   - Verify mask quality on different slices
   - Masks should follow lesion boundaries smoothly

### Viewing Tracked Results

1. **Output Video**

   - Automatically generated video showing masks overlaid
   - Available in the "Output Video" section
   - Can be downloaded for review
2. **Mask Files**

   - Individual mask PNGs saved in `/tmp/output_masks/{session_id}/`
   - Combined frames in `/tmp/output_combined/{session_id}/`

### Radiologist Review

After segmentation and tracking, you can review and confirm the AI's findings:

1. **Review Panel**

   - Located in the "🩺 Radiologist Review Panel" section
   - Shows the AI's tumor prediction status
2. **Verify Prediction**

   - Check the "AI Prediction" display
   - Toggle "Tumor Present" checkbox if correction needed
3. **Mark as Reviewed**

   - Click "✅ Mark as Reviewed by Radiologist"
   - This updates the patient database with your review
   - Status appears in the review status textbox

---

## Analysis and Reporting

### Generating Reports and Analysis

After completing segmentation and tracking on the Segmentation tab:

1. **Navigate to Analysis & Reporting Tab**

   - Click "📊 Analysis & Reporting" tab
2. **Generate AI Reports**

   - Click "✨ Generate AI Reports" button
   - The system will:
     - Convert 2D masks back to 3D NIfTI segmentation
     - Calculate tumor volume in mm³ and mL
     - Run parcellation analysis (affected brain regions)
     - Generate clinical and patient-friendly reports
     - Create DOCX files for download
     - Update patient database with results
3. **Monitor Progress**

   - Reports appear in the text boxes
   - DOCX files become available for download
   - Volume data is stored for trend analysis

### Accessing Reports

#### Clinical Report

- Detailed medical findings for radiologists
- Volume measurements and anatomical locations
- Technical terminology and analysis details

#### Patient-Friendly Report

- Simplified language explaining findings
- Key results in understandable terms
- Recommendations if applicable

#### Download Reports

- Click download buttons for DOCX files
- Files are saved in the patient folder permanently
- Ready for sharing or printing

### Volume Trend Analysis

1. **View Volume Plot**

   - Line graph showing tumor volume across sessions
   - X-axis: Session dates/IDs
   - Y-axis: Volume in mL
   - Automatically loads when tab is selected
2. **Refresh Plot**

   - Click "🔄 Refresh Plot" to update with latest data
3. **Interpretation**

   - Increasing trend: Possible tumor growth
   - Decreasing trend: Reduction/recovery
   - Stable: No significant change

---

## Advanced Features

### Manual Report Regeneration

If you need to regenerate reports (e.g., after corrections):

1. **Navigate to Analysis & Reporting Tab**
2. **Click "✨ Generate AI Reports"**
3. **Wait for Processing** (~30-60 seconds)
4. **Review New Reports**

### Custom Frame Rates

For faster processing or more detailed analysis:

1. **Adjust "Scale" Slider** (before loading video)
   - Higher values: More frames extracted (slower, more detail)
   - Lower values: Fewer frames (faster, less detail)
   - Range: 1.0 to video FPS

### Working with Annotations JSON

To specify exact slices for annotation:

1. **Create JSON File**

   - Location: `common_data/{patient_id}/json/annotation.json`
2. **Format:**

   ```json
   {
     "sess_01": {
       "slice_id": 133
     },
     "sess_02": {
       "slice_id": 150
     }
   }
   ```
3. **Effect**

   - System automatically navigates to specified slice
   - Useful for consistent annotation across patients

### Batch Processing

Use the folder watcher for automated processing:

1. **Configure Watcher**

   ```bash
   python folder_watcher.py
   ```
2. **Drop Scans**

   - Place NIfTI files in monitored directory
   - System automatically processes new files

---

## Tips and Best Practices

### For Best Segmentation Results

1. **Start on Clear Slice**

   - Choose a frame where lesion boundaries are distinct
   - Middle of the lesion often works well
2. **Stroke Annotation Tips**

   - Cover the interior of the lesion
   - Don't worry about exact boundaries (SAM2 refines)
   - Multiple small strokes can work better than one large stroke
3. **Click Annotation Tips**

   - Start with one positive click in lesion center
   - Add more positive clicks if parts are missed
   - Use negative clicks to exclude incorrectly included regions
4. **Multi-Object Scenarios**

   - Segment one lesion completely before moving to next
   - Use consistent object IDs across sessions for same lesion
   - Document which object ID corresponds to which lesion

### For Efficient Workflow

1. **Prepare Scans in Advance**

   - Ensure NIfTI files are properly formatted
   - Name files descriptively (include modality: flair_001.nii.gz)
   - Organize by patient and session
2. **Use Keyboard Shortcuts** (if available)

   - Tab through interface elements
   - Enter to confirm actions
3. **Monitor GPU Memory**

   - Close unnecessary applications
   - Process one patient at a time if memory is limited
4. **Regular Data Backup**

   - Backup `common_data/` directory regularly
   - Export important reports to external storage

### For Longitudinal Studies

1. **Consistent Naming**

   - Use systematic patient IDs
   - Sessions should be chronologically numbered
2. **Annotation Consistency**

   - Try to annotate same slice across sessions
   - Use annotation JSON to standardize slice selection
3. **Volume Tracking**

   - Check volume plot after each new session
   - Look for trends over time

---

## Appendix: File Structure Reference

### Patient Data Directory

```
common_data/
├── comman_format.json              # Global patient database
├── pid_001/
│   ├── mri_scans/
│   │   ├── sess_01/
│   │   │   ├── flair.nii.gz       # Original scan
│   │   │   ├── t1.nii.gz
│   │   │   └── t2.nii.gz
│   │   └── sess_02/
│   │       └── flair.nii.gz
│   ├── json/
│   │   └── annotation.json         # Slice indices
│   ├── patient_results.json        # Per-session results
│   ├── pid_001_sess_01_seg.nii.gz # Segmentation output
│   ├── pid_001_sess_01_clinical_report.docx
│   ├── pid_001_sess_01_patient_report.docx
│   └── pid_001_sess_01_parcellation.txt
```

### Temporary Files

```
/tmp/ or project_root/tmp/
├── output_frames/{session_id}/     # Extracted video frames
├── output_masks/{session_id}/      # Segmentation masks (PNG)
├── output_combined/{session_id}/   # Frames + masks overlay
├── output_files/{session_id}/      # NIfTI exports, reports, DOCX files
├── gradio_downloads/               # Files prepared for download
```

---

## Getting Help

### Resources

- **README.md**: Overview and technical details
- **Console Logs**: Detailed debug information
- **GitHub Issues**: Report bugs or request features
- **Community**: Share tips and solutions

### Contact

For persistent issues or questions:

- Check console output for error details
- Include error messages when reporting issues
- Provide steps to reproduce problems
- Share sample data if possible (anonymized)

---

**Last Updated**: February 2026  
**Version**: 1.1  
**For**: GEMMA3S Application
