# Gemma3S: Reimagining Radiology with Spot, Segment & Simplify

## Overview

**Gemma3S** is an end-to-end radiology suite built on Google’s **MedSigLIP** and **MedGemma** along with MedSAM2.  It enables automated diagnosis, precise lesion segmentation, and longitudinal tracking, generating both clinical and patient-friendly reports.

## Key Features of the MedGemma Pillars

### 🎯 Spot (Anamoly Detector & Patient Priority Manager):
- **MedSigLIP powered diagnosis**: Automated premilinary diagnosis with confidence scores.
- **Comprehensive Patient Database**: Organized patient records with hierarchical structure (Patient → Session → Scan).
- **MRI Scan Upload**: Direct upload of NIfTI format MRI scans with automatic organization.
- **Session Tracking**: Support for multiple imaging sessions per patient for longitudinal studies.
- **Smart Patient Overview**: Color-coded patient table showing tumor status, review status, and confidence scores.

### 🎯 Segment (MedSigLIP guided Interactive Lesion Delineator):
- **MedSAM2 Integration**: Precise delineation (segmentation) of anomalous region using MedSAM2.
- **Video Based Workflow**: Converts 3D NIfTI volumes to video format for intuitive slice-by-slice annotation.
- **Interactive Annotation Tools**:
  - Stroke based annotation with brush tool.
  - Click based annotation (positive/negative points).
  - Real time mask preview and refinement.
- **Bidirectional Propagation**: Automatically propagates annotations forward and backward through slices.

### 🏥 Simplify (Report Generator and a Patient Assistant):
- **MedGemma Integration**:Reports generated automatically after Segment using MedGemma.
- **Dual Report System**:
  - **Clinical Report**: Detailed radiological findings for healthcare professionals.
  - **Patient Report**: Simplified, patient-friendly explanation of findings.
- **Document Export**: Reports available as Markdown and .docx  format for download.

### 📊 Advanced Analysis Tools
- **Volume Measurement**: Automatic calculation of lesion volumes in mm3.
- **NIfTI Export**: Segmentation masks exported in standard NIfTI format preserving original headers and spatial transformation.
- **Hallucination Control**: Parcellation grounding for hallucination-free report generation.
- **Longitudinal Tracking**: Volume trend plots showing lesion progression across multiple sessions.

### 🎨 Modern User Interface
- **Gradio 6.x Framework**: Clean, responsive web-based interface.
- **Multi-Tab Navigation**: Organized workflow across Upload, Overview, Segmentation, Analysis, and Reports tabs.
- **Real-Time Updates**: Live preview of segmentation masks and tracking results.
- **Download Support**: Easy download of segmentation files, reports, and analysis results.

## Technical Architecture

### Core Technologies
- **Deep Learning Framework**: PyTorch with CUDA acceleration
- **Anomaly Detection**: LoRA finetuned MedSigLIP
- **Segmentation Model**: MedSAM2 
- **Report Generation**: MedGemma 
- **Medical Imaging**: NiBabel for NIfTI file handling
- **UI Framework**: Gradio 6.6.0+
- **Video Processing**: MoviePy, FFmpeg, OpenCV
- **Data Handling**: NumPy, Pandas for numerical operations

### System Requirements
- **GPU**: NVIDIA GPU with CUDA support (recommended: 16GB+ VRAM)
- **RAM**: 32GB+ recommended
- **OS**: Linux (tested on Ubuntu)
- **Python**: 3.8+


## Project Structure
```
gemma3s/
├── app_medgemma_new_gradio.py  # Main Gradio application
├── simplify_report.py           # MedGemma report generation module
├── pipeline.py                  # Automated analysis pipeline
├── folder_watcher.py            # File system monitoring for auto-processing
├── README.md                    # This file
└── usage_instruction.md         # Detailed usage guide

common_data/                     # Patient data directory
├── comman_format.json          # Patient database summary
├── pid_001/                    # Patient folder
│   ├── mri_scans/
│   │   └── sess_01/            # Session folder
│   │       └── *.nii.gz        # NIfTI scan files
│   ├── json/                   # Annotation metadata
│   ├── patient_results.json    # Session analysis results
│   └── *_seg.nii.gz           # Segmentation outputs
└── ...

checkpoints/                     # Model checkpoints
├── MedSAM2_latest.pt
├── MedSAM2_CTLesion.pt
└── ...

sam2/                           # SAM2 model implementation
└── configs/                    # Model configuration files
```

## Installation

### Prerequisites
```bash
# Install CUDA toolkit (if not already installed)
# Visit: https://developer.nvidia.com/cuda-downloads
# Install Python dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Setup
```bash
# Clone the repository
git clone <repository-url>
cd MedSAM2/gemma3s

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
export HF_TOKEN="your_huggingface_token"  # Required for MedGemma
export GRADIO_SERVER_PORT=18863           # Optional: custom port

# Download model checkpoints (if not included)
# Place checkpoint files in ../checkpoints/ directory
```

### Configuration
The application automatically detects available checkpoints and configs. Ensure:
1. Model checkpoints are in `../checkpoints/` directory
2. SAM2 configs are in `../sam2/configs/` directory
3. `common_data/` directory exists for patient data storage

## Quick Start

### Launch the Application
```bash
python app_medgemma_new_gradio.py
```
The application will start on `http://0.0.0.0:18863` (or your configured port).

### Basic Workflow
1. **Upload Scans**: Navigate to "Upload Scans" tab and upload NIfTI files.
2. **Select Patient**: Review patient overview and select patient for segmentation.
3. **Segment**: Use interactive tools to make prompts on the MedSigLIP picked target frame.
4. **Track**: Run automatic propagation to segment entire volume.
5. **Analyze**: View volume measurements and parcellation results.
6. **Review Reports**: Access AI-generated clinical and patient reports.

For detailed step-by-step instructions, see [usage_instruction.md](usage_instruction.md).

## Data Format

### Input Files
- **Format**: NIfTI (.nii.gz or .nii)
- **Modalities**: FLAIR, T1, T2, DWI, etc.
- **Dimensions**: 3D volumes (any size, automatically handled)

### Output Files
- **Segmentation Masks**: NIfTI format (.nii.gz) binary lesion masks
- **Medical Reports**: .docx and Markdown formats

## API and Extension

### Custom Pipelines

The `pipeline.py` module can be used for batch processing:

```python
from pipeline import run_pipeline_for_patient

success, message = run_pipeline_for_patient("pid_001")
```

### Report Generation
MedGemma can be used independently:
```python
from simplify_report import MedGemmaSimplify
gemma = MedGemmaSimplify(hf_token="your_token")
clinical, patient = gemma.generate_reports(context_data)
```


## Acknowledgments
This project stands on the shoulders of giants and would not be possible without the following outstanding works:

Gemma3S is primarily powered by Google's **MedGemma** and **MedSigLIP**, which serve as the central multimodal engines enabling  automated diagnosis, vision–language alignment, and automated clinical reporting.
- **Paper**  Sellergren, Andrew, et al. "Medgemma technical report." arXiv preprint arXiv:2507.05201 (2025).   
- **Models**: https://huggingface.co/google/medgemma-1.5-4b-it    https://huggingface.co/google/medsiglip-448

**This codebase builds upon [MedSAM2](https://github.com/bowang-lab/MedSAM2)** by the Bo Wang Lab at University of Toronto. MedSAM2 provides the foundational segmentation framework that powers GEMMA3S.
- **Paper**: Ma, Jun, et al. "Medsam2: Segment anything in 3d medical images and videos." arXiv preprint arXiv:2504.03600 (2025).
- **Models**: https://huggingface.co/wanglab/MedSAM2
- **Project Page**: https://medsam2.github.io/


## Contact and Support
For questions, issues, or feature requests:
- **Email**: ggd5551@gmail.com
