# GEMMA3S: Spot, Segment & Simplify

> **⚠️ Attribution Notice**: This project builds heavily on the [MedSAM2](https://github.com/bowang-lab/MedSAM2) codebase developed by the Bo Wang Lab. We extend our sincere gratitude to the MedSAM2 team for their groundbreaking work in medical image segmentation. The core segmentation engine, SAM2 model integration, and video-based segmentation pipeline are adapted from their original implementation. Please see the [Acknowledgments](#acknowledgments) section for detailed credits.

## Overview

GEMMA3S (Generative Medical Model for Analysis - Spot, Segment & Simplify) is an advanced AI-powered medical imaging platform that combines state-of-the-art segmentation technology with generative AI to provide comprehensive brain MRI analysis. The system enables clinicians and researchers to segment brain lesions, track their progression over time, and generate both clinical and patient-friendly medical reports automatically.

## Key Features

### 🎯 Interactive Medical Image Segmentation

- **MedSAM2 Integration**: Leverages the latest MedSAM2 model for accurate medical image segmentation
- **Video-Based Workflow**: Converts 3D NIfTI volumes to video format for intuitive slice-by-slice annotation
- **Interactive Annotation Tools**:
  - Stroke-based annotation with brush tool
  - Click-based annotation (positive/negative points)
  - Real-time mask preview and refinement
- **Bidirectional Propagation**: Automatically propagates annotations forward and backward through video frames

### 🏥 Patient Management System

- **Comprehensive Patient Database**: Organized patient records with hierarchical structure (Patient → Session → Scan)
- **MRI Scan Upload**: Direct upload of NIfTI format MRI scans with automatic organization
- **Session Tracking**: Support for multiple imaging sessions per patient for longitudinal studies
- **Smart Patient Overview**: Color-coded patient table showing tumor status, review status, and confidence scores

### 📊 Advanced Analysis Tools

- **Volume Measurement**: Automatic calculation of lesion volumes in milliliters with voxel counting
- **NIfTI Export**: Segmentation masks exported in standard NIfTI format preserving original headers and spatial information
- **Parcellation Analysis**: Brain region identification using FreeSurfer atlas parcellation
- **Longitudinal Tracking**: Volume trend plots showing lesion progression across multiple sessions

### 🤖 AI-Powered Report Generation

- **Dual Report System**:
  - **Clinical Report**: Detailed radiological findings for healthcare professionals
  - **Patient Report**: Simplified, patient-friendly explanation of findings
- **MedGemma Integration**: Uses Google's specialized medical language model for report generation
- **Automated Workflow**: Reports generated automatically after segmentation completion
- **Document Export**: Reports available as both Markdown and DOCX formats

### 🎨 Modern User Interface

- **Gradio 6.x Framework**: Clean, responsive web-based interface
- **Multi-Tab Navigation**: Organized workflow across Upload, Overview, Segmentation, Analysis, and Reports tabs
- **Real-Time Updates**: Live preview of segmentation masks and tracking results
- **Download Support**: Easy download of segmentation files, reports, and analysis results

## Technical Architecture

### Core Technologies

- **Deep Learning Framework**: PyTorch with CUDA acceleration
- **Segmentation Model**: SAM2 (Segment Anything Model 2) with MedSAM2 checkpoints
- **Medical Imaging**: NiBabel for NIfTI file handling
- **UI Framework**: Gradio 6.6.0+
- **Report Generation**: MedGemma (Medical Gemma variant)
- **Video Processing**: MoviePy, FFmpeg, OpenCV
- **Data Handling**: NumPy, Pandas for numerical operations

### System Requirements

- **GPU**: NVIDIA GPU with CUDA support (recommended: 16GB+ VRAM)
- **RAM**: 32GB+ recommended
- **Storage**: SSD with sufficient space for patient data and models
- **OS**: Linux (tested on Ubuntu)
- **Python**: 3.8+

### Model Checkpoints

The system supports multiple checkpoint configurations:

- `MedSAM2_latest.pt` - Latest general medical segmentation model
- `MedSAM2_CTLesion.pt` - Optimized for CT lesion segmentation
- `MedSAM2_MRI_LiverLesion.pt` - Specialized for MRI liver lesions
- `EfficientTAM` variants - Lightweight models for faster inference
- SAM2.1 Hiera configurations

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

1. **Upload Scans**: Navigate to "Upload Scans" tab and upload NIfTI files
2. **Select Patient**: Review patient overview and select patient for segmentation
3. **Segment**: Use interactive tools to annotate lesions on the target frame
4. **Track**: Run automatic propagation to segment entire volume
5. **Analyze**: View volume measurements and parcellation results
6. **Review Reports**: Access AI-generated clinical and patient reports

For detailed step-by-step instructions, see [usage_instruction.md](usage_instruction.md).

## Data Format

### Input Files

- **Format**: NIfTI (.nii.gz or .nii)
- **Modalities**: FLAIR, T1, T2, DWI, etc.
- **Dimensions**: 3D volumes (any size, automatically handled)

### Output Files

- **Segmentation Masks**: NIfTI format (.nii.gz) with binary masks
- **Volume Reports**: Text files with quantitative measurements
- **Parcellation Results**: JSON with brain region annotations
- **Medical Reports**: DOCX and Markdown formats

### Annotation Files

- **Location**: `common_data/<patient_id>/json/`
- **Format**: JSON with slice indices and metadata
- **Example**:
  ```json
  {
    "sess_01": {
      "slice_id": 133,
      "scan_name": "flair.nii.gz"
    }
  }
  ```

## Features in Detail

### Segmentation Workflow

1. **NIfTI to Video Conversion**: Slices extracted at configurable frame rates
2. **Initial Annotation**: User annotates target slice (usually from JSON metadata)
3. **Mask Refinement**: Interactive editing with stroke or click tools
4. **Propagation**: SAM2 propagates mask bidirectionally through all frames
5. **NIfTI Export**: Masks reconstructed into 3D volume with original spatial information

### Volume Calculation

- Voxel counting in segmented regions
- Automatic volume computation using NIfTI header voxel dimensions
- Conversion to clinical units (mm³ → mL)
- Historical tracking across sessions

### Parcellation Integration

- Registration-based parcellation using FreeSurfer atlas
- Identification of affected brain regions
- Quantification of lesion overlap with anatomical structures
- Integration into clinical reports

### Report Generation Pipeline

1. **Context Gathering**: Volume data, parcellation results, scan metadata
2. **Clinical Report**: Detailed findings, measurements, anatomical locations
3. **Patient Report**: Simplified explanation with accessible language
4. **Storage**: Reports saved to patient folder and JSON database
5. **Export**: Available as downloadable DOCX files

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

## Performance Optimization

### GPU Memory Management

- Automatic mixed precision with bfloat16
- SDPA attention optimization for CUDA
- Lazy model loading (models loaded on first use)
- Session-based resource cleanup

### Inference Speed

- Frame interval adjustment for faster processing
- Efficient SAM2 propagation algorithms
- Background process management for concurrent users

### Storage Optimization

- Temporary files automatically cleaned after 15 minutes
- Compressed NIfTI format (.nii.gz)
- Incremental JSON updates

## Troubleshooting

### Common Issues

**GPU Out of Memory**

- Reduce video frame rate (increase frame interval)
- Use smaller/TinyVIT model variants
- Process shorter video segments

**Model Loading Errors**

- Verify checkpoint file paths in console output
- Check CUDA compatibility
- Ensure sufficient disk space (models are several GB)

**Upload Failures**

- Confirm NIfTI file format (.nii or .nii.gz)
- Check file permissions in `common_data/` directory
- Review console for detailed error messages

**Report Generation Timeout**

- Verify HF_TOKEN environment variable is set
- Check internet connection (MedGemma requires API access)
- Review MedGemma service status

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request with detailed description

## Citation

If you use GEMMA3S in your research, please cite:

```bibtex
@software{gemma3s,
  title={GEMMA3S: Generative Medical Model for Analysis - Spot, Segment & Simplify},
  author={Your Team},
  year={2026},
  url={https://github.com/your-repo/gemma3s}
}
```

## License

See [LICENSE](../LICENSE) file for details.

## Acknowledgments

This project stands on the shoulders of giants and would not be possible without the following outstanding works:

### Core Foundation: MedSAM2

**This codebase heavily builds upon [MedSAM2](https://github.com/bowang-lab/MedSAM2)** by the Bo Wang Lab at University of Toronto. MedSAM2 provides the foundational segmentation framework that powers GEMMA3S.

- **Repository**: https://github.com/bowang-lab/MedSAM2
- **Paper**: [MedSAM2: Segment Anything in 3D Medical Images and Videos](https://arxiv.org/abs/2504.03600)
- **Models**: https://huggingface.co/wanglab/MedSAM2
- **Project Page**: https://medsam2.github.io/

**Key components adapted from MedSAM2**:

- SAM2 video predictor implementation (`sam2/sam2_video_predictor.py`)
- Medical image predictor (`sam2/sam2_image_predictor.py`)
- Video-based segmentation workflow and frame propagation logic
- Model architecture and checkpoint loading (`sam2/build_sam.py`)

**Citation for MedSAM2**:

```bibtex
@article{medsam2,
  title={MedSAM2: Segment Anything in 3D Medical Images and Videos},
  author={Wang Lab Team},
  journal={arXiv preprint arXiv:2504.03600},
  year={2025},
  url={https://github.com/bowang-lab/MedSAM2}
}
```

### Additional Dependencies

- **SAM2 (Meta AI)**: [Segment Anything Model 2](https://github.com/facebookresearch/sam2) - Foundation model for promptable segmentation
- **MedGemma (Google)**: Medical language model for clinical report generation
- **FreeSurfer**: [Brain parcellation atlas and tools](https://surfer.nmr.mgh.harvard.edu/)
- **Gradio**: [UI framework](https://gradio.app/) for building the web interface
- **NiBabel**: [NIfTI file I/O library](https://nipy.org/nibabel/)
- **PyTorch**: Deep learning framework

### Our Contributions

GEMMA3S extends MedSAM2 with:

- AI-powered medical report generation using MedGemma
- Patient management and longitudinal tracking system
- Automated parcellation analysis and brain region identification
- Dual report generation (clinical + patient-friendly)
- Enhanced Gradio 6.x user interface with modern workflow
- Integrated volume measurement and trend analysis
- Document export capabilities (DOCX, Markdown)

We are deeply grateful to the MedSAM2 team and all open-source contributors whose work made this project possible.

## Contact and Support

For questions, issues, or feature requests:

- **GitHub Issues**: [Create an issue](https://github.com/your-repo/issues)
- **Email**: support@yourproject.org
- **Documentation**: See [usage_instruction.md](usage_instruction.md)

---

**Version**: 1.0
**Last Updated**: February 2026
**Status**: Production Ready
