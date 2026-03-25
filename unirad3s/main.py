import os
from simplify_report import MedGemmaSimplify

def main():

    base_path = os.path.expanduser("/mnt/bb586fde-943d-4653-af27-224147bfba7e/Medgemma/MedSAM2/common_data/pid_002/mri_scans")

    mr_volume_path = os.path.join(
        base_path,
        "sess_03/BraTS20_Training_001_t1ce.nii"
    )

    segmentation_mask_path = os.path.join(
        base_path,
        "sess_03/pid_002_sess_03_seg.nii.gz"
    )

    slice_index = 72

    manual_report_context = (
    "Background anatomical context:\n"
    "- Known tumor volume: approximately 211 ml\n"
    "- Previously reported regional involvement includes:\n"
        " - Left cerebral white matter\n"
    " - Left superior temporal cortex\n"
    " - Left middle temporal cortex\n"
    " - Left inferior temporal cortex\n"
    " - Left insular cortex\n"
    " - Left thalamus\n"
    " - Right cerebral white matter\n\n"
    "This information is provided for contextual reference only."
    )
    hf_token = os.getenv("HF_TOKEN")

    if hf_token is None:
        print("Warning: HF_TOKEN not found in environment variables.")

    print("Loading MedGemma Simplify module...")
    simplifier = MedGemmaSimplify(hf_token=hf_token)

    print("Running report generation pipeline...")
    clinical_report, patient_report = simplifier.generate_reports(
        mr_volume_path=mr_volume_path,
        slice_index=slice_index,
        segmentation_mask_path=segmentation_mask_path,
        manual_report_context=manual_report_context,
        clinical_docx="Clinical_Report_Gemma.docx",
        patient_docx="Patient_Friendly_Gemma.docx"
    )

    print("\nClinical Report Generated")
    print("Saved: Clinical_Report_Gemma.docx")

    print("\nPatient-Friendly Report Generated")
    print("Saved: Patient_Friendly_Gemma.docx")


if __name__ == "__main__":
    main()
