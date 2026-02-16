import os
from simplify_report import MedGemmaSimplify

def main():

    base_path = os.path.expanduser("~/ashu/Med_Gemma/Data")

    mr_volume_path = os.path.join(
        base_path,
        "BraTS20_Training_001/BraTS20_Training_001_t1ce.nii"
    )

    segmentation_mask_path = os.path.join(
        base_path,
        "BraTS20_Training_001/BraTS20_Training_001_seg.nii"
    )

    slice_index = 72

    manual_report_context = "211 ml tumor extending into Right-Cerebral-White-Matter, ctx-rh-superiortemporal, ctx-rh-middletemporal, Right-Thalamus, ctx-rh-insula, ctx-rh-inferiortemporal, Left-Cerebral-White-Matter regions."

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
        clinical_docx="Clinical_MRI_Report.docx",
        patient_docx="Patient_Friendly_Report.docx"
    )

    print("\nClinical Report Generated")
    print("Saved: Clinical_MRI_Report.docx")

    print("\nPatient-Friendly Report Generated")
    print("Saved: Patient_Friendly_Report.docx")


if __name__ == "__main__":
    main()
