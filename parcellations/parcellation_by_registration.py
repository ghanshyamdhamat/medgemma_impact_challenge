import numpy as np
import nibabel as nib
import ants
import sys
import re

def process_parcellation(parcellation, segmentation_path, lut_path, volume, dataset_name="Brats"):
    lobe_map = {
        'Frontal': [
            'superiorfrontal', 'rostralmiddlefrontal', 'caudalmiddlefrontal',
            'parsopercularis', 'parstriangularis', 'parsorbitalis',
            'lateralorbitofrontal', 'medialorbitofrontal',
            'precentral', 'paracentral', 'frontalpole'
        ],
        'Parietal': [
            'superiorparietal', 'inferiorparietal', 'supramarginal',
            'postcentral', 'precuneus'
        ],
        'Temporal': [
            'superiortemporal', 'middletemporal', 'inferiortemporal',
            'bankssuperiortemporal', 'fusiform', 'transversetemporal',
            'entorhinal', 'temporalpole', 'parahippocampal'
        ],
        'Occipital': [
            'lateraloccipital', 'lingual', 'cuneus', 'pericalcarine'
        ],
        'Cingulate': [
            'rostralanteriorcingulate', 'caudalanteriorcingulate',
            'posteriorcingulate', 'isthmuscingulate'
        ]
    }
    lut_dict = {}
    with open(lut_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            try:
                label_no = int(parts[0])
                label_name = parts[1]
                lut_dict[label_no] = label_name
            except ValueError:
                continue

    # Map labels to names
    labels = parcellation.flatten()
    missing_labels = [label for label in labels if label not in lut_dict]
    label_names = {label: lut_dict.get(label, 'Unknown') for label in labels}
    # print("Label names:", label_names)
    # Replace "Right-" and "-rh-" with "left-" in all values

    def convert_label(value):
            # Handle cortical regions: ctx-lh-region or ctx-rh-region
            m = re.match(r'ctx-(lh|rh)-(.+)', value)
            if m:
                hemi, region = m.groups()
                # Flip hemisphere for BRATS
                hemi_word = "left" if hemi == "lh" else "right"
                result = f"{hemi_word} {region} cortex".replace("-", " ")
            # Handle white matter: wm-lh-region or wm-rh-region
            elif re.match(r'wm-(lh|rh)-(.+)', value):
                m = re.match(r'wm-(lh|rh)-(.+)', value)
                hemi, region = m.groups()
                hemi_word = "left" if hemi == "lh" else "right"
                result = f"{hemi_word} {region} white matter".replace("-", " ")
            # Handle subcortical: Left-X or Right-X
            elif value.startswith("Right-"):
                result = ("Right " + value[6:]).replace("-", " ")
            elif value.startswith("Left-"):
                result = ("Left " + value[5:]).replace("-", " ")
            else:
                result = value.replace("-", " ") 
            # Add space before temporal, frontal, parietal, occipital, cingulate if connected
            result = re.sub(r'(\w)(temporal|frontal|parietal|occipital|cingulate)', r'\1 \2', result, flags=re.IGNORECASE)
            return result

    if dataset_name == "Brats": # Left ↔ Right labels are interchanged because BRATS uses inverted orientation / transformed coordinates.   
        def convert_label(value):
            # Handle cortical regions: ctx-lh-region or ctx-rh-region
            m = re.match(r'ctx-(lh|rh)-(.+)', value)
            if m:
                hemi, region = m.groups()
                # Flip hemisphere for BRATS
                hemi_word = "left" if hemi == "rh" else "right"
                result = f"{hemi_word} {region} cortex".replace("-", " ")
            # Handle white matter: wm-lh-region or wm-rh-region
            elif re.match(r'wm-(lh|rh)-(.+)', value):
                m = re.match(r'wm-(lh|rh)-(.+)', value)
                hemi, region = m.groups()
                hemi_word = "left" if hemi == "rh" else "right"
                result = f"{hemi_word} {region} white matter".replace("-", " ")
            # Handle subcortical: Left-X or Right-X
            elif value.startswith("Right-"):
                result = ("Left " + value[6:]).replace("-", " ")
            elif value.startswith("Left-"):
                result = ("Right " + value[5:]).replace("-", " ")
            else:
                result = value.replace("-", " ")
            
            # Add space before temporal, frontal, parietal, occipital, cingulate if connected
            result = re.sub(r'(\w)(temporal|frontal|parietal|occipital|cingulate)', r'\1 \2', result, flags=re.IGNORECASE)
            return result

    label_names = {label: convert_label(value) for label, value in label_names.items()}

    parcel = parcellation
    mask = nib.load(segmentation_path).get_fdata()
    mask = (mask > 0.5) * 1
    parcel_masked = np.multiply(parcel, mask)

    unique_values = np.unique(parcel_masked)
    unique_values = unique_values[unique_values != 0]
    parcels = []

    for val in unique_values:
        voxel_count = np.sum(parcel_masked == val)
        label = label_names.get(val, 'Unknown')
        parcels.append((val, label, voxel_count))

    # Sort by voxel_count descending
    parcels_sorted = sorted(parcels, key=lambda x: x[2], reverse=True)

    # Keep only first occurrence of each unique label, up to 7
    seen_labels = set()
    unique_parcels = []
    for val, label, voxel_count in parcels_sorted:
        if label not in seen_labels:
            unique_parcels.append((val, label, voxel_count))
            seen_labels.add(label)
        if len(unique_parcels) == 7:
            break

    # Filter out Unknown and small regions
    unique_parcels_filtered = [parcel for parcel in unique_parcels if parcel[1] != 'Unknown' and parcel[2] >= 100]

    # Group by hemisphere, keeping volume order within each group
    left_parcels = [p for p in unique_parcels_filtered if p[1].lower().startswith('left')]
    right_parcels = [p for p in unique_parcels_filtered if p[1].lower().startswith('right')]
    other_parcels = [p for p in unique_parcels_filtered if not p[1].lower().startswith('left') and not p[1].lower().startswith('right')]
    
    # Majority hemisphere first; minority hemisphere only if voxel_count > 500
    if len(left_parcels) >= len(right_parcels):
        right_parcels = [p for p in right_parcels if p[2] > 500]
        unique_parcels_filtered = left_parcels + right_parcels + other_parcels
    else:
        left_parcels = [p for p in left_parcels if p[2] > 500]
        unique_parcels_filtered = right_parcels + left_parcels + other_parcels

    volume_ml = int(volume / 1000)

    regions_formatted = "\n".join(f" - {parcel[1].replace('-', ' ').capitalize()}"
                              for parcel in unique_parcels_filtered)

    regions_formatted_list = [parcel[1] for parcel in unique_parcels_filtered]

    lines = []

    # First line
    lines.append("\\\"Background anatomical context:\\n\"")

    # Volume line
    lines.append(f"\"- Known tumor volume: approximately {volume_ml} ml\\n\"")
    # Header line
    lines.append("\"- Previously reported regional involvement includes:\\n\"")

    # Dynamic regions
    for region in regions_formatted_list:
        lines.append(f"\" - {region}\\n\"")

    # Extra literal and real newline between regions and final line
    lines.append("\"\\n\"")   # literal "\n"
    # OR: lines.append("\"\\n\\n\"") if you want literal \n\n

    # Final note line
    lines.append("\"This information is provided for contextual reference only.\"")

    # Join with REAL newline characters between lines
    parcellation_context = "\n".join(lines)


    print(parcellation_context)

  



def register_images(mni_t1_path, mni_parcellation_path, input_t1_path):
    fixed = ants.image_read(input_t1_path)
    moving = ants.image_read(mni_t1_path)

    # Nonlinear registration
    reg = ants.registration(
        fixed=fixed,
        moving=moving,
        type_of_transform="SyN",
        niterations=[100, 100, 100, 100]  # Example values for iterations
    )

    moving_mask = ants.image_read(mni_parcellation_path)
    warped_mask = ants.apply_transforms(
        fixed=fixed,
        moving=moving_mask,
        transformlist=reg['fwdtransforms'],
        interpolator="nearestNeighbor"
    )
    return warped_mask.numpy()  

def main(input_t1_path, sam_output_path, mni_t1_path, mni_parcellation_path, lut_path, dataset_name):
    warped_mask = register_images(mni_t1_path, mni_parcellation_path, input_t1_path)
    sam_output = nib.load(sam_output_path)
    sam_mask = sam_output.get_fdata()
    voxel_count_sam = np.sum(sam_mask > 0.5)
    volume = voxel_count_sam * np.prod(sam_output.header.get_zooms())
    process_parcellation(warped_mask, sam_output_path, lut_path, volume, dataset_name)



if __name__ == "__main__":
    if len(sys.argv) != 7:
        print("Usage: parcellation_by_registration.py <input_t1_path> <sam_output_path> <mni_t1_path> <mni_parcellation_path> <lut_path><dataset_name>")
    else:
        main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4],sys.argv[5], sys.argv[6])
