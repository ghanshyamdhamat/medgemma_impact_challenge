import numpy as np
import nibabel as nib
import ants
import sys

def process_parcellation(parcellation, segmentation_path, lut_path, volume):
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

    # Keep only first occurrence of each unique label
    seen_labels = set()
    unique_parcels = []
    for val, label, voxel_count in parcels_sorted:
        if label not in seen_labels:
            unique_parcels.append((val, label, voxel_count))
            seen_labels.add(label)
        if len(unique_parcels) == 7:
            break

    unique_parcels_filtered = [parcel for parcel in unique_parcels if parcel[1] != 'Unknown' and parcel[2] >= 100]
    print(f"{int(volume/1000)} ml tumor extending into {', '.join(parcel[1] for parcel in unique_parcels_filtered)} regions.")
  



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

def main(input_t1_path, sam_output_path, mni_t1_path, mni_parcellation_path, lut_path):
    warped_mask = register_images(mni_t1_path, mni_parcellation_path, input_t1_path)
    sam_output = nib.load(sam_output_path)
    sam_mask = sam_output.get_fdata()
    voxel_count_sam = np.sum(sam_mask > 0.5)
    volume = voxel_count_sam * np.prod(sam_output.header.get_zooms())
    process_parcellation(warped_mask, sam_output_path, lut_path, volume)



if __name__ == "__main__":
    if len(sys.argv) != 6:
        print("Usage: parcellation_by_registration.py <input_t1_path> <sam_output_path> <mni_t1_path> <mni_parcellation_path> <lut_path>")
    else:
        main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4],sys.argv[5])
