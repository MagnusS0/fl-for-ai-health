import argparse
import json
import os
import shutil
import h5py
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def get_volume_info(h5_path):
    """Extract metadata from H5 file"""
    with h5py.File(h5_path, "r") as f:
        labels_present = set()
        num_slices = len(f["labels"])
        for i in range(num_slices):
            slice_labels = set(map(int, set(f["labels"][f"slice_{i}"][:].flatten())))
            if 0 in slice_labels:
                slice_labels.remove(0)
            labels_present.update(slice_labels)
    return {"num_slices": num_slices, "labels_present": labels_present}


def main():
    parser = argparse.ArgumentParser(
        description="Split BRATS dataset into client partitions"
    )
    parser.add_argument("--data-dir", required=True, help="Base directory for dataset")
    parser.add_argument(
        "--num-clients", type=int, required=True, help="Number of client partitions"
    )
    parser.add_argument(
        "--modalities", default="FLAIR_T1w_t1gd_T2w", help="Imaging modalities to use"
    )
    parser.add_argument(
        "--copy", action="store_true", help="Copy files instead of moving"
    )
    parser.add_argument("--slice-dir", default="axial", help="Slice direction")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    # Get all H5 files in the directory
    h5_files = [f for f in os.listdir(args.data_dir) if f.endswith(".h5")]

    # Group volumes and collect metadata
    volume_stats = {}
    for h5_file in tqdm(h5_files, desc="Analyzing volumes"):
        vol_idx = int(h5_file.split("_")[1].split(".")[0])
        h5_path = os.path.join(args.data_dir, h5_file)
        volume_stats[vol_idx] = {"file_path": h5_path, **get_volume_info(h5_path)}

    # Prepare for stratified split
    volume_indices = list(volume_stats.keys())
    strat_labels = [
        "_".join(sorted(map(str, volume_stats[v]["labels_present"])))
        for v in volume_indices
    ]

    # Split volumes into client partitions
    client_volumes = [[] for _ in range(args.num_clients)]
    remaining_vols = volume_indices
    remaining_labels = strat_labels

    # Stratified splitting
    for client_id in range(args.num_clients - 1):
        train_vols, test_vols = train_test_split(
            remaining_vols,
            test_size=1 / (args.num_clients - client_id),
            stratify=remaining_labels,
            random_state=args.seed + client_id,
        )
        client_volumes[client_id] = test_vols
        remaining_vols = train_vols
        remaining_labels = [strat_labels[volume_indices.index(v)] for v in train_vols]
    client_volumes[-1] = remaining_vols

    # Create client directories and move files
    base_dir = os.path.dirname(args.data_dir)
    for client_id, vols in enumerate(client_volumes):
        client_dir = os.path.join(
            base_dir,
            f"preprocessed_{args.modalities}_{args.slice_dir}_client_{client_id}",
        )
        os.makedirs(client_dir, exist_ok=True)

        # Prepare client metadata
        metadata_list = []

        # Move H5 files and update metadata
        for vol_idx in tqdm(vols, desc=f"Processing client {client_id}"):
            vol_info = volume_stats[vol_idx]
            src_path = vol_info["file_path"]
            dst_path = os.path.join(client_dir, os.path.basename(src_path))

            if args.copy:
                shutil.copy2(src_path, dst_path)
            else:
                shutil.move(src_path, dst_path)

            # Read H5 file to get slice information
            with h5py.File(src_path, "r") as h5f:
                images_group = h5f["images"]
                labels_group = h5f["labels"]

                # Process each slice
                for slice_name in images_group.keys():
                    slice_idx = int(
                        slice_name.split("_")[1]
                    )  # Get index from "slice_X"
                    label_data = labels_group[slice_name][:]
                    labels_present = [int(l) for l in np.unique(label_data)]

                    # Create slice-level metadata
                    slice_metadata = {
                        "volume_idx": vol_idx,
                        "slice_idx": slice_idx,
                        "h5_path": os.path.basename(dst_path),
                        "labels_present": labels_present,
                    }
                    metadata_list.append(slice_metadata)

        # Save client metadata
        metadata_path = os.path.join(client_dir, "metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata_list, f, indent=2)

        total_slices = len(metadata_list)
        print(f"Client {client_id}: {total_slices} slices from {len(vols)} volumes")


if __name__ == "__main__":
    main()
