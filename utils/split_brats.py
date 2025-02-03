import argparse
import json
import os
import shutil
from sklearn.model_selection import train_test_split
from collections import defaultdict
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(description='Split BRATS dataset into client partitions')
    parser.add_argument('--data-dir', required=True, help='Base directory for dataset')
    parser.add_argument('--dataset-json-path', required=True, help='Path to dataset.json')
    parser.add_argument('--num-clients', type=int, required=True, help='Number of client partitions')
    parser.add_argument('--modalities', default='FLAIR_T1w_t1gd_T2w', help='Imaging modalities to use')
    parser.add_argument('--copy', action='store_true', help='Copy files instead of moving')
    parser.add_argument('--slice-dir', default='axial', help='Slice direction')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()

    # Load original training metadata
    train_dir = os.path.join(os.path.dirname(args.data_dir), 
                             f'preprocessed_{args.modalities}_{args.slice_dir}_train')
    metadata_path = os.path.join(train_dir, 'metadata.json')
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    # Group slices by volume
    volume_stats = defaultdict(lambda: {'slices': [], 'labels': set()})
    for entry in metadata:
        vol_idx = entry['volume_idx']
        volume_stats[vol_idx]['slices'].append(entry)
        volume_stats[vol_idx]['labels'].update(entry['labels_present'])

    # Prepare for stratified split
    volume_indices = list(volume_stats.keys())
    strat_labels = ['_'.join(sorted(map(str, volume_stats[v]['labels']))) 
                    for v in volume_indices]

    # Split volumes into client partitions
    client_volumes = [[] for _ in range(args.num_clients)]
    remaining_vols = volume_indices
    remaining_labels = strat_labels

    # Stratified splitting
    for client_id in range(args.num_clients - 1):
        train_vols, test_vols = train_test_split(
            remaining_vols,
            test_size=1/(args.num_clients - client_id),
            stratify=remaining_labels,
            random_state=args.seed + client_id
        )
        client_volumes[client_id] = test_vols
        remaining_vols = train_vols
        remaining_labels = [strat_labels[volume_indices.index(v)] for v in train_vols]
    client_volumes[-1] = remaining_vols

    # Create client directories
    base_dir = os.path.dirname(args.data_dir)
    for client_id, vols in enumerate(client_volumes):
        client_dir = os.path.join(base_dir, 
                                  f'preprocessed_{args.modalities}_{args.slice_dir}_client_{client_id}')
        os.makedirs(client_dir, exist_ok=True)
        
        client_metadata = []
        for vol in tqdm(vols, desc=f'Processing client {client_id} volumes'):
            client_metadata.extend(volume_stats[vol]['slices'])
        
        # Move image and label files
        for slice_entry in tqdm(client_metadata, desc=f'Moving files for client {client_id}', leave=False):
            # Process image files
            original_image_path = slice_entry["image_path"]
            new_image_path = os.path.join(client_dir, os.path.basename(original_image_path))
            if args.copy:
                shutil.copy2(original_image_path, new_image_path)
            else:
                shutil.move(original_image_path, new_image_path)
            slice_entry["image_path"] = new_image_path

            # Process label file
            original_label_path = slice_entry["label_path"]
            new_label_path = os.path.join(client_dir, os.path.basename(original_label_path))
            if args.copy:
                shutil.copy2(original_label_path, new_label_path)
            else:
                shutil.move(original_label_path, new_label_path)
            slice_entry["label_path"] = new_label_path
        
        # Updated client metadata
        client_metadata_path = os.path.join(client_dir, 'metadata.json')
        with open(client_metadata_path, 'w') as f:
            json.dump(client_metadata, f)
        
        print(f'Client {client_id}: {len(client_metadata)} slices from {len(vols)} volumes')

if __name__ == '__main__':
    main()
