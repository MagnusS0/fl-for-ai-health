"""
This script is used to load the BRATS dataset and preprocess it into 2D slices.
"""

import os
import json
import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Resize
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from rich import print
from dotenv import load_dotenv
import os

load_dotenv()
    
class BRATSDataset2D(Dataset):
    """
    BRATS dataset for 2D segmentation
    """
    @staticmethod
    def build_datasets(data_dir, dataset_json_path, modality_to_use='FLAIR', slice_direction='axial', 
                      test_ratio=0.15, seed=42):
        """
        Static method to build both train and test datasets.
        Should be called before creating dataset instances if preprocessed data doesn't exist.
        """
        print("\n" + "="*80)
        print("Starting dataset build process...")
        print(f"Settings: modality={modality_to_use}, direction={slice_direction}, test_ratio={test_ratio}, seed={seed}")
        print("="*80 + "\n")
        
        base_preprocess_dir = os.path.dirname(data_dir)
        train_dir = os.path.join(base_preprocess_dir, f'preprocessed_{modality_to_use}_{slice_direction}_train')
        test_dir = os.path.join(base_preprocess_dir, f'preprocessed_{modality_to_use}_{slice_direction}_test')
        
        # Check if already preprocessed
        if os.path.exists(train_dir) and os.path.exists(test_dir):
            print("Preprocessed datasets already exist at:")
            print(f"Train: {train_dir}")
            print(f"Test: {test_dir}")
            return
        
        print(f"Creating preprocessing directories:")
        print(f"Train: {train_dir}")
        print(f"Test: {test_dir}")
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)
        
        with open(dataset_json_path, 'r') as f:
            dataset_info = json.load(f)
            
        modality_dict = {
            "FLAIR": 0, "T1w": 1, "t1gd": 2, "T2w": 3
        }
        modality_channel = modality_dict.get(modality_to_use)
        
        # Get all training data
        all_data = dataset_info['training']
        print(f"\nFound {len(all_data)} total volumes in dataset")
        
        # Analyze label distribution for each volume
        print("\nAnalyzing label distribution in volumes...")
        volume_labels = []
        for data_item in tqdm(all_data, desc="Analyzing volumes"):
            label_path = os.path.join(data_dir, data_item['label'].replace('./', ''))
            label_nii = nib.load(label_path)
            label_3d = label_nii.get_fdata()
            
            labels, counts = np.unique(label_3d, return_counts=True)
            label_dist = {int(label): count for label, count in zip(labels, counts)}
            volume_labels.append(label_dist)
        
        # Create stratification array
        print("\nCreating stratified split...")
        strat_array = []
        for label_dist in volume_labels:
            signature = "_".join(str(label) for label in sorted(label_dist.keys()) if label > 0)
            strat_array.append(signature)
        
        # Perform stratified split
        indices = np.arange(len(all_data))
        train_indices, test_indices = train_test_split(
            indices, test_size=test_ratio, random_state=seed, stratify=strat_array
        )
        
        print(f"\nSplit complete: {len(train_indices)} train volumes, {len(test_indices)} test volumes")
        
        # Process train set
        BRATSDataset2D._process_split(
            all_data, train_indices, data_dir, train_dir, 
            modality_channel, slice_direction, 'train'
        )
        
        # Process test set
        BRATSDataset2D._process_split(
            all_data, test_indices, data_dir, test_dir, 
            modality_channel, slice_direction, 'test'
        )
        
        print("\n" + "="*80)
        print("Dataset building complete!")
        print("="*80 + "\n")

    @staticmethod
    def _process_split(all_data, indices, data_dir, output_dir, modality_channel, slice_direction, split_name):
        """Helper method to process each split"""
        data_to_process = [all_data[i] for i in indices]
        
        print(f"\nProcessing {split_name} set:")
        print(f"Found {len(data_to_process)} volumes to process")
        print(f"Using {Parallel(n_jobs=-1).n_jobs} cores for parallel processing")

        # Process volumes in parallel
        results = Parallel(n_jobs=-1, verbose=10)(
            delayed(BRATSDataset2D._process_single_volume)(
                data_item,
                data_dir,
                output_dir,
                modality_channel,
                slice_direction,
                volume_idx
            )
            for volume_idx, data_item in enumerate(tqdm(data_to_process, desc="Scheduling volumes"))
        )

        # Aggregate results
        metadata = []
        total_slices = 0
        total_labeled_slices = 0
        for vol_metadata, vol_slices, vol_labeled in results:
            metadata.extend(vol_metadata)
            total_slices += vol_slices
            total_labeled_slices += vol_labeled

        print(f"\n{split_name.capitalize()} set processing complete:")
        print(f"Total slices processed: {total_slices}")
        print(f"Slices with labels: {total_labeled_slices}")
        print(f"Percentage of slices with labels: {(total_labeled_slices/total_slices)*100:.2f}%")
        
        print(f"Saving metadata for {split_name} set...")
        with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f)

    @staticmethod
    def get_cropping_indices(volume_3d, margin=10):
        """
        Calculates cropping indices for a 3D volume based on non-zero values.

        Args:
            volume_3d (np.ndarray): The 3D volume to find cropping indices for.
            margin (int): Optional margin to add around the bounding box.

        Returns:
            tuple: (x_min, x_max, y_min, y_max, z_min, z_max) cropping indices.
        """
        non_zero_mask = volume_3d > 0
        if not np.any(non_zero_mask):
            return None

        coords = np.argwhere(non_zero_mask)
        x_min, y_min, z_min = coords.min(axis=0)
        x_max, y_max, z_max = coords.max(axis=0)

        # Add margin
        x_min = max(0, x_min - margin)
        y_min = max(0, y_min - margin)
        z_min = max(0, z_min - margin)
        x_max = min(volume_3d.shape[0], x_max + margin + 1)
        y_max = min(volume_3d.shape[1], y_max + margin + 1)
        z_max = min(volume_3d.shape[2], z_max + margin + 1)

        return (x_min, x_max, y_min, y_max, z_min, z_max)

    @staticmethod
    def _process_single_volume(data_item, data_dir, output_dir, modality_channel, slice_direction, volume_idx):
        """Process a single volume in parallel"""
        image_path = os.path.join(data_dir, data_item['image'].replace('./', ''))
        label_path = os.path.join(data_dir, data_item['label'].replace('./', ''))

        image_nii = nib.load(image_path)
        image_3d = image_nii.get_fdata()

        label_nii = nib.load(label_path)
        label_3d = label_nii.get_fdata()

        # Extract entire volume for this modality
        volume_data = image_3d[..., modality_channel]

        # Intelligent cropping
        cropping_indices = BRATSDataset2D.get_cropping_indices(
            volume_data, 
            margin=2
        )
        if cropping_indices is not None:
            x_min, x_max, y_min, y_max, z_min, z_max = cropping_indices
            image_3d_cropped = volume_data[x_min:x_max, y_min:y_max, z_min:z_max]
            label_3d_cropped = label_3d[x_min:x_max, y_min:y_max, z_min:z_max]
        else:
            image_3d_cropped = image_3d
            label_3d_cropped = label_3d

        # Calculate volume-level statistics
        eps = 1e-8
        volume_mean = np.mean(image_3d_cropped)
        volume_std = np.std(image_3d_cropped) + eps

        vol_metadata = []
        vol_slices = 0
        vol_labeled = 0

        if slice_direction == 'axial':
            num_slices = image_3d_cropped.shape[2]
            vol_slices = num_slices

            for slice_idx in range(num_slices):
                image_slice = image_3d_cropped[:, :, slice_idx]
                label_slice = label_3d_cropped[:, :, slice_idx]

                if np.any(label_slice > 0):
                    vol_labeled += 1
                    # Normalize using volume statistics
                    image_slice = (image_slice - volume_mean) / volume_std

                    # Save slice
                    image_path = os.path.join(output_dir, f'image_vol{volume_idx}_slice{slice_idx}.npy')
                    label_path = os.path.join(output_dir, f'label_vol{volume_idx}_slice{slice_idx}.npy')

                    np.save(image_path, image_slice)
                    np.save(label_path, label_slice)

                    vol_metadata.append({
                        'volume_idx': volume_idx,
                        'slice_idx': slice_idx,
                        'image_path': image_path,
                        'label_path': label_path,
                        'labels_present': [int(l) for l in np.unique(label_slice) if l > 0]
                    })

        return vol_metadata, vol_slices, vol_labeled


    def __init__(self, data_dir, dataset_json_path, modality_to_use='FLAIR', slice_direction='axial', 
                 transform=None, preprocessed_dir=None, split='train', test_ratio=0.15, seed=42):
        """
        Initilize the dataset.

        The dataset is split into train and test sets. Using the original training set with labels.
        On the first run, the dataset is preprocessed from 3D volumes to 2D slices.
        The preprocessed data is saved to disk and can be loaded from there on subsequent runs.

        Args:
            data_dir (str): Path to the dataset directory.
            dataset_json_path (str): Path to the dataset.json file.
            modality_to_use (str): The modality to use for the dataset.
            slice_direction (str): The direction to slice the dataset.
            transform (torchvision.transforms.Compose): The transform to apply to the dataset.
            num_workers (int): The number of workers to use for the dataset.
            preprocessed_dir (str): The directory to save/load the preprocessed dataset.
            split (str): The split to use for the dataset.
            test_ratio (float): The ratio of the dataset to use for the test set.
            seed (int): The seed to use for the dataset.
        """
        print(f"\nInitializing BRATSDataset2D ({split} split)")
        self.data_dir = data_dir
        self.dataset_json_path = dataset_json_path
        self.modality_to_use = modality_to_use
        self.slice_direction = slice_direction
        self.transform = transform
        self.split = split
        self.test_ratio = test_ratio
        self.seed = seed
        self.dataset_stats = {}
        
        # Update preprocessed directory to include split
        if preprocessed_dir is None:
            self.preprocessed_dir = os.path.join(
                os.path.dirname(data_dir), 
                f'preprocessed_{modality_to_use}_{slice_direction}_{split}'
            )
        else:
            self.preprocessed_dir = preprocessed_dir
            
        # Initialize lists that will store the processed data
        self.slice_paths = [] 
        self.slice_indices = []
        self.volume_indices = []
        
        # Build datasets if they don't exist
        if not os.path.exists(self.preprocessed_dir):
            print(f"Preprocessed data not found at: {self.preprocessed_dir}")
            print("Triggering dataset build process...")
            self.build_datasets(data_dir, dataset_json_path, modality_to_use, 
                              slice_direction, test_ratio, seed)
        else:
            print(f"Found preprocessed data at: {self.preprocessed_dir}")
            
        self._load_preprocessed_metadata()
        print(f"Loaded {len(self.slice_paths)} slices for {split} split")

    def _load_preprocessed_metadata(self):
        """Load metadata for preprocessed slices"""
        print(f"Loading metadata from: {self.preprocessed_dir}")
        with open(os.path.join(self.preprocessed_dir, 'metadata.json'), 'r') as f:
            metadata = json.load(f)
        
        for item in tqdm(metadata, desc="Loading metadata"):
            self.volume_indices.append(item['volume_idx'])
            self.slice_indices.append(item['slice_idx'])
            self.slice_paths.append((item['image_path'], item['label_path']))

    def __len__(self):
        return len(self.slice_paths)

    def __getitem__(self, idx):
        """Load a preprocessed slice"""
        image_path, label_path = self.slice_paths[idx]
        
        # Load preprocessed numpy arrays
        image_slice = np.load(image_path)
        label_slice = np.load(label_path)
        
        # Convert to torch tensors
        image_slice = torch.from_numpy(image_slice).float()
        label_slice = torch.from_numpy(label_slice).long()
        
        # Add channel dimension for image
        image_slice = image_slice.unsqueeze(0)
        
        # Apply transforms if any
        if self.transform:
            image_slice = self.transform(image_slice)
            label_slice = label_slice.unsqueeze(0)
            label_slice = self.transform(label_slice)
            label_slice = label_slice.squeeze(0)
            label_slice = label_slice.long()
        
        return image_slice, label_slice
    
if __name__ == "__main__":
    dataset = BRATSDataset2D(
        data_dir=os.getenv("BRATS_DATA_DIR"),
        dataset_json_path=os.getenv("BRATS_DATASET_JSON_PATH"),
        modality_to_use="FLAIR",
        slice_direction="axial",
        transform=Resize((128, 128)),
        split='train',
    )
    test_dataset = BRATSDataset2D(
        data_dir=os.getenv("BRATS_DATA_DIR"),
        dataset_json_path=os.getenv("BRATS_DATASET_JSON_PATH"),
        modality_to_use="FLAIR",
        slice_direction="axial",
        transform=Resize((128, 128)),
        split='test',
    )
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=16)

    print(len(dataset))
    print(dataset[0])
    
    # Get one batch and create a 4x4 grid
    images, labels = next(iter(dataloader))

    print(images.shape)
    print(labels.shape)
    
    # Create a figure with a 4x4 grid
    fig, axes = plt.subplots(4, 8, figsize=(20, 10))
    
    for idx in range(16):
        row = idx // 4
        col = (idx % 4) * 2  # Multiply by 2 because we're showing image and label side by side
        
        # Get image and label
        image = images[idx].squeeze().numpy()
        label = labels[idx].numpy()
        
        # Get the actual volume and slice index for this item
        batch_idx = idx
        volume_idx = dataset.volume_indices[batch_idx]
        slice_idx = dataset.slice_indices[batch_idx]
        
        # Plot image
        axes[row, col].imshow(image, cmap='gray')
        axes[row, col].axis('off')
        axes[row, col].set_title(f'Vol {volume_idx}, Slice {slice_idx}')
        
        # Plot label
        im = axes[row, col + 1].imshow(label, cmap='tab10', vmin=0, vmax=3)
        axes[row, col + 1].axis('off')
        
        # Print unique values in label to see what classes are present
        unique_labels = np.unique(label)
        if len(unique_labels) > 1:  # If there are any non-zero labels
            print(f"Volume {volume_idx}, Slice {slice_idx} contains labels: {unique_labels}")
    
    # Add a colorbar
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), orientation='horizontal', fraction=0.02, pad=0.04)
    cbar.set_ticks([0, 1, 2, 3])
    cbar.set_ticklabels(['Background', 'Edema', 'Non-enhancing tumor', 'Enhancing tumour'])
    
    plt.tight_layout()
    plt.show()
