"""
This script is used to load the BRATS dataset and preprocess it into 2D slices.
"""

import os
import json
import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Resize, InterpolationMode
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from rich import print
from dotenv import load_dotenv
from typing import List, Tuple, Dict, Optional
from utils.normalizer import PercentileNormalizer
import h5py

load_dotenv()

class BRATSDataset2D(Dataset):
    """
    BRATS dataset for 2D segmentation
    """

    @staticmethod
    def build_datasets(
        data_dir: str,
        dataset_json_path: str,
        modality_to_use: List[str] = ["FLAIR"],
        slice_direction: str = "axial",
        test_ratio: float = 0.15,
        seed: int = 42,
        filter_empty_slices: bool = True,
    ) -> None:
        """
        Static method to build both train and test datasets.
        Should be called before creating dataset instances if preprocessed data doesn't exist.
        """
        print("\n" + "=" * 80)
        print("Starting dataset build process...")
        print(
            f"Settings: modality={modality_to_use}, direction={slice_direction}, test_ratio={test_ratio}, seed={seed}"
        )
        print("=" * 80 + "\n")

        base_preprocess_dir = os.path.dirname(data_dir)
        train_dir = os.path.join(
            base_preprocess_dir,
            f"preprocessed_{'_'.join(modality_to_use)}_{slice_direction}_train",
        )
        test_dir = os.path.join(
            base_preprocess_dir,
            f"preprocessed_{'_'.join(modality_to_use)}_{slice_direction}_test",
        )

        # Check if already preprocessed
        if os.path.exists(train_dir) and os.path.exists(test_dir):
            print("Preprocessed datasets already exist at:")
            print(f"Train: {train_dir}")
            print(f"Test: {test_dir}")
            return

        print("Creating preprocessing directories:")
        print(f"Train: {train_dir}")
        print(f"Test: {test_dir}")
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)

        with open(dataset_json_path, "r") as f:
            dataset_info = json.load(f)

        modality_dict = {"FLAIR": 0, "T1w": 1, "t1gd": 2, "T2w": 3}
        modality_channels = [modality_dict.get(m) for m in modality_to_use]

        # Get all training data
        all_data = dataset_info["training"]
        print(f"\nFound {len(all_data)} total volumes in dataset")

        print("\nAnalyzing label distribution in volumes...")
        volume_labels = []
        for data_item in tqdm(all_data, desc="Analyzing volumes"):
            label_path = os.path.join(data_dir, data_item["label"].replace("./", ""))
            label_nii = nib.load(label_path)
            label_3d = label_nii.get_fdata()

            labels, counts = np.unique(label_3d, return_counts=True)
            label_dist = {int(label): count for label, count in zip(labels, counts)}
            volume_labels.append(label_dist)

        print("\nCreating stratified split...")
        strat_array = []
        for label_dist in volume_labels:
            signature = "_".join(
                str(label) for label in sorted(label_dist.keys()) if label > 0
            )
            strat_array.append(signature)

        # Perform stratified split
        indices = np.arange(len(all_data))
        train_indices, test_indices = train_test_split(
            indices, test_size=test_ratio, random_state=seed, stratify=strat_array
        )

        print(
            f"\nSplit complete: {len(train_indices)} train volumes, {len(test_indices)} test volumes"
        )

        # Process train set
        BRATSDataset2D._process_split(
            all_data,
            train_indices,
            data_dir,
            train_dir,
            modality_channels,
            slice_direction,
            "train",
            filter_empty_slices,
        )

        # Process test set
        BRATSDataset2D._process_split(
            all_data,
            test_indices,
            data_dir,
            test_dir,
            modality_channels,
            slice_direction,
            "test",
            filter_empty_slices,
        )

        print("\n" + "=" * 80)
        print("Dataset building complete!")
        print("=" * 80 + "\n")

    @staticmethod
    def _process_split(
        all_data,
        indices,
        data_dir,
        output_dir,
        modality_channels,
        slice_direction,
        split_name,
        filter_empty_slices=True,
    ):
        """Helper method to process each split"""
        data_to_process = [all_data[i] for i in indices]

        print(f"\nProcessing {split_name} set:")
        print(f"Found {len(data_to_process)} volumes to process")
        print(f"Using {Parallel(n_jobs=-1).n_jobs} cores for parallel processing")

        # Process volumes in parallel
        results = Parallel(n_jobs=-1, verbose=10, backend="loky", prefer="threads")(
            delayed(BRATSDataset2D._process_single_volume)(
                data_item,
                data_dir,
                output_dir,
                modality_channels,
                slice_direction,
                volume_idx,
                filter_empty_slices,
            )
            for volume_idx, data_item in enumerate(
                tqdm(data_to_process, desc="Scheduling volumes")
            )
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
        print(
            f"Percentage of slices with labels: {(total_labeled_slices / total_slices) * 100:.2f}%"
        )

        print(f"Saving metadata for {split_name} set...")
        with open(os.path.join(output_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f)

    @staticmethod
    def get_cropping_indices(
        volume_mask: np.ndarray, 
        margin: int = 10
    ) -> Optional[Tuple[int, int, int, int, int, int]]:
        """
        Calculates cropping indices based on a combined mask of all modalities.

        Args:
            volume_mask (np.ndarray): The 3D mask to find cropping indices for.
            margin (int): Optional margin to add around the bounding box.

        Returns:
            tuple: (x_min, x_max, y_min, y_max, z_min, z_max) cropping indices.
        """
        if volume_mask.ndim == 4:
            non_zero_mask = np.any(volume_mask > 0, axis=-1)
        else:
            non_zero_mask = volume_mask > 0

        coords = np.argwhere(non_zero_mask)
        x_min, y_min, z_min = coords.min(axis=0)
        x_max, y_max, z_max = coords.max(axis=0)

        # Add margin
        x_min = max(0, x_min - margin)
        y_min = max(0, y_min - margin)
        z_min = max(0, z_min - margin)
        x_max = min(volume_mask.shape[0], x_max + margin + 1)
        y_max = min(volume_mask.shape[1], y_max + margin + 1)
        z_max = min(volume_mask.shape[2], z_max + margin + 1)

        return (x_min, x_max, y_min, y_max, z_min, z_max)

    @staticmethod
    def _process_single_volume(
        data_item, data_dir, output_dir, modality_channels, slice_direction, volume_idx, filter_empty_slices=True
    ):
        """Process a single volume with multiple modalities using BrainLesion-Preprocessing"""
        image_path = os.path.join(data_dir, data_item["image"].replace("./", ""))
        label_path = os.path.join(data_dir, data_item["label"].replace("./", ""))

        # Load the 3D volume
        image_nii = nib.load(image_path)
        image_3d = image_nii.get_fdata()

        label_nii = nib.load(label_path)
        label_3d = label_nii.get_fdata()

        # Extract modality data
        volume_data = []
        for modality_channel in modality_channels:
            modality_data = image_3d[..., modality_channel]
            # Percentile normalization
            normalizer = PercentileNormalizer(lower_percentile=0.1, upper_percentile=99.9, lower_limit=0, upper_limit=1)
            modality_data = normalizer.normalize(modality_data)
            volume_data.append(modality_data)

        # Combine modalities
        combined_volume = np.stack(volume_data, axis=-1)

        # Intelligent cropping
        cropping_indices = BRATSDataset2D.get_cropping_indices(combined_volume, margin=2)
        if cropping_indices is not None:
            x_min, x_max, y_min, y_max, z_min, z_max = cropping_indices
            combined_volume = combined_volume[x_min:x_max, y_min:y_max, z_min:z_max]
            label_3d = label_3d[x_min:x_max, y_min:y_max, z_min:z_max]
        else:
            combined_volume = combined_volume
            label_3d = label_3d


        vol_metadata = []
        vol_labeled = 0

        slice_axis = {"axial": 2, "sagittal": 0, "coronal": 1}[slice_direction]
        
        h5_filename = os.path.join(output_dir, f"volume_{volume_idx}.h5")

        with h5py.File(h5_filename, 'w') as h5f:
            images_group = h5f.create_group('images')
            labels_group = h5f.create_group('labels')

            for slice_idx in range(combined_volume.shape[slice_axis]):
                if slice_axis == 0:
                    image_slice = combined_volume[slice_idx, :, :, :]
                    label_slice = label_3d[slice_idx, :, :]
                elif slice_axis == 1:
                    image_slice = combined_volume[:, slice_idx, :, :]
                    label_slice = label_3d[:, slice_idx, :]
                else:
                    image_slice = combined_volume[:, :, slice_idx, :]
                    label_slice = label_3d[:, :, slice_idx]

                has_labels = np.sum(label_slice) > 0

                def __save_slice():
                    """Helper to save slice and update metadata"""
                    images_group.create_dataset(f'slice_{slice_idx}', data=image_slice, compression='gzip')
                    labels_group.create_dataset(f'slice_{slice_idx}', data=label_slice, compression='gzip')

                    vol_metadata.append({
                        "volume_idx": volume_idx,
                        "slice_idx": slice_idx,
                        "h5_path": h5_filename,
                        "labels_present": [int(l) for l in np.unique(label_slice)],
                    })

                if has_labels:
                    vol_labeled += 1

                if filter_empty_slices:
                    if has_labels:
                        __save_slice()
                else:
                    # Save all slices
                    __save_slice()

        return vol_metadata, len(vol_metadata), vol_labeled

    def __init__(
        self,
        data_dir: str,
        dataset_json_path: str,
        modality_to_use: List[str] = ["FLAIR", "T1w", "t1gd", "T2w"],
        slice_direction: str = "axial",
        transform_image: Optional[torch.nn.Module] = None,
        transform_label: Optional[torch.nn.Module] = None,
        preprocessed_dir: Optional[str] = None,
        split: str = "train",
        test_ratio: float = 0.15,
        seed: int = 42,
        filter_empty_slices: bool = True,
    ):
        """
        Initilize the dataset.

        The dataset is split into train and test sets. Using the original training set with labels.
        On the first run, the dataset is preprocessed from 3D volumes to 2D slices.
        The preprocessed data is saved to disk and can be loaded from there on subsequent runs.

        Args:
            data_dir (str): Path to the dataset directory.
            dataset_json_path (str): Path to the dataset.json file.
            modality_to_use (list[str]): The modalities to use for the dataset.
            slice_direction (str): The direction to slice the dataset.
            transform_image (torchvision.transforms.Compose): The transform to apply to the image.
            transform_label (torchvision.transforms.Compose): The transform to apply to the label.
            num_workers (int): The number of workers to use for the dataset.
            preprocessed_dir (str): The directory to save/load the preprocessed dataset.
            split (str): The split to use for the dataset.
            test_ratio (float): The ratio of the dataset to use for the test set.
            seed (int): The seed to use for the dataset.
            filter_empty_slices (bool): Whether to filter out empty slices.
        """
        print(f"\nInitializing BRATSDataset2D ({split} split)")
        self.data_dir = data_dir
        self.dataset_json_path = dataset_json_path
        self.modality_to_use = modality_to_use
        self.slice_direction = slice_direction
        self.transform_image = transform_image
        self.transform_label = transform_label
        self.split = split
        self.test_ratio = test_ratio
        self.seed = seed
        self.dataset_stats = {}

        # Update preprocessed directory to include split
        if preprocessed_dir is None:
            self.preprocessed_dir = os.path.join(
                os.path.dirname(data_dir),
                f"preprocessed_{'_'.join(modality_to_use)}_{slice_direction}_{split}",
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
            self.build_datasets(
                data_dir,
                dataset_json_path,
                modality_to_use,
                slice_direction,
                test_ratio,
                seed,
                filter_empty_slices,
            )
        else:
            print(f"Found preprocessed data at: {self.preprocessed_dir}")

        self._load_preprocessed_metadata()
        print(f"Loaded {len(self.slice_paths)} slices for {split} split")

    def _get_class_weights(self) -> torch.Tensor:
        """
        Calculate class weigths based on inverse frequency of pixels in each class.
        """
        weights_path = os.path.join(self.preprocessed_dir, "class_weights.pt")
        if os.path.exists(weights_path):
            return torch.load(weights_path)
        
        class_counts = torch.zeros(4, dtype=torch.float32)
        
        for idx in tqdm(range(len(self)), desc="Calculating class weights"):
            _, label = self[idx] 
            label = label.squeeze().long()
            
            for cls in range(4):
                class_counts[cls] += (label == cls).sum().item()

        class_counts = class_counts[1:]

        # Calculate inverse frequency
        frequencies = class_counts / class_counts.sum()
        weights = 1.0 / (frequencies + 1e-8)

        # Cache
        torch.save(weights, weights_path)
        
        return weights

    def _load_preprocessed_metadata(self):
        """Load metadata for preprocessed slices"""
        print(f"Loading metadata from: {self.preprocessed_dir}")
        with open(os.path.join(self.preprocessed_dir, "metadata.json"), "r") as f:
            metadata = json.load(f)

        for item in tqdm(metadata, desc="Loading metadata"):
            self.volume_indices.append(item["volume_idx"])
            self.slice_indices.append(item["slice_idx"])
            self.slice_paths.append((item["h5_path"], item["slice_idx"]))

    def __len__(self):
        return len(self.slice_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a single dataset sample.
        
        Args:
            idx: Index of sample to retrieve
              
        Returns:
            Tuple containing:
                - image: Tensor of shape (C, H, W)
                - label: Tensor of shape (H, W) with class labels

        Raises:
            FileNotFoundError: If preprocessed files are missing
        """
        h5_path, slice_idx = self.slice_paths[idx]

        try:
            with h5py.File(h5_path, 'r') as h5f:
                images_group = h5f['images']
                labels_group = h5f['labels']

                image_slice = images_group[f'slice_{slice_idx}'][:]
                label_slice = labels_group[f'slice_{slice_idx}'][:]
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"Preprocessed data missing for index {idx}: {str(e)}"
            ) from e

        # Convert to tensors and handle device placement
        image_slice = torch.from_numpy(image_slice).float().permute(2, 0, 1)
        label_slice = torch.from_numpy(label_slice).long()

        # Apply transforms
        if self.transform_image:
            image_slice = self.transform_image(image_slice)
        if self.transform_label:
            label_slice = self.transform_label(label_slice.unsqueeze(0)).squeeze(0)

        return image_slice, label_slice


if __name__ == "__main__":
    dataset = BRATSDataset2D(
        data_dir=os.getenv("BRATS_DATA_DIR"),
        dataset_json_path=os.getenv("BRATS_DATASET_JSON_PATH"),
        modality_to_use=["FLAIR", "T1w", "t1gd", "T2w"],
        slice_direction="axial",
        transform_image=Resize((128, 128)),
        transform_label=Resize((128, 128), interpolation=InterpolationMode.NEAREST),
        split="train",
        filter_empty_slices=False,
    )
    test_dataset = BRATSDataset2D(
        data_dir=os.getenv("BRATS_DATA_DIR"),
        dataset_json_path=os.getenv("BRATS_DATASET_JSON_PATH"),
        modality_to_use=["FLAIR", "T1w", "t1gd", "T2w"],
        slice_direction="axial",
        transform_image=Resize((128, 128)),
        transform_label=Resize((128, 128), interpolation=InterpolationMode.NEAREST),
        split="test",
        filter_empty_slices=False,
    )
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=16)

    #weight = dataset._get_class_weights()
    #print(weight)

    #log_weight = torch.log(weight+1)
    #normalized_weight = log_weight / log_weight.sum()
    #print(normalized_weight)
    print(len(dataset))
    print(dataset[0])

    # Get one batch and create a 4x4 grid
    images, labels = next(iter(dataloader))

    print(images.shape)
    print(labels.shape)

    # Create a figure with 4 samples showing all modalities
    fig, axes = plt.subplots(4, 5, figsize=(25, 20))  # 4 samples, 5 columns (4 modalities + label)

    for idx in range(4):
        # Get image and label
        image = images[idx].permute(1, 2, 0).numpy()  # Shape (H, W, C)
        label = labels[idx].numpy()

        # Get the actual volume and slice index for this item
        volume_idx = dataset.volume_indices[idx]
        slice_idx = dataset.slice_indices[idx]

        # Plot each modality separately
        for mod_idx, mod_name in enumerate(["FLAIR", "T1w", "t1gd", "T2w"]):
            axes[idx, mod_idx].imshow(image[:, :, mod_idx], cmap="gray", vmin=0, vmax=1)
            axes[idx, mod_idx].axis("off")
            axes[idx, mod_idx].set_title(f"{mod_name}\nVol {volume_idx}, Slice {slice_idx}")

        # Plot label
        im = axes[idx, 4].imshow(label, cmap="tab10", vmin=0, vmax=3)
        axes[idx, 4].axis("off")
        axes[idx, 4].set_title("Ground Truth Segmentation")

    # Add a colorbar
    cbar = fig.colorbar(
        im, ax=axes.ravel().tolist(), 
        orientation="horizontal", 
        fraction=0.02, 
        pad=0.04
    )
    cbar.set_ticks([0, 1, 2, 3])
    cbar.set_ticklabels(["Background", "Edema", "Non-enhancing tumor", "Enhancing tumour"])

    plt.tight_layout()
    plt.show()
