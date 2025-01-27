"""
This script is used to load the MedMNIST dataset and push it to the Hugging Face Hub.
"""

from medmnist import PathMNIST
from datasets import Dataset, DatasetDict
from huggingface_hub import login
from dotenv import load_dotenv
import os

load_dotenv()


def load_medmnist(split: str):
    dataset = PathMNIST(
        split=split, 
        download=True, 
        size=64, 
        root="data",
        transform=None
    )
    return dataset


if __name__ == "__main__":
    # Load all splits
    splits = ['train', 'test', 'val']
    combined_dict = {'image': [], 'label': [], 'split': []}
    
    for split in splits:
        pytorch_dataset = load_medmnist(split)
        for image, label in pytorch_dataset:
            combined_dict['image'].append(image)
            combined_dict['label'].append(label.item())
            combined_dict['split'].append(split)

    # Create Hugging Face dataset
    hf_dataset = Dataset.from_dict(combined_dict)

    # Create DatasetDict
    dataset_dict = DatasetDict({
        "train": hf_dataset.filter(lambda x: x["split"] == "train"),
        "test": hf_dataset.filter(lambda x: x["split"] == "test"),
        "val": hf_dataset.filter(lambda x: x["split"] == "val")
    })
    
    # Save and push to hub
    dataset_dict.save_to_disk("data/medmnist/combined")
    
    login(token=os.getenv("HF_TOKEN"), write_permission=True)
    
    dataset_dict.push_to_hub(
        "MagnusSa/medmnist", 
        commit_message="PathMNIST - Combined train/test/val splits",
        token=os.getenv("HF_TOKEN")
    )



