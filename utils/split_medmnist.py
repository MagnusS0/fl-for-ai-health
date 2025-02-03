import argparse
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner


DATASET_DIRECTORY = "data/medmnist"


def save_dataset_to_disk(dataset_name: str, num_partitions: int, split: str):
    """This function downloads the a HF dataset and generates N partitions.

    Each will be saved into the DATASET_DIRECTORY.
    """
    partitioner = IidPartitioner(num_partitions=num_partitions)
    fds = FederatedDataset(
        dataset=dataset_name,
        partitioners={split: partitioner},
    )

    for partition_id in range(num_partitions):
        partition = fds.load_partition(partition_id)
        partition_train_test = partition.train_test_split(test_size=0.2, seed=42)
        file_path = f"./{DATASET_DIRECTORY}/{dataset_name}_part_{partition_id + 1}"
        partition_train_test.save_to_disk(file_path)
        print(f"Written: {file_path}")


if __name__ == "__main__":
    # Initialize argument parser
    parser = argparse.ArgumentParser(
        description="Save a HF dataset partitions to disk"
    )

    parser.add_argument(
        "--dataset-name",
        type=str,
        nargs="?",
        default="MagnusSa/medmnist",
        help="Name of the HF dataset to download",
    )
    parser.add_argument(
        "--num-partitions",
        type=int,
        default=2,
        help="Number of partitions to create",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Split to build from",
    )

    # Parse the arguments
    args = parser.parse_args()

    # Call the function with the provided argument
    save_dataset_to_disk(
        dataset_name=args.dataset_name,
        num_partitions=args.num_partitions,
        split=args.split,
    )