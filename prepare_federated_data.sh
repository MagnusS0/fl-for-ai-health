#!/bin/bash
set -e  

# Check .env
if [ ! -f .env ]; then
    echo "Error: .env file not found!"
    echo "Please create .env file with required paths:"
    echo "BRATS_DATA_DIR=/path/to/Task01_BrainTumour"
    echo "BRATS_JSON_PATH=/path/to/Task01_BrainTumour/dataset.json"
    echo "MEDMNIST_DIRECTORY=/path/to/medmnist/storage"
    echo "REMOTE_PATH_BRATS=/path/to/remote/brats"
    echo "REMOTE_PATH_MEDMNIST=/path/to/remote/medmnist"
    exit 1
fi

source .env

# Required environment variables
REQUIRED_VARS=(
    "BRATS_DATA_DIR"
    "BRATS_JSON_PATH"
    "MEDMNIST_DIRECTORY"
    "REMOTE_USER"
    "REMOTE_PATH_BRATS"
    "REMOTE_PATH_MEDMNIST"
    "CLIENT_LIST"
)

# Check required variables
for var in "${REQUIRED_VARS[@]}"; do
    if [ -z "${!var}" ]; then
        echo "Error: $var is not set in .env file"
        exit 1
    fi
done

# Convert comma-separated CLIENT_LIST to array
IFS=',' read -ra CLIENTS <<< "$CLIENT_LIST"
NUM_CLIENTS=${#CLIENTS[@]}

mkdir -p "$MEDMNIST_DIRECTORY"

echo "=== Starting Dataset Preparation For N-Clients: $NUM_CLIENTS ==="

# 1. Build BRATS dataset
echo "Building BRATS dataset..."
if python data/brats.py; then
    echo "✓ BRATS preprocessing complete"
else
    echo "✗ BRATS preprocessing failed"
    exit 1
fi

# 2. Prepare MedMNIST dataset
echo "Preparing MedMNIST dataset..."
if python utils/split_medmnist.py --dataset-name MagnusSa/medmnist --num-partitions $NUM_CLIENTS --split train; then
    echo "✓ MedMNIST preparation complete"
else
    echo "✗ MedMNIST preparation failed"
    exit 1
fi

# 3. Split BRATS dataset
echo "Splitting BRATS dataset..."
BRATS_PREPROCESSED_DIR=$(dirname "$BRATS_DATA_DIR")/preprocessed_FLAIR_T1w_t1gd_T2w_axial_train
if python utils/split_brats.py \
    --data-dir "$BRATS_PREPROCESSED_DIR" \
    --num-clients $NUM_CLIENTS \
    --modalities FLAIR_T1w_t1gd_T2w \
    --slice-dir axial; then
    echo "✓ BRATS splitting complete"
else
    echo "✗ BRATS splitting failed"
    exit 1
fi

# 4. Make distribution scripts executable
chmod +x utils/distribute_brats.sh
chmod +x utils/distribute_medmnist.sh

# 5. Prepare partition strings
# For BRATS, partitions start at 0 (e.g. 0,1,2,...)
BRATS_PARTITIONS=()
for ((i=0; i<$NUM_CLIENTS; i++)); do
   BRATS_PARTITIONS+=("$i")
done
BRATS_PARTITION_STRING=$(IFS=,; echo "${BRATS_PARTITIONS[*]}")

# For MedMNIST, partitions start at 1 (e.g. 1,2,3,...)
MEDMNIST_PARTITIONS=()
for ((i=1; i<=$NUM_CLIENTS; i++)); do
   MEDMNIST_PARTITIONS+=("$i")
done
MEDMNIST_PARTITION_STRING=$(IFS=,; echo "${MEDMNIST_PARTITIONS[*]}")

# Create comma-separated string for clients
CLIENT_STRING=$(IFS=,; echo "${CLIENTS[*]}")

echo "=== Starting Dataset Distribution ==="

echo "Distributing BRATS dataset..."
if ./utils/distribute_brats.sh \
    -s "$(dirname "$BRATS_DATA_DIR")" \
    -c "$CLIENT_STRING" \
    -p "$BRATS_PARTITION_STRING" \
    -u "$REMOTE_USER" \
    -d "$REMOTE_PATH_BRATS"; then
    echo "✓ BRATS distribution complete"
else
    echo "✗ BRATS distribution failed"
    exit 1
fi

echo "Distributing MedMNIST dataset..."
if ./utils/distribute_medmnist.sh \
    -s "$MEDMNIST_DIRECTORY" \
    -c "$CLIENT_STRING" \
    -p "$MEDMNIST_PARTITION_STRING" \
    -u "$REMOTE_USER" \
    -d "$REMOTE_PATH_MEDMNIST"; then
    echo "✓ MedMNIST distribution complete"
else
    echo "✗ MedMNIST distribution failed"
    exit 1
fi

echo "=== Dataset Preparation and Distribution Complete ==="