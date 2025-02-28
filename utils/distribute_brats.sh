#!/bin/bash

show_help() {
    echo "Usage: $0 [OPTIONS]"
    echo "Options:"
    echo "  -s, --source-dir DIR     Source directory containing BRATS splits"
    echo "  -c, --clients LIST       Comma-separated list of client hosts/IPs"
    echo "  -p, --partitions LIST    Comma-separated list of partition numbers"
    echo "  -u, --user USER          Remote username for SSH/rsync"
    echo "  -d, --dest-path PATH     Remote base path for dataset"
    echo ""
    echo "Example:"
    echo "  $0 -s /data/brats -c client2.com,client3.com -p 2,3 -u user -d /app/data"
}

# Add partition parameter parsing
while [[ $# -gt 0 ]]; do
    case $1 in
        -s|--source-dir)
            SOURCE_DIR="$2"
            shift 2
            ;;
        -c|--clients)
            IFS=',' read -ra CLIENTS <<< "$2"
            shift 2
            ;;
        -p|--partitions)
            IFS=',' read -ra PARTITIONS <<< "$2"
            shift 2
            ;;
        -u|--user)
            REMOTE_USER="$2"
            shift 2
            ;;
        -d|--dest-path)
            REMOTE_PATH="$2"
            shift 2
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Validate arrays match
if [ ${#CLIENTS[@]} -ne ${#PARTITIONS[@]} ]; then
    echo "Error: Number of clients and partitions must match"
    exit 1
fi

# Process each client with specific partition
for i in "${!CLIENTS[@]}"; do
    CLIENT="${CLIENTS[$i]}"
    PARTITION="${PARTITIONS[$i]}"
    CLIENT_SOURCE="$SOURCE_DIR/preprocessed_FLAIR_T1w_t1gd_T2w_axial_client_$PARTITION"
    REMOTE_DEST="$REMOTE_PATH/preprocessed_FLAIR_T1w_t1gd_T2w_axial_train"
    
    echo "Processing partition $PARTITION for client $CLIENT..."
    
    if [ ! -d "$CLIENT_SOURCE" ]; then
        echo "Error: Client source directory not found: $CLIENT_SOURCE"
        continue
    fi
    
    # Test SSH connection
    if ! ssh -q "$REMOTE_USER@$CLIENT" exit; then
        echo "Error: Cannot connect to $CLIENT"
        continue
    fi
    
    echo "Copying data to $CLIENT..."
    
    ssh "$REMOTE_USER@$CLIENT" "mkdir -p $REMOTE_PATH"
    
    # Remove existing train directory if present
    ssh "$REMOTE_USER@$CLIENT" "rm -rf $REMOTE_DEST"
    
    # Copy and rename directory
    rsync -av --progress "$CLIENT_SOURCE/" "$REMOTE_USER@$CLIENT:$REMOTE_DEST"
    
    if [ $? -eq 0 ]; then
        echo "Successfully distributed data to $CLIENT"
    else
        echo "Error: Failed to copy data to $CLIENT"
    fi
done

echo "Distribution complete!"