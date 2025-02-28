#!/bin/bash

show_help() {
    echo "Usage: $0 [OPTIONS]"
    echo "Options:"
    echo "  -s, --source-dir DIR     Source directory containing MedMNIST splits"
    echo "  -c, --clients LIST       Comma-separated list of client hosts/IPs"
    echo "  -p, --partitions LIST    Comma-separated list of partition numbers (default: 2,3,4,...)"
    echo "  -u, --user USER          Remote username for SSH/rsync"
    echo "  -d, --dest-path PATH     Remote base path for dataset"
    echo ""
    echo "Example:"
    echo "  $0 -s /data/medmnist -c client2.com,client3.com,client4.com -p 2,3,4 -u user -d /app/data"
}


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

# Validate inputs
if [ -z "$SOURCE_DIR" ] || [ -z "$CLIENTS" ] || [ -z "$REMOTE_USER" ] || [ -z "$REMOTE_PATH" ]; then
    echo "Error: Missing required arguments"
    show_help
    exit 1
fi

# If partitions not specified, generate them starting from 2
if [ -z "$PARTITIONS" ]; then
    for i in "${!CLIENTS[@]}"; do
        PARTITIONS[$i]=$((i + 2))
    done
fi

# Validate array lengths match
if [ ${#CLIENTS[@]} -ne ${#PARTITIONS[@]} ]; then
    echo "Error: Number of clients and partitions must match"
    exit 1
fi

# Process each client with its specific partition
for i in "${!CLIENTS[@]}"; do
    CLIENT="${CLIENTS[$i]}"
    PARTITION="${PARTITIONS[$i]}"
    CLIENT_SOURCE="$SOURCE_DIR/medmnist_part_$PARTITION"
    REMOTE_DEST="$REMOTE_PATH"
    
    echo "Processing partition $PARTITION for client $CLIENT..."
    
    # Rest of the script remains the same but uses $PARTITION instead of $((i+1))
    if [ ! -d "$CLIENT_SOURCE" ]; then
        echo "Error: Partition directory not found: $CLIENT_SOURCE"
        continue
    fi
    
    ssh "$REMOTE_USER@$CLIENT" "mkdir -p $REMOTE_DEST"
    rsync -av --progress "$CLIENT_SOURCE/" "$REMOTE_USER@$CLIENT:$REMOTE_DEST/medmnist_part_$PARTITION"
done