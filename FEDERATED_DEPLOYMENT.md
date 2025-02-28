# Federated Deployment Instructions

This document guides you through deploying the federated learning application on a distributed system.
It assumes you are familier with Docker in the Flower Framework you can follow the Quickstart guide [here](https://flower.ai/docs/framework/v1.13.0/en/docker/tutorial-quickstart-docker.html).

**Prerequisites:**

*   **Flower:** Ensure you are using Flower version **1.13.0 or 1.13.1**.
*   **Docker:** Docker installed and configured on all participating machines (server and clients).
* **Network Connectivity:** All client machines must be able to communicate with the server machine.
*   **`.env` File:** Create a `.env` file in the project root. This file is *crucial* for configuring data paths and remote access.  It should contain the following:

    ```toml
    BRATS_DATA_DIR=/path/to/Task01_BrainTumour  # Local path to BRATS data (if using BRATS)
    BRATS_JSON_PATH=/path/to/Task01_BrainTumour/dataset.json  # Local path to BRATS dataset.json
    MEDMNIST_DIRECTORY=/path/to/medmnist/storage  # Local path to store MedMNIST (if using MedMNIST)
    REMOTE_USER=your_remote_username   # SSH username for client machines
    REMOTE_PATH_BRATS=/remote/path/to/brats  # Remote path on clients for BRATS data
    REMOTE_PATH_MEDMNIST=/remote/path/to/medmnist  # Remote path on clients for MedMNIST data
    CLIENT_LIST=client1.exampleclient2.example,client3.example  # Comma-separated list of client DNS names or IPs
    ```
    **Important:** `CLIENT_LIST` must contain one entry for each client, and determines the number of clients (`NUM_CLIENTS`) used in later steps. If running a ClientApp on the same machine as the ServerApp this machine also needs to be inlcuded in the list.

## Step 1: Dataset Preparation and Distribution

The `prepare_federated_data.sh` script automates dataset preprocessing, splitting, and distribution to client machines.

1.  **Ensure `.env` is Correct:**  Double-check that all paths in your `.env` file are accurate, both for local paths and remote paths on the client machines.
2. You have the download the BRATS dataset from [Medical Decathlon](http://medicaldecathlon.com/)

3.  **Run `prepare_federated_data.sh`:** Execute the script from the project root:

    ```bash
    chmod +x prepare_federated_data.sh
    ./prepare_federated_data.sh
    ```

    This script performs the following actions:

    *   **BRATS:**
        *   Preprocesses the 3D BRATS data into 2D slices (if not already preprocessed).
        *   Splits the BRATS dataset into partitions for each client, using stratified splitting.
        *   Copies the appropriate partition to each client machine via `rsync` and `ssh`.
    *   **MedMNIST:**
        *   Downloads the MedMNIST dataset (if not already downloaded).
        *   Splits the MedMNIST dataset into partitions for each client.
        *   Copies the appropriate partition to each client machine via `rsync` and `ssh`.

    **Troubleshooting:** If you encounter errors, ensure:
    *   The `.env` file is present and correctly configured.
    *   SSH keys are set up for passwordless access to the client machines. (Or you will have to retype it multiple times)
    *   The remote paths specified in `.env` exist or can be created.

## Step 2: Docker Image

1.  **Configure `pyproject.toml`:**
    *   **Choose Task:**  Uncomment the lines for either classification or segmentation, and set the `model` and other parameters as desired (see examples in `README.md`).  *Crucially*, set paths that will be valid *inside* the Docker container:
        ```toml
            [tool.flwr.app.config]
            # ... other config ...
            # For MedMNIST:
            from-disk = true
            disk-path = "/app/data"  # Path inside the container
            # For BRATS:
            data-dir = "/app/data"  # Path inside the container
            dataset-json-path = "/app/data/dataset.json" # Path inside the container
        ``` 

    *   **Local Deployment Address:** Add the `local-deployment` section to set the server address:
        ```toml
        [tool.flwr.federations.local-deployment]
        address = "0.0.0.0:9093"  # Server address and port
        insecure = true
        ```

2.  **Build Docker Image:** 
<details>
<summary><b>Example Setup Scripts (Click to Expand)</b></summary>

**Note:**  These assume the existence of helper scripts (`flower.config`, `flower-helper.sh`) which are not provided here but contain environment-specific configurations (like network settings, VLANs, etc.).

**`flower-serverapp` script:**
```bash
#!/bin/bash -eu

. ./flower.config
HOSTS="${HOSTS_SUPERLINK}"
. ./flower-helper.sh

superLinkIP="$(echo "${superLinkIP}" | sed -e "s/\.[0-9]*$/.100/")"
echo "SuperLinkIP=${superLinkIP}"


# ====== Build ==============================================================
cp Dockerfile.serverapp  ~/fl-build # Copy to a build directory
cd ~/fl-build
docker build -f Dockerfile.serverapp -t flwr_serverapp:0.0.1 .

# ====== Run ================================================================
docker run --rm \
   -v /path/to/your/MedMNIST:/app/data/MedMNIST \  # Adjust paths as needed!
   -v /path/to/your/MedicalDecathlon:/app/data/MedicalDecathlon \ #Adjust paths
   --name serverapp \
   --detach \
   flwr_serverapp:0.0.1 \
   --insecure \
   --address="0.0.0.0:9093"
```
**`Dockerfile.clientapp`:**

```docker
FROM flwr/serverapp:1.13.0

WORKDIR /app

COPY pyproject.toml .
RUN sed -i 's/.*flwr\[simulation\].*//' pyproject.toml \
   && python -m pip install -U --no-cache-dir .

ENTRYPOINT ["flwr-serverapp"]
```
</details>



## Step 3: Starting the Federated Learning System

1.  **Server:** On the server machine, start the Flower SuperLink. Then run the ServerApp.

2.  **Clients:** On *each* client machine, start a Flower SuperNode. You'll need to specify the `partition-id` and `num-partitions` using `--node-config`.  **Important:**  The `partition-id` should correspond to the partition number assigned to that client by `prepare_federated_data.sh`. The `num-partitions` should be equal to the total number of clients.
  
    **Example:**
    ```bash
    #!/bin/bash -eu

    . ./flower.config
    HOSTS="${HOSTS_SUPERNODE}"
    . ./flower-helper.sh


    # ====== Run ================================================================
    docker run --rm \
      -p 9094:9094 \
      --network "hostonly${VLAN}" \
      --name "supernode-${HOSTNAME}" \
      --detach \
      flwr/supernode:1.13.0  \
      --insecure \
      --superlink "${superLinkIP}:9092" \
      --node-config "partition-id=0 num-partitions=4" \
      --clientappio-api-address 0.0.0.0:9094 \
      --isolation process
    ```

    **Note for BRATS:** For the BRATS dataset, the `prepare_federated_data.sh` script creates client directories named `preprocessed_..._train` on all nodes there `partition-id`should be set to `0` and same for `num-partitions`. 

    **Note for MedMNIST:** MedMNIST partition IDs start from 1, so `partition-id=1` corresponds to the directory `medmnist_part_1`.

    Then start up ClientApps.

3. **Start a run** \
On the server machine run:
    ```bash
    flwr run . local-deployment
    ```

## Step 4: Monitoring with TensorBoard

1.  **Locate Log Directory:** The `ServerApp` generates TensorBoard logs in the `tb_logs` directory. If running in Docker, this directory is *inside* the server container.
2.  **Extract Logs (Docker):** If using Docker, copy the `tb_logs` directory from the *server* container to your local machine:

    ```bash
    docker cp <server_container_id>:/app/tb_logs ./server_tb_logs
    ```

    Replace `<server_container_id>` with the actual ID of your server container.
3.  **Launch TensorBoard:**

    ```bash
    tensorboard --logdir ./server_tb_logs
    ```

    Access TensorBoard in your browser (usually at `http://localhost:6006`).