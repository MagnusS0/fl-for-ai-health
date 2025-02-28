# Federated Learning for Medical Imaging

A Flower-based federated learning application for medical image analysis tasks (classification and segmentation) using PyTorch.

## Features

-   **Two Medical Use Cases**:
    -   🧠 Brain Tumor Segmentation (BRATS 3D to 2D slices)
    -   📊 Pathological Classification (MedMNIST dataset)
-   **Supported Models**:
    -   Classification: ResNet-18, TinyViT
    -   Segmentation: U-Net, SegFormer
-   **Federated Learning Features**:
    -   TensorBoard integration for training monitoring
    -   Best model checkpointing
    -   Centralized evaluation

## Installation

```bash
git clone https://github.com/smadigi/fl_2.git
cd fl_2
pip install -e .
```

## Quickstart (Centralized Simulation)

For a quick, centralized simulation (all clients on one machine), follow these steps:

1.  **Install Requirements:** (See Installation above)

2.  **Configure Task (MedMNIST Classification):**  Edit `pyproject.toml`:

    ```toml
    [tool.flwr.app.components]
    # Classification
    serverapp = "fl_for_ai_health.classification.class_server:app"
    clientapp = "fl_for_ai_health.classification.class_client:app"
    # Segmentation (comment out or remove for classification)
    # serverapp = "fl_for_ai_health.segmentation.seg_server:app"
    # clientapp = "fl_for_ai_health.segmentation.seg_client:app"

    [tool.flwr.app.config]
    num-server-rounds = 20  # Adjust as needed
    fraction-fit = 0.5
    local-epochs = 1
    batch-size = 32   # Or other appropriate size
    learning-rate = 3e-4
    img-size = 64
    in-channels = 3
    num-classes = 9
    from-disk = false  # Downloads MedMNIST
    model="resnet-18" # or "tiny-vit"
    ```

3. **BRATS Simulation**
   - Download the dataset from [Medical Decathlon](http://medicaldecathlon.com/)
   - Create a `.env` file in the project root, and add the following paths (adjust to your actual paths):
        ```
        BRATS_DATA_DIR=/path/to/Task01_BrainTumour
        BRATS_JSON_PATH=/path/to/Task01_BrainTumour/dataset.json
        ```

    - Update `pyproject.toml` for segmentation:

    ```toml
    [tool.flwr.app.components]
    # Classification (comment out or remove for segmentation)
    # serverapp = "fl_for_ai_health.classification.class_server:app"
    # clientapp = "fl_for_ai_health.classification.class_client:app"
    # Segmentation
    serverapp = "fl_for_ai_health.segmentation.seg_server:app"
    clientapp = "fl_for_ai_health.segmentation.seg_client:app"


    [tool.flwr.app.config]
    num-server-rounds = 20 # Adjust rounds
    fraction-fit = 0.5
    local-epochs = 1
    data-dir = "/path/to/Task01_BrainTumour"   
    dataset-json-path = "/path/to/Task01_BrainTumour/dataset.json"
    in-channels = 4
    num-classes = 4
    img-size = 96
    batch-size = 16   # Adjust as needed
    learning-rate = 3e-4
    weight-decay = 1e-4
    val-split = 0.2
    num-workers = 4
    model="u-net" # or "segformer"
    ```
    - Run Simulation
    ```bash
    flwr run .
    ```

4.  **Run Simulation:**

    Adjust number of nodes and the compute per node:
    ```toml
    [tool.flwr.federations.local-simulation]
    options.num-supernodes = 4
    options.backend.client-resources.num-cpus = 4 # each ClientApp will use 4 CPU cores
    ```
    Then start the run with:
    ```bash
    flwr run .
    ```
    This will run the simulation on a local computer. The data is split automatically by Flower.
## Federated Deployment

For a *distributed*, federated setup across multiple machines, see the detailed instructions in [`FEDERATED_DEPLOYMENT.md`](FEDERATED_DEPLOYMENT.md).  This involves:

1.  **Dataset Preparation and Distribution:** Using the provided `prepare_federated_data.sh` script.
2.  **Docker Setup:** Building and running Docker containers on each client and the server.
3.  **Configuration:**  Setting up `pyproject.toml` and `.env` correctly.

## Monitoring Training

TensorBoard logs are saved in `tb_logs/`.  Launch TensorBoard with:

```bash
tensorboard --logdir tb_logs/
```

## Resources

- [Flower Documentation](https://flower.ai/docs/)
- [MedMNIST Paper](https://medmnist.com/)
- [BRATS Challenge](https://www.med.upenn.edu/cbica/brats/)
- [TinyViT Paper](https://arxiv.org/abs/2207.10666)
- [SegFormer Paper](https://arxiv.org/abs/2105.15203)
- [U-Net Paper](https://arxiv.org/abs/1505.04597)
- [ResNet Paper](https://arxiv.org/abs/1512.03385)