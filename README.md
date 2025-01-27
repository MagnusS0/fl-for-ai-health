# fl-for-AI-health: Federated Learning for Medical Imaging

A Flower-based federated learning application for medical image analysis tasks (classification and segmentation) using PyTorch.

## Features

- **Two Medical Use Cases**:
  - 🧠 Brain Tumor Segmentation (BRATS 3D to 2D slices)
  - 📊 Pathological Classification (MedMNIST dataset)
- **Supported Models**:
  - Classification: ResNet-18, TinyViT
  - Segmentation: U-Net, SegFormer
- **Federated Learning Features**:
  - TensorBoard integration for training monitoring
  - Best model checkpointing
  - Centralized evaluation


## Installation
```bash
git clone https://github.com/MagnusS0/fl-for-AI-health.git
cd fl-for-AI-health
pip install -e .
```

## Running Simulations

### For Classification (MedMNIST):
```
bash
flwr run .
```

### For Segmentation (BRATS):

Make sure you have downloaded the BRATS dataset and set the paths in the `.env`file.

1. Update `pyproject.toml`:

```bash
[tool.flwr.app.components]
# serverapp = "fl_for_ai_health.classification.class_server:app"
# clientapp = "fl_for_ai_health.classification.class_client:app"
serverapp = "fl_for_ai_health.segmentation.seg_server:app"
clientapp = "fl_for_ai_health.segmentation.seg_client:app"

[tool.flwr.app.config]
in-channels = 1
num-classes = 4
model="u-net" or "segformer"
```

2. Run simulation:
```bash
flwr run .
```

On the first run this will build the dataset from 3D to 2D axial slices this mike take some time.
On connecutive runs this will run much faster. Alternativly run the dataset script first.

## Configuration

Key configuration options in `pyproject.toml`:

```bash
[tool.flwr.app.config]
num-server-rounds = 10 # Total federation rounds
fraction-fit = 0.5 # Fraction of clients used for training
local-epochs = 1 # Local client epochs
batch-size = 64 # Training batch size
learning-rate = 4e-3 # Initial learning rate
img-size = 64 # Input image size
model = "tiny-vit" # Model architecture
in-channels = 3 # Input channels
num-classes = 9 # Output classes
```

## Monitoring Training

TensorBoard logs are saved in `tb_logs/`. Launch TensorBoard with:
```bash
tensorboard --logdir tb_logs/
```

## Dataset Preparation

1. **MedMNIST**:
   - Automatically downloaded via Hugging Face Datasets
   - Preprocessed into train/test/val splits

2. **BRATS**:
   - Requires manual download from [Medical Decathlon](http://medicaldecathlon.com/)
   - Preprocessing handled by `data/brats.py`

## Resources

- [Flower Documentation](https://flower.ai/docs/)
- [MedMNIST Paper](https://medmnist.com/)
- [BRATS Challenge](https://www.med.upenn.edu/cbica/brats/)
- [TinyViT Paper](https://arxiv.org/abs/2207.10666)
- [SegFormer Paper](https://arxiv.org/abs/2105.15203)
- [U-Net Paper](https://arxiv.org/abs/1505.04597)
- [ResNet Paper](https://arxiv.org/abs/1512.03385)