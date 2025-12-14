# YOLOv5 Person Detection with Model Compression

Person detection project using YOLOv5. We applied model compression techniques (Pruning, Quantization) to reduce model size while keeping good performance.

For detailed code explanation, check out the blog post:
https://medium.com/@azsxfv92/6-apply-pruning-fde333565644

## 📋 Project Overview

- **Goal**: Person detection and model compression using YOLOv5
- **Model**: YOLOv5s (Small)
- **Techniques**:
  - Structured Pruning
  - Weight Quantization
  - Fine-tuning
- **Dataset**: COCO Dataset (Person class only)

## 🚀 Quick Start

### 1. Setup

```bash
# Clone repository
git clone <your-repository-url>
cd YOLOv5_Person_Detection_Project

# Create conda environment (recommended)
conda create -n yolov5 python=3.10
conda activate yolov5

# Install dependencies
pip install -r requirements.txt
```

**Note**: `requirements.txt` contains all packages from the actual yolov5 virtual environment used in this project.

### 2. Prepare Dataset

#### Download COCO Dataset

```bash
# Create datasets folder
mkdir -p ../datasets/coco

# Download COCO 2017 dataset (auto)
cd data/scripts
bash get_coco.sh
cd ../..
```

Or download manually:
- Download link: http://cocodataset.org/#download
- Need: train2017.zip, val2017.zip, annotations_trainval2017.zip

#### Filter Person Class

Run the first cell in the notebook to extract only person class from COCO dataset.

```python
# Run in YOLOv5_Person_Detection_Small_Annotated.ipynb
# Cell 1: Filter Person class from COCO dataset
```

### 3. Download Pretrained Model

```bash
# Download YOLOv5s pretrained model
wget https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s.pt
```

### 4. Run Notebook

```bash
# Start Jupyter Lab
jupyter lab

# Or Jupyter Notebook
jupyter notebook
```

Open `YOLOv5_Person_Detection_Small_Annotated.ipynb` in browser and run cells in order.

## 📁 Project Structure

```
YOLOv5_Person_Detection_Project/
├── YOLOv5_Person_Detection_Small_Annotated.ipynb  # Main notebook
├── requirements.txt                                # Package dependencies
├── README.md                                       # This file
├── .gitignore                                      # Git ignore file
├── LICENSE                                         # License
│
├── models/                                         # YOLOv5 model definitions
│   ├── yolo.py                                     # YOLO model class
│   ├── common.py                                   # Common layers
│   └── hub/*.yaml                                  # Model config files
│
├── utils/                                          # Utility functions
│   ├── dataloaders.py                              # Data loaders
│   ├── general.py                                  # General functions
│   ├── loss.py                                     # Loss functions
│   ├── metrics.py                                  # Metrics
│   ├── plots.py                                    # Visualization
│   └── torch_utils.py                              # PyTorch utils
│
├── data/                                           # Dataset configs
│   ├── person_final.yaml                           # Person detection config
│   ├── coco.yaml                                   # COCO dataset config
│   ├── hyps/                                       # Hyperparameters
│   └── scripts/                                    # Data download scripts
│
├── train.py                                        # Training script
├── train_fine_tuning.py                           # Fine-tuning script
├── val.py                                          # Validation script
├── detect.py                                       # Inference script
└── export.py                                       # Model export (ONNX, TensorRT, etc)
```

## 📊 Notebook Steps

The notebook has these steps:

1. **Data Preparation**: Filter only Person class from COCO
2. **Model Training**: Train person detection with YOLOv5s
3. **Structured Pruning**: Apply channel-wise pruning
4. **Fine-tuning**: Recover performance after pruning
5. **Weight Quantization**: Apply weight quantization (K-means clustering)
6. **Fine-tuning**: Recover performance after quantization
7. **Performance Comparison**: Compare Original vs Pruned vs Quantized models

## 🎯 Main Features

### 1. Structured Pruning
- Remove less important channels to reduce model size
- Calculate channel importance based on L1-norm
- Apply 70% pruning ratio

### 2. Weight Quantization
- Apply weight quantization using K-means clustering
- Support 8-bit, 4-bit, 2-bit quantization
- Save as sparse matrix for better memory efficiency

### 3. Fine-tuning
- Recover performance after Pruning/Quantization
- Fast training using subset of training data
- Adjust learning rate and early stopping


**Made with YOLOv5** 🚀
