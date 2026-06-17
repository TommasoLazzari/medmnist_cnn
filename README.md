# MedMNIST CNN

A comparative study of lightweight convolutional neural networks for blood cell classification on the BloodMNIST dataset.

The project investigates how architectural components such as data augmentation, batch normalization, spatial attention, and transfer learning affect predictive performance, computational cost, and inference efficiency. Starting from a simple two-layer CNN, progressively more complex architectures are evaluated under identical training and evaluation conditions.

## Dataset

The experiments are conducted on **BloodMNIST**, a multi-class medical image classification benchmark from the MedMNIST collection.

* 8 blood cell classes
* RGB images
* Resolution: 64 × 64
* Official train/validation/test splits

## Models

| Model | Description                      |
| ----- | -------------------------------- |
| A     | Baseline CNN                     |
| B     | Baseline CNN + Data Augmentation |
| C     | Model B + Batch Normalization    |
| D     | Model C + Spatial Attention      |
| E     | Model A + Batch Normalization    |
| F     | Model E + Spatial Attention      |
| G     | ResNet50 Transfer Learning       |

## Repository Structure

```text
medmnist_cnn/
│
├── main.py          # evaluation and model comparison
├── train.py                       # training pipeline
├── Report.pdf         # project report
│
├── utils/
│   ├── data_loader.py             # dataset loading and preprocessing
│   └── visualization.py           # training and evaluation plots
│
├── models/
│   ├── baseline_cnn.py
│   ├── augmented_cnn.py
│   ├── bn_cnn.py
│   ├── attention_cnn.py
│   ├── bn_noaug_cnn.py
│   ├── attention_noaug_cnn.py
│   └── resnet50_transfer.py
│
├── saved_models/                  # trained models (.keras)
│
└── results/
    ├── comparison_results.json
    ├── metadata_*.json
    ├── curves/
    └── evaluation/
```

## Installation

```bash
pip install tensorflow numpy matplotlib scikit-learn medmnist
```

## Training

Train all architectures:

```bash
python train.py --model all --epochs 50 --batch-size 64 --seed 42
```

Train a single model:

```bash
python train.py --model A
```

## Evaluation

After training, evaluate all saved models:

```bash
python main.py
```

The script computes:

* Accuracy
* Precision
* Recall
* F1-score
* ROC-AUC
* Inference latency
* Confusion matrices

and stores the results inside the `results/` directory.

## Report

A detailed description of the methodology, experiments, and results is available in:

```text
Report.pdf
```
