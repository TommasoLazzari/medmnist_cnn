# MedMNIST CNN

Technical pipeline for training and evaluating CNN models on BloodMNIST.

## Authors

Ludergnani Brenno

Lazzari Tommaso

## Project structure

* `main.py`: evaluation script. Loads trained models from `saved\\\\\\\_models/`, computes test metrics, saves confusion matrices and `results/comparison\\\\\\\_results.json`.
* `train.py`: training script. Loads BloodMNIST, builds selected models, trains them, saves `.keras` models, learning curves, and metadata JSON files.
* `utils/data\\\\\\\_loader.py`: downloads BloodMNIST, scales images to `\\\\\\\[0,1]`, standardizes them using training-set channel statistics, and creates `tf.data.Dataset` objects.
* `utils/visualization.py`: functions for learning curves, confusion matrices, and ROC plots.
* `models/`: model definitions:

  * `baseline\\\\\\\_cnn.py`: basic 2-block CNN.
  * `augmented\\\\\\\_cnn.py`: baseline CNN with data augmentation.
  * `bn\\\\\\\_cnn.py`: augmented CNN with Batch Normalization.
  * `attention\\\\\\\_cnn.py`: augmented BN-CNN with spatial attention.
  * `bn\\\\\\\_noaug\\\\\\\_cnn.py`: BN-CNN without augmentation.
  * `attention\\\\\\\_noaug\\\\\\\_cnn.py`: BN-CNN with spatial attention, without augmentation.
  * `resnet50\\\\\\\_transfer.py`: frozen ImageNet ResNet50 backbone with custom classifier head.
* `results/`: generated metrics, metadata, learning curves, and confusion matrices.
* `saved\\\\\\\_models/`: trained Keras models.

## Requirements

```bash
pip install tensorflow numpy matplotlib scikit-learn medmnist
```

## Training

Train all models:

```bash
python train.py --model all --epochs 50 --batch-size 64 --seed 42
```

Train a single model:

```bash
python train.py --model A --epochs 50 --batch-size 64 --seed 42
```

Available model keys: `A`, `B`, `C`, `D`, `E`, `F`, `G`.

## Evaluation

After training, run:

```bash
python Ludergnani\\\\\\\_Lazzari.py
```

This evaluates all saved models in `saved\\\\\\\_models/` and writes:

```text
results/comparison\\\\\\\_results.json
results/evaluation/\\\\\\\*\\\\\\\_confusion\\\\\\\_matrix.png
```

## Outputs

Training produces:

```text
saved\\\\\\\_models/model\\\\\\\_<letter>.keras
results/metadata\\\\\\\_<letter>.json
results/curves/\\\\\\\*\\\\\\\_history.png
```

Evaluation produces:

```text
results/comparison\\\\\\\_results.json
results/evaluation/\\\\\\\*\\\\\\\_confusion\\\\\\\_matrix.png
```

