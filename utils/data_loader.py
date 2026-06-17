import numpy as np
import tensorflow as tf
from medmnist import BloodMNIST

def load_bloodmnist_data(img_size=64):
    """
    Downloads and loads the BloodMNIST dataset at the specified resolution,
    normalizes it using training set statistics, and returns train, val,
    and test numpy arrays.
    """
    print(f"--- Loading BloodMNIST data with image resolution: {img_size}x{img_size} ---")
    
    # Download and load raw data
    train_dataset_raw = BloodMNIST(split="train", download=True, size=img_size)
    val_dataset_raw = BloodMNIST(split="val", download=True, size=img_size)
    test_dataset_raw = BloodMNIST(split="test", download=True, size=img_size)

    # Convert to float32 and scale to [0, 1]
    x_train = train_dataset_raw.imgs.astype("float32") / 255.0
    y_train = train_dataset_raw.labels.squeeze()
    
    x_val = val_dataset_raw.imgs.astype("float32") / 255.0
    y_val = val_dataset_raw.labels.squeeze()
    
    x_test = test_dataset_raw.imgs.astype("float32") / 255.0
    y_test = test_dataset_raw.labels.squeeze()

    # Calculate channel-wise training mean and std dev
    train_mean = np.mean(x_train, axis=(0, 1, 2), keepdims=True)
    train_std = np.std(x_train, axis=(0, 1, 2), keepdims=True) + 1e-7

    # Standardize data using training statistics
    x_train = (x_train - train_mean) / train_std
    x_val = (x_val - train_mean) / train_std
    x_test = (x_test - train_mean) / train_std

    print(f"Train set: {x_train.shape}, labels: {y_train.shape}")
    print(f"Val set:   {x_val.shape}, labels: {y_val.shape}")
    print(f"Test set:  {x_test.shape}, labels: {y_test.shape}")
    print(f"Train channel-wise mean: {train_mean.squeeze()}")
    print(f"Train channel-wise std:  {train_std.squeeze()}")
    
    return x_train, y_train, x_val, y_val, x_test, y_test


def make_datasets(x_train, y_train, x_val, y_val, x_test, y_test, batch_size=64, seed=42):
    """
    Wraps numpy arrays into tf.data.Dataset objects, applying shuffling,
    batching, and prefetching.
    """
    autotune = tf.data.AUTOTUNE

    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    val_ds = tf.data.Dataset.from_tensor_slices((x_val, y_val))
    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))

    train_ds = train_ds.shuffle(
        buffer_size=len(x_train),
        seed=seed,
        reshuffle_each_iteration=True,
    ).batch(batch_size).prefetch(autotune)

    val_ds = val_ds.batch(batch_size).prefetch(autotune)
    test_ds = test_ds.batch(batch_size).prefetch(autotune)

    return train_ds, val_ds, test_ds
