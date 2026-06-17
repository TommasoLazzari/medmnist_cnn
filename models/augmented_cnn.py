import tensorflow as tf
from tensorflow.keras import layers, Model

def create_augmented_model(input_shape=(64, 64, 3), num_classes=8):
    #Model B: Model A + Data Augmentation. Includes random flips, rotation, and zoom.
    
    inputs = layers.Input(shape=input_shape, name="input_image")
    #Data Augmentation: random flips + random rotation + random zoom
    x = layers.RandomFlip("horizontal_and_vertical", name="random_flip")(inputs)
    x = layers.RandomRotation(factor=0.1, fill_mode="reflect", name="random_rotation")(x)
    x = layers.RandomZoom(height_factor=0.1, fill_mode="reflect", name="random_zoom")(x)
    #First layer: Conv + MaxPool
    x = layers.Conv2D(32, (3, 3), padding="same", activation="relu", name="conv_1")(x)
    x = layers.MaxPooling2D((2, 2), name="maxpool_1")(x)
    #Second layer: Conv + MaxPool
    x = layers.Conv2D(64, (3, 3), padding="same", activation="relu", name="conv_2")(x)
    x = layers.MaxPooling2D((2, 2), name="maxpool_2")(x)
    #Classifier: Dense Head
    x = layers.Flatten(name="flatten")(x)
    x = layers.Dense(128, activation="relu", name="dense_dense")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)
    
    model = Model(inputs=inputs, outputs=outputs, name="Model_B_Augmented_CNN")
    return model
