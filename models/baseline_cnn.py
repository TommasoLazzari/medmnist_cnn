import tensorflow as tf
from tensorflow.keras import layers, Model

def create_baseline_model(input_shape=(64, 64, 3), num_classes=8):
    
    inputs = layers.Input(shape=input_shape, name="input_image")
    #First layer: Conv + MaxPool
    x = layers.Conv2D(32, (3, 3), padding="same", activation="relu", name="conv_1")(inputs)
    x = layers.MaxPooling2D((2, 2), name="maxpool_1")(x)
    #Second layer: Conv + MaxPool
    x = layers.Conv2D(64, (3, 3), padding="same", activation="relu", name="conv_2")(x)
    x = layers.MaxPooling2D((2, 2), name="maxpool_2")(x)
    #Classifier: Dense Head
    x = layers.Flatten(name="flatten")(x)
    x = layers.Dense(128, activation="relu", name="dense_dense")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)
    
    model = Model(inputs=inputs, outputs=outputs, name="Model_A_Baseline_CNN")
    return model
