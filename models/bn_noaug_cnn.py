import tensorflow as tf
from tensorflow.keras import layers, Model

def create_bn_noaug_model(input_shape=(64, 64, 3), num_classes=8):
    #Model C: Model A + Batch Normalization. Batch Normalization is added before activations for maximum stability.
    inputs = layers.Input(shape=input_shape, name="input_image")
    #First layer: Conv + BN + MaxPool
    x = layers.Conv2D(32, (3, 3), padding="same", use_bias=False, name="conv_1")(inputs)
    x = layers.BatchNormalization(name="bn_1")(x)
    x = layers.Activation("relu", name="relu_1")(x)
    x = layers.MaxPooling2D((2, 2), name="maxpool_1")(x)
    #Second layer: Conv + BN + MaxPool
    x = layers.Conv2D(64, (3, 3), padding="same", use_bias=False, name="conv_2")(x)
    x = layers.BatchNormalization(name="bn_2")(x)
    x = layers.Activation("relu", name="relu_2")(x)
    x = layers.MaxPooling2D((2, 2), name="maxpool_2")(x)
    #Classifier: Dense Head + BN
    x = layers.Flatten(name="flatten")(x)
    x = layers.Dense(128, use_bias=False, name="dense_dense")(x)
    x = layers.BatchNormalization(name="bn_dense")(x)
    x = layers.Activation("relu", name="relu_dense")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)
    
    model = Model(inputs=inputs, outputs=outputs, name="Model_E_BN_NoAug_CNN")
    return model
