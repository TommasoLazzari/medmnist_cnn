import tensorflow as tf
from tensorflow.keras import layers, Model

def spatial_attention_block(input_feature, name_prefix="spatial_att"):
    """
    Computes a 2D spatial attention map using channel-wise pooling (Avg and Max)
    and a Conv2D layer, then applies it as a scale mask.
    """
    #Channel-wise average pooling
    avg_pool = layers.Lambda(
        lambda x: __import__('tensorflow').reduce_mean(x, axis=-1, keepdims=True),
        name=f"{name_prefix}_avg"
    )(input_feature)
    #Channel-wise max pooling
    max_pool = layers.Lambda(
        lambda x: __import__('tensorflow').reduce_max(x, axis=-1, keepdims=True),
        name=f"{name_prefix}_max"
    )(input_feature)
    
    #Avg pooling and max pooling's activations concatenation
    concat = layers.Concatenate(axis=-1, name=f"{name_prefix}_concat")([avg_pool, max_pool])
    
    #Convolution to produce the attention map
    attention_map = layers.Conv2D(
        filters=1,
        kernel_size=7,
        padding="same",
        activation="sigmoid",
        use_bias=False,
        name=f"{name_prefix}_conv"
    )(concat)
    
    #Application of the attention map over the input features' tensor
    refined = layers.Multiply(name=f"{name_prefix}_scale")([input_feature, attention_map])
    
    return refined


def create_attention_model(input_shape=(64, 64, 3), num_classes=8):
    """
    Model D: Model C + Spatial Attention
    Injects a Spatial Attention Block after the convolutional feature extractor,
    refining spatial features before flattening.
    """
    inputs = layers.Input(shape=input_shape, name="input_image")
    
    #Data Augmentation: random flips + random rotation + random zoom
    x = layers.RandomFlip("horizontal_and_vertical", name="random_flip")(inputs)
    x = layers.RandomRotation(factor=0.1, fill_mode="reflect", name="random_rotation")(x)
    x = layers.RandomZoom(height_factor=0.1, fill_mode="reflect", name="random_zoom")(x)
    
    #First layer: Conv + BN + MaxPool
    x = layers.Conv2D(32, (3, 3), padding="same", use_bias=False, name="conv_1")(x)
    x = layers.BatchNormalization(name="bn_1")(x)
    x = layers.Activation("relu", name="relu_1")(x)
    x = layers.MaxPooling2D((2, 2), name="maxpool_1")(x)
    
    #Second layer: Conv + BN + MaxPool + Attention
    x = layers.Conv2D(64, (3, 3), padding="same", use_bias=False, name="conv_2")(x)
    x = layers.BatchNormalization(name="bn_2")(x)
    x = layers.Activation("relu", name="relu_2")(x)
    x = layers.MaxPooling2D((2, 2), name="maxpool_2")(x)
    x = spatial_attention_block(x, name_prefix="spatial_attention")
    
    #Classifier: Dense Head + BN
    x = layers.Flatten(name="flatten")(x)
    x = layers.Dense(128, use_bias=False, name="dense_dense")(x)
    x = layers.BatchNormalization(name="bn_dense")(x)
    x = layers.Activation("relu", name="relu_dense")(x)
    
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)
    
    model = Model(inputs=inputs, outputs=outputs, name="Model_D_Attention_CNN")
    return model
