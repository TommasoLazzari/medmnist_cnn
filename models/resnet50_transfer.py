import tensorflow as tf
from tensorflow.keras import layers, Model

'''
ResNet50, con i pesi di IMageNet
backbone congelata
nuova head di classificazione addestrata:
- global average pooling
- dense(256)
- dense(8)
'''


def create_resnet50_transfer_model(input_shape=(64, 64, 3), num_classes=8):
    
    inputs = layers.Input(shape=input_shape, name="input_image")

    x = layers.Rescaling(scale=255.0, name="rescale_to_255")(inputs)
    x = tf.keras.applications.resnet50.preprocess_input(x)

    base_model = tf.keras.applications.ResNet50(
        include_top=False,
        weights="imagenet",
        input_shape=input_shape,
        pooling=None,
    )
    base_model.trainable = False

    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = layers.Dense(256, activation="relu", name="dense_256")(x)
    x = layers.Dropout(0.30, name="dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = Model(inputs=inputs, outputs=outputs, name="Model_H_ResNet50_Transfer")
    return model