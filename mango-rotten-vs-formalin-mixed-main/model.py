"""
model.py
--------
Defines two model architectures for the binary classification task
(formalin-treated mango = 0, rotten mango = 1):

    1. build_custom_cnn()   -> a CNN built from scratch (satisfies the
                                "custom CNN architecture" option in the
                                assignment brief).
    2. build_transfer_model() -> MobileNetV2 transfer learning (satisfies
                                the "pre-trained transfer learning model"
                                option; MobileNetV2 is lightweight enough
                                to train on a laptop CPU and to deploy
                                for free on Streamlit Community Cloud).

Both models accept raw uint8 RGB images of shape
(config.IMG_HEIGHT, config.IMG_WIDTH, 3) and include their own
preprocessing/rescaling layer internally, so callers (train_model.py,
app.py) never need to manually normalize pixel values -- this avoids the
classic bug where training-time preprocessing and inference-time
preprocessing drift apart.
"""

import tensorflow as tf
from tensorflow.keras import layers, models

import config


def build_custom_cnn(input_shape=(config.IMG_HEIGHT, config.IMG_WIDTH, config.IMG_CHANNELS)):
    """A compact CNN trained from scratch.

    Architecture: 4 conv blocks (Conv2D -> BatchNorm -> ReLU -> MaxPool)
    with increasing filter depth, followed by GlobalAveragePooling and a
    small dense head with dropout for regularization (important given a
    mini-project-sized dataset).
    """
    inputs = layers.Input(shape=input_shape, name="input_image")

    x = layers.Rescaling(1.0 / 255.0, name="rescale")(inputs)

    # Block 1
    x = layers.Conv2D(32, 3, padding="same", activation=None)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D()(x)

    # Block 2
    x = layers.Conv2D(64, 3, padding="same", activation=None)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D()(x)

    # Block 3
    x = layers.Conv2D(128, 3, padding="same", activation=None)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D()(x)

    # Block 4
    x = layers.Conv2D(256, 3, padding="same", activation=None)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D()(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(1, activation="sigmoid", name="prediction")(x)

    model = models.Model(inputs, outputs, name="custom_cnn_mango_classifier")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.LEARNING_RATE),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.Precision(name="precision"),
                 tf.keras.metrics.Recall(name="recall"),
                 tf.keras.metrics.AUC(name="auc")],
    )
    return model


def build_transfer_model(input_shape=(config.IMG_HEIGHT, config.IMG_WIDTH, config.IMG_CHANNELS),
                          fine_tune_at=None):
    """MobileNetV2 transfer-learning model.

    Parameters
    ----------
    fine_tune_at : int or None
        If given, unfreeze MobileNetV2 layers from this index onward for
        fine-tuning (call this AFTER a first round of head-only training;
        see train_model.py `--fine-tune` flag).
    """
    inputs = layers.Input(shape=input_shape, name="input_image")

    # MobileNetV2 expects inputs preprocessed to [-1, 1]
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)

    base_model = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False

    if fine_tune_at is not None:
        base_model.trainable = True
        for layer in base_model.layers[:fine_tune_at]:
            layer.trainable = False

    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(1, activation="sigmoid", name="prediction")(x)

    model = models.Model(inputs, outputs, name="mobilenetv2_mango_classifier")

    lr = config.LEARNING_RATE / 10 if fine_tune_at is not None else config.LEARNING_RATE
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.Precision(name="precision"),
                 tf.keras.metrics.Recall(name="recall"),
                 tf.keras.metrics.AUC(name="auc")],
    )
    return model, base_model


if __name__ == "__main__":
    # Quick sanity check: build both models and print their summaries.
    print("=" * 70)
    print("CUSTOM CNN")
    print("=" * 70)
    cnn = build_custom_cnn()
    cnn.summary()

    print("\n" + "=" * 70)
    print("MOBILENETV2 TRANSFER MODEL")
    print("=" * 70)
    transfer_model, _ = build_transfer_model()
    transfer_model.summary()
