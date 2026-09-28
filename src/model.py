"""CNN architecture: treats a log-mel-spectrogram as an image and classifies genre.

Four conv blocks progressively shrink the time/frequency axes while growing
channel depth (32 -> 64 -> 128 -> 256), which is the standard "extract more
abstract features as you go deeper" pattern. GlobalAveragePooling replaces a
big Flatten+Dense to keep parameter count sane and reduce overfitting on a
dataset as small as GTZAN (~1000 base tracks).
"""

from tensorflow.keras import layers, models


def conv_block(x, filters):
    x = layers.Conv2D(filters, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2), padding="same")(x)
    x = layers.Dropout(0.25)(x)
    return x


def build_cnn(input_shape, num_classes):
    inputs = layers.Input(shape=input_shape, name="mel_spectrogram")

    x = conv_block(inputs, 32)
    x = conv_block(x, 64)
    x = conv_block(x, 128)
    x = conv_block(x, 256)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="genre")(x)

    model = models.Model(inputs, outputs, name="genre_cnn")
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
