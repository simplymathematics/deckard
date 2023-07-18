import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D


__all__ = [
    "MNISTNet",
]


# Step 1: Load the MNIST dataset
class MNISTNet:
    def __init__(self, optimizer, loss, metrics):
        model = Sequential()
        model.add(
            Conv2D(
                filters=1,
                kernel_size=(5, 5),
                strides=1,
                activation="relu",
                input_shape=(28, 28, 1),
            ),
        )
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(
            Conv2D(
                filters=10,
                kernel_size=(5, 5),
                strides=1,
                activation="relu",
                input_shape=(23, 23, 4),
            ),
        )
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dense(100, activation="relu"))
        model.add(Dense(10, activation="softmax"))
        self.model = model
        self.optimizer = tf.keras.optimizers.get(optimizer)
        self.loss = tf.keras.losses.get(loss)
        self.metrics = metrics if isinstance(metrics, (list, dict)) else [metrics]

    def __call__(self):
        self.model.compile(
            optimizer=self.optimizer, loss=self.loss, metrics=self.metrics,
        )
        return self.model
