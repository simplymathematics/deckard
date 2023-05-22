import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D


__all__ = [
    "TFNet",
]


# Step 1: Load the MNIST dataset
class TFNet(Model):
    """
    Standard TensorFlow model for unit testing.
    """

    def __init__(self, optimizer: dict, loss_object: dict, output_shape=10):
        super(TFNet, self).__init__()
        self.conv1 = Conv2D(filters=4, kernel_size=5, activation="relu")
        self.conv2 = Conv2D(filters=10, kernel_size=5, activation="relu")
        self.maxpool = MaxPool2D(
            pool_size=(2, 2),
            strides=(2, 2),
            padding="valid",
            data_format=None,
        )
        self.flatten = Flatten()
        self.dense1 = Dense(100, activation="relu")
        self.logits = Dense(output_shape, activation="linear")
        identifier = dict(loss_object)
        self.loss_object = tf.keras.losses.get(identifier=identifier)
        self.optimizer = tf.keras.optimizers.get(identifier=dict(optimizer))

        self.train_step = train_step

    def call(self, x):
        """
        Call function to evaluate the model.
        :param x: Input to the model
        :return: Prediction of the model
        """
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.maxpool(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.logits(x)
        return x


def train_step(model, images, labels):
    with tf.GradientTape() as tape:
        predictions = model(images, training=True)
        loss = model.loss_object(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
