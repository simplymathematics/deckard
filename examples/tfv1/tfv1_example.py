import tensorflow.compat.v1 as tf


__all__ = [
    "MNISTNet",
]


# Step 1: Load the MNIST dataset
class MNISTNet:
    def __init__(self, input_shape=(28, 28, 1), num_classes=10):
        self.input_ph = tf.placeholder(tf.float32, shape=[None, *input_shape])
        self.labels_ph = tf.placeholder(tf.float32, shape=[None, num_classes])
        self.layer1 = tf.layers.conv2d(
            self.input_ph, filters=4, kernel_size=5, activation=tf.nn.relu,
        )
        self.layer2 = tf.layers.max_pooling2d(self.layer1, 2, 2)
        self.layer3 = tf.layers.conv2d(
            self.layer2, filters=10, kernel_size=5, activation=tf.nn.relu,
        )
        self.layer4 = tf.layers.max_pooling2d(self.layer3, 2, 2)
        self.layer5 = tf.layers.flatten(self.layer4)
        self.layer6 = tf.layers.dense(self.layer5, 100, activation=tf.nn.relu)
        self.output_layer = tf.layers.dense(self.layer6, 10)
