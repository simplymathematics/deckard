# train RESNET50 on MNIST dataset using transfer learning
# import libraries
import os
import sys
import time
import numpy as np
import logging

# import resnet from tf.keras.applications
from tensorflow.keras.applications.resnet50 import ResNet50

# import preprocessing from tf.keras.applications
from tensorflow.keras.applications.resnet50 import preprocess_input
logger = logging.getLogger(__name__)


# import tf
import tensorflow as tf

# create main function
def main(level:int = 1, filename:str = 'mnist_resnet.h5', data = 'mnist'):
    # set logging level
    logging.basicConfig(level = level)
    # set logging format with timestamp
    logging.Formatter.converter = time.gmtime
    logging.Formatter.format = "[%(asctime)s] %(levelname)s: %(message)s"
    # set logging format for console
    logging.getLogger().setLevel(level)
    # set start time
    start_time = time.time()
    logger.info("Start time: %s", time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(start_time)))
    # set seed
    np.random.seed(0)
    # set tf seed
    tf.random.set_seed(0)
    # set tf eager execution
    # load resnet50 model
    model = ResNet50(weights = "imagenet")
    logger.info("Model loaded")
    # load data
    # data string to uppercase
    data = data.upper()
    # check if data is valid
    if data == 'MNIST':
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        logger.info("Data loaded")
    elif data == 'CIFAR10':
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
        logger.info("Data loaded")
    else:
        logger.error("Invalid data")
        sys.exit(1)
    # normalize data with preprocessing
    x_train = preprocess_input(x_train)
    x_test = preprocess_input(x_test)
    # convert data to float32
    x_train = x_train.astype(np.float32)
    x_test = x_test.astype(np.float32)
    # reshape data
    x_train = np.expand_dims(x_train, axis = 3)
    x_test = np.expand_dims(x_test, axis = 3)
    logger.info("Data reshaped")
    # convert data to tensor
    x_train = tf.convert_to_tensor(x_train, dtype = tf.float32)
    x_test = tf.convert_to_tensor(x_test, dtype = tf.float32)
    logger.info("Data converted to tensor")
    # convert labels to tensor
    y_train = tf.convert_to_tensor(y_train, dtype = tf.int32)
    y_test = tf.convert_to_tensor(y_test, dtype = tf.int32)
    logger.info("Labels converted to tensor")
    # create dataset
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    logger.info("Dataset created")
    # create iterators
    # train_iter = train_dataset.shuffle(60000).batch(32).repeat().make_one_shot_iterator()
    # test_iter = test_dataset.batch(32).make_one_shot_iterator()
    logger.info("Iterators created")
    # define input
    x = tf.placeholder(tf.float32, [None, 28, 28, 1])
    logger.info("Input defined")
    # define output
    y = tf.placeholder(tf.int32, [None])
    logger.info("Output defined")
    # define loss
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels = y, logits = model(x)))
    logger.info("Loss defined")
    # define optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate = 0.001)
    logger.info("Optimizer defined")
    # define train op
    train_op = optimizer.minimize(loss)
    logger.info("Train op defined")
    # define accuracy
    correct_pred = tf.equal(tf.argmax(model(x), 1), y)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    logger.info("Accuracy defined")
    # create session
    sess = tf.Session()
    logger.info("Session created")
    # initialize variables
    sess.run(tf.global_variables_initializer())
    logger.info("Variables initialized")
    # add new classifier layers
    # flatten
    new_model = tf.keras.layers.Flatten()(model.output)
    # classify layer
    new_model = tf.keras.layers.Dense(10, activation = "softmax")(new_model)
    # new ouput layer
    new_model = tf.keras.Model(inputs = model.input, outputs = new_model)
    # compile new model
    new_model.compile(optimizer = optimizer, loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels = y, logits = new_model(x))), metrics = ["accuracy"])
    logger.info("New model created")
    # train new model
    new_model.fit(x_train, y_train, epochs = 10, batch_size = 32, validation_data = (x_test, y_test))
    logger.info("New model trained")
    # evaluate new model
    new_model.evaluate(x_test, y_test)
    logger.info("New model evaluated")
    # save new model
    new_model.save(filename)
    logger.info("New model saved")
    # set end time
    end_time = time.time()
    # log accuracy
    logger.info("Accuracy: %s", sess.run(accuracy, feed_dict = {x: x_test, y: y_test}))
    # log time
    logger.info("End time: %s", time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(end_time)))
    # define elapsed time
    elapsed_time = end_time - start_time
    # log elapsed time
    logger.info("Elapsed time: %s", time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
    # evaluate model
    logger.info("Accuracy: %s", sess.run(accuracy, feed_dict = {x: x_test, y: y_test}))
    # close session
    sess.close()

    assert os.path.exists(filename)
    assert isinstance(model, tf.keras.Model)
    assert isinstance(new_model, tf.keras.Model)
    # assert isinstance(train_iter, tf.data.Iterator)
    # assert isinstance(test_iter, tf.data.Iterator)
    assert isinstance(x, tf.Tensor)
    assert isinstance(y, tf.Tensor)
    assert isinstance(loss, tf.Tensor)
    assert isinstance(optimizer, tf.train.Optimizer)
    assert isinstance(train_op, tf.Operation)
    assert isinstance(correct_pred, tf.Tensor)
    assert isinstance(accuracy, tf.Tensor)
    assert isinstance(sess, tf.Session)
    assert isinstance(start_time, float)
    assert isinstance(end_time, float)
    assert isinstance(elapsed_time, float)
    assert isinstance(x_train, np.ndarray)
    assert isinstance(y_train, np.ndarray)
    assert isinstance(x_test, np.ndarray)
    assert isinstance(y_test, np.ndarray)
    assert isinstance(correct_pred, np.ndarray)
    assert isinstance(accuracy, float)
    assert sess.run(tf.report_uninitialized_variables()) == []
    logger.info("Session closed")
    return 0

if __name__ == "__main__":
    # import argparse
    import argparse
    # accept command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--level", type = int, default = logger.INFO, help = "set logging level")
    parser.add_argument("-f", "--filename", type = str, default = "model.h5", help = "set filename")
    parser.add_argument("-d", "--data", type = str, default = "mnist", help = "set data. Choose mnist or cifar10.")
    args = parser.parse_args()
    level = args.level
    filename = args.filename
    data = args.data
    
    # set up logging
    logging.basicConfig(level = level, format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    # assert main() == 0
    assert main(level = level, filename = filename, data = data) == 0

    