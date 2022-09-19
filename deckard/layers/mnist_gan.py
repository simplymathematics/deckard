# # train a GAN on mnist with keras

# # import modules
# import os
# import sys
# import time

# import numpy as np

# import tensorflow as tf
# from tensorflow.python.keras.layers import Input, Dense, Reshape, Flatten, Dropout, Lambda
# from tensorflow.python.keras.layers import BatchNormalization, Activation, ZeroPadding2D
# from tensorflow.python.keras.layers import LeakyReLU
# from tensorflow.python.keras.layers import UpSampling2D, Conv2D, Conv2DTranspose
# from tensorflow.python.keras.models import Sequential, Model
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.datasets import mnist
# import logging
# import matplotlib.pyplot as plt
# from tensorflow.python.ops.nn_impl import normalize
# # image display function
# logger = logging.getLogger(__name__)

# def load_data(dataset = 'mnist') -> tuple:
#     """
#     load mnist data
#     """
#     logger.info("loading data")
#     if dataset != 'mnist':
#         raise NotImplementedError
#     (x_train, y_train), (x_test, y_test) = mnist.load_data()
#     x_train = (x_train.astype(np.float32) -127.5 )/127.5
#     x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
#     return (x_train, y_train, x_test, y_test)

# # sample the data
# def sample_data(x_train:np.ndarray, y_train:np.ndarray, batch_size:int) -> tuple:
#     """
#     sample data
#     """
#     idx = np.random.randint(0, x_train.shape[0], batch_size)
#     return (x_train[idx], y_train[idx])

# # load batch
# def load_batch(x_train:np.ndarray, y_train:np.ndarray, batch_size:int) -> tuple:
#     """
#     load batch
#     """
#     idx = np.random.randint(0, x_train.shape[0], batch_size)
#     return (x_train[idx], y_train[idx])

# def detect_image_size(x_train:np.ndarray) -> tuple:
#     """
#     detect image size
#     """
#     return x_train[0].shape


# def build_discriminator(input_shape:tuple, latent_dim = 100) -> Model:
#     """
#     build discriminator
#     """
#     logger.info("building discriminator")
#     model = Sequential()
#     model.add(Flatten(input_shape=latent_dim))
#     model.add(Dense(512))
#     model.add(LeakyReLU(alpha=0.2))
#     model.add(Dense(256))
#     model.add(LeakyReLU(alpha=0.2))
#     model.add(Dense(1, activation='sigmoid'))
#     model.summary()
#     return model

# def build_generator(input_shape:int) -> Model:
#     """
#     build generator
#     """
#     logger.info("building generator")
#     model = Sequential()
#     model.add(Dense(256, input_dim=100))
#     model.add(LeakyReLU(alpha=0.2))
#     model.add(BatchNormalization(momentum=0.8))
#     model.add(Dense(512))
#     model.add(LeakyReLU(alpha=0.2))
#     model.add(BatchNormalization(momentum=0.8))
#     model.add(Dense(1024))
#     model.add(LeakyReLU(alpha=0.2))
#     model.add(BatchNormalization(momentum=0.8))
#     model.add(Dense(np.prod(input_shape), activation='tanh'))
#     model.add(Reshape(input_shape))
#     model.summary()
#     return model

# def build_gan(generator:Model, discriminator:Model) -> Model:
#     """
#     build gan
#     """
#     logger.info("building gan")
#     # define gan
#     gan_input = Input(shape=(100,))
#     # connect gan input to generator's output
#     x = generator(gan_input)
#     # connect gan input to discriminator's output
#     gan_output = discriminator(x)
#     # define gan model
#     gan = Model(gan_input, gan_output)
#     return gan


# def train_gan(generator:Model, discriminator:Model, gan:Model, x_train:np.ndarray, y_train:np.ndarray, epochs:int, batch_size:int, sample_interval:int = 10, save_interval:int = 10, save_dir:str = None, loss = 'binary_crossentropy', optimizer = 'adam') -> tuple:
#     """
#     train gan
#     """
#     logger.info("training gan")
#     # compile models
#     generator.compile(loss=loss, optimizer=optimizer)
#     discriminator.compile(loss=loss, optimizer=optimizer)
#     gan.compile(loss=loss, optimizer=optimizer)
#     # train the model
#     d_loss_list = []
#     g_loss_list = []
#     if save_dir is None:
#         save_dir  = os.path.join(os.getcwd())
#     assert os.path.exists(save_dir)
#     for epoch in range(epochs):
#         for _ in range(batch_size):
#             # generate real images
#             real_images, _ = load_batch(x_train, y_train, batch_size)
#             # generate latent vectors
#             latent_vectors = np.random.normal(0, 1, size=(batch_size, 100))
#             # generate fake images
#             fake_images = generator.predict(latent_vectors)
#             # get discriminated real images
#             discriminated_real_images = discriminator.predict(real_images)
#             discriminated_fake_images = discriminator.predict(fake_images)
#             # train the discriminator
#             discriminator_loss = np.mean(discriminated_real_images) - np.mean(discriminated_fake_images)
#             discriminator.train_on_batch(real_images, np.ones((batch_size, 1)))
#             discriminator.train_on_batch(fake_images, np.zeros((batch_size, 1)))
#             # train the generator
#             generator_loss = -np.mean(discriminator.predict(fake_images))
#             gan.train_on_batch(latent_vectors, np.ones((batch_size, 1)))
#         # log loss values
#         d_loss_list.append(discriminator_loss)
#         g_loss_list.append(generator_loss)
#         logger.info("Epoch: {}/{}".format(str(epoch + 1), epochs))
#         logger.info("discriminator loss: {}".format(discriminator_loss))
#         logger.info("generator loss: {}".format(generator_loss))
#         # # sample images
#         # if epoch % sample_interval == 0 or epoch == epochs - 1:
#         #     assert save_array_as_image(fake_images, epoch, save_dir)
#         # save the model
#         if epoch % save_interval == 0 or epoch == epochs - 1:
#             generator.save(os.path.join(save_dir, "generator_model_{}.h5".format(epoch)))
#             discriminator.save(os.path.join(save_dir, "discriminator_model_{}.h5".format(epoch)))
#     return (d_loss_list, g_loss_list)

# def build_generator(input_shape:int) -> Model:
#     """
#     build generator
#     """
#     logger.info("building generator")
#     model = Sequential()
#     model.add(Dense(256, input_dim=100))
#     model.add(LeakyReLU(alpha=0.2))
#     model.add(BatchNormalization(momentum=0.8))
#     model.add(Dense(512))
#     model.add(LeakyReLU(alpha=0.2))
#     model.add(BatchNormalization(momentum=0.8))
#     model.add(Dense(1024))
#     model.add(LeakyReLU(alpha=0.2))
#     model.add(BatchNormalization(momentum=0.8))
#     model.add(Dense(np.prod(input_shape), activation='tanh'))
#     model.add(Reshape(input_shape))
#     model.summary()
#     return model

# def build_gan(generator:Model, discriminator:Model) -> Model:
#     """
#     build gan
#     """
#     logger.info("building gan")
#     # define gan
#     gan_input = Input(shape=(100,))
#     # connect gan input to generator's output
#     x = generator(gan_input)
#     # connect gan input to discriminator's output
#     gan_output = discriminator(x)
#     # define gan model
#     gan = Model(gan_input, gan_output)
#     return gan

# def train_gan(generator:Model, discriminator:Model, gan:Model, x_train:np.ndarray, y_train:np.ndarray, epochs:int, batch_size:int, sample_interval:int = 10, save_interval:int = 10, save_dir:str = None, loss = 'binary_crossentropy', optimizer = 'adam') -> tuple:
#     """
#     train gan
#     """
#     logger.info("training gan")
#     # compile models
#     generator.compile(loss=loss, optimizer=optimizer)
#     discriminator.compile(loss=loss, optimizer=optimizer)
#     gan.compile(loss=loss, optimizer=optimizer)
#     # train the model
#     d_loss_list = []
#     g_loss_list = []
#     if save_dir is None:
#         save_dir  = os.path.join(os.getcwd())
#     assert os.path.exists(save_dir)
#     for epoch in range(epochs):
#         for _ in range(batch_size):
#             # generate real images
#             real_images, _ = load_batch(x_train, y_train, batch_size)
#             # generate latent vectors
#             latent_vectors = np.random.normal(0, 1, size=(batch_size, 100))
#             # generate fake images
#             fake_images = generator.predict(latent_vectors)
#             # get discriminated real images
#             discriminated_real_images = discriminator.predict(real_images)
#             discriminated_fake_images = discriminator.predict(fake_images)
#             # train the discriminator
#             discriminator_loss = np.mean(discriminated_real_images) - np.mean(discriminated_fake_images)
#             discriminator.train_on_batch(real_images, np.ones((batch_size, 1)))
#             discriminator.train_on_batch(fake_images, np.zeros((batch_size, 1)))
#             # train the generator
#             generator_loss = -np.mean(discriminator.predict(fake_images))
#             gan.train_on_batch(latent_vectors, np.ones((batch_size, 1)))
#         # log loss values
#         d_loss_list.append(discriminator_loss)
#         g_loss_list.append(generator_loss)
#         logger.info("Epoch: {}/{}".format(str(epoch + 1), epochs))
#         logger.info("discriminator loss: {}".format(discriminator_loss))
#         logger.info("generator loss: {}".format(generator_loss))
#         # # sample images
#         # if epoch % sample_interval == 0 or epoch == epochs - 1:
#         #     assert save_array_as_image(fake_images, epoch, save_dir)
#         # save the model
#         if epoch % save_interval == 0 or epoch == epochs - 1:
#             generator.save(os.path.join(save_dir, "generator_model_{}.h5".format(epoch)))
#             discriminator.save(os.path.join(save_dir, "discriminator_model_{}.h5".format(epoch)))
#     return (d_loss_list, g_loss_list)

# def save_array_as_image(images:np.ndarray, epoch:int, save_dir:str) -> None:
#     """
#     save array as image
#     """
#     from PIL import Image
#     assert os.path.exists(save_dir)
#     # save the image
#     image = images[0] * 127.5 + 127.5
#     image = np.clip(image, 0, 255).astype('uint8')
#     image = Image.fromarray(image)
#     image.save(os.path.join(save_dir, "image_{}.png".format(epoch)))
#     assert os.path.exists(os.path.join(save_dir, "image_{}.png".format(epoch)))
#     return True


# if __name__ == '__main__':
#     import argparse
#     # inherit logging level from root logger if ther is one
#     # setup command line argument parser
#     parser = argparse.ArgumentParser(description='train gan')
#     parser.add_argument('--epochs', type=int, default=1, help='number of epochs')
#     parser.add_argument('--batch-size', type=int, default=64, help='batch size')
#     parser.add_argument('--sample-interval', type=int, default=10, help='sample interval')
#     parser.add_argument('--save-interval', type=int, default=10, help='save interval')
#     parser.add_argument('--save-dir', type=str, default=None, help='save directory')
#     parser.add_argument('--loss', type=str, default='binary_crossentropy', help='loss function')
#     parser.add_argument('--optimizer', type=str, default='adam', help='optimizer')
#     parser.add_argument('--verbose', type=int, default=logger.INFO, help='verbose')
#     parser.add_argument('--dataset', type=str, default='mnist', help='dataset')
#     # parse command line arguments
#     args = parser.parse_args()
#     epochs = args.epochs
#     batch_size = args.batch_size
#     sample_interval = args.sample_interval
#     save_interval = args.save_interval
#     save_dir = args.save_dir
#     loss = args.loss
#     optimizer = args.optimizer
#     verbose = args.verbose
#     dataset = args.dataset
#     logging.basicConfig(level=verbose, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
#     # load data
#     (x_train, y_train, x_test, y_test) = load_data(dataset)
#     # build model
#     input_shape = detect_image_size(x_train)
#     generator = build_generator(input_shape)
#     latent_dim = 100
#     discriminator = build_discriminator(latent_dim, input_shape)
#     gan = build_gan(generator, discriminator)
#     # train model
#     (d_losses, g_losses) = train_gan(discriminator=discriminator, generator=generator, gan=gan, x_train = x_train, y_train = y_train, epochs = epochs, batch_size = batch_size, sample_interval = sample_interval, save_interval = save_interval, save_dir = save_dir, loss = loss, optimizer = optimizer)

#     assert os.path.exists(os.path.join(os.getcwd(), "generator_model_10.h5"))
#     assert os.path.exists(os.path.join(os.getcwd(), "discriminator_model_10.h5"))
#     assert isinstance(d_losses, list)
#     assert isinstance(g_losses, list)
#     assert len(d_losses) == epochs
#     assert len(g_losses) == epochs
#     assert isinstance(d_losses[0], float)
#     assert isinstance(g_losses[0], float)
#     assert isinstance(d_losses[-1], float)
#     assert isinstance(g_losses[-1], float)
