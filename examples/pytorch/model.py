# """
# The script demonstrates a simple example of using ART with PyTorch. The example train a small model on the MNIST dataset
# and creates adversarial examples using the Fast Gradient Sign Method. Here we use the ART classifier to train the model,
# it would also be possible to provide a pretrained model to the ART classifier.
# The parameters are chosen for reduced computational requirements of the script and not optimised for accuracy.
# """
# import torch.nn as nn
# import torch.nn.functional as F
# import numpy as np

# from art.attacks.evasion import FastGradientMethod
# from art.estimators.classification import PyTorchClassifier
# from art.utils import load_mnist
# from argparse import Namespace
# import pickle
# import torch
# from tqdm import tqdm

# from deckard.base.utils import factory

# # Step 0: Define the neural network model, return logits instead of activation in forward method
# arch = {
#     "conv": (4, 10),
#     "lin": (100,),
#     "stride": 1,
#     "kernel_size": 5,
#     "pooling": (2, 2),
#     "padding": 0,
#     "input_channels": 1,
#     "input_shape": (1, 28, 28),
#     "num_classes": 10,
# }

# for param in arch:
#     locals()[param] = arch[param]


# init_params = {
#     "optimizer": {
#         "name": "torch.optim.Adam",
#         "lr": 0.001,
#     },
#     "loss": {
#         "name": "torch.nn.CrossEntropyLoss",
#     },
# }
# fit = {
#     "epochs": 10,
#     "batch_size": 256,
# }


# class SampleNet(nn.Module):
#     def __init__(self):
#         super(SampleNet, self).__init__()
#         print("conv[0] = ", conv[0])
#         self.conv_1 = nn.Conv2d(
#             in_channels=input_channels,
#             out_channels=conv[0],
#             kernel_size=kernel_size,
#             stride=stride,
#             padding=padding,
#         )
#         print("conv[1] = ", conv[1])
#         self.conv_2 = nn.Conv2d(
#             in_channels=conv[0],
#             out_channels=conv[1],
#             kernel_size=kernel_size,
#             stride=stride,
#             padding=padding,
#         )
#         print(
#             "conv[0]*conv[0]*conv[1] *input_channels = ",
#             conv[0] * conv[0] * conv[1] * input_channels,
#         )
#         print("lin[0] = ", lin[0])
#         self.fc_1 = nn.Linear(
#             in_features=conv[0] * conv[0] * conv[1] * input_channels,
#             out_features=lin[0],
#         )
#         self.fc_2 = nn.Linear(in_features=lin[0], out_features=num_classes)
#         # input("Press Enter to continue...")

#     def forward(self, x):
#         x = F.relu(self.conv_1(x))
#         x = F.max_pool2d(x, pooling[0], pooling[1])
#         x = F.relu(self.conv_2(x))
#         x = F.max_pool2d(x, pooling[0], pooling[1])
#         x = x.view(-1, conv[0] * conv[0] * conv[1] * input_channels)
#         x = F.relu(self.fc_1(x))
#         x = self.fc_2(x)
#         return x


# # Step 1: Load the MNIST dataset

# (x_train, y_train), (x_test, y_test), min_pixel_value, max_pixel_value = load_mnist()

# # Step 1a: Swap axes to PyTorch's NCHW format

# x_train = np.transpose(x_train, (0, 3, 1, 2)).astype(np.float32)
# x_test = np.transpose(x_test, (0, 3, 1, 2)).astype(np.float32)

# # Step 2: Create the model
# model = SampleNet()


# # init_param_dict = {
# #     "loss" : loss,
# #     "optimizer" : optimizer,
# # }


# # Step 3: Create the ART classifier
# classifier = PyTorchClassifier(
#     model=model,
#     optimizer=factory(
#         init_params["optimizer"].pop("name"),
#         params=model.parameters(),
#         **init_params["optimizer"],
#     ),
#     loss=factory(init_params["loss"].pop("name"), **init_params["loss"]),
#     clip_values=(min_pixel_value, max_pixel_value),
#     input_shape=input_shape,
#     nb_classes=num_classes,
# )
# # Step 4: Train the ART classifier
# epochs = fit.pop("epochs", 10)
# for tqdm in tqdm(range(epochs)):
#     classifier.fit(x_train, y_train, **fit, nb_epochs=1)

# train_loss = classifier.compute_loss(x_train, y_train)
# test_loss = classifier.compute_loss(x_test, y_test)
# print(f"Train loss: {np.mean(train_loss)}")
# print(f"Test loss: {np.mean(test_loss)}")
# input("Press Enter to continue...")
# # Step 5: Evaluate the ART classifier on benign test examples

# predictions = classifier.predict(x_test)
# accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(
#     y_test,
# )
# print("Accuracy on benign test examples: {}%".format(accuracy * 100))
# input("Press Enter to continue...")


# # Step 6: Generate adversarial test examples
# attack = FastGradientMethod(estimator=classifier, eps=0.2)
# x_test_adv = attack.generate(x=x_test)

# # Step 7: Evaluate the ART classifier on adversarial test examples

# predictions = classifier.predict(x_test_adv)
# accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(
#     y_test,
# )
# print("Accuracy on adversarial test examples: {}%".format(accuracy * 100))

# # Step 8: Save the pytorch classifier (stored in the ART.model attribute)
# torch.save(classifier.model, "classifier.pt")
# print("Saved classifier.pt")

# # Step 9: Save the data
# with open("control.pkl", "wb") as f:
#     ns = Namespace(X_train=x_train, y_train=y_train, X_test=x_test, y_test=y_test)
#     pickle.dump(ns, f)
# print("Saved control.pkl")

# # Step 10: Load the model
# assert isinstance(torch.load("classifier.pt"), SampleNet)
# print("Successfully loaded classifier.pt")
