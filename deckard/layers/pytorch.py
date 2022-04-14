# # Use pytorch to train an image recognition model on mnist

# # import pytorch
# import pytorch as torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# from torch.autograd import Variable
# from torchvision import datasets, transforms


# # import the data
# train_loader = torch.utils.data.DataLoader(
#     datasets.MNIST('../data', train=True, download=True,
#                      transform=transforms.Compose([
#                             transforms.ToTensor(),
#                             transforms.Normalize((0.1307,), (0.3081,))
#                         ])))
# test_loader = torch.utils.data.DataLoader(
#     datasets.MNIST('../data', train=False, transform=transforms.Compose([
#                             transforms.ToTensor(),
#                             transforms.Normalize((0.1307,), (0.3081,))
#     ])))
    


# # define the model
# class Torch(nn.Module):
#     def __init__(self):
#         super(Torch, self).__init__()
#         self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
#         self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
#         self.conv2_drop = nn.Dropout2d()
#         self.fc1 = nn.Linear(320, 50)
#         self.fc2 = nn.Linear(50, 10)


#     def forward(self, x):
#         x = F.relu(F.max_pool2d(self.conv1(x), 2))
#         x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
#         x = x.view(-1, 320)
#         x = F.relu(self.fc1(x))
#         x = F.dropout(x, training=self.training)
#         x = self.fc2(x)
#         return F.log_softmax(x)
    
#     def predict(self, x):
#         return self.forward(x)
    
#     def backward(self, x, y):
#         return self.loss(x, y)
    
#     def loss(self, x, y):
#         return F.nll_loss(self.forward(x), y)

# # train the model
# def train(model, train_loader, optimizer, epoch):
#     model.train()
#     for batch_idx, (data, target) in enumerate(train_loader):
#         data, target = Variable(data), Variable(target)
#         optimizer.zero_grad()
#         output = model(data)
#         loss = F.nll_loss(output, target)
#         loss.backward()
#         optimizer.step()
#         if batch_idx % 10 == 0:
#             print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
#                 epoch, batch_idx * len(data), len(train_loader.dataset),
#                 100. * batch_idx / len(train_loader), loss.data[0]))

# # test the model
# def test(model, test_loader):
#     model.eval()
#     test_loss = 0
#     correct = 0
#     for data, target in test_loader:
#         data, target = Variable(data, volatile=True), Variable(target)
#         output = model(data)
#         test_loss += F.nll_loss(output, target, size_average=False).data[0] # sum up batch loss
#         pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
#         correct += pred.eq(target.data.view_as(pred)).cpu().sum()

#     test_loss /= len(test_loader.dataset)
#     print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
#         test_loss, correct, len(test_loader.dataset),
#         100. * correct / len(test_loader.dataset)))

# if __name__ == 'main':
#     model = Torch()
#     optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
#     for epoch in range(1, 2):
#         train(model, train_loader, optimizer, epoch)
#         test(model, test_loader)