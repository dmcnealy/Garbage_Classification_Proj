## Some of this code was borrowed from pytorch.org/tutorials

import numpy as np
import torch
import torchvision
import torchvision.datasets.folder as folder
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
from pathlib import Path


batch_sz = 1
current_dir = str(Path.cwd().parent)
print(current_dir)

## make train and test sets avail
transfrm = transforms.Compose([transforms.ToTensor()])
trainset = torchvision.datasets.ImageFolder(root=os.path.join(current_dir, "DATASET/TRAIN"), transform=transfrm)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_sz, shuffle=True, num_workers=2)
testset = torchvision.datasets.ImageFolder(root=os.path.join(current_dir, "DATASET/TEST"), transform=transfrm)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_sz, shuffle=True, num_workers=2)
classes = ('N', 'O', 'R')


# functions to show an image
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.preshape = nn.AdaptiveAvgPool3d((3,32,32))
        self.conv1 = nn.Conv2d(3, 6, 3)
        # self.pool1 = nn.MaxPool2d(2, 2)


        self.conv2 = nn.Conv2d(6, 10, 5)
        self.pool2 = nn.MaxPool2d(2, 2)


        self.conv3 = nn.Conv2d(10, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.preshape(x)
        x = self.pool1(F.relu(self.conv1(x)))
        print("#OK")
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x




def train_model():

    dtiter = iter(trainloader)
    images, labels = dtiter.__next__()
    # # show images
    # imshow(torchvision.utils.make_grid(images))
    # # print labels
    # print("label", classes[labels])

    net = Net()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(2):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    PATH = current_dir + '/DATASET/grbg_net.pth'
    torch.save(net.state_dict(), PATH)


def test_model():
    #WHICH DATASET (train now)
    dataiter = iter(trainloader)
    images, labels = dataiter.next()

    PATH = current_dir + '/DATASET/grbg_net.pth'
    net = Net()
    net.load_state_dict(torch.load(PATH))

    correct = 0
    total = 0
    with torch.no_grad():
        for data in trainloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 training set images: %d %%' % (
            100 * correct / total))

    # correct = 0
    # total = 0
    # with torch.no_grad():
    #     for data in testloader:
    #         images, labels = data
    #         outputs = net(images)
    #         _, predicted = torch.max(outputs.data, 1)
    #         total += labels.size(0)
    #         correct += (predicted == labels).sum().item()
    #
    # print('Accuracy of the network on the 10000 test images: %d %%' % (
    #         100 * correct / total))

    ## test indiv classes
    # class_correct = list(0. for i in range(10))
    # class_total = list(0. for i in range(10))
    # with torch.no_grad():
    #     for data in testloader:
    #         images, labels = data
    #         outputs = net(images)
    #         _, predicted = torch.max(outputs, 1)
    #         c = (predicted == labels).squeeze()
    #         for i in range(batch_sz):
    #             label = labels[i]
    #             class_correct[label] += c.item()
    #             class_total[label] += 1
    #
    # for i in range(len(classes)):
    #     print('Accuracy of %5s : %2d %%' % (
    #         classes[i], 100 * class_correct[i] / class_total[i]))



