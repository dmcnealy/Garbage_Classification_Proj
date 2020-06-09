## Some of this code was borrowed from pytorch.org/tutorials
## Github https://github.com/dmcnealy/Garbage_Classification_Proj.git

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
from pathlib import Path


batch_sz = 1

current_dir = str(Path.cwd())

## make train and test sets avail
transfrm = transforms.Compose([transforms.RandomCrop((500,500), pad_if_needed=True, padding_mode='constant'), transforms.ToTensor()])
trainset = torchvision.datasets.ImageFolder(root=os.path.join(current_dir, "DATASET_2CLASS/TRAIN"), transform=transfrm)
print(np.array(trainset).shape)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_sz, shuffle=True, num_workers=2)
testset = torchvision.datasets.ImageFolder(root=os.path.join(current_dir, "DATASET_2CLASS/TEST"), transform=transfrm)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_sz, shuffle=True, num_workers=2)
classes = ('O', 'R')
# classes = ('N', 'O', 'R')




# functions to show an image
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.preshape = nn.AdaptiveAvgPool3d((3,500,500))

        self.conv1 = nn.Conv2d(3, 6, 3)
        self.pool1 = nn.AvgPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 8, 3)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(8, 10, 3)
        self.conv4 = nn.Conv2d(10,16,3)

        self.fc1 = nn.Linear(784, 400)
        self.fc2 = nn.Linear(400, 100)
        self.fc3 = nn.Linear(100, 40)
        self.fc4 = nn.Linear(40, 3)

    def forward(self, x):
        x = self.preshape(x)

        x = self.pool1(F.relu(self.conv1(x)))

        x = self.pool2(F.relu(self.conv2(x)))

        x = torch.tanh(self.conv3(x))

        x = F.relu(self.conv4(x))

        x = x.view(-1, 784)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = self.fc4(x)

        return x




def train_model():

    print("Training with batch size ", batch_sz, ".")
    dtiter = iter(trainloader)
    images, labels = dtiter.__next__()
    # # show images
    # imshow(torchvision.utils.make_grid(images))
    # # print labels
    # print("label", classes[labels])

    net = Net()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters()) #lr=0.001, momentum=0.9)

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data

        # zero the para1meter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        print('Batch [%d] loss: %.3f' %
              (i + 1, loss.item()))

    PATH = current_dir + '/DATASET/grbg_net.pth'
    torch.save(net.state_dict(), PATH)


def test_model(mode="test"):

    theloader = testloader if (mode =="test") else trainloader

    PATH = current_dir + '/DATASET/grbg_net.pth'
    net = Net()
    net.load_state_dict(torch.load(PATH))

    # test indiv classes
    class_correct = list(0. for i in range(len(classes)))
    class_total = list(0. for i in range(len(classes)))

    class_misses = np.zeros((len(classes), len(classes)))
    with torch.no_grad():
        for data in theloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)

            for i in range(len(labels)):   #pred, lab in zip(predicted, labels):
                if(predicted[i] != labels[i]):
                    class_misses[labels[i]][predicted[i]] += 1
                else:
                    class_correct[labels[i]] += 1
                class_total[labels[i]] += 1

    correct = np.sum(np.array(class_correct))
    total = correct + np.sum(np.array(class_misses))

    if(mode == "train"):
        print('Accuracy of the network on the training images: %d %%' % (
            100 * correct / total))
    else:
        print('Accuracy of the network on the test images: %d %%' % (
                100 * correct / total))


    for i in range(len(classes)):
        print('Accuracy of %5s : %2d %%' % (
            classes[i], 100 * class_correct[i] / class_total[i]))


    print("\n                      ", classes)
    for i in range(len(classes)):
        print(classes[i], " misinterpreted as ", class_misses[i])





