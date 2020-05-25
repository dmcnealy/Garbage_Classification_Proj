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
current_dir = str(Path.cwd())
print(current_dir)

## make train and test sets avail
transfrm = transforms.Compose([transforms.ToTensor()])
trainset = torchvision.datasets.ImageFolder(root=os.path.join(current_dir, "DATASET_2CLASS/TRAIN"), transform=transfrm)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_sz, shuffle=True, num_workers=2)
testset = torchvision.datasets.ImageFolder(root=os.path.join(current_dir, "DATASET_2CLASS/TEST"), transform=transfrm)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_sz, shuffle=True, num_workers=2)
classes = ('O', 'R')# classes = ('N', 'O', 'R')


# functions to show an image
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.preshape = nn.AdaptiveAvgPool3d((3,50,50))
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

        # print(x.shape)
        x = self.pool1(F.relu(self.conv1(x)))
  #      print(x.shape)

        x = self.pool2(F.relu(self.conv2(x)))
   #     print(x.shape)

        x = torch.tanh(self.conv3(x))
    #    print(x.shape)

        x = F.relu(self.conv4(x))
     #   print(x.shape)

      #  print("VIEWING")
        x = x.view(-1, 784)
       # print(x.shape)

        #print("LINEAR FROM 784 to 3")
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = self.fc4(x)
        # print("final: ", x.shape)

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

    for epoch in range(1):  # how many loops over dataset
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
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


def test_model(mode="test"):
    theloader = testloader
    if(mode == "train"):
        theloader = trainloader

    PATH = current_dir + '/DATASET/grbg_net.pth'
    net = Net()
    net.load_state_dict(torch.load(PATH))


    correct = 0
    total = 0

    # test indiv classes
    class_correct = list(0. for i in range(len(classes)))
    class_total = list(0. for i in range(len(classes)))

    class_misses = np.zeros((len(classes), len(classes)))
    with torch.no_grad():
        for data in theloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            c = (predicted == labels).squeeze()

            if (predicted[0] != labels):
                # print(predicted)
                # print(labels)
                # print(classes[predicted[0]], " predcicted. Actual : ", classes[labels])

                class_misses[labels][predicted[0]] += 1

            for i in range(batch_sz):
                label = labels[i]
                class_correct[label] += c.item()
                class_total[label] += 1

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





