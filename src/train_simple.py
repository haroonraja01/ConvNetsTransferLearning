import os
import time
import matplotlib
# import the necessary packages
from sklearn.metrics import classification_report
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
import torch.optim as optim
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets, models, transforms
import argparse
import torch
from simple_model import FirstModel
matplotlib.use("Agg")

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", type=str, required=False, help="path to output trained model")
ap.add_argument("-p", "--plot", type=str, required=False, help="path to output loss/accuracy plot")
args = vars(ap.parse_args())

data_dir = '/Users/haroonraja/Google Drive/Colab Notebooks/ConvNetsTransferLearning/data'
batch_size = 32
lr = 0.01
momentum = 0.9
num_epochs = 15

# set the device we will be using to train the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load data
transform = {
            'train': transforms.Compose([transforms.Resize(255),
                                         transforms.RandomRotation(30),
                                         transforms.RandomResizedCrop(224),
                                         transforms.RandomHorizontalFlip(),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.5, 0.5, 0.5],
                                                              [0.5, 0.5, 0.5])]),
            'val': transforms.Compose([transforms.Resize(255),
                                       transforms.CenterCrop(224),
                                       transforms.ToTensor()])
        }
# dataset = datasets.ImageFolder(data_dir, transform=transform['train'])
dataset = {}
dataloader = {}
for x in ['train', 'val']:
    dataset[x] = datasets.ImageFolder(os.path.join(data_dir, x), transform=transform[x])
    dataloader[x] = DataLoader(dataset[x], batch_size=batch_size, shuffle=True)

# Setting up model, optimizer, loss, and scheduler
model = FirstModel()
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
loss_function = nn.CrossEntropyLoss()
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
train_loss = []
val_loss = []
model.to(device)
for epoch in range(num_epochs):
    print('Epoch number: {}'.format(epoch))
    train_loss_curr = []
    val_loss_curr = []
    for learning_mode in ['train', 'val']:
        if learning_mode == 'train':
            model.train()
        else:
            model.eval()

        for i, data in enumerate(dataloader[learning_mode]):
            images, labels = data[0].to(device), data[1].to(device)
            # print(labels[0], np.shape(images))
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_function(outputs, labels)
            if learning_mode == 'train':
                train_loss_curr.append(loss.item())
            else:
                val_loss_curr.append(loss.item())

            if learning_mode == 'train':
                loss.backward()
                optimizer.step()
            # outputs = model(images)
            _, predictions = torch.max(outputs, 1)
            print(learning_mode + ' loss:', loss.item())
            # print(loss_function(outputs, labels), labels[0], preds)

            # imshow(images[0], normalize=False)
    train_loss.append(np.mean(train_loss_curr))
    val_loss.append(np.mean(val_loss_curr))
    scheduler.step()

# Plot loss
fig, ax = plt.subplots()
ax.plot(train_loss, label='Train loss')
ax.plot(val_loss, label='Val loss')
ax.set(xlabel='Epochs', ylabel='Average loss values',
       title='About as simple as it gets, folks')
legend = ax.legend(loc='upper right', shadow=True, fontsize='x-large')

fig.savefig("/Users/haroonraja/Google Drive/Colab Notebooks/ConvNetsTransferLearning/output/test.pdf")
plt.close(fig)
