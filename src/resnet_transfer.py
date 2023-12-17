import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
from PIL import Image
from tempfile import TemporaryDirectory

cudnn.benchmark = True


class ResnetTransfer:
    def __int__(self, data_dir='../data/hymenoptera_data/', batch_size=32):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.lr = 0.01
        self.momentum = 0.9
        self.num_epochs = 5

    def transform_data(self):
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
            dataset[x] = datasets.ImageFolder(os.path.join(self.data_dir, x), transform=transform[x])
            dataloader[x] = torch.utils.data.DataLoader(dataset[x], batch_size=self.batch_size, shuffle=True)
        return dataloader

    def train_model(self, dataloader, loss_function, model, optimizer, scheduler):
        for epoch in range(self.num_epochs):
            print('Epoch number: {}'.format(epoch))
            for learning_mode in ['train', 'val']:
                if learning_mode == 'train':
                    model.train()
                else:
                    model.eval()

                for i, data in enumerate(dataloader[learning_mode]):
                    images, labels = data
                    # print(labels[0], np.shape(images))
                    optimizer.zero_grad()
                    outputs = model(images)
                    loss = loss_function(outputs, labels)
                    if learning_mode == 'train':
                        loss.backward()
                        optimizer.step()
                    # outputs = model(images)
                    _, predictions = torch.max(outputs, 1)
                    print(learning_mode + ' loss:', loss.item())
                    # print(loss_function(outputs, labels), labels[0], preds)

                    # imshow(images[0], normalize=False)
            scheduler.step()

    def optimizer_config(self, model, algo='sgd'):
        if algo == 'sgd':
            return optim.SGD(model.parameters(), lr=self.lr, momentum=self.momentum)


