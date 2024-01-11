import torch
import os
import numpy as np
import torch.nn as nn
from torch import optim
from torch.utils.data import random_split
from datetime import datetime

from train_loaddata import DataLoad

from resnet import ResNet

model_path = '../model/{0:%Y-%m-%d}/'.format(datetime.now())


def train(device, model, data_dir, epochs=200, batch_size=5, lr=0.00001):
    data = DataLoad(data_dir)

    if not os.path.exists(model_path):
        os.makedirs(model_path)

    net = model.to(device)

    val_length = int(len(data) / 9)
    train_length = len(data) - val_length
    train_data, val_data = random_split(dataset=data, lengths=[train_length, val_length],
                                        generator=torch.Generator())
    # print("train: ", len(train_data))
    # print("val: ", len(val_data))
    train_data = torch.utils.data.DataLoader(dataset=train_data,
                                             batch_size=batch_size,
                                             shuffle=True)
    val_data = torch.utils.data.DataLoader(dataset=val_data,
                                           batch_size=batch_size,
                                           shuffle=True)
    # print("train/batch: ", len(train_data))
    # print("val/batch: ", len(val_data))

    # weight_decay=1e-8, momentum=0.9
    optimizer = optim.RMSprop(net.parameters(), lr=lr,
                              weight_decay=1e-8, momentum=0.9)

    #     criterion = nn.BCEWithLogitsLoss()
    #     criterion = nn.SmoothL1Loss(reduction="sum")
    criterion = nn.L1Loss()

    best_loss = float("inf")
    best_val_loss = float("inf")

    for epoch in range(epochs):

        print("epoch: {}".format(epoch))
        net.train()

        for train, label in train_data:
            optimizer.zero_grad()

            train = train.to(device=device, dtype=torch.float32)
            label = label.to(device=device, dtype=torch.float32)

            # print(train.shape)

            pred = net(train)
            loss = criterion(pred, label)

            if loss < best_loss:
                best_loss = loss
                torch.save(net.state_dict(), model_path + "train_model.pth")

            loss.backward()
            optimizer.step()
        print('Loss/train', best_loss.item())

        val_loss_line = []

        net.eval()  # supposed to add this
        with torch.no_grad():
            for image, label in val_data:
                image = image.to(device=device, dtype=torch.float32)
                label = label.to(device=device, dtype=torch.float32)
                output = net(image)
                val_loss = criterion(output, label)
                val_loss_line.append(val_loss.item())

        if np.mean(val_loss_line) < best_val_loss:
            best_val_loss = np.mean(val_loss_line)
            torch.save(net.state_dict(), model_path + "val_model.pth")
        print("loss/val", best_val_loss)

    return 0


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(device)

    data_dir = '../data/train/train_shaped.csv'

    model = ResNet(input_size=1, num_classes=1)

    train(device, model, data_dir)
