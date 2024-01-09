import torch
from torch.utils.data import Dataset
from torch.utils.data import random_split
import pandas as pd
import numpy as np


class DataLoad(Dataset):
    def __init__(self, data_dir):
        self.data = pd.read_csv(data_dir)  # "../data/train/train_shaped.csv"
        self.data = self.data.values[0::, 0::]
        self.train = np.asarray(self.data[0::, 3:13], dtype=float)
        self.label = np.asarray(self.data[0::, 13:14], dtype=float)
        print(self.data.shape)
        print(self.train.shape)
        print(self.label.shape)
        # self.labels = self.data[::][13]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # print(index)
        train = np.squeeze(self.train[index])
        label = np.squeeze(self.label[index])
        # print(self.train[index], self.label[index])
        return train, label


if __name__ == "__main__":
    data_dir = "../data/train/train_shaped.csv"
    dataset = DataLoad(data_dir)
    print("数据个数: ", len(dataset))
    # data = torch.utils.data.DataLoader(dataset=dataset, batch_size=1, shuffle=True)
    val_length = int(len(dataset)/9)
    train_length = len(dataset) - val_length
    train_data, val_data = random_split(dataset=dataset, lengths=[train_length, val_length],
                                        generator=torch.Generator())
    train = torch.utils.data.DataLoader(dataset=train_data, batch_size=1, shuffle=True)
    val = torch.utils.data.DataLoader(dataset=val_data, batch_size=1, shuffle=True)
    print("train:", len(train))
    print("val: ", len(val))
    print("ok")

    # print(train_data.dtype)
    # print(data.dtype)
    # print(train[0])

    print(dataset[0])
    # train = torch.tensor(data=dataset, dtype=torch.float)
    # i=0
    print("start iter")
    for a, b in train:
        print(a.shape, b.shape)
        print(a, b)
    # a, b = next(iter(train))
    # print(a.shape, b.shape)
    # print(a, b)
