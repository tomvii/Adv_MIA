import csv
import os

import PIL.Image as pili
import numpy as np
import torch
import torchvision
from torch.utils.data import Dataset


class medDataset(Dataset):
    def __init__(self, csvF, dataRoot, trans):
        self.x = []
        self.y = []
        self.__len__ = 0
        self.class2num = dict()
        self.sampleWeight = []
        self.trans = trans
        with open(csvF) as f:
            fcsv = csv.reader(f)
            for row in fcsv:
                self.__len__ += 1
                print('\r{}'.format(self.__len__), end='')
                self.x.append(os.path.join(dataRoot, row[0]))
                self.y.append(int(row[1]))
                if int(row[1]) not in self.class2num.keys():
                    self.class2num[int(row[1])] = 0
                self.class2num[int(row[1])] += 1
        for i in range(len(self.y)):
            self.sampleWeight.append(self.__len__ / self.class2num[self.y[i]])

    def __getitem__(self, index):
        return self.trans(pili.open(self.x[index]).convert('RGB')), self.y[index]

    def __len__(self):
        return self.__len__

    def getSampleWeight(self):
        return torch.Tensor(self.sampleWeight)


class multiLabelDataset(Dataset):
    def __init__(self, csvPath, dataRoot, trans):
        headerNum = 1
        self.class2idx = dict()
        self.idx2class = dict()
        self.imgs = []
        self.target = []
        self.classNum = 0
        self.trans = trans
        self.__len__ = 0
        with open(csvPath, 'r') as f:
            fcsv = csv.reader(f)
            for i, row in enumerate(fcsv):
                if i < headerNum:
                    for c in row:
                        self.class2idx[c] = self.classNum
                        self.idx2class[self.classNum] = c
                        self.classNum += 1
                    continue
                self.__len__ += 1
                print('\r{}'.format(self.__len__), end='')
                self.imgs.append(os.path.join(dataRoot, row[0]))
                self.target.append(torch.Tensor([int(c) for c in row[1][1:]]))

    def __getitem__(self, index):
        return self.trans(pili.open(self.imgs[index]).convert('RGB')), self.target[index]

    def __len__(self):
        return self.__len__


def dealTargetBin(target):
    tarTensor = torchvision.transforms.ToTensor()(target)
    tarTensor = torchvision.transforms.Grayscale()(tarTensor).unsqueeze(0)
    tarTensor[tarTensor >= 0.5] = 1
    tarTensor[tarTensor < 0.5] = 0
    tarTensor = tarTensor.to(torch.int)
    return tarTensor


class segDataset(Dataset):
    def __init__(self, inputDir, targetDir, csvPath, imgTrans, targetTrans, class2idx=None, aug=None):
        self.x = []
        self.y = []
        self.length = 0
        self.class2idx = dict()
        if class2idx is not None:
            self.class2idx = class2idx
        self.classNum = 0
        self.aug = aug
        self.channelNum = None
        self.imgTrans = imgTrans
        self.targetTrans = targetTrans
        with open(csvPath, 'r') as f:
            csvReader = csv.reader(f)
            for i, row in enumerate(csvReader):
                target = pili.open(os.path.join(targetDir, row[1])).convert('L')
                if target is not None:
                    target = np.array(targetTrans(target))
                    classes = np.unique(target)
                    for intensity in classes:
                        if intensity not in self.class2idx.keys():
                            self.class2idx[intensity] = self.classNum
                            self.classNum += 1
                        target[target == intensity] = self.class2idx[intensity]
                    # now the intensity mask is class mask
                    self.length += 1
                    self.x.append(os.path.join(inputDir, row[0]))
                    self.y.append(os.path.join(targetDir, row[1]))
                    print('\r{}'.format(self.length), end='')
        self.classNum = len(self.class2idx.keys())
        print('\nimgs shape:{}\ttarget shape:{}'.format(len(self.x), len(self.y)))
        img = self.imgTrans(pili.open(self.x[0]).convert('RGB'))
        self.channelNum = img.shape[0]

    def __getitem__(self, index):
        img = self.imgTrans(pili.open(self.x[index]).convert('RGB'))
        target = self.targetTrans(pili.open(self.y[index]).convert('L'))
        target = np.array(target)
        classes = np.unique(target)
        for intensity in classes:
            target[target == intensity] = self.class2idx[intensity]
        target = torch.as_tensor(target, dtype=torch.long)
        target = torch.nn.functional.one_hot(target, self.classNum)
        target = torch.permute(target, (2, 0, 1)).to(torch.float)
        if self.aug is not None:
            return self.aug(img, target)
        return img, target

    def __len__(self):
        return self.length


if __name__ == '__main__':
    imgsPath = r'D:\medicalData\mel2017\img'
    targetsPath = r'D:\medicalData\mel2017\target'
    csvPath = r'D:\medicalData\mel2017\testSeg.csv'
    dataH = 256
    dataW = 256
    imgTrans = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                               torchvision.transforms.Resize((dataH, dataW))])
    targetTrans = torchvision.transforms.Compose([torchvision.transforms.Resize((dataH, dataW))])
    aug = torchvision.transforms.Compose([torchvision.transforms.RandomHorizontalFlip(),
                                          torchvision.transforms.RandomVerticalFlip()])
    dataset = segDataset(imgsPath, targetsPath, csvPath, imgTrans, targetTrans)
    dataloader = torch.utils.data.DataLoader(dataset, 8, True)
    for x, y in dataloader:
        print('x:{}  y:{}'.format(x.size(), y.size()))
        print(torch.unique(torchvision.transforms.RandomRotation(90)(y[0:1, :, :, :])))

