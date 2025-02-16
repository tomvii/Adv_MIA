import torch
import numpy as np
from model import resnet


def getCyclicSche(optimizer, num_epochs, numSamples, batchSize, maxLR):
    """
    Initialize scheduler.
    """
    update_steps = int(np.floor(numSamples / batchSize) + 1)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=maxLR, pct_start=0.25,
                                                    steps_per_epoch=update_steps, epochs=int(num_epochs))
    return scheduler


if __name__ == '__main__':
    net = resnet.ResNet18(2)
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    epochNum = 50
    numSample = 1000
    batchSize = 16
    maxLR = 0.1
    sche = getCyclicSche(optimizer, epochNum, numSample, batchSize, maxLR)
    for epoch in range(1, epochNum + 1):
        print('epoch', epoch)
        for i in range(int(np.floor(numSample / batchSize) + 1)):
            print('current lr:{}'.format(sche.get_lr()[0]))
            optimizer.step()
            sche.step()
