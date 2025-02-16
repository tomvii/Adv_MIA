import argparse
import time

import torch
import torchvision
import model.unet as unet
import model.segnet as segnet
import dataset.dataset as dataset
import torch.utils.data as data
import os
import tqdm
import torchvision.transforms.functional as ttf
import metric
from advattack import _pgd_whitebox
import numpy as np

config = {
    'mel': {
        'name': 'melanoma',
        'imgPath': "./data/mel2017/img/",
        'maskPath': "./data/mel2017/target/",
        'trainCsvPath': "./data/mel2017/trainSeg.csv",
        'testCsvPath': "./data/mel2017/testSeg.csv",
        'imgTrans': torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                    torchvision.transforms.Resize((256, 256))]),
        'targetTrans': torchvision.transforms.Compose([torchvision.transforms.Resize((256, 256),  # must be NEAREST!!!!
                                                                                     interpolation=ttf.InterpolationMode.NEAREST)])
    },

    'xray': {
        'name': 'xraySeg',
        'imgPath': "./data/covidSeg/image/",
        'maskPath': "./data/covidSeg/target/",
        'trainCsvPath': "./data/covidSeg/trainSeg.csv",
        'testCsvPath': "./data/covidSeg/testSeg.csv",
        'imgTrans': torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                    torchvision.transforms.Resize((256, 256))]),
        'targetTrans': torchvision.transforms.Compose([torchvision.transforms.Resize((256, 256),  # must be NEAREST!!!!
                                                                                     interpolation=ttf.InterpolationMode.NEAREST)])
    }
}

modelConstructor = {
    'unet': unet.UNet,
    'segNet': segnet.SegNet,
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--method', type=str)
    parser.add_argument('--eps', type=int)
    parser.add_argument('--model', type=str)
    parser.add_argument('--epoch', type=int)
    parser.add_argument('--bs', type=int)
    parser.add_argument('--gpu', nargs='+')
    parser.add_argument('--wd', type=float)
    parser.add_argument('--lr', type=float)
    args = parser.parse_args()
    gpus = [int(s) for s in args.gpu]
    mainGPU = gpus[0]
    print('use cuda', gpus, 'main card', mainGPU)
    device = torch.device("cuda:{}".format(mainGPU) if torch.cuda.is_available() else "cpu")
    config = config[args.dataset]
    batchSize = args.bs
    saveDir = r'./segckpt'
    epochNum = args.epoch
    lr = args.lr
    wd = args.wd
    method = args.method
    trainAdvParam = {
        0: (0, 0, 0, '0', '0', '0'),
        1: (1 / 255, 10, 0.5 / 255, '1-255', '10', '05-255'),
        2: (2 / 255, 10, 0.5 / 255, '2-255', '10', '05-255'),
        4: (4 / 255, 10, 1 / 255, '4-255', '10', '1-255'),
        8: (8 / 255, 10, 2 / 255, '8-255', '10', '2-255')
    }
    advP = trainAdvParam[args.eps]
    os.makedirs(saveDir, exist_ok=True)

    # ------------------------load data---------------------------#
    trainDataset = dataset.segDataset(config['imgPath'],
                                      config['maskPath'],
                                      config['trainCsvPath'],
                                      config['imgTrans'],
                                      config['targetTrans'])
    testDataset = dataset.segDataset(config['imgPath'],
                                     config['maskPath'],
                                     config['testCsvPath'],
                                     config['imgTrans'],
                                     config['targetTrans'],
                                     trainDataset.class2idx)
    trainLoader = data.DataLoader(trainDataset, batchSize, shuffle=True)
    testLoader = data.DataLoader(testDataset, batchSize)
    classNum = trainDataset.classNum
    print('train class to idx:{}'.format(trainDataset.class2idx))
    print('test class to idx:{}'.format(testDataset.class2idx))
    result = []
    print(method + ':')
    assert method == 'nat' or method == 'pgdat'
    result.append(method + ':')
    if method == 'pgdat':
        result.append('_'.join(advP[3:]))
    # ------------------------create model------------------------#
    model = modelConstructor[args.model](trainDataset.channelNum, classNum).to(device)
    modelName = model.__class__.__name__
    model = torch.nn.DataParallel(model, gpus, mainGPU)
    # ------------------------define loss------------------------#
    lossF = torch.nn.CrossEntropyLoss().to(device)
    # ------------------------define optimizer-------------------------#
    opt = torch.optim.Adam(model.parameters(), lr, weight_decay=wd)
    result.append('wd:{} lr:{}'.format(wd, lr))
    if method == 'nat':
        modelPath = os.path.join(saveDir, modelName + '_{}_NAT_{}_{}'.format(args.dataset,
                                                                              lossF.__class__.__name__,
                                                                              opt.__class__.__name__))
    else:
        modelPath = os.path.join(saveDir, modelName + '_{}_PGDAT_{}_{}_eps{}_step{}_stepSize{}'.format(args.dataset,
                                                                                                        lossF.__class__.__name__,
                                                                                                        opt.__class__.__name__,
                                                                                                        advP[
                                                                                                            3],
                                                                                                        advP[
                                                                                                            4],
                                                                                                        advP[
                                                                                                            5]))
    print(modelPath)
    bestDice = 0
    bestAt = 0
    bestAdvDice = 0
    totalTrainTime = 0
    #----------------train---------------#
    for epoch in range(1, epochNum + 1):
        print('Epoch', epoch)
        trainLoss = 0
        print('#-----training-----#')
        epochTime = 0
        for i, (batchX, batchY) in tqdm.tqdm(enumerate(trainLoader), total=len(trainLoader)):
            t1 = time.time()
            model.train()
            if method == 'pgdat':
                model.eval()
                batchX = _pgd_whitebox(model,
                                       torch.nn.CrossEntropyLoss,
                                       batchX.to(device),
                                       batchY.to(device),
                                       advP[0],
                                       advP[1],
                                       advP[2])
                model.train()
            opt.zero_grad()
            pred = model(batchX.to(device))
            loss = lossF(pred, batchY.to(device))
            loss.backward()
            opt.step()
            trainLoss += loss.item() * len(batchX)
            epochTime += time.time() - t1
        print('train loss:{} time:{}s'.format(trainLoss / len(trainDataset), epochTime))
        totalTrainTime += epochTime
        # -----------------------------------test---------------------------#
        print('#-----validating-----#')
        with torch.no_grad():
            model.eval()
            valLoss = 0
            valDice = 0
            valIoU = 0
            valAdvLoss = 0
            valAdvDice = 0
            valAdvIoU = 0
            for i, (batchX, batchY) in tqdm.tqdm(enumerate(testLoader), total=len(testLoader)):
                pred = model(batchX.to(device))
                loss = lossF(pred, batchY.to(device))
                valLoss += loss.item() * len(batchX)
                valIoU += np.mean(metric.IoU(pred, batchY)) * len(batchX)
                pred = torch.nn.functional.one_hot(pred.argmax(dim=1), classNum).permute(0, 3, 1, 2).float()
                valDice += metric.dice_coeff(pred, batchY.to(device)).item() * len(batchX)
                if method == 'pgdat':
                    advX = _pgd_whitebox(model,
                                         torch.nn.CrossEntropyLoss,
                                         batchX.to(device),
                                         batchY.to(device),
                                         advP[0],
                                         20,
                                         advP[2])
                    predAdv = model(advX.to(device))
                    valAdvLoss += lossF(predAdv, batchY.to(device)).item() * len(batchX)
                    valAdvIoU += np.mean(metric.IoU(predAdv, batchY)) * len(batchX)
                    predAdv = torch.nn.functional.one_hot(predAdv.argmax(dim=1), classNum).permute(0, 3, 1,
                                                                                                   2).float()
                    valAdvDice += metric.dice_coeff(predAdv, batchY.to(device)).item() * len(batchX)
            valLoss /= len(testDataset)
            valDice /= len(testDataset)
            valIoU /= len(testDataset)
            r = 'val loss:{} val dice:{} val IoU:{}'.format(valLoss, valDice, valIoU)
            if method == 'pgdat':
                valAdvLoss /= len(testDataset)
                valAdvDice /= len(testDataset)
                valAdvIoU /= len(testDataset)
                r += ' val adv loss:{} val adv dice:{} val adv IoU:{}'.format(valAdvLoss, valAdvDice, valAdvIoU)
            print(r)
            if method == 'pgdat':
                if valAdvDice > bestAdvDice:
                    print('update best adv dice:', valAdvDice, 'previoius:', bestAdvDice, 'save to:',
                          modelPath)
                    bestAdvDice = valAdvDice
                    bestDice = valDice
                    bestAt = epoch
                    torch.save(model.module.state_dict(), modelPath)
            else:
                if valDice > bestDice:
                    print('update best dice:', valDice, 'previoius:', bestDice, 'save to:',
                          modelPath)
                    bestDice = valDice
                    bestAt = epoch
                    torch.save(model.module.state_dict(), modelPath)

    if method == 'nat':
        r = 'Best validation dice:{} at epoch {}'.format(bestDice, bestAt)
    else:
        r = 'Best validation adv dice:{} and nat dice:{} at epoch {}'.format(bestAdvDice, bestDice, bestAt)
    print(r)
    result.append(r)
    for r in result:
        print(r)
