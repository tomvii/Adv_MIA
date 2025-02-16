import argparse
import os.path

import numpy as np
import torch
import torchvision
import tqdm
from torch.utils.data import SubsetRandomSampler
import model.mobilenet_v2
import dataset.dataset
import model.chexnet
import model.resnet
from advattack import _pgd_whitebox
from scheduler import getCyclicSche

xrayArg = {
    'dataName': 'xray',
    'trainCsvPath': "./data/xray/trainMultiLabel_1w.csv",
    'testCsvPath': "./data/xray/testMultiLabel_1w.csv",
    'imgPath': "./data/xray/img/",
    'trainTrans': torchvision.transforms.Compose([torchvision.transforms.RandomHorizontalFlip(),
                                                  torchvision.transforms.Resize((256, 256)),
                                                  torchvision.transforms.CenterCrop((224, 224)),
                                                  torchvision.transforms.ToTensor()]),
    'valTrans': torchvision.transforms.Compose([torchvision.transforms.Resize((256, 256)),
                                                torchvision.transforms.CenterCrop((224, 224)),
                                                torchvision.transforms.ToTensor()])
}

modelConstructor = {
    'res18': model.resnet.ResNet18,
    'chexNet': model.chexnet.chexNet,
    'mv2': model.mobilenet_v2.mobilenet_v2
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str)
    parser.add_argument('--model', type=str)
    parser.add_argument('--eps', type=int)
    parser.add_argument('--epoch', type=int)
    parser.add_argument('--bs', type=int)
    parser.add_argument('--gpu', nargs='+')
    parser.add_argument('--wd', type=float)
    parser.add_argument('--lr', type=float)
    args = parser.parse_args()
    gpus = [int(s) for s in args.gpu]
    GPUIndex = gpus[0]
    torch.cuda.set_device(GPUIndex)
    print('use cuda', gpus, 'main card', GPUIndex)
    device = torch.device("cuda:{}".format(GPUIndex))
    saveDir = './clsCkpt/'  # ckpt save path
    os.makedirs(saveDir, exist_ok=True)
    batchSize = args.bs
    epochNum = args.epoch
    method = args.method
    lr = args.lr
    wd = args.wd
    trainAdvParam = {
        0: (0, 0, 0, '0-255', '0', '0-255'),
        1: (1 / 255, 10, 0.5 / 255, '1-255', '10', '05-255'),
        2: (2 / 255, 10, 0.5 / 255, '2-255', '10', '05-255'),
        4: (4 / 255, 10, 1 / 255, '4-255', '10', '1-255'),
        8: (8 / 255, 10, 2 / 255, '8-255', '10', '2-255')
    }
    advP = trainAdvParam[args.eps]

    # -----------------------load data--------------------------#
    arg = xrayArg
    print('Training dataset:', arg['dataName'])
    trainDataSet = dataset.dataset.multiLabelDataset(arg['trainCsvPath'], arg['imgPath'], arg['trainTrans'])
    traindataset_size = len(trainDataSet)
    print('Training class num:{} sample num:{}'.format(trainDataSet.classNum, traindataset_size))
    valDataset = dataset.dataset.multiLabelDataset(arg['testCsvPath'], arg['imgPath'], arg['valTrans'])
    valdataset_size = len(valDataset)
    print('Validating class num:{} sample num:{}'.format(valDataset.classNum, valdataset_size))
    train_loader = torch.utils.data.DataLoader(trainDataSet, batch_size=batchSize, num_workers=2)
    validation_loader = torch.utils.data.DataLoader(valDataset, batch_size=batchSize, num_workers=2)
    result = []
    bestNatAuc = 0
    bestClassAuc = dict()
    for k in valDataset.idx2class.keys():
        bestClassAuc[valDataset.idx2class[k]] = 0
    bestAt = 1
    # -----------------------load model-------------------------#
    m = modelConstructor[args.model](trainDataSet.classNum).to(device)
    modelName = m.__class__.__name__
    m = torch.nn.DataParallel(m, gpus, GPUIndex)
    if method == 'nat':
        modelPath = os.path.join(saveDir,
                                 modelName + '_{}_multilabel_{}_maxLR{}_weightDecay{}'.format(arg['dataName'], lr,
                                                                                              method,
                                                                                              wd))
    else:
        modelPath = os.path.join(saveDir,
                                 modelName + '_{}_multilabel_{}_adv{}_maxLR{}_weightDecay{}'.format(arg['dataName'],
                                                                                                    '_'.join(advP[3:]),
                                                                                                    lr,
                                                                                                    method,
                                                                                                    wd))
    # -----------------------define optimizer and loss----------#
    lossF = torch.nn.BCEWithLogitsLoss().to(device)
    opt = torch.optim.SGD(m.parameters(), lr, weight_decay=wd, momentum=0.9)
    sche = getCyclicSche(opt, epochNum, traindataset_size, batchSize, lr)
    print('using', opt.__class__.__name__)
    for epoch in range(1, epochNum + 1):
        print('\nEpoch {} / {}'.format(epoch, epochNum))
        print('#------------training------------#')
        lossAccum = 0
        for i, (batch_x, batch_y) in tqdm.tqdm(enumerate(train_loader), total=len(train_loader)):
            m.train()
            if method == 'pgdat':
                advx = _pgd_whitebox(m, torch.nn.BCEWithLogitsLoss, batch_x.to(device), batch_y.to(device),
                                     advP[0], advP[1], advP[2])
            else:
                advx = batch_x.to(device)
            m.train()
            pred = m(advx)
            loss = lossF(pred, batch_y.to(device))
            opt.zero_grad()
            loss.backward()
            opt.step()
            sche.step()
            lossAccum += loss.item()
        print('train loss:{}'.format(lossAccum / len(train_loader)), end='\n')
        print('\n#------------validating------------#')
        with torch.no_grad():
            m.eval()
            predBatch = []
            targetBatch = []
            for i, (batch_x, batch_y) in tqdm.tqdm(enumerate(validation_loader),
                                                   total=len(validation_loader)):
                if method == 'pgdat':
                    advx = _pgd_whitebox(m, torch.nn.BCEWithLogitsLoss, batch_x.to(device), batch_y.to(device),
                                         advP[0], 20, advP[2])
                    predBatch.append(m(advx.to(device)))
                else:
                    predBatch.append(m(batch_x.to(device)))
                targetBatch.append(batch_y.to(device))
            predBatch = torch.concat(predBatch, dim=0).to(device)
            targetBatch = torch.concat(targetBatch, dim=0).to(device)
            loss = lossF(predBatch, targetBatch)
            aucList = model.chexnet.compute_AUCs(targetBatch, predBatch)
            aucAvg = np.array(aucList).mean()
            print(' loss: {} aucAvg: {} bestauc: {}'.format(loss.item(), aucAvg, bestNatAuc))
            if aucAvg > bestNatAuc:
                print('updating best acc(previous:{})'.format(bestNatAuc))
                bestNatAuc = aucAvg
                for i, auc in enumerate(aucList):
                    bestClassAuc[valDataset.idx2class[i]] = auc
                torch.save(m.module.state_dict(), modelPath)
                bestAt = epoch
    r = 'wd:{}'.format(wd) + ' lr:{}'.format(lr) + ' bestAucAvg:{}'.format(
        bestNatAuc) + ' at epoch:{} '.format(bestAt) + str(bestClassAuc)
    print(r)
    result.append(r)
    result.append('\n')
    for r in result:
        print(r)
