import argparse
import os

import torch
import torchvision
import tqdm
from torch.utils.data import SubsetRandomSampler
import model.mobilenet_v2
import core.utils.hat as hat
import core.utils.mart as mart
import core.utils.trades as trades
import dataset.dataset
import model.chexnet
import model.resnet
from advattack import _pgd_whitebox
from scheduler import getCyclicSche

config = {
    'mesidorBin': {
        'dataName': 'mesidor',
        'trainCsvPath': "./medicalData/mesidor/trainbin.csv",
        'testCsvPath': "./medicalData/mesidor/testbin.csv",
        'imgPath': "./medicalData/mesidor/img/",
        'beta': {'pgdat': 0, 'trades': 12, 'mart': 6, 'hat': 12},
        'trainTrans': torchvision.transforms.Compose([torchvision.transforms.Resize((299, 299)),
                                                      torchvision.transforms.RandomRotation(20),
                                                      torchvision.transforms.RandomHorizontalFlip(),
                                                      torchvision.transforms.RandomVerticalFlip(),
                                                      torchvision.transforms.ToTensor()]),
        'valTrans': torchvision.transforms.Compose([torchvision.transforms.Resize((299, 299)),
                                                    torchvision.transforms.ToTensor()])},
    'mesidorMulti': {
        'dataName': 'mesidor',
        'trainCsvPath': "./medicalData/mesidor/train.csv",
        'testCsvPath': "./medicalData/mesidor/test.csv",
        'imgPath': "./medicalData/mesidor/img/",
        'beta': {'pgdat': 0, 'trades': 6, 'mart': 6, 'hat': 3},
        'trainTrans': torchvision.transforms.Compose([torchvision.transforms.Resize((299, 299)),
                                                      torchvision.transforms.RandomRotation(20),
                                                      torchvision.transforms.RandomHorizontalFlip(),
                                                      torchvision.transforms.RandomVerticalFlip(),
                                                      torchvision.transforms.ToTensor()]),
        'valTrans': torchvision.transforms.Compose([torchvision.transforms.Resize((299, 299)),
                                                    torchvision.transforms.ToTensor()])},

    'melBin': {
        'dataName': 'melanoma',
        'trainCsvPath': "./medicalData/mel2017/trainbin.csv",
        'testCsvPath': "./medicalData/mel2017/testbin.csv",
        'imgPath': "./medicalData/mel2017/img/",
        'beta': {'pgdat': 0, 'trades': 6, 'mart': 6, 'hat': 6},
        'trainTrans': torchvision.transforms.Compose([torchvision.transforms.Resize((256, 256)),
                                                      torchvision.transforms.CenterCrop((224, 224)),
                                                      torchvision.transforms.RandomRotation(20),
                                                      torchvision.transforms.RandomHorizontalFlip(),
                                                      torchvision.transforms.RandomVerticalFlip(),
                                                      torchvision.transforms.ToTensor()]),
        'valTrans': torchvision.transforms.Compose([torchvision.transforms.Resize((256, 256)),
                                                    torchvision.transforms.CenterCrop((224, 224)),
                                                    torchvision.transforms.ToTensor()])
    },

    'melMulti': {
        'dataName': 'melanoma',
        'trainCsvPath': "./medicalData/mel2017/train.csv",
        'testCsvPath': "./medicalData/mel2017/test.csv",
        'imgPath': "./medicalData/mel2017/img/",
        'beta': {'pgdat': 0, 'trades': 6, 'mart': 6, 'hat': 6},
        'trainTrans': torchvision.transforms.Compose([torchvision.transforms.Resize((256, 256)),
                                                      torchvision.transforms.CenterCrop((224, 224)),
                                                      torchvision.transforms.RandomRotation(20),
                                                      torchvision.transforms.RandomHorizontalFlip(),
                                                      torchvision.transforms.RandomVerticalFlip(),
                                                      torchvision.transforms.ToTensor()]),
        'valTrans': torchvision.transforms.Compose([torchvision.transforms.Resize((256, 256)),
                                                    torchvision.transforms.CenterCrop((224, 224)),
                                                    torchvision.transforms.ToTensor()])
    },

    'xrayBin': {
        'dataName': 'xray',
        'trainCsvPath': "./medicalData/xray/trainbin.csv",
        'testCsvPath': "./medicalData/xray/testbin.csv",
        'imgPath': "./medicalData/xray/img/",
        'beta': {'pgdat': 0, 'trades': 6, 'mart': 6, 'hat': 6},
        'trainTrans': torchvision.transforms.Compose([torchvision.transforms.RandomHorizontalFlip(),
                                                      torchvision.transforms.Resize((256, 256)),
                                                      torchvision.transforms.CenterCrop((224, 224)),
                                                      torchvision.transforms.ToTensor(),
                                                      torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                                                                       [0.229, 0.224, 0.225])]),
        'valTrans': torchvision.transforms.Compose([torchvision.transforms.Resize((256, 256)),
                                                    torchvision.transforms.CenterCrop((224, 224)),
                                                    torchvision.transforms.ToTensor(),
                                                    torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                                                                     [0.229, 0.224, 0.225])])
    }


}

modelConstructor = {
    'res18': model.resnet.ResNet18,
    'chexNet': model.chexnet.chexNet,
    'mv2': model.mobilenet_v2.mobilenet_v2
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--method', type=str)
    parser.add_argument('--model', type=str)
    parser.add_argument('--helperModel', type=str, default=None)
    parser.add_argument('--helperWeight', type=str, default=None)
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
    print('train par:')
    print(advP)
    helperModel = None
    # -----------------------load data--------------------------#
    arg = config[args.dataset]
    beta = arg['beta']
    print('Training dataset:', arg['dataName'])
    trainDataSet = dataset.dataset.medDataset(arg['trainCsvPath'], arg['imgPath'], arg['trainTrans'])
    traindataset_size = len(trainDataSet)
    print('Training class num:{} sample num:{}'.format(trainDataSet.class2num.keys().__len__(), traindataset_size))
    valDataset = dataset.dataset.medDataset(arg['testCsvPath'], arg['imgPath'], arg['valTrans'])
    valdataset_size = len(valDataset)
    print('Validating class num:{} sample num:{}'.format(valDataset.class2num.keys().__len__(), valdataset_size))
    train_loader = torch.utils.data.DataLoader(trainDataSet,
                                               batch_size=batchSize,
                                               sampler=torch.utils.data.WeightedRandomSampler(
                                                   trainDataSet.getSampleWeight(), len(trainDataSet), replacement=True),
                                               num_workers=8)
    validation_loader = torch.utils.data.DataLoader(valDataset,
                                                    batch_size=batchSize,
                                                    num_workers=8)
    # ----------------------------------------load helper model------------------------------------------#
    results = []
    modelPaths = []
    results.append('data:{} class:{}'.format(arg['dataName'], valDataset.class2num.keys().__len__()))
    if method != 'nat':
        results.append(method + ' beta ' + str(beta[method]) + ':')
    if 'hat' == method:
        print('using hat, load helper from:', args.helperWeight)
        helperModel = modelConstructor[args.helperModel].to(device)
        l = torch.load(args.helperWeight, map_location=device)
        helperModel.load_state_dict(l if not hasattr(l, 'state_dict') else l.state_dict())
        helperModel = torch.nn.DataParallel(helperModel, gpus, GPUIndex)
    else:
        helperModel = None
    bestAdvAcc = 0
    natAcc = 0
    bestAt = 1
    advAcc = dict()
    # -----------------------load model-------------------------#
    m = modelConstructor[args.model](trainDataSet.class2num.keys().__len__()).to(device)
    modelName = m.__class__.__name__
    m = torch.nn.DataParallel(m, gpus, GPUIndex)
    if method == 'nat':
        modelPath = os.path.join(saveDir,
                                 modelName + '_{}_{}_class{}_maxLR{}_weightDecay{}'.format(args.dataset,
                                                                                           method,
                                                                                           trainDataSet.class2num.keys().__len__(),
                                                                                           lr,
                                                                                           wd))
    else:
        modelPath = os.path.join(saveDir,
                                 modelName + '_{}_{}_beta{}_class{}_ADV{}_{}_{}_maxLR{}_weightDecay{}'.format(
                                     args.dataset,
                                     method,
                                     beta[method],
                                     trainDataSet.class2num.keys().__len__(),
                                     advP[3],
                                     advP[4],
                                     advP[5], lr,
                                     wd))
    modelPaths.append(modelPath)
    # -----------------------define optimizer and loss----------#
    lossF = torch.nn.CrossEntropyLoss().to(device)
    opt = torch.optim.SGD(m.parameters(), lr=lr, weight_decay=wd, momentum=0.9)
    sche = getCyclicSche(opt, epochNum, traindataset_size, batchSize, lr)
    print('using', opt.__class__.__name__)
    # ------------------------------train------------------------#
    for epoch in range(1, epochNum + 1):
        print('\nEpoch {} / {}'.format(epoch, epochNum))
        print('#------------training------------#')
        lossAccum = 0
        for i, (batch_x, batch_y) in tqdm.tqdm(enumerate(train_loader), total=len(train_loader)):
            m.train()
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            if method == 'trades':
                loss, batch_metrics = trades.trades_loss(m, batch_x, batch_y, opt, advP[2], advP[0],
                                                         advP[1], beta=beta[method])
            elif method == 'mart':
                loss, batch_metrics = mart.mart_loss(m, batch_x, batch_y, opt, advP[2], advP[0], advP[1],
                                                     beta=beta[method])
            elif method == 'hat':
                assert helperModel is not None
                loss, batch_metrics = hat.hat_loss(m, batch_x, batch_y, opt, advP[2], advP[0], advP[1],
                                                   hr_model=helperModel, beta=beta[method])
            elif method == 'pgdat':
                model.eval()
                advX = _pgd_whitebox(m, torch.nn.CrossEntropyLoss, batch_x.to(device), batch_y.to(device),
                                     advP[0], advP[1], advP[2])
                m.train()
                pred = m(advX)
                loss = lossF(pred, batch_y.to(device))
            elif method == 'nat':
                m.train()
                pred = m(batch_x.to(device))
                loss = lossF(pred, batch_y.to(device))
            else:
                print('no such method:', method)
                exit(1)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sche.step()
            lossAccum += loss.item()
        print('loss:{}'.format(lossAccum / len(train_loader)), end='\n')
        print('\n#------------validating------------#')
        with torch.no_grad():
            m.eval()
            predBatch = []
            advPredBatch = []
            targetBatch = []
            for i, (batch_x, batch_y) in tqdm.tqdm(enumerate(validation_loader),
                                                   total=len(validation_loader)):
                predBatch.append(m(batch_x.to(device)))
                advx = _pgd_whitebox(m, torch.nn.CrossEntropyLoss, batch_x.to(device), batch_y.to(device),
                                     advP[0],
                                     20, advP[2])
                advPredBatch.append(m(advx.to(device)))
                targetBatch.append(batch_y.unsqueeze(1))
            predBatch = torch.concat(predBatch).to(device)
            advPredBatch = torch.concat(advPredBatch).to(device)
            targetBatch = torch.concat(targetBatch).to(device).squeeze()
            loss = lossF(predBatch, targetBatch)
            acc = (predBatch.argmax(1) == targetBatch).to(torch.int).cpu().numpy().mean()
            advAcc = (advPredBatch.argmax(1) == targetBatch).to(torch.int).cpu().numpy().mean()
            print(' loss: {} acc: {} advAcc:{}'.format(loss.item(), acc, advAcc))
            if method != 'nat':
                if advAcc > bestAdvAcc:
                    print(
                        'updating best advAcc(previous:{}) and save model to {}'.format(bestAdvAcc, modelPath))
                    bestAdvAcc = advAcc
                    natAcc = acc
                    torch.save(m.module.state_dict(), modelPath)
                    bestAt = epoch
            else:
                if acc > natAcc:
                    print(
                        'updating best advAcc(previous:{}) and save model to {}'.format(bestAdvAcc, modelPath))
                    bestAdvAcc = advAcc
                    natAcc = acc
                    torch.save(m.module.state_dict(), modelPath)
                    bestAt = epoch
    r = 'eps:{} step:{} stepSize:{}'.format(advP[0], advP[1], advP[2]) + ' weight_decay:{}'.format(
        wd) + ' maxLR:{}'.format(lr) + ' best advAcc at:{}'.format(
        bestAt) + ' epochs with advAcc:{} natAcc:{}'.format(bestAdvAcc, natAcc)
    print(r)
    results.append(r)
    for r in results:
        print(r)
