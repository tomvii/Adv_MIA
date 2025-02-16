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

xrayArg = {'dataName': 'xray',
           'modelConstructor': model.resnet.ResNet18,
           'testCsvPath': "./data/xray/testMultiLabel_1w.csv",
           'imgPath': "./data/xray/img/",
           'valTrans': torchvision.transforms.Compose([torchvision.transforms.Resize((256, 256)),
                                                       torchvision.transforms.CenterCrop((224, 224)),
                                                       torchvision.transforms.ToTensor()])}

modelConstructor = {
    'res18': model.resnet.ResNet18,
    'chexNet': model.chexnet.chexNet,
    'mv2': model.mobilenet_v2.mobilenet_v2
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--targetModel', type=str)
    parser.add_argument('--targetWeight', type=str)
    parser.add_argument('--surrogateModel', type=str)
    parser.add_argument('--surrogateWeight', type=str)
    parser.add_argument('--bs', type=int)
    parser.add_argument('--gpu', nargs='+')
    args = parser.parse_args()
    gpus = [int(s) for s in args.gpu]
    GPUIndex = gpus[0]
    if args.surrogateModel is None:
        args.surrogateModel = args.targetModel
    if args.surrogateWeight is None:
        args.surrogateWeight = args.targetWeight
    results = []
    torch.cuda.set_device(GPUIndex)
    print('use cuda', gpus, 'main card', GPUIndex)
    device = torch.device("cuda:{}".format(GPUIndex) if torch.cuda.is_available() else "cpu")
    batchSize = args.bs
    advParam = [
        (1 / 255, 20, 0.5 / 255, '1-255', '20', '05-255', 0.0015),
        (2 / 255, 20, 0.5 / 255, '2-255', '20', '05-255', 0.0015),
        (4 / 255, 20, 1 / 255, '4-255', '20', '1-255', 0.003),
        (8 / 255, 20, 2 / 255, '8-255', '20', '2-255', 0.003)
    ]
    # -----------------------load data--------------------------#
    arg = xrayArg
    print('dataset:', arg['dataName'])
    valDataset = dataset.dataset.multiLabelDataset(arg['testCsvPath'], arg['imgPath'], arg['valTrans'])
    valdataset_size = len(valDataset)
    print('Validating class num:{} sample num:{}'.format(valDataset.classNum, valdataset_size))
    validation_loader = torch.utils.data.DataLoader(valDataset, batch_size=batchSize, num_workers=2)
    # ----------------------------------------load surrogate model------------------------------------------#
    sm = modelConstructor[args.surrogateModel](valDataset.classNum).to(device)
    loaded = torch.load(args.surrogateWeight, map_location=device)
    sm.load_state_dict(loaded)
    sm = torch.nn.DataParallel(sm, gpus, GPUIndex)
    # -----------------------load target model-------------------------#
    m = modelConstructor[args.targetModel](valDataset.classNum).to(device)
    loaded = torch.load(args.targetWeight, map_location=device)
    m.load_state_dict(loaded)
    m = torch.nn.DataParallel(m, gpus, GPUIndex)
    with torch.no_grad():
        m.eval()
        predBatch = []
        targetBatch = []
        for i, (batch_x, batch_y) in tqdm.tqdm(enumerate(validation_loader),
                                               total=len(validation_loader)):
            predBatch.append(torch.sigmoid(m(batch_x.to(device))))
            targetBatch.append(batch_y.to(device))
        predBatch = torch.concat(predBatch, dim=0).to(device)
        targetBatch = torch.concat(targetBatch, dim=0).to(device)
        aucList = model.chexnet.compute_AUCs(targetBatch, predBatch)
        aucAvg = np.array(aucList).mean()
        r = 'aucAvg: {}'.format(aucAvg)
        print(r)
        results.append(r)
    print('\n#------------adv validating------------#')
    for advP in advParam:
        for p in advParam:
            print('adv param:', p)
            with torch.no_grad():
                sm.eval()
                m.eval()
                predBatch = []
                targetBatch = []
                for i, (batch_x, batch_y) in tqdm.tqdm(enumerate(validation_loader), total=len(validation_loader)):
                    advx = _pgd_whitebox(sm, torch.nn.BCEWithLogitsLoss, batch_x.to(device), batch_y.to(device), p[0],
                                         p[1], p[2])
                    predBatch.append(torch.sigmoid(m(advx.to(device))))
                    targetBatch.append(batch_y.to(device))
                predBatch = torch.concat(predBatch, dim=0).to(device)
                targetBatch = torch.concat(targetBatch, dim=0).to(device)
                aucList = model.chexnet.compute_AUCs(targetBatch, predBatch)
                class2Auc = dict()
                for i, auc in enumerate(aucList):
                    class2Auc[valDataset.idx2class[i]] = auc
                aucAvg = np.array(aucList).mean()
                r = str(p[3:]) + 'aucAvg: {} auc:{}'.format(aucAvg, class2Auc)
                print(r)
                results.append(r)
    print('dataset:', arg['dataName'])
    print('Validating class num:{} sample num:{}'.format(valDataset.classNum, valdataset_size))
    for r in results:
        print(r)
