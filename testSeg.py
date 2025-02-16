import argparse

import numpy as np
import torch.utils.data as data
import torchvision
import torchvision.transforms.functional as ttf
import tqdm

import dataset.dataset as dataset
import metric
import model.segnet as segnet
import model.unet as unet
from advattack import _pgd_whitebox
from model.resnet import *

config = {
    'mel': {
        'name': 'melanoma',
        'imgPath': "./data/mel2017/img/",
        'maskPath': "./data/mel2017/target/",
        'testCsvPath': "./data/mel2017/testSeg.csv",
        'imgTrans': torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                    torchvision.transforms.Resize((256, 256))]),
        'targetTrans': torchvision.transforms.Compose([torchvision.transforms.Resize((256, 256),  # must be NEAREST!!!!
                                                                                     interpolation=ttf.InterpolationMode.NEAREST)])
    },

    'xray': {
        'name': 'xray',
        'imgPath': "./data/covidSeg/image/",
        'maskPath': "./data/covidSeg/target/",
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


def _pgddice_whitebox(model,
                      X,
                      y,
                      epsilon,
                      num_steps,
                      step_size):
    X_pgd = Variable(X.data, requires_grad=True)
    random_noise = torch.FloatTensor(*X_pgd.shape).uniform_(-epsilon, epsilon).to(X_pgd.device)
    X_pgd = Variable(X_pgd.data + random_noise, requires_grad=True)

    for _ in range(num_steps):
        with torch.enable_grad():
            loss = metric.dice_loss(torch.nn.functional.softmax(model(X_pgd), dim=1), y)
        loss.backward()
        eta = step_size * X_pgd.grad.data.sign()
        X_pgd = Variable(X_pgd.data + eta, requires_grad=True)
        eta = torch.clamp(X_pgd.data - X.data, -epsilon, epsilon)
        X_pgd = Variable(X.data + eta, requires_grad=True)
        X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0), requires_grad=True)
    return X_pgd


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--targetModel', type=str)
    parser.add_argument('--targetWeight', type=str)
    parser.add_argument('--surrogateModel', type=str)
    parser.add_argument('--surrogateWeight', type=str)
    parser.add_argument('--bs', type=int)
    parser.add_argument('--gpu', nargs='+')
    args = parser.parse_args()
    gpus = [int(s) for s in args.gpu]
    mainGPU = gpus[0]
    if args.surrogateModel is None:
        args.surrogateModel = args.targetModel
    if args.surrogateWeight is None:
        args.surrogateWeight = args.targetWeight
    print('use cuda', gpus, 'main card', mainGPU)
    device = torch.device("cuda:{}".format(mainGPU) if torch.cuda.is_available() else "cpu")
    arg = config[args.dataset]
    batchSize = args.bs
    testAdvParam = [(1 / 255, 20, 0.5 / 255, '1-255', '20', '05-255'),
                    (2 / 255, 20, 0.5 / 255, '2-255', '20', '05-255'),
                    (4 / 255, 20, 1 / 255, '4-255', '20', '1-255'),
                    (8 / 255, 20, 2 / 255, '8-255', '20', '2-255')]
    lossList = ['ce', 'dice']
    # ------------------------load data---------------------------#
    testDataset = dataset.segDataset(arg['imgPath'],
                                     arg['maskPath'],
                                     arg['testCsvPath'],
                                     arg['imgTrans'],
                                     arg['targetTrans'])
    testLoader = data.DataLoader(testDataset, batchSize)
    classNum = testDataset.classNum
    print(arg['name'])
    print('test class to idx:{}'.format(testDataset.class2idx))
    class2I = {0: np.array([0, 0, 0]), 1: np.array([0, 0, 255])}
    #-------------------create surrogate model-------------#
    smodel = modelConstructor[args.surrogateModel](testDataset.channelNum, classNum).to(device)
    smodelName = smodel.__class__.__name__
    l = torch.load(args.surrogateWeight, map_location=device)
    if hasattr(l, 'state_dict'):
        smodel.load_state_dict(l.state_dict())
    else:
        smodel.load_state_dict(l)
    del l
    smodel = torch.nn.DataParallel(smodel, gpus, mainGPU)
    result = []
    result.append(args.targetWeight)
    # ------------------------create target model------------------------#
    model = modelConstructor[args.targetModel](testDataset.channelNum, classNum).to(device)
    modelName = model.__class__.__name__
    print(args.targetWeight)
    l = torch.load(args.targetWeight, map_location=device)
    if hasattr(l, 'state_dict'):
        model.load_state_dict(l.state_dict())
    else:
        model.load_state_dict(l)
    del l
    model = torch.nn.DataParallel(model, gpus, mainGPU)
    print('#-----validating-----#')
    lossF = torch.nn.CrossEntropyLoss().to(device)
    with torch.no_grad():
        model.eval()
        smodel.eval()
        valDice = 0
        iou = 0
        ceLoss = 0
        saveIdx = 0
        for i, (batchX, batchY) in tqdm.tqdm(enumerate(testLoader), total=len(testLoader)):
            pred = model(batchX.to(device))
            ceLoss += lossF(pred, batchY.to(device)).item() * len(batchX)
            dice = np.zeros((len(batchX),))
            miou = np.zeros((len(batchX),))
            iou += metric.IoU(pred, batchY) * len(batchX)
            pred = torch.nn.functional.one_hot(pred.argmax(dim=1), classNum).permute(0, 3, 1, 2).float()
            valDice += metric.multiclass_dice_coeff(pred, batchY.to(device)).item() * len(batchX)
            for j in range(len(batchX)):
                dice[j] = metric.multiclass_dice_coeff(pred[j:j + 1, :, :, :],
                                                       batchY[j:j + 1, :, :, :].to(device)).item()
                miou[j] = metric.IoU(pred[j:j + 1, :, :, :], batchY[j:j + 1, :, :, :])
                assert 0 <= miou[j] <= 1
                assert 0 <= dice[j] <= 1
            saveIdx += len(batchX)

        r = 'test dice:{}'.format(valDice / len(testDataset)) + ' test iou:{}'.format(
            iou / len(testDataset)) + ' test ceLoss:{}'.format(ceLoss / len(testDataset))
        print(r)
        result.append(r)
        for advLoss in lossList:
            result.append(advLoss)
            for p in testAdvParam:
                valAdvDice = 0
                valAdvIoU = 0
                for i, (batchX, batchY) in tqdm.tqdm(enumerate(testLoader), total=len(testLoader)):
                    if advLoss == 'ce':
                        advX = _pgd_whitebox(smodel,
                                             torch.nn.CrossEntropyLoss,
                                             batchX.to(device),
                                             batchY.to(device),
                                             p[0],
                                             p[1],
                                             p[2])
                    elif advLoss == 'dice':
                        advX = _pgddice_whitebox(smodel,
                                                 batchX.to(device),
                                                 batchY.to(device),
                                                 p[0],
                                                 p[1],
                                                 p[2])
                    else:
                        print('no', advLoss)
                        exit(1)
                    predAdv = model(advX.to(device))
                    valAdvIoU += metric.IoU(predAdv, batchY) * len(batchX)
                    predAdv = torch.nn.functional.one_hot(predAdv.argmax(dim=1), classNum).permute(0, 3, 1,
                                                                                                   2).float()
                    valAdvDice += metric.multiclass_dice_coeff(predAdv, batchY.to(device)).item() * len(batchX)
                valAdvDice /= len(testDataset)
                valAdvIoU /= len(testDataset)
                r = 'eps:{} step:{} stepSize:{} dice:{} IoU:{}'.format(p[3], p[4], p[5], valAdvDice,
                                                                       valAdvIoU)
                print(r)
                result.append(r)
        result.append('\n')
    print(arg['name'])
    print(arg['surrogateWeight'])
    for r in result:
        print(r)
