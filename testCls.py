import argparse
import time

import torch
import torchvision
import tqdm
from torch.utils.data import SubsetRandomSampler

import dataset.dataset
import metric
import model.chexnet
import model.resnet
import model.mobilenet_v2
import simba.simba as ss
from advattack import _pgd_whitebox, _fgsm_whitebox, _cw_whitebox
from autoattack import AutoAttack

config = {

    'mesidorBin': {'dataName': 'mesidorBin',
                   'testCsvPath': "./data/mesidor/testbin.csv",
                   'imgPath': "./data/mesidor/img/",
                   'imgSize': 299,
                   'valTrans': torchvision.transforms.Compose([torchvision.transforms.Resize((299, 299)),
                                                               torchvision.transforms.ToTensor()])},

    'mesidorMul': {'dataName': 'mesidorMul',
                   'testCsvPath': "./data/mesidor/test.csv",
                   'imgPath': "./data/mesidor/img/",
                   'imgSize': 299,
                   'valTrans': torchvision.transforms.Compose([torchvision.transforms.Resize((299, 299)),
                                                               torchvision.transforms.ToTensor()])},

    'melBin': {'dataName': 'melanomaBin',
               'testCsvPath': "./data/mel2017/testbin.csv",
               'imgPath': "./data/mel2017/img/",
               'imgSize': 224,
               'valTrans': torchvision.transforms.Compose([torchvision.transforms.Resize((256, 256)),
                                                           torchvision.transforms.CenterCrop((224, 224)),
                                                           torchvision.transforms.ToTensor()])},

    'melMul': {'dataName': 'melanomaMul',
               'testCsvPath': "./data/mel2017/test.csv",
               'imgPath': "./data/mel2017/img/",
               'imgSize': 224,
               'valTrans': torchvision.transforms.Compose([torchvision.transforms.Resize((256, 256)),
                                                           torchvision.transforms.CenterCrop((224, 224)),
                                                           torchvision.transforms.ToTensor()])},
    'xrayBin': {
        'dataName': 'xray',
        'testCsvPath': "./data/xray/testbin.csv",
        'imgPath': "./data/xray/img/",
        'imgSize': 224,
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
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--targetModel', type=str, required=True)
    parser.add_argument('--targetWeight', type=str, required=True)
    parser.add_argument('--surrogateModel', type=str)
    parser.add_argument('--surrogateWeight', type=str)
    parser.add_argument('--bs', type=int, required=True)
    parser.add_argument('--gpu', nargs='+', required=True)
    args = parser.parse_args()
    gpus = [int(s) for s in args.gpu]
    GPUIndex = gpus[0]
    torch.cuda.set_device(GPUIndex)
    print('use cuda', gpus, 'main card', GPUIndex)
    device = torch.device("cuda:{}".format(GPUIndex) if torch.cuda.is_available() else "cpu")
    batchSize = args.bs
    if args.surrogateModel is None:
        args.surrogateModel = args.targetModel
    if args.surrogateWeight is None:
        args.surrogateWeight = args.targetWeight
    advParam = [
        (1 / 255, 20, 0.5 / 255, '1-255', '20', '05-255', 0.0015),
        (2 / 255, 20, 0.5 / 255, '2-255', '20', '05-255', 0.0015),
        (4 / 255, 20, 1 / 255, '4-255', '20', '1-255', 0.003),
        (8 / 255, 20, 2 / 255, '8-255', '20', '2-255', 0.003)
    ]
    attackMethod = ['pgd', 'aa', 'fgsm', 'cw', 'square', 'simba']

    # -----------------------load data--------------------------#
    arg = config[args.dataset]
    results = []
    print('dataset:', arg['dataName'])
    valDataset = dataset.dataset.medDataset(arg['testCsvPath'], arg['imgPath'], arg['valTrans'])
    valdataset_size = len(valDataset)
    print('Validating class num:{} sample num:{}'.format(valDataset.class2num.keys().__len__(), valdataset_size))
    validation_loader = torch.utils.data.DataLoader(valDataset,
                                                    batch_size=batchSize,
                                                    num_workers=2)
    # ----------------------------------------load surrogate model------------------------------------------#
    results.append('data:{} class:{}'.format(arg['dataName'], valDataset.class2num.keys().__len__()))
    sm = modelConstructor[args.surrogateModel](valDataset.class2num.keys().__len__()).to(device)
    loaded = torch.load(args.surrogateWeight, map_location=device)
    sm.load_state_dict(loaded)
    sm = torch.nn.DataParallel(sm, gpus, GPUIndex)
    # -----------------------load target model-------------------------#
    m = modelConstructor[args.targetModel](valDataset.class2num.keys().__len__()).to(device)
    loaded = torch.load(args.targetWeight, map_location=device)
    m.load_state_dict(loaded)
    m = torch.nn.DataParallel(m, gpus, GPUIndex)
    print(args.targetWeight)
    results.append(args.targetWeight)
    with torch.no_grad():
        print('\n#------------nat validating------------#')
        m.eval()
        predBatch = []
        targetBatch = []
        for batch_x, batch_y in validation_loader:
            predBatch.append(m(batch_x.to(device)))
            targetBatch.append(batch_y.unsqueeze(1))
        predBatch = torch.concat(predBatch).cpu().numpy()
        targetBatch = torch.concat(targetBatch).squeeze().cpu().numpy()
        acc, recall, prec, f1 = metric.multiMetric(predBatch, targetBatch, 1)
        r = 'nat acc: {} recall:{} prec:{} f1:{}'.format(acc, recall.mean(), prec.mean(), f1.mean())
        print(r)
        results.append(r)
        for attm in attackMethod:
            print(attm + ':')
            results.append(attm + ':')
            for p in advParam:
                with torch.no_grad():
                    sm.eval()
                    m.eval()
                    if attm == 'square':
                        aa = AutoAttack(sm, valDataset.class2num.keys().__len__(),
                                        norm='Linf', eps=p[0], version='standard', verbose=False)
                    if attm == 'aa':
                        aa = AutoAttack(sm, valDataset.class2num.keys().__len__(),
                                        norm='Linf', eps=p[0], version='standard', verbose=False)
                    if attm == 'simba':
                        simbaIter = 5000
                        ssAttack = ss.SimBA(sm, arg['dataName'], arg['imgSize'])
                    eps = p[0]
                    step = p[1]
                    stepSize = p[2]
                    predBatch = []
                    targetBatch = []
                    totalTime = 0
                    for batch_x, batch_y in tqdm.tqdm(validation_loader, total=len(validation_loader)):
                        if attm == 'square':
                            t1 = time.time()
                            batch_x = aa.square.perturb(batch_x.to(device), batch_y.to(device))
                            totalTime += time.time() - t1
                            predBatch.append(m(batch_x.to(device)))
                            targetBatch.append(batch_y.unsqueeze(1))
                        elif attm == 'fgsm':
                            t1 = time.time()
                            batch_x = _fgsm_whitebox(sm, batch_x.to(device), batch_y.to(device), p[0])
                            totalTime += time.time() - t1
                            predBatch.append(m(batch_x.to(device)))
                            targetBatch.append(batch_y.unsqueeze(1))
                        elif attm == 'cw':
                            t1 = time.time()
                            batch_x = _cw_whitebox(sm, batch_x.to(device), batch_y.to(device),
                                                   p[0],
                                                   30, p[6], valDataset.class2num.keys().__len__())
                            totalTime += time.time() - t1
                            step = 30
                            stepSize = p[6]
                            predBatch.append(m(batch_x.to(device)))
                            targetBatch.append(batch_y.unsqueeze(1))
                        elif attm == 'aa':
                            t1 = time.time()
                            batch_x = aa.run_standard_evaluation(batch_x, batch_y,
                                                                 bs=validation_loader.batch_size)
                            totalTime += time.time() - t1
                            predBatch.append(m(batch_x.to(device)))
                            targetBatch.append(batch_y.unsqueeze(1))
                        elif attm == 'pgd':
                            t1 = time.time()
                            batch_x = _pgd_whitebox(sm, torch.nn.CrossEntropyLoss, batch_x.to(device),
                                                    batch_y.to(device),
                                                    p[0],
                                                    p[1], p[2])
                            totalTime += time.time() - t1
                            predBatch.append(m(batch_x.to(device)))
                            targetBatch.append(batch_y.unsqueeze(1))
                        elif attm == 'simba':
                            t1 = time.time()
                            batch_x = ssAttack.simba_batch(batch_x.to(device), batch_y.to(device), simbaIter,
                                                           arg['imgSize'], 1,
                                                           p[0], log_every=simbaIter + 10)
                            totalTime += time.time() - t1
                            predBatch.append(m(batch_x.to(device)))
                            targetBatch.append(batch_y.unsqueeze(1))
                    predBatch = torch.concat(predBatch).cpu().numpy()
                    targetBatch = torch.concat(targetBatch).squeeze().cpu().numpy()
                    acc, recall, prec, f1 = metric.multiMetric(predBatch, targetBatch, 1)
                    r = '\teps:{} step:{} stepSize:{} acc: {} recall: {} prec: {} f1: {} totalTIme:{}s'.format(p[3],
                                                                                                               p[4],
                                                                                                               p[5],
                                                                                                               acc,
                                                                                                               recall.mean(),
                                                                                                               prec.mean(),
                                                                                                               f1.mean(),
                                                                                                               totalTime)
                    print(r)
                    results.append(r)

    print('dataset:', arg['dataName'])
    print('Validating class num:{} sample num:{}'.format(valDataset.class2num.keys().__len__(), valdataset_size))
    for r in results:
        print(r)
