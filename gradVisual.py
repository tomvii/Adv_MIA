import argparse
import os
import shutil

import cv2 as cv
import torch
import torchvision
import tqdm
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import model.chexnet
import dataset.dataset
import model.mobilenet_v2
import model.resnet
from advattack import _pgd_whitebox

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

    'melanomaBin': {'dataName': 'melanomaBin',
                    'testCsvPath': "./data/mel2017/testbin.csv",
                    'imgPath': "./data/mel2017/img/",
                    'imgSize': 224,
                    'valTrans': torchvision.transforms.Compose([torchvision.transforms.Resize((256, 256)),
                                                                torchvision.transforms.CenterCrop((224, 224)),
                                                                torchvision.transforms.ToTensor()])},

    'melanomaMul': {'dataName': 'melanomaMul',
                    'testCsvPath': "./data/mel2017/test.csv",
                    'imgPath': "./data/mel2017/img/",
                    'imgSize': 224,
                    'valTrans': torchvision.transforms.Compose([torchvision.transforms.Resize((256, 256)),
                                                                torchvision.transforms.CenterCrop((224, 224)),
                                                                torchvision.transforms.ToTensor()])},
    'xrayBin': {'dataName': 'xray',
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
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--model', type=str)
    parser.add_argument('--weight', type=str)
    parser.add_argument('--saveDir', type=str)
    parser.add_argument('--bs', type=int)
    parser.add_argument('--gpu', nargs='+')
    args = parser.parse_args()
    gpus = [int(s) for s in args.gpu]
    GPUIndex = gpus[0]
    torch.cuda.set_device(GPUIndex)
    print('use cuda', gpus, 'main card', GPUIndex)
    device = torch.device("cuda:{}".format(GPUIndex) if torch.cuda.is_available() else "cpu")
    batchSize = args.bs
    arg = config[args.dataset]
    saveDir = args.saveDir
    if os.path.exists(saveDir):
        shutil.rmtree(saveDir)
    os.makedirs(saveDir, exist_ok=True)
    whiteAdvParam = [(1 / 255, 20, 0.5 / 255, '1-255', '20', '05-255'),
                     (2 / 255, 20, 0.5 / 255, '2-255', '20', '05-255'),
                     (4 / 255, 20, 1 / 255, '4-255', '20', '1-255'),
                     (8 / 255, 20, 2 / 255, '8-255', '20', '2-255')]
    # --------------load data---------------#
    valDataset = dataset.dataset.medDataset(arg['testCsvPath'], arg['imgPath'], arg['valTrans'])
    valdataset_size = len(valDataset)
    print('Validating class num:{} sample num:{}'.format(valDataset.class2num.keys().__len__(), valdataset_size))
    validation_loader = torch.utils.data.DataLoader(valDataset,
                                                    batch_size=batchSize,
                                                    num_workers=2)

    # ---------------load model--------------#
    m = modelConstructor[args.model](valDataset.class2num.keys().__len__())
    l = torch.load(args.weight, map_location=device)
    m.load_state_dict(l)
    m = m.to(device)
    target_layers = [m.layer4[-1]]
    cam = GradCAM(model=m, target_layers=target_layers)
    base = 0
    m.eval()
    for batchX, batchY in tqdm.tqdm(validation_loader, total=len(validation_loader)):
        batchX, batchY = batchX.to(device), batchY.to(device)
        targets = []
        with torch.no_grad():
            pred = torch.softmax(m(batchX), dim=1).cpu().numpy()
        imgPaths = []
        for i in range(batchY.shape[0]):
            label = int(batchY[i].item())
            targets.append(ClassifierOutputTarget(label))
            imgPaths.append('class{}_confidence{:.3f}_{}.jpg'.format(label, pred[i, label], base + i))
        # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
        grayscale_cam = cam(input_tensor=batchX, targets=targets)
        # In this example grayscale_cam has only one image in the batch:
        for batchIdx in range(grayscale_cam.shape[0]):
            visualization = show_cam_on_image(
                cv.cvtColor(batchX[batchIdx, :].permute(1, 2, 0).cpu().numpy(), cv.COLOR_RGB2BGR),
                grayscale_cam[batchIdx, :],
                image_weight=0.5)
            imgPath = os.path.join(saveDir, imgPaths[batchIdx])
            cv.imwrite(imgPath, visualization)
        for p in whiteAdvParam:
            imgPaths = []
            for i in range(batchY.shape[0]):
                label = int(batchY[i].item())
                targets.append(ClassifierOutputTarget(label))
                imgPaths.append('class{}_confidence{:.3f}_{}map_eps{}.jpg'.format(label, pred[i, label],
                                                                                  base + i, p[3]))
            # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
            grayscale_cam = cam(
                input_tensor=_pgd_whitebox(m, torch.nn.CrossEntropyLoss, batchX, batchY, p[0], p[1], p[2]),
                targets=targets)
            # In this example grayscale_cam has only one image in the batch:
            for batchIdx in range(grayscale_cam.shape[0]):
                visualization = show_cam_on_image(
                    cv.cvtColor(batchX[batchIdx, :].permute(1, 2, 0).cpu().numpy(), cv.COLOR_RGB2BGR),
                    grayscale_cam[batchIdx, :],
                    image_weight=0.5)
                imgPath = os.path.join(saveDir, imgPaths[batchIdx])
                cv.imwrite(imgPath, visualization)
        base += len(batchX)
