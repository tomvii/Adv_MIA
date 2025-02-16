import torch
import numpy as np
from torch import Tensor


def binaryMetric(pred, target, thres):  # pred is [0,1], target ∈ {0, 1}, thres is [0, 1]
    assert pred.shape == target.shape
    assert 0 <= thres <= 1
    predictClass = pred.__ge__(thres).astype(int)
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    total = target.shape[0]
    for i in range(predictClass.shape[0]):
        if predictClass[i] == target[i]:
            if predictClass[i] == 1:
                tp += 1
            else:
                tn += 1
        else:
            if predictClass[i] == 1:
                fp += 1
            else:
                fn += 1
    acc = (tp + tn) / total
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    prec = tp / (tp + fp) if tp + fp > 0 else 0
    return acc, recall, prec


def multiMetric(pred, target, axis):
    predictClass = np.argmax(pred, axis)
    classNum = pred.shape[axis]
    assert predictClass.shape == target.shape
    tpArr = np.zeros((classNum,))
    predArr = np.zeros((classNum,))
    targetArr = np.zeros((classNum,))
    acc = 0
    recall = np.zeros((classNum,))
    prec = np.zeros((classNum,))
    f1 = np.zeros((classNum,))
    for i in range(predictClass.shape[0]):
        if predictClass[i] == target[i]:
            tpArr[predictClass[i]] += 1
            acc += 1
        predArr[predictClass[i]] += 1
        targetArr[target[i]] += 1
    for i in range(classNum):
        recall[i] = tpArr[i] / targetArr[i] if targetArr[i] > 0 else 0
        prec[i] = tpArr[i] / predArr[i] if predArr[i] > 0 else 0
        f1[i] = 0
        if recall[i] + prec[i] > 0:
            f1[i] = 2 * recall[i] * prec[i] / (recall[i] + prec[i])
    return acc / target.shape[0], recall, prec, f1


def dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon=1e-6):
    # Average of Dice coefficient for all batches, or for a single mask
    assert input.size() == target.size()
    if input.dim() == 2 and reduce_batch_first:
        raise ValueError(f'Dice: asked to reduce batch but got tensor without batch dimension (shape {input.shape})')

    if input.dim() == 2 or reduce_batch_first:
        inter = torch.dot(input.reshape(-1), target.reshape(-1))
        sets_sum = torch.sum(input) + torch.sum(target)
        if sets_sum.item() == 0:
            sets_sum = 2 * inter

        return (2 * inter + epsilon) / (sets_sum + epsilon)
    else:
        # compute and average metric for each batch element
        dice = 0
        for i in range(input.shape[0]):
            dice += dice_coeff(input[i, ...], target[i, ...])
        return dice / input.shape[0]


def multiclass_dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon=1e-6):
    # Average of Dice coefficient for all classes
    assert input.size() == target.size()
    dice = 0
    for channel in range(input.shape[1]):
        dice += dice_coeff(input[:, channel, ...], target[:, channel, ...], reduce_batch_first, epsilon)

    return dice / input.shape[1]


def dice_loss(input: Tensor, target: Tensor, multiclass: bool = False):
    # Dice loss (objective to minimize) between 0 and 1
    assert input.size() == target.size()
    fn = multiclass_dice_coeff if multiclass else dice_coeff
    return 1 - fn(input, target, reduce_batch_first=True)


def IoU(pred, target): # [batchSize, channel, h, w]
    classNum = target.shape[1]
    predArr = pred.argmax(1).cpu().numpy() + 1
    targetArr = target.argmax(1).cpu().numpy() + 1
    iou = 0
    for i in range(len(predArr)):
        inter, _ = np.histogram(predArr[i, :, :] * (predArr[i, :, :] == targetArr[i, :, :]), bins=classNum, range=(1, classNum))
        areaPred, _ = np.histogram(predArr[i, :, :], bins=classNum, range=(1, classNum))
        areaTarget, _ = np.histogram(targetArr[i, :, :], bins=classNum, range=(1, classNum))
        areaUnion = areaPred + areaTarget - inter
        iou += np.mean(inter / areaUnion)
    return iou / len(predArr)



if __name__ == '__main__':
    pred = np.array([[0.5, 0.2, 0.3], [0.2, 0.5, 0.3], [0.2, 0.3, 0.5]])
    target = np.array([0, 2, 1])
    print(multiMetric(pred, target, 1))
