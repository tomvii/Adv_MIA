from model.resnet import *


def _pgd_whitebox(model,
                  lossC,
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
            loss = lossC()(model(X_pgd), y)
        loss.backward()
        eta = step_size * X_pgd.grad.data.sign()
        X_pgd = Variable(X_pgd.data + eta, requires_grad=True)
        eta = torch.clamp(X_pgd.data - X.data, -epsilon, epsilon)
        X_pgd = Variable(X.data + eta, requires_grad=True)
        X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0), requires_grad=True)
    return X_pgd


def _fgsm_whitebox(model,
                   X,
                   y,
                   epsilon):
    X_pgd = Variable(X.data, requires_grad=True)

    # random_noise = torch.FloatTensor(*X_pgd.shape).uniform_(-epsilon, epsilon).cuda()
    # X_pgd = Variable(X_pgd.data + random_noise, requires_grad=True)

    with torch.enable_grad():
        loss = nn.CrossEntropyLoss()(model(X_pgd), y)
    loss.backward()
    eta = epsilon * X_pgd.grad.data.sign()
    X_pgd = Variable(X_pgd.data + eta, requires_grad=True)
    eta = torch.clamp(X_pgd.data - X.data, -epsilon, epsilon)
    X_pgd = Variable(X.data + eta, requires_grad=True)
    X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0), requires_grad=True)
    return X_pgd


def _cw_whitebox(model,
                 X,
                 y,
                 epsilon,
                 num_steps,
                 step_size,
                 classNum):  # 8/255: 0.003    4/255: 0.003     2/255: 0.0015   1/255: 0.0015
    X_pgd = Variable(X.data, requires_grad=True)

    random_noise = torch.FloatTensor(*X_pgd.shape).uniform_(-epsilon, epsilon).cuda()
    X_pgd = Variable(X_pgd.data + random_noise, requires_grad=True)

    for _ in range(num_steps):
        with torch.enable_grad():
            loss = cwloss(model(X_pgd), y, num_classes=classNum)
        loss.backward()
        eta = step_size * X_pgd.grad.data.sign()
        X_pgd = Variable(X_pgd.data + eta, requires_grad=True)
        eta = torch.clamp(X_pgd.data - X.data, -epsilon, epsilon)
        X_pgd = Variable(X.data + eta, requires_grad=True)
        X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0), requires_grad=True)
    return X_pgd


def cwloss(output, target, confidence=50, num_classes=10):
    # compute the probability of the label class versus the maximum other
    target = target.data
    target_onehot = torch.zeros(target.size() + (num_classes,))
    target_onehot = target_onehot.cuda()
    target_onehot.scatter_(1, target.unsqueeze(1), 1.)
    target_var = Variable(target_onehot, requires_grad=False)
    real = (target_var * output).sum(1)
    other = ((1. - target_var) * output - target_var * 10000.).max(1)[0]
    loss = -torch.clamp(real - other + confidence, min=0.)  # equiv to max(..., 0.)
    loss = torch.sum(loss)
    return loss
