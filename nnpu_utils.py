from dataset import CIFAR_Dataset
from torch.utils.data import DataLoader
from utils import PULoss, get_batch
from model import CNN
import torch
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms

def train_nnpu(model, optimizer, label_loader, unlabel_loader, test_loader, max_epoch, num_iter):
    label_iter, unlabel_iter = iter(label_loader), iter(unlabel_loader)
    best_weight = None
    best_acc = 0.0
    criterion = PULoss(Probability_P=0.4, nnPU=True)
    for e in range(max_epoch):
        for i in range(num_iter):
            X_l, y_l, label_iter = get_batch(label_iter, label_loader)
            X_u, y_u, unlabel_iter = get_batch(unlabel_iter, unlabel_loader)

            X = torch.cat([X_l, X_u], 0)
            y = torch.cat([y_l, y_u], 0)

            logits = model(X)

            _, loss = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        if e % 1 == 0:
            acc = test(model, test_loader)
            print('Epoch {} Test Acc {}'.format(e, acc))

    return model.state_dict()

def test(model, test_loader):
    acc = 0.0
    cnt = 0
    model.eval()
    for i, (data, label) in enumerate(test_loader):
        data, label = data.cuda(), label.cuda()

        logits = model(data)
        if logits.shape[1] == 1:
            pred = torch.sign(model(data)).long().squeeze()
        else:
            pred = logits.max(1)[1]
            label = label>0
        acc += float((pred==label).float().sum().detach().cpu())
        cnt += data.shape[0]

    model.train()
    return acc / cnt

