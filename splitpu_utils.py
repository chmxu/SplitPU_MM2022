from dataset import CIFAR_Dataset
from torch.utils.data import DataLoader
from utils import PULoss, get_batch
from model import CNN
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from torchvision import transforms
import math
from utils import PULoss, get_batch, JSDLoss, hard_loss, sim_loss
from torchvision.transforms import RandomCrop, RandomHorizontalFlip, Compose

def split(model, teacher, optimizer, 
    label_loader, unlabel_loader, test_loader, unlabel_loader2, num_iter, thres=0.92):
    max_epoch = 50
    label_iter, unlabel_iter = iter(label_loader), iter(unlabel_loader)

    for e in range(max_epoch):
        train_acc = 0.0
        remain_x, remain_y, filter_x, filter_y = [], [], [], []
        for i in range(num_iter):
            X_l, y_l, label_iter = get_batch(label_iter, label_loader)
            X_u, y_u, unlabel_iter = get_batch(unlabel_iter, unlabel_loader)
            num_label = X_l.shape[0]

            X = torch.cat([X_l, X_u], 0)

            logits = model(X)
            pred = (logits>0).float().squeeze()
            with torch.no_grad():
                teacher_logits = teacher(X).detach()
                teacher_pred = (teacher_logits>0).float().squeeze()

            cls_loss = nn.BCELoss()(F.sigmoid(logits[:, 0]), teacher_pred)
            loss = cls_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_acc, remain_x, filter_x = test_on_train_data(model, unlabel_loader2, teacher)
        print('Epoch {}, Acc {}'.format(e, train_acc))
        if train_acc > thres: break

    remain_x = remain_x[torch.randperm(remain_x.shape[0])]

    with torch.no_grad():
        filter_logits = teacher(filter_x.cuda())[:, 0].detach().cpu()
        low_th = torch.quantile(filter_logits, 0.15)
        high_th = torch.quantile(filter_logits, 0.85)
        idx = torch.nonzero((filter_logits<low_th)+(filter_logits>high_th))[:, 0]
        filter_idx = torch.nonzero((filter_logits>=low_th)*(filter_logits<=high_th))[:, 0]
        remain_x = torch.cat([remain_x, filter_x[idx]], 0)

    return remain_x, filter_x

def test_on_train_data(model, test_loader, teacher):
    acc = 0.0
    cnt = 0
    model.eval()
    remain_x, filter_x = [], []
    test_iter = iter(test_loader)
    num_iter = len(test_loader)

    for i in range(num_iter):
        X, y_l, test_iter = get_batch(test_iter, test_loader)
     
        X = X.cuda()
        pred = (model(X)>0).float().squeeze()
        with torch.no_grad():
            teacher_logits = teacher(X).detach()
            teacher_pred = (teacher_logits>0).float().squeeze()
        acc += float((pred==teacher_pred).float().sum().detach().cpu())
        cnt += X.shape[0]

        remain_idx = torch.nonzero(pred==teacher_pred)[:, 0]
        filter_idx = torch.nonzero(pred!=teacher_pred)[:, 0]
        remain_x.append(X[remain_idx].cpu())
        filter_x.append(X[filter_idx].cpu())

    model.train()
    remain_x = torch.cat(remain_x, 0)
    filter_x = torch.cat(filter_x, 0)
    return acc / cnt, remain_x, filter_x

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

def train_splitpu(model, teacher, optimizer, easy_data, hard_data, 
    label_loader, test_loader, soft_label=True, 
    lam={'hard':0.3, 'sim':0.1, 'feat':0.1}):
    max_epoch = 100
    label_iter = iter(label_loader)
    best_weight = None
    best_acc = 0.0
    num_iter = len(label_loader)
    batch_size = math.ceil(easy_data.shape[0] / num_iter)
    hard_batch_size = math.ceil(hard_data.shape[0] / num_iter)
    criterion = JSDLoss(weight=0.7, reduction='sum')

    weak_aug = Compose([RandomCrop(32, padding=4),
                    RandomHorizontalFlip()])
    strong_aug = Compose([RandomCrop(32, padding=4),
                    transforms.RandomApply([
                        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
                        ], p=0.8),
                    RandomHorizontalFlip()])

    for e in range(max_epoch):
        easy_data = easy_data[torch.randperm(easy_data.shape[0])]
        hard_data = hard_data[torch.randperm(hard_data.shape[0])]
        for i in range(num_iter):
            label_X, label_y, label_iter = get_batch(label_iter, label_loader)
            label_y = label_y.view(-1, 1).float()
            easy_X = easy_data[batch_size*i:batch_size*(i+1)].cuda()
            hard_X = hard_data[hard_batch_size*i:hard_batch_size*(i+1)].cuda()

            ce_X = easy_X.clone().cuda()
            X = torch.cat([label_X, ce_X], 0)
            X = weak_aug(X)
            weak_hard_X = weak_aug(hard_X)
            strong_hard_X = strong_aug(hard_X)
            hard_X_ = torch.cat([weak_hard_X, strong_hard_X], 0)
            inp = torch.cat([X, hard_X_], 0)
            
            with torch.no_grad():
                t_logits = teacher(ce_X).detach()
                t_feat = teacher(hard_X_, return_feat=True)[0]

            feat_low, z, p, logits = model(inp, return_feat=True)
            feat_low, z, p = feat_low[X.shape[0]:], z[X.shape[0]:], p[X.shape[0]:]
            hard_logits = logits[X.shape[0]:]
            easy_logits = logits[:X.shape[0]]

            l_hard = hard_loss(hard_logits, soft_label)
            l_sim = sim_loss(z, p)
            l_feat = nn.MSELoss()(feat_low, t_feat)
        
            if soft_label:
                t_logits = F.sigmoid(t_logits)
            else:
                t_logits = (t_logits>0).float()
            t_logits = torch.cat([1-t_logits, t_logits], 1)
            
            jsd_loss = criterion(easy_logits[label_X.shape[0]:], t_logits)                        
            ce_loss = nn.BCEWithLogitsLoss(reduction='sum')(easy_logits[:label_X.shape[0]], label_y)
            l_easy = (jsd_loss + ce_loss) / X.shape[0]
            loss = l_easy + lam['hard'] * l_hard + lam['sim'] * l_sim + lam['feat'] * l_feat

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        acc = test(model, test_loader) 
        print('Epoch {} Acc {}'.format(e, acc))
    return model.state_dict()
