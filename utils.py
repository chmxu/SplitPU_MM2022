import torch.nn as nn 
import torch
import numpy as np
import torch.nn.functional as F

def get_batch(loader, ori_loader, idx=False):
    if idx:
        try:
            data, label, idx = next(loader)
            return data.cuda(), label.cuda(), idx.cuda(), loader
            #return data.cuda(), label.cuda(), idx, loader
        except StopIteration:
            loader = iter(ori_loader)
            data, label, idx = next(loader)
            return data.cuda(), label.cuda(), idx.cuda(), loader
    else:
        try:
            data, label = next(loader)
            return data.cuda(), label.cuda(), loader
        except StopIteration:
            loader = iter(ori_loader)
            data, label = next(loader)
            return data.cuda(), label.cuda(), loader

def sigmoid_loss(input, reduction='elementwise_mean'):
    # y must be -1/+1
    # NOTE: torch.nn.functional.sigmoid is 1 / (1 + exp(-x)). BUT sigmoid loss should be 1 / (1 + exp(x))
    loss = torch.sigmoid(-input)
    
    return loss

def pu_risk_estimators_sigmoid(y_pred, y_true, prior):
    # y_true is -1/1
    #one_u = torch.ones(y_true.size()).cuda()
    one_u = torch.ones(y_true.size())
    positive = (y_true == 1).float()
    unlabeled = (y_true == -1).float()
    P_size = max(1., torch.sum(positive))
    u_size = max(1. ,torch.sum(unlabeled))
    y_positive = sigmoid_loss(y_pred).view(-1)
    y_unlabeled = sigmoid_loss(-y_pred).view(-1)
    positive_risk = (prior * y_positive * positive / P_size).sum()
    negative_risk = ((unlabeled / u_size - prior * positive / P_size) * y_unlabeled).sum()
    return positive_risk, negative_risk


def pu_loss(y_pred, y_true, loss_fn, Probility_P=0.25, BETA=0, gamma=1.0, Yi=1e-8, L=None, nnPU = True, eps = None):
    P_p, P_n, P_u = 0, 0, 0
    R_p, R_n = pu_risk_estimators_sigmoid(y_pred, y_true, Probility_P)

    M_reg = torch.zeros(1)
    if L is not None:
        FL = torch.mm((2 * y_pred - 1).transpose(0, 1), L)
        R_manifold = torch.mm(FL, (2 * y_pred - 1))
        M_reg = Yi * R_manifold
    if (not nnPU) or (loss_fn == 'Xent'):
        return None, R_p + R_n 

    if -BETA > R_n:
        #print("NEGATIVE")
        #print(R_n)
        return R_p - BETA, -gamma*R_n#, Probility_P * P_p, P_u, Probility_P * P_
        # return -gamma * PU_2, torch.sum(M_reg), Probility_P * P_p, P_u, Probility_P * P_n
        # return Probility_P * P_p
    else:
        #print("POSITIVE")
        #print(R_p, R_n, R_p + R_n)
        return R_p + R_n, R_p + R_n#, Probility_P * P_p, P_u, Probility_P * P_n
        # return PU_1, torch.sum(M_reg), Probility_P * P_p, P_u, Probility_P * P_n

class PULoss(nn.Module):
    '''
    only works for binary classification
    '''
    def __init__(self, loss_fn='sigmoid', Probability_P=0.25, BETA=0, gamma=1.0, Yi=1e-8, nnPU=True):
        super(PULoss, self).__init__()
        self.loss_fn = loss_fn
        self.Probability_P = Probability_P
        self.BETA = BETA
        self.gamma = gamma
        self.Yi = Yi
        self.nnPU = nnPU

    def update_p(self, p):
        self.Probability_P = p

    def forward(self, y_pred, y_true, L=None, eps = None):
        return pu_loss(y_pred, y_true, self.loss_fn, self.Probability_P, self.BETA, self.gamma, self.Yi, L, nnPU = self.nnPU, eps = eps)

def custom_kl_div(prediction, target, reduction='sum'):
    output_pos = target * (target.clamp(min=1e-7).log() - prediction)
    zeros = torch.zeros_like(output_pos)
    output = torch.where(target > 0, output_pos, zeros)
    output = torch.sum(output, axis=1)
    if reduction == 'sum':
        return output.sum()
    else:
        return output.mean()

class JSDLoss(torch.nn.Module):
    def __init__(self, weight=0.7, reduction='mean'):
        super(JSDLoss, self).__init__()
        self.weights = [weight, 1-weight]
        self.reduction = reduction
        assert abs(1.0 - sum(self.weights)) < 0.001
        self.scale = -1.0 / ((1.0-self.weights[0]) * np.log((1.0-self.weights[0])))
    
    def forward(self, pred, target):
        preds = list()
        if type(pred) == list:
            for i, p in enumerate(pred):
                preds.append(F.softmax(p, dim=1)) 
        else:
            #preds.append(F.softmax(pred, dim=1))
            pred = F.sigmoid(pred)
            preds.append(torch.cat([1-pred, pred], 1))

        distribs = [target] + preds
        assert len(self.weights) == len(distribs)

        mean_distrib = sum([w*d for w,d in zip(self.weights, distribs)])
        mean_distrib_log = mean_distrib.clamp(1e-7, 1.0).log()
        
        jsw = sum([w*custom_kl_div(mean_distrib_log, d, self.reduction) for w,d in zip(self.weights, distribs)])
        return self.scale * jsw


def hard_loss(logits, soft_label=True):
    weak_logits, strong_logits = logits.chunk(2)
    p_label = F.sigmoid(weak_logits.detach())
    targets = (p_label>0).float()
    mask = p_label.ge(0.95).float()

    if soft_label:
        loss = (F.binary_cross_entropy_with_logits(strong_logits, p_label,
                        reduction='none') * mask.unsqueeze(-1)).sum(-1).mean()
    else:
        loss = (F.binary_cross_entropy_with_logits(strong_logits, targets,
                          reduction='none') * mask).mean()
    return loss

def sim_loss(z, p):
    criterion = nn.CosineSimilarity(dim=1).cuda()
    z1, z2 = z.detach().chunk(2)
    p1, p2 = p.chunk(2)

    loss = -(criterion(p1, z2).mean() + criterion(p2, z1).mean()) * 0.5
    return loss
