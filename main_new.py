from dataset import CIFAR_Dataset
from torch.utils.data import DataLoader
from utils import PULoss, get_batch
from model import CNN
import torch
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms
from nnpu_utils import train_nnpu
from splitpu_utils import split, train_splitpu
import argparse

def main_nnpu(root, label_loader, unlabel_loader, test_lodaer):
    max_epoch = 20
    model = CNN().cuda()

    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=5e-4)
    criterion = PULoss(Probability_P=0.4, nnPU=True)

    nnpu_weight = train_nnpu(model, optimizer, 
        label_loader, unlabel_loader, test_loader, 
        max_epoch=max_epoch, num_iter=len(unlabel_loader))
    return nnpu_weight

def main_splitpu(nnpu_weight, root, label_loader, unlabel_loader, unlabel_loader2, test_lodaer):
    lr = 5e-5
    model = CNN().cuda()
    nnpu_model = CNN().cuda()

    nnpu_model.load_state_dict(nnpu_weight)
    nnpu_model.eval()
    optimizer = optim.SGD(model.parameters(), lr=1e-3, weight_decay=5e-3)
    criterion = PULoss(Probability_P=0.4)

    remain_x, filter_x = split(model, nnpu_model, optimizer, 
        label_loader, unlabel_loader, test_loader, unlabel_loader2, 
        num_iter=len(unlabel_loader))

    lam = [{'hard':0.3, 'sim':0.1, 'feat':0.3}, {'hard':0.01, 'sim':0.0, 'feat':0.0}]
    for i in range(2):
        model = CNN(dual=True).cuda()
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-2)
        teacher_weight = train_splitpu(model, nnpu_model, optimizer,
            remain_x, filter_x, label_loader, test_loader, soft_label=True, lam=lam[i])
        nnpu_model.load_state_dict(teacher_weight, strict=False)
        nnpu_model.eval()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train image model with cross entropy loss')
    parser.add_argument('-r', '--root', type=str, default='./cifar')
    parser.add_argument('--nnpu', type=str, default=None)
    args = parser.parse_args()


    root = args.root
    data_transforms = {
            'train': transforms.Compose([
                transforms.ToPILImage(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]),
            'val': transforms.Compose([
                transforms.ToPILImage(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
        }

    batch_size = 500
    n_labeled = 1000

    cifar_data = CIFAR_Dataset(root, n_labeled=n_labeled, n_unlabeled=50000, transform=data_transforms)
    label_dataset, unlabel_dataset, test_dataset, _ = cifar_data.get_dataset()

    label_batch_size = int(batch_size / (len(unlabel_dataset)/len(label_dataset)))

    label_loader = DataLoader(label_dataset, batch_size=label_batch_size, shuffle=True)
    unlabel_loader = DataLoader(unlabel_dataset, batch_size=batch_size, shuffle=True)
    unlabel_loader2 = DataLoader(unlabel_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    if args.nnpu is None:
        nnpu_weight = main_nnpu(root, label_loader, unlabel_loader, test_loader)
    else:
        nnpu_weight = torch.load(args.nnpu)
    main_splitpu(nnpu_weight, root, label_loader, unlabel_loader, unlabel_loader2, test_loader)
