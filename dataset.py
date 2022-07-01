import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import logging
import os
import copy
import os.path as osp
import pickle
from collections import Counter

def _load_datafile(filename):
    with open(filename, 'rb') as fo:
        data_dict = pickle.load(fo, encoding='bytes')
        #print(data_dict.keys())
        assert data_dict[b'data'].dtype == np.uint8
        image_data = data_dict[b'data']
        image_data = image_data.reshape((image_data.shape[0], 3, 32, 32)).transpose(0, 2, 3, 1)
        return image_data, np.array(data_dict[b'labels'])


class CIFAR_Dataset(object):
    def __init__(self, root, n_labeled=1000, n_unlabeled=50000,
        transform=None, seed=None, pos_labels=[0,1,8,9], neg_labels=[2,3,4,5,6,7]):
        self.root = root
        self.pos_labels = pos_labels
        self.neg_labels = neg_labels
        self.n_labeled = n_labeled
        self.n_unlabeled = n_unlabeled
        self.transform = transform

        self.X_tr, self.Y_tr, self.X_te, self.Y_te = self.get_cifar(self.root)

    def binarize(self, labels):
        Y_ = - np.ones_like(labels)
        for label in self.pos_labels:
            Y_[labels==label] = 1
        return Y_

    def get_dataset(self):
        X_l, y_l, X_u, y_u = self.process_pu(self.X_tr, self.Y_tr)
        return SimpleDataSet(X_l, y_l, self.transform['train'], mode='L'), SimpleDataSet(X_u, y_u, self.transform['train'], mode='U'), SimpleDataSet(self.X_te, self.Y_te, self.transform['val'], mode='T'), SimpleDataSet(self.X_tr, self.Y_tr, self.transform['train'], mode='T')

    def get_cifar(self, root):
        train_filenames = ['data_batch_{}'.format(ii + 1) for ii in range(5)]
        eval_filename = 'test_batch'

        x_tr = np.zeros((50000, 32, 32, 3), dtype='uint8')
        y_tr = np.zeros(50000, dtype='int32')

        for ii, fname in enumerate(train_filenames):
            cur_images, cur_labels = _load_datafile(osp.join(root, fname))
            x_tr[ii * 10000 : (ii+1) * 10000, ...] = cur_images
            y_tr[ii * 10000 : (ii+1) * 10000, ...] = cur_labels

        x_te, y_te = _load_datafile(osp.join(root, eval_filename))

        y_tr = self.binarize(y_tr)
        y_te = self.binarize(y_te)

        return x_tr, y_tr, x_te, y_te

    def process_pu(self, X, Y):
        rand_idx = np.arange(X.shape[0])
        np.random.shuffle(rand_idx)
        X_, Y_ = X[rand_idx], Y[rand_idx]
        pos_idx = np.where(Y_==1)[0]
        label_idx = pos_idx[:self.n_labeled]
        unlabel_idx = np.concatenate([pos_idx[self.n_labeled:], np.where(Y_==-1)[0]], 0)

        unlabel_idx = unlabel_idx[:self.n_unlabeled]

        X_l, Y_l = X_[label_idx], Y_[label_idx]
        X_u, Y_u = X_[unlabel_idx], Y_[unlabel_idx]
        return X_l, Y_l, X_u, Y_u


class SimpleDataSet(Dataset):
    def __init__(self, data, label, transform, mode='train'):
        self.data = data
        self.label = label
        self.transform = transform
        self.mode = mode

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        data = self.data[index]
        label = self.label[index]
        if self.mode == 'L':
            return self.transform(data), 1
        elif self.mode == 'U':
            return self.transform(data), -1
        else:
            return self.transform(data), label 

class DebugDataSet(Dataset):
    def __init__(self, ori_loader, teacher, oracle, num_clean_delete=1000, num_noisy_delete=1000):
        clean_data, clean_label, noisy_data, noisy_label = self.filter_data(ori_loader, teacher, oracle)
        print('In total {} clean samples, {} noisy samples'.format(clean_data.shape[0], noisy_data.shape[0]))
        #assert num_delete < noisy_data.shape[0]

        clean_idx = torch.randperm(clean_data.shape[0])
        clean_data, clean_label = clean_data[clean_idx][:-num_clean_delete], clean_label[clean_idx][:-num_clean_delete]
        noisy_idx = torch.randperm(noisy_data.shape[0])
        noisy_data, noisy_label = noisy_data[noisy_idx][:-num_noisy_delete], noisy_label[noisy_idx][:-num_noisy_delete]

        self.data = torch.cat([clean_data, noisy_data], 0)
        self.label = torch.cat([clean_label, noisy_label], 0)
        print('After deletion, {} samples in total, {} clean samples, {} noisy samples'.format(self.data.shape[0], clean_data.shape[0], noisy_data.shape[0]))

    def filter_data(self, ori_loader, teacher, oracle):
        clean_data, clean_label, noisy_data, noisy_label = [], [], [], []
        for i, (data, label) in enumerate(ori_loader):
            data, label = data.cuda(), label.cuda()

            with torch.no_grad():
                teacher_logits = teacher(data)
                oracle_logits = oracle(data)

                teacher_pred = (teacher_logits>0).float().squeeze()
                oracle_pred = (oracle_logits>0).float().squeeze()

            clean_index = torch.nonzero(teacher_pred==oracle_pred)[:, 0]
            noisy_index = torch.nonzero(teacher_pred!=oracle_pred)[:, 0]
            
            clean_data.append(data[clean_index].cpu())
            clean_label.append(teacher_pred[clean_index].detach().cpu())
            noisy_data.append(data[noisy_index].cpu())
            noisy_label.append(teacher_pred[noisy_index].detach().cpu())

        clean_data = torch.cat(clean_data, 0)
        clean_label = torch.cat(clean_label, 0)
        noisy_data = torch.cat(noisy_data, 0)
        noisy_label = torch.cat(noisy_label, 0)
        return clean_data, clean_label, noisy_data, noisy_label

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx], self.label[idx]


class IndexDataset(Dataset):
    def __init__(self, dataset):
        self.data = dataset.data
        self.label = dataset.label
        self.transform = dataset.transform
        self.mode = dataset.mode

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        data = self.data[index]
        label = self.label[index]
        if self.mode == 'L':
            return self.transform(data), 1, index
        elif self.mode == 'U':
            return self.transform(data), -1, index
        else:
            return self.transform(data), label, index
