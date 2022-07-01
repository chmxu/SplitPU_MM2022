import torch
import torch.nn as nn
import torch.nn.functional as F

def weights_init(m):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)

class MultiLayerPerceptron(nn.Module):
    def __init__(self, dim):
        super(MultiLayerPerceptron, self).__init__()
        
        self.l1 = nn.Linear(dim, 300, bias=False)
        self.bn1 = nn.BatchNorm1d(300)
        self.l2 = nn.Linear(300, 300, bias=False)
        self.bn2 = nn.BatchNorm1d(300)
        self.l3 = nn.Linear(300, 300, bias=False)
        self.bn3 = nn.BatchNorm1d(300)
        self.l4 = nn.Linear(300, 300, bias=False)
        self.bn4 = nn.BatchNorm1d(300)
        self.l5 = nn.Linear(300, 1)
        
        self.apply(weights_init)

    def forward(self, x):
        x = self.l1(x)
        x = x.view(-1, 300)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.l2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.l3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.l4(x)
        x = self.bn4(x)
        x = F.relu(x)
        x = self.l5(x)
        return x

class CNN(nn.Module):

    def __init__(self, num_output=1, dual=False):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 96, kernel_size = 3, padding=1)
        self.bn1 = nn.BatchNorm2d(96)
        self.conv2 = nn.Conv2d(96, 96, kernel_size = 3, padding=1)
        self.bn2 = nn.BatchNorm2d(96)
        self.conv3 = nn.Conv2d(96, 96, kernel_size = 3, stride = 2, padding=1)
        self.bn3 = nn.BatchNorm2d(96)
        self.conv4 = nn.Conv2d(96, 192, kernel_size = 3, padding=1)
        self.bn4 = nn.BatchNorm2d(192)
        self.conv5 = nn.Conv2d(192, 192, kernel_size = 3, padding=1)
        self.bn5 = nn.BatchNorm2d(192)
        self.conv6 = nn.Conv2d(192, 192, kernel_size = 3, stride = 2, padding=1)
        self.bn6 = nn.BatchNorm2d(192)
        self.conv7 = nn.Conv2d(192, 192, kernel_size = 3, padding=1)
        self.bn7 = nn.BatchNorm2d(192)
        self.conv8 = nn.Conv2d(192, 192, kernel_size = 1)
        self.bn8 = nn.BatchNorm2d(192)
        self.conv9 = nn.Conv2d(192, 10, kernel_size = 1)
        self.bn9 = nn.BatchNorm2d(10)
        self.l1 = nn.Linear(640, 1000)
        self.l2 = nn.Linear(1000, 1000)
        self.l3 = nn.Linear(1000, num_output)
        
        self.dual = dual
        if dual:
            self.predictor = nn.Sequential(nn.Linear(1000, 512, bias=False),
                                        nn.BatchNorm1d(512),
                                        nn.ReLU(), # hidden layer
                                        nn.Linear(512, 1000)) # output layer

        self.apply(weights_init)

    def forward(self, x, return_feat=False):

        x1 = self.conv1(x)
        x1 = self.bn1(x1)
        x1 = F.relu(x1)
        x2 = self.conv2(x1)
        x2 = self.bn2(x2)
        x2 = F.relu(x2)
        x3 = self.conv3(x2)
        x3 = self.bn3(x3)
        x3 = F.relu(x3)
        x4 = self.conv4(x3)
        x4 = self.bn4(x4)
        x4 = F.relu(x4)
        x5 = self.conv5(x4)
        x5 = self.bn5(x5)
        x5 = F.relu(x5)
        x6 = self.conv6(x5)
        x6 = self.bn6(x6)
        x6 = F.relu(x6)
        x7 = self.conv7(x6)
        x7 = self.bn7(x7)
        x7 = F.relu(x7)
        x8 = self.conv8(x7)
        x8 = self.bn8(x8)
        x8 = F.relu(x8)
        x9 = self.conv9(x8)
        x9 = self.bn9(x9)
        x9 = F.relu(x9)
        x9 = x9.view(-1, 640)
        x = self.l1(x9)
        x = F.relu(x)
        x = self.l2(x)
        x = F.relu(x)
        if return_feat: 
            if self.dual:
                p = self.predictor(x)
                return x1, x, p, self.l3(x)
            return x1, x
        x = self.l3(x)
        return x

if __name__ == '__main__':
    x = torch.zeros((1, 3, 32, 32))
    model = CNN()
    print(model(x))
