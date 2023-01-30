import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
# norm = 'GN'

class Net(nn.Module):
    def __init__(self, norm='GN', num=1):
        super(Net, self).__init__()
        self.norm = norm
        self.num = num
        self.conv1 = nn.Conv2d(1, 16, 3, padding=0, bias=False) # RF:1+(3-1)1=3; ji=1,jo=1; 28x28x1 -> 26x26x16
        self.bn1 = nn.BatchNorm2d(16)
        self.drop1 = nn.Dropout(0.1)
        self.conv3 = nn.Conv2d(16, 32, 3, padding=0, bias=False) # RF:4+(3-1)2=8; ji=2,jo=2; 26x26x16 -> 24x24x16
        self.bn2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(2, 2) # RF:8+(2-1)2=10; ji=2,jo=4; 24x24x16 -> 12x12x16
        self.drop2 = nn.Dropout(0.1)
        self.conv5 = nn.Conv2d(32, 10, 1, padding=0, bias=False) # RF:10+(3-1)4=18; ji=4,jo=4; 12x12x16 -> 10x10x16
        self.bn3 = nn.BatchNorm2d(10) 
        self.drop3 = nn.Dropout(0.1)
        self.conv6 = nn.Conv2d(10, 16, 3, padding=0, bias=False) # RF:18+(2-1)4=22; ji=4,jo=8; 10x10x16 -> 16x16x16
        self.bn4 = nn.BatchNorm2d(16)
        self.drop4 = nn.Dropout(0.1)
        self.conv10 = nn.Conv2d(16, 16, 3, padding=0, bias=False) # RF:18+(2-1)4=22; ji=4,jo=8; 16x16x16 -> 14x14x16
        self.bn10 = nn.BatchNorm2d(16)
        self.drop10 = nn.Dropout(0.1)
        self.conv7 = nn.Conv2d(16, 16, 3, padding=0, bias=False) # RF:18+(2-1)4=22; ji=4,jo=8; 5x5x16 -> 3x3x16
        self.gap = nn.AdaptiveAvgPool2d((1,1)) 

        self.lin = nn.Linear(16, 10)

    def forward(self, x):

        x = F.relu(self.conv1(x))

        if self.norm == 'BN':
          n_chans = x.shape[1]
          running_mean = torch.zeros(n_chans) 
          running_std = torch.ones(n_chans)
          x = F.batch_norm(x, running_mean, running_std, training=True)
          x = F.relu(self.conv3(self.drop1(x)))
          n_chans = x.shape[1]
          running_mean = torch.zeros(n_chans) 
          running_std = torch.ones(n_chans)
          x = F.batch_norm(x, running_mean, running_std, training=True)
          x = F.relu(self.conv5(self.drop2(self.pool2(x))))
          n_chans = x.shape[1]
          running_mean = torch.zeros(n_chans) 
          running_std = torch.ones(n_chans)
          x = F.batch_norm(x, running_mean, running_std, training=True)
          x = F.relu(self.conv6(x))
          n_chans = x.shape[1]
          running_mean = torch.zeros(n_chans) 
          running_std = torch.ones(n_chans)
          x = F.batch_norm(x, running_mean, running_std, training=True)
          x = F.relu(self.conv10(self.drop4(x)))
          n_chans = x.shape[1]
          running_mean = torch.zeros(n_chans) 
          running_std = torch.ones(n_chans)
          x = F.batch_norm(x, running_mean, running_std, training=True)

        elif self.norm == 'GN':
          x = F.group_norm(x,self.num)
          x = F.relu(self.conv3(self.drop1(x)))
          x = F.group_norm(x, self.num)
          x = F.relu(self.conv5(self.drop2(self.pool2(x))))
          x = F.group_norm(x, self.num)
          x = F.relu(self.conv6(x))
          x = F.group_norm(x, self.num)
          x = F.relu(self.conv10(self.drop4(x)))
          x = F.group_norm(x, self.num)

        else:
          x = F.layer_norm(x, [16, 26, 26])
          x = F.relu(self.conv3(self.drop1(x)))
          x = F.layer_norm(x, [32, 24, 24])
          x = F.relu(self.conv5(self.drop2(self.pool2(x))))
          x = F.layer_norm(x, [10, 12, 12])
          x = F.relu(self.conv6(x))
          x = F.layer_norm(x, [16, 10, 10])
          x = F.relu(self.conv10(self.drop4(x)))
          x = F.layer_norm(x, [16, 8, 8])

        x = self.conv7(self.drop10(x))

        x = self.gap(x)

        x = x.view(-1, 16)

        x = self.lin(x)
        
        return F.log_softmax(x)

# model = Net(norm)
