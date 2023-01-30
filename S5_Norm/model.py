import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

'''
  Summary of `Net` Neural Network Class:

  -> In order to optimize the Number of Params and make the model edge deployable, a squeeze and expand
  model architecture is used. Initially we apply convolutions with 16 output channels tilla RF=5 is reached.
  Then a single maxpool is used at this point to add invariance. Later we expand upto 32 channels then use a
  transition block to squeeze back to 10 channels with a pointwise convolution(mix channels) and then expand
  back to 16 with a default 3x3 convolutions. Later we add some more layers to increase the capacity, add a GAP
  layer to preservere spatial knowledge and obtain amplitudes of classes. Finally a FCL is added to match the output
  shape. 
  
  -> In between the model makes use of a control flow to decide which type of normalization stratergy is to be used
  based on the use input (BN+L1, LN, GN). Depending on the num and norm we implement this in the network.

  -> Functional programming is use via torch.nn.functional to simplify the code and reduce overhead.
  
  Attributes:
    norm: Type of Normalization used Eg:"BN": L1+BatchNorm, "LN":LayerNorm, "GP": Group Norm
    x: Image from the batch
    num: Number of groups for Group Norm 
    device: Default arg is 'cuda:0', i.e gpu. It can be changed to CPU.
  
  Output and Output Shape:
    out: Predicted MNIST Digit, shape: [batch_size,10]
'''

class Net(nn.Module):
    def __init__(self, norm='GN', num=1, device='cuda:0'):
        super(Net, self).__init__()
        self.norm = norm
        self.num = num
        self.device = device
        self.conv1 = nn.Conv2d(1, 16, 3, padding=0, bias=False) # RF:1+(3-1)1=3; ji=1,jo=1; 28x28x1 -> 26x26x16
        self.bn1 = nn.BatchNorm2d(16)
        self.drop1 = nn.Dropout(0.1)
        self.conv3 = nn.Conv2d(16, 32, 3, padding=0, bias=False) # RF:3+(3-1)1=5; ji=1,jo=1; 26x26x16 -> 24x24x32
        self.bn2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(2, 2) # RF:5+(2-1)2=7; ji=1,jo=2; 24x24x32 -> 12x12x32
        self.drop2 = nn.Dropout(0.1)
        self.conv5 = nn.Conv2d(32, 10, 1, padding=0, bias=False) # RF:7+(1-1)2=7; ji=2,jo=2; 12x12x32 -> 12x12x10
        self.bn3 = nn.BatchNorm2d(10) 
        self.drop3 = nn.Dropout(0.1)
        self.conv6 = nn.Conv2d(10, 16, 3, padding=0, bias=False) # RF:7+(3-1)2=11; ji=2,jo=2; 12x12x10 -> 10x10x16
        self.bn4 = nn.BatchNorm2d(16)
        self.drop4 = nn.Dropout(0.1)
        self.conv10 = nn.Conv2d(16, 16, 3, padding=0, bias=False) # RF:11+(3-1)2=15; ji=2,jo=2; 10x10x16 -> 8x8x16
        self.bn10 = nn.BatchNorm2d(16)
        self.drop10 = nn.Dropout(0.1)
        self.conv7 = nn.Conv2d(16, 16, 3, padding=0, bias=False) # RF:15+(3-1)2=19; ji=2,jo=2; 8x8x16 -> 6x6x16
        self.gap = nn.AdaptiveAvgPool2d((1,1)) 

        self.lin = nn.Linear(16, 10)

    def forward(self, x):

        x = F.relu(self.conv1(x))

        if self.norm == 'BN':
          n_chans = x.shape[1]
          # Running mean and variance is calulated by the model by setting training=True
          # Hence we dont need to provide these params ourselves
          running_mean = torch.zeros(n_chans).to(self.device) 
          running_std = torch.ones(n_chans).to(self.device)
          x = F.batch_norm(x, running_mean, running_std, training=True, momentum=0.9)
          x = F.relu(self.conv3(self.drop1(x)))
          x = F.batch_norm(x, running_mean, running_std, training=True, momentum=0.9)
          # Transition Block: Squeeze
          x = F.relu(self.conv5(self.drop2(self.pool2(x))))
          x = F.batch_norm(x, running_mean, running_std, training=True, momentum=0.9)
          # Transition Block: Expand
          x = F.relu(self.conv6(x))
          x = F.batch_norm(x, running_mean, running_std, training=True, momentum=0.9)
          x = F.relu(self.conv10(self.drop4(x)))
          x = F.batch_norm(x, running_mean, running_std, training=True, momentum=0.9)

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
