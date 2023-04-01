## Implementation and Goal Achieved

Trained CIFAR-10 by making use of strided convolutions instead of max-pooling to minimize feature map size reduction, dilated convolutions after initial convolutions to maximize receptive field incrementation, and depth-wise separable convolutions along with 1x1 to effectively mix and merge with minimal overheads.

## Model Structure
```
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, 5, 1, 1), # 32x32 - jin=1, jout=1, rf = 1+(4)*1 = 5
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 16, 3, 2, padding=1), #strided, 32x32 - jin=1, jout=2, rf = 5+(2)*1 = 7
            nn.ReLU(),
            nn.BatchNorm2d(16)
        )
        self.conv2 =  nn.Sequential(
            nn.Conv2d(16, 32, 3, 1, 1,dilation=1), # dilation, 16x16 - jo=2 - rf=7+(5-1)*2=15
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, 3, 1, 1, groups=32), # depthwise-seperable, 16x16 - jo=2 - rf=15+(3-1)*2=19
            nn.ReLU(),
            nn.BatchNorm2d(64)
        )
        self.conv3 =  nn.Sequential(
            nn.Conv2d(64, 128, 3, 2, padding=1), # strided, 16x16 - jo=4 - rf=19+(3-1)*2=23
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, 3, 1, padding=1), # 8x8 - jo=4 - rf=23+(3-1)*4=31
            nn.ReLU(),
            nn.BatchNorm2d(128),
        )
        self.conv4 =  nn.Sequential(
            nn.Conv2d(128, 32, 3, 2, padding=1), # strided, 8x8 - jo=8 - rf=31+(3-1)*4=39
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, 1), # 4x4, Mix and Merge with 1x1
            nn.Conv2d(32, 32, 3, 1, 1), # 4x4 - jo=8 - rf=39+(3-1)*8=55
            nn.ReLU(),
            nn.BatchNorm2d(32)
        )
        self.gap = nn.AdaptiveAvgPool2d((1,1)) #gap
        self.fc1 = nn.Linear(32, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.gap(x)
        x = x.view(-1,32)
        x = self.fc1(x)
        return x

```


## Model Summary

```

----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 16, 30, 30]           1,216
              ReLU-2           [-1, 16, 30, 30]               0
       BatchNorm2d-3           [-1, 16, 30, 30]              32
            Conv2d-4           [-1, 16, 15, 15]           2,320
              ReLU-5           [-1, 16, 15, 15]               0
       BatchNorm2d-6           [-1, 16, 15, 15]              32
            Conv2d-7           [-1, 32, 15, 15]           4,640
              ReLU-8           [-1, 32, 15, 15]               0
       BatchNorm2d-9           [-1, 32, 15, 15]              64
           Conv2d-10           [-1, 64, 15, 15]             640
             ReLU-11           [-1, 64, 15, 15]               0
      BatchNorm2d-12           [-1, 64, 15, 15]             128
           Conv2d-13            [-1, 128, 8, 8]          73,856
             ReLU-14            [-1, 128, 8, 8]               0
      BatchNorm2d-15            [-1, 128, 8, 8]             256
           Conv2d-16            [-1, 128, 8, 8]         147,584
             ReLU-17            [-1, 128, 8, 8]               0
      BatchNorm2d-18            [-1, 128, 8, 8]             256
           Conv2d-19             [-1, 32, 4, 4]          36,896
             ReLU-20             [-1, 32, 4, 4]               0
      BatchNorm2d-21             [-1, 32, 4, 4]              64
           Conv2d-22             [-1, 32, 4, 4]           1,056
           Conv2d-23             [-1, 32, 4, 4]           9,248
             ReLU-24             [-1, 32, 4, 4]               0
      BatchNorm2d-25             [-1, 32, 4, 4]              64
AdaptiveAvgPool2d-26             [-1, 32, 1, 1]               0
           Linear-27                   [-1, 10]             330
================================================================

```

## CIFAR-10 with Albumentations

![image](https://user-images.githubusercontent.com/22516287/229272174-4102d106-4cf5-4607-b85f-5559891e12e5.png)

## Test Accuracy:
```
75%
```

## Transformations using Albumentations

```
train_transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.ShiftScaleRotate(p=0.5),
    A.CoarseDropout(max_holes = 1, max_height=16, max_width=16, min_height=16, min_width=16, min_holes = 1, fill_value=-1.69, mask_fill_value = None),
    A.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5],
    ),
    ToTensorV2()
])

test_transform = A.Compose([
    # A.HorizontalFlip(p=0.5),
    # A.ShiftScaleRotate(p=0.5),
    # A.CoarseDropout(max_holes = 1, max_height=16, max_width=16, min_height=16, min_width=16, min_holes = 1, fill_value=-1.69, mask_fill_value = None),
    A.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5],
    ),
    ToTensorV2()
])

class Cifar10(torchvision.datasets.CIFAR10):
    def __init__(self, root="~/data/cifar10", train=True, download=True, transform=None):
        super().__init__(root=root, train=train, download=download, transform=transform)

    def __getitem__(self, index):
        image, label = self.data[index], self.targets[index]

        if self.transform is not None:
            transformed = self.transform(image=image)
            image = transformed["image"]

        return image, label

trainset = Cifar10(root='./data', train=True,download=True, transform=train_transform)
testset = Cifar10(root='./data', train=False,download=True, transform=test_transform)

trainloader = torch.utils.data.DataLoader(trainset,batch_size = 4, shuffle=True)
testloader = torch.utils.data.DataLoader(testset,batch_size = 4, shuffle=True)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

```
