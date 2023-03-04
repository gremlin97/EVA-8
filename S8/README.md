## Goal

* Create a simplified and well organized folder/file structure to host the model, utils and main driver code files seperately and import it for use in the colab file.
* Created a custom-optimized fast Resnet Architecture inspired by the DAWN CIFAR-10 Benchmark winner. Achieve 90% test accuracy on CIFAR-10 with a low training time using OCP and augmentations via albumentations.

### Custom Resnet Model Structure

```
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

class ResnetF(nn.Module):
  def __init__(self):
    super().__init__()
    self.prep = nn.Sequential(
        nn.Conv2d(3, 64 , 3 , 1 ,1),
        nn.ReLU(),
        nn.BatchNorm2d(64)
    )
    self.layer1 = nn.Sequential(
        nn.Conv2d(64, 128, 3, 1, 1),
        nn.MaxPool2d(2,2),
        nn.ReLU(),
        nn.BatchNorm2d(128)
    )
    self.residual1 = nn.Sequential(
        nn.Conv2d(128, 128, 3, 1, 1),
        nn.ReLU(),
        nn.BatchNorm2d(128),
        nn.Conv2d(128, 128, 3, 1, 1),
        nn.ReLU(),
        nn.BatchNorm2d(128)
    )
    self.layer2 = nn.Sequential(
        nn.Conv2d(128, 256, 3, 1, 1),
        nn.MaxPool2d(2,2),
        nn.ReLU(),
        nn.BatchNorm2d(256)
    )
    self.layer3 = nn.Sequential(
        nn.Conv2d(256, 512, 3, 1, 1),
        nn.MaxPool2d(2,2),
        nn.ReLU(),
        nn.BatchNorm2d(512),
        nn.Dropout(0.1)
    )
    self.residual2 = nn.Sequential(
      nn.Conv2d(512, 512, 3, 1, 1),
      nn.ReLU(),
      nn.BatchNorm2d(512),
      nn.Conv2d(512, 512, 3, 1, 1),
      nn.ReLU(),
      nn.BatchNorm2d(512)
    )
    self.maxpool = nn.MaxPool2d(4,2)
    self.fc = nn.Linear(512,10)
    self.softmax = nn.Softmax()
    
  
  def forward(self, x):
    x = self.prep(x)
    residual1 = self.layer1(x)
    x = self.residual1(residual1)
    x += residual1
    x = self.layer2(x)
    residual2 = self.layer3(x)
    x = self.residual2(residual2)
    x += residual2
    x = self.maxpool(x)
    x = x.view(x.size(0), -1)
    x = self.fc(x)
    x = self.softmax(x)
    return x
 
 ```


### Model Summary
```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 64, 32, 32]           1,792
              ReLU-2           [-1, 64, 32, 32]               0
       BatchNorm2d-3           [-1, 64, 32, 32]             128
            Conv2d-4          [-1, 128, 32, 32]          73,856
         MaxPool2d-5          [-1, 128, 16, 16]               0
              ReLU-6          [-1, 128, 16, 16]               0
       BatchNorm2d-7          [-1, 128, 16, 16]             256
            Conv2d-8          [-1, 128, 16, 16]         147,584
              ReLU-9          [-1, 128, 16, 16]               0
      BatchNorm2d-10          [-1, 128, 16, 16]             256
           Conv2d-11          [-1, 128, 16, 16]         147,584
             ReLU-12          [-1, 128, 16, 16]               0
      BatchNorm2d-13          [-1, 128, 16, 16]             256
           Conv2d-14          [-1, 256, 16, 16]         295,168
        MaxPool2d-15            [-1, 256, 8, 8]               0
             ReLU-16            [-1, 256, 8, 8]               0
      BatchNorm2d-17            [-1, 256, 8, 8]             512
           Conv2d-18            [-1, 512, 8, 8]       1,180,160
        MaxPool2d-19            [-1, 512, 4, 4]               0
             ReLU-20            [-1, 512, 4, 4]               0
      BatchNorm2d-21            [-1, 512, 4, 4]           1,024
          Dropout-22            [-1, 512, 4, 4]               0
           Conv2d-23            [-1, 512, 4, 4]       2,359,808
             ReLU-24            [-1, 512, 4, 4]               0
      BatchNorm2d-25            [-1, 512, 4, 4]           1,024
           Conv2d-26            [-1, 512, 4, 4]       2,359,808
             ReLU-27            [-1, 512, 4, 4]               0
      BatchNorm2d-28            [-1, 512, 4, 4]           1,024
        MaxPool2d-29            [-1, 512, 1, 1]               0
           Linear-30                   [-1, 10]           5,130
          Softmax-31                   [-1, 10]               0
================================================================
Total params: 6,575,370
Trainable params: 6,575,370
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.01
Forward/backward pass size (MB): 6.50
Params size (MB): 25.08
Estimated Total Size (MB): 31.60
----------------------------------------------------------------
```

### Training Logs:
```
    Accuracy is: 37.122
    LR is : [0.04266729403103062]
    Epoch is: 0
    Accuracy is: 53.902
    LR is : [0.07051438100407266]
    Epoch is: 1
    Accuracy is: 62.012
    LR is : [0.09836146797711469]
    Epoch is: 2
    Accuracy is: 67.01
    LR is : [0.12620855495015673]
    Epoch is: 3
    Accuracy is: 69.902
    LR is : [0.14658278022665575]
    Epoch is: 4
    Accuracy is: 72.616
    LR is : [0.13887936009963878]
    Epoch is: 5
    Accuracy is: 74.99
    LR is : [0.13117593997262178]
    Epoch is: 6
    Accuracy is: 77.318
    LR is : [0.12347251984560481]
    Epoch is: 7
    Accuracy is: 78.404
    LR is : [0.11576909971858781]
    Epoch is: 8
    Accuracy is: 79.556
    LR is : [0.10806567959157085]
    Epoch is: 9
    Accuracy is: 80.874
    LR is : [0.10036225946455386]
    Epoch is: 10
    Accuracy is: 81.862
    LR is : [0.09265883933753688]
    Epoch is: 11
    Accuracy is: 83.066
    LR is : [0.0849554192105199]
    Epoch is: 12
    Accuracy is: 84.192
    LR is : [0.07725199908350291]
    Epoch is: 13
    Accuracy is: 84.824
    LR is : [0.06954857895648592]
    Epoch is: 14
    Accuracy is: 86.146
    LR is : [0.061845158829468935]
    Epoch is: 15
    Accuracy is: 86.926
    LR is : [0.054141738702451966]
    Epoch is: 16
    Accuracy is: 87.696
    LR is : [0.04643831857543497]
    Epoch is: 17
    Accuracy is: 88.402
    LR is : [0.03873489844841799]
    Epoch is: 18
    Accuracy is: 89.13
    LR is : [0.031031478321401018]
    Epoch is: 19
    Accuracy is: 89.952
    LR is : [0.02332805819438402]
    Epoch is: 20
    Accuracy is: 90.718
    LR is : [0.015624638067367025]
    Epoch is: 21
    Accuracy is: 91.468
    LR is : [0.007921217940350056]
    Epoch is: 22
    Accuracy is: 92.146
    LR is : [0.00021779781333305936]
    Epoch is: 23
    Finished Training
    Accuracy of the model on the 10000 test images: 87 %
```
### Test Accuracy on Test Dataloader (10k Images)

```
Accuracy of the network on the 10000 test images: 87 %
```

### One Cycle Policy Details

![LR](https://github.com/gremlin97/EVA-8/blob/main/S8/lr.PNG)

***Max LR:*** 0.1552225357427048

```
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, 
                                                max_lr=ler_rate,
                                                steps_per_epoch=len(trainloader), 
                                                epochs=24,
                                                pct_start=0.2,
                                                div_factor=10,
                                                three_phase=False, 
                                                final_div_factor=100,
                                                anneal_strategy='linear'
                                                ) 
```

### Augmentation Strategy:
* Cutout 8x8
* Random Crop:32x32
* Horizontal Flip: probablity~0.5

### Link to Utils Folder (Contains Utility Files, Model Files and other supporting driver code)
https://github.com/gremlin97/EVA-Utils
