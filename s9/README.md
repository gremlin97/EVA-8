## Goal and Implementation Steps

* Create a simple bare-bones custom ViT (Vision Transformer) based self-attention block to understand how to create Key-Query-Value and calculate scaled dot product attention.
* We take the CIFAR10 Image, apply 3 convolutions through a convolution block and convert/scale the image to a feature map of size 3x3x48. Subsequently, we apply a GAP layer to get a 1x1x48 and then flatten this to 48 values.
* Then this input (let's say X) is passed through 3 Linear layers, K, Q, and V and converted to the corresponding Key, Query and Values for the corresponding input image. Then scaled dot product attention is applied to obtain the updated values. This is scaled back to 48 values and the previous block structure is repeated (Ultimus Block) 4 times.
* Finally we apply an FC layer to obtain the output layer corresponding the class size.

## Custom Ultimus Model Structure

```
import torch.nn as nn
import torch.nn.functional as F


class Ultimus(nn.Module):
    def __init__(self):
        super(Ultimus, self).__init__()
        self.k = nn.Linear(48,8)
        self.q = nn.Linear(48,8)
        self.v = nn.Linear(48,8)
        self.out = nn.Linear(8,48)

    def forward(self, x):
        k = self.k(x) # Calculating k,q,v values from learnanble k,q,v learnable layers
        q = self.q(x)
        v = self.v(x)
        score = F.softmax(torch.matmul(q,k.T)/torch.sqrt(torch.tensor(k.shape[1])),dim=1) # score = softmax((k x q.Transpose)/root(k.shape))
        attention = torch.matmul(score,v) # Scaled dot-product attention (score x v)
        out = self.out(attention)
        return out

class Transformer(nn.Module):
  def __init__(self):
    super(Transformer,self).__init__()
    self.conv1 = nn.Conv2d(3, 16, 3, 1, 1)  # Set of three convolutions and gap to downscale image from 32x32x3 to 1x1x48
    self.conv2 = nn.Conv2d(16, 32, 3, 1, 1) # (in_c,out_c,kernel_size,stride,padding)
    self.conv3 = nn.Conv2d(32, 48, 3, 1, 1)
    self.gap = nn.AdaptiveAvgPool2d((1,1))
    self.ultimusBlock = Ultimus()
    self.cap = nn.Linear(48,10) # Final Prediction Layer
  
  def forward(self, x):
      x = self.conv3(F.relu(self.conv2(F.relu(self.conv1(x)))))
      x = self.gap(x)
      x = torch.flatten(x, 1)

      main = x
      # print(main.shape)
      residue = self.ultimusBlock(main) # Block 1
      # print(residue.shape)
      main = main + residue # Skip Connection input->2
      residue = self.ultimusBlock(main) # Block 2
      main = main + residue # Skip Connection 1->2
      residue = self.ultimusBlock(main) # Block 3
      main = main + residue # Skip Connection 2->3
      residue = self.ultimusBlock(main) #Block 4
      main = main + residue # Skip Connection 3->output

      # x = self.ultimusBlock(self.ultimusBlock(self.ultimusBlock(self.ultimusBlock(x))))
      main = self.cap(main)
      return main

```

## Model Summary

```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 16, 32, 32]             448
            Conv2d-2           [-1, 32, 32, 32]           4,640
            Conv2d-3           [-1, 48, 32, 32]          13,872
 AdaptiveAvgPool2d-4             [-1, 48, 1, 1]               0
            Linear-5                    [-1, 8]             392
            Linear-6                    [-1, 8]             392
            Linear-7                    [-1, 8]             392
            Linear-8                   [-1, 48]             432
           Ultimus-9                   [-1, 48]               0
           Linear-10                    [-1, 8]             392
           Linear-11                    [-1, 8]             392
           Linear-12                    [-1, 8]             392
           Linear-13                   [-1, 48]             432
          Ultimus-14                   [-1, 48]               0
           Linear-15                    [-1, 8]             392
           Linear-16                    [-1, 8]             392
           Linear-17                    [-1, 8]             392
           Linear-18                   [-1, 48]             432
          Ultimus-19                   [-1, 48]               0
           Linear-20                    [-1, 8]             392
           Linear-21                    [-1, 8]             392
           Linear-22                    [-1, 8]             392
           Linear-23                   [-1, 48]             432
          Ultimus-24                   [-1, 48]               0
           Linear-25                   [-1, 10]             490
================================================================
Total params: 25,882
Trainable params: 25,882
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.01
Forward/backward pass size (MB): 0.75
Params size (MB): 0.10
Estimated Total Size (MB): 0.86
----------------------------------------------------------------

```

### Training Logs:

```
[1,  2000] loss: 1.890
      Accuracy is: 31.2
      LR is : [0.002875125008333889]
      Epoch is: 0
      [2,  2000] loss: 1.651
      Accuracy is: 39.596
      LR is : [0.004750250016667778]
      Epoch is: 1
      [3,  2000] loss: 1.534
      Accuracy is: 44.552
      LR is : [0.006625375025001668]
      Epoch is: 2
      [4,  2000] loss: 1.485
      Accuracy is: 46.276
      LR is : [0.008500500033335558]
      Epoch is: 3
      [5,  2000] loss: 1.450
      Accuracy is: 47.712
      LR is : [0.009895771]
      Epoch is: 4
      [6,  2000] loss: 1.417
      Accuracy is: 43.414
      LR is : [0.0093754585]
      Epoch is: 5
      [7,  2000] loss: 2.355
      Accuracy is: 11.368
      LR is : [0.008855146]
      Epoch is: 6
      [8,  2000] loss: 2.247
      Accuracy is: 16.334
      LR is : [0.0083348335]
      Epoch is: 7
      [9,  2000] loss: 2.034
      Accuracy is: 24.206
      LR is : [0.007814521]
      Epoch is: 8
      [10,  2000] loss: 1.908
      Accuracy is: 28.388
      LR is : [0.0072942085]
      Epoch is: 9
      [11,  2000] loss: 2.029
      Accuracy is: 27.852
      LR is : [0.006773896]
      Epoch is: 10
      [12,  2000] loss: 1.991
      Accuracy is: 27.834
      LR is : [0.0062535835]
      Epoch is: 11
      [13,  2000] loss: 2.620
      Accuracy is: 23.024
      LR is : [0.005733271]
      Epoch is: 12
      [14,  2000] loss: 2.439
      Accuracy is: 20.378
      LR is : [0.0052129585]
      Epoch is: 13
      [15,  2000] loss: 2.007
      Accuracy is: 27.008
      LR is : [0.004692646]
      Epoch is: 14
      [16,  2000] loss: 2.094
      Accuracy is: 26.106
      LR is : [0.004172333499999999]
      Epoch is: 15
      [17,  2000] loss: 2.133
      Accuracy is: 27.968
      LR is : [0.003652021]
      Epoch is: 16
      [18,  2000] loss: 1.899
      Accuracy is: 31.358
      LR is : [0.0031317085]
      Epoch is: 17
      [19,  2000] loss: 3.167
      Accuracy is: 25.052
      LR is : [0.0026113959999999993]
      Epoch is: 18
      [20,  2000] loss: 2.364
      Accuracy is: 26.042
      LR is : [0.0020910835000000003]
      Epoch is: 19
      [21,  2000] loss: 1.943
      Accuracy is: 28.592
      LR is : [0.0015707710000000003]
      Epoch is: 20
      [22,  2000] loss: 1.848
      Accuracy is: 33.438
      LR is : [0.0010504584999999986]
      Epoch is: 21
      [23,  2000] loss: 1.741
      Accuracy is: 37.39
      LR is : [0.0005301460000000004]
      Epoch is: 22
      [24,  2000] loss: 1.683
      Accuracy is: 39.042
      LR is : [9.83349999999876e-06]
      Epoch is: 23
      Finished Training

```

### Test Accuracy on Test Dataloader (10k Images)

```
Accuracy of the model on the 10000 test images: 41 %
Achieved a max/best accuracy of 47.7 % on the test images
```

### Model Loss

![Loss](https://github.com/gremlin97/EVA-8/blob/main/s9/images/loss.png)

### Link to Utils Folder (Contains Utility Files, Model Files and other supporting driver code)
https://github.com/gremlin97/EVA-Utils
