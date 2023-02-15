## Goal

* Create a simplified and well organized folder/file structure to host the model, utils and main driver code files seperately and import it for use in the colab file.
* Train a Resnet18 model (20 epochs) without bottleneck layer on CIFAR-10 dataset and achieve decent (~80) test accuracy on the same.
* Find and display 10 misclassified images by the model
* Apply GradCAM (>5px) on these 10 aforementioned images and undersand which regions in these class activations are responsible for these outputs.
* Use augmentation techniques such as random crop and cutout for the same using the albumentations library.

### Hyperparameters:
* SGD with *0.9 momentum*
* *Learning rate*: 0.001

### Augmentation Strategy:
* Cutout 16x16
* Random Crop:32x32

### Training Logs:
```
[1,  2000] loss: 1.061
[1,  4000] loss: 1.057
[1,  6000] loss: 1.024
[1,  8000] loss: 0.946
[1, 10000] loss: 0.909
[1, 12000] loss: 0.924
Accuracy is: 65.316
[2,  2000] loss: 0.799
[2,  4000] loss: 0.787
[2,  6000] loss: 0.789
[2,  8000] loss: 0.776
[2, 10000] loss: 0.766
[2, 12000] loss: 0.747
Accuracy is: 72.728
[3,  2000] loss: 0.649
[3,  4000] loss: 0.658
[3,  6000] loss: 0.648
[3,  8000] loss: 0.638
[3, 10000] loss: 0.637
[3, 12000] loss: 0.625
Accuracy is: 77.45
[4,  2000] loss: 0.532
[4,  4000] loss: 0.542
[4,  6000] loss: 0.552
[4,  8000] loss: 0.546
[4, 10000] loss: 0.553
[4, 12000] loss: 0.550
Accuracy is: 80.906
[5,  2000] loss: 0.449
[5,  4000] loss: 0.480
[5,  6000] loss: 0.472
[5,  8000] loss: 0.463
[5, 10000] loss: 0.480
[5, 12000] loss: 0.479
Accuracy is: 83.568
[6,  2000] loss: 0.388
[6,  4000] loss: 0.396
[6,  6000] loss: 0.410
[6,  8000] loss: 0.406
[6, 10000] loss: 0.418
[6, 12000] loss: 0.421
Accuracy is: 85.802
[7,  2000] loss: 0.329
[7,  4000] loss: 0.338
[7,  6000] loss: 0.349
[7,  8000] loss: 0.355
[7, 10000] loss: 0.380
[7, 12000] loss: 0.359
Accuracy is: 87.78
[8,  2000] loss: 0.261
[8,  4000] loss: 0.285
[8,  6000] loss: 0.298
[8,  8000] loss: 0.310
[8, 10000] loss: 0.306
[8, 12000] loss: 0.313
Accuracy is: 89.782
[9,  2000] loss: 0.242
[9,  4000] loss: 0.237
[9,  6000] loss: 0.242
[9,  8000] loss: 0.273
[9, 10000] loss: 0.273
[9, 12000] loss: 0.283
Accuracy is: 90.946
[10,  2000] loss: 0.208
[10,  4000] loss: 0.212
[10,  6000] loss: 0.228
[10,  8000] loss: 0.234
[10, 10000] loss: 0.238
[10, 12000] loss: 0.236
Accuracy is: 92.076
[11,  2000] loss: 0.190
[11,  4000] loss: 0.192
[11,  6000] loss: 0.196
[11,  8000] loss: 0.204
[11, 10000] loss: 0.206
[11, 12000] loss: 0.207
Accuracy is: 93.15
[12,  2000] loss: 0.157
[12,  4000] loss: 0.172
[12,  6000] loss: 0.170
[12,  8000] loss: 0.183
[12, 10000] loss: 0.172
[12, 12000] loss: 0.200
Accuracy is: 93.948
[13,  2000] loss: 0.142
[13,  4000] loss: 0.153
[13,  6000] loss: 0.157
[13,  8000] loss: 0.158
[13, 10000] loss: 0.152
[13, 12000] loss: 0.163
Accuracy is: 94.712
[14,  2000] loss: 0.125
[14,  4000] loss: 0.136
[14,  6000] loss: 0.144
[14,  8000] loss: 0.148
[14, 10000] loss: 0.150
[14, 12000] loss: 0.142
Accuracy is: 95.222
[15,  2000] loss: 0.126
[15,  4000] loss: 0.134
[15,  6000] loss: 0.122
[15,  8000] loss: 0.134
[15, 10000] loss: 0.130
[15, 12000] loss: 0.145
Accuracy is: 95.626
[16,  2000] loss: 0.115
[16,  4000] loss: 0.119
[16,  6000] loss: 0.121
[16,  8000] loss: 0.128
[16, 10000] loss: 0.132
[16, 12000] loss: 0.132
Accuracy is: 95.81
[17,  2000] loss: 0.105
[17,  4000] loss: 0.106
[17,  6000] loss: 0.111
[17,  8000] loss: 0.117
[17, 10000] loss: 0.112
[17, 12000] loss: 0.106
Accuracy is: 96.348
[18,  2000] loss: 0.101
[18,  4000] loss: 0.107
[18,  6000] loss: 0.098
[18,  8000] loss: 0.114
[18, 10000] loss: 0.116
[18, 12000] loss: 0.109
Accuracy is: 96.256
[19,  2000] loss: 0.090
[19,  4000] loss: 0.102
[19,  6000] loss: 0.095
[19,  8000] loss: 0.103
[19, 10000] loss: 0.103
[19, 12000] loss: 0.102
Accuracy is: 96.688
[20,  2000] loss: 0.076
[20,  4000] loss: 0.086
[20,  6000] loss: 0.084
[20,  8000] loss: 0.101
[20, 10000] loss: 0.086
[20, 12000] loss: 0.100
Accuracy is: 97.072
Finished Training

```

### Test Accuracy on Test Dataloader (10k Images)

```
Accuracy of the network on the 10000 test images: 86 %

```

### Loss Graph for the Model Training


#### Loss Graph

![Loss](https://github.com/gremlin97/EVA-8/blob/main/S7/images/loss_7.png)

### Misclassified Images by Resnet18 model on CIFAR10 test dataset
The images are displayed in a 5x2 grid and each image has its corresponding wrong prediction by the model.

![GN](https://github.com/gremlin97/EVA-8/blob/main/S7/images/missclassified.png)

### GradCAM output on 10 Misclassified Images by Resnet18 model on CIFAR10 test dataset

#### Misclassified Images

<p float="left">
  <img src="https://github.com/gremlin97/EVA-8/blob/main/S7/images/1.png" width="200" />
  <img src="https://github.com/gremlin97/EVA-8/blob/main/S7/images/2.png" width="200" /> 
  <img src="https://github.com/gremlin97/EVA-8/blob/main/S7/images/3.png" width="200" />
  <img src="https://github.com/gremlin97/EVA-8/blob/main/S7/images/4.png" width="200" />
  <img src="https://github.com/gremlin97/EVA-8/blob/main/S7/images/5.png" width="200" /> 
  <img src="https://github.com/gremlin97/EVA-8/blob/main/S7/images/6.png" width="200" />
  <img src="https://github.com/gremlin97/EVA-8/blob/main/S7/images/7.png" width="200" />
  <img src="https://github.com/gremlin97/EVA-8/blob/main/S7/images/8.png" width="200" /> 
  <img src="https://github.com/gremlin97/EVA-8/blob/main/S7/images/9.png" width="200" />
  <img src="https://github.com/gremlin97/EVA-8/blob/main/S7/images/10.png" width="200" />
</p>

#### Corresponding GradCAM Output (For their actual corresponding class)

<p float="left">
  <img src="https://github.com/gremlin97/EVA-8/blob/main/S7/images/1g.png" width="200" />
  <img src="https://github.com/gremlin97/EVA-8/blob/main/S7/images/2g.png" width="200" /> 
  <img src="https://github.com/gremlin97/EVA-8/blob/main/S7/images/3g.png" width="200" />
  <img src="https://github.com/gremlin97/EVA-8/blob/main/S7/images/4g.png" width="200" />
  <img src="https://github.com/gremlin97/EVA-8/blob/main/S7/images/5g.png" width="200" /> 
  <img src="https://github.com/gremlin97/EVA-8/blob/main/S7/images/6g.png" width="200" />
  <img src="https://github.com/gremlin97/EVA-8/blob/main/S7/images/7g.png" width="200" />
  <img src="https://github.com/gremlin97/EVA-8/blob/main/S7/images/8g.png" width="200" /> 
  <img src="https://github.com/gremlin97/EVA-8/blob/main/S7/images/9g.png" width="200" />
  <img src="https://github.com/gremlin97/EVA-8/blob/main/S7/images/10g.png" width="200" />
</p>




