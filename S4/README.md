# Part 1

**Goal**: Backprop a Simple Neural Network using excel.

**Conclusion**: Decreasing the Learn rate decreases the convergence rate of the error function i.e. the decrease in error is lower across the Epochs.

## Screenshots

![Part 1](https://github.com/gremlin97/EVA-8/blob/main/S4/part1.PNG)
![Part 2](https://github.com/gremlin97/EVA-8/blob/main/S4/part2.PNG)
![Part 3](https://github.com/gremlin97/EVA-8/blob/main/S4/part3.PNG)

# Part 2

## Goal

To train a neural network model to predict digits from 0-9 using the MNIST Digits.

### Constraints
* Less than 20k params
* Less than 20 Epochs
* Minimum Accuracy of 99.4 % on test data
* Use BN, Dropout, and GAP Layers

## Results
* Max Accuracy of 99.48% achieved on the test data
* 17,098 Parameters

## Max Accuracy Achieved
99.48%

## Training Logs:

```
# Maximum Accuracy Training Log:

'''
Epoch:  13
loss=0.003916250541806221 batch_id=1874: 100%|██████████| 1875/1875 [00:26<00:00, 70.80it/s]

Test set: Average loss: 0.0175, Accuracy: 9948/10000 (99%)

'''

# Training Logs:

'''
Epoch:  1
  0%|          | 0/1875 [00:00<?, ?it/s]<ipython-input-22-e8be6f7f95f0>:27: UserWarning: Implicit dimension choice for log_softmax has been deprecated. Change the call to include dim=X as an argument.
  return F.log_softmax(x)
loss=0.04736865311861038 batch_id=1874: 100%|██████████| 1875/1875 [00:27<00:00, 67.42it/s]

Test set: Average loss: 0.0382, Accuracy: 9878/10000 (99%)

Epoch:  2
loss=0.12485630810260773 batch_id=1874: 100%|██████████| 1875/1875 [00:26<00:00, 70.35it/s]

Test set: Average loss: 0.0319, Accuracy: 9894/10000 (99%)

Epoch:  3
loss=0.00622600968927145 batch_id=1874: 100%|██████████| 1875/1875 [00:27<00:00, 68.48it/s]

Test set: Average loss: 0.0272, Accuracy: 9909/10000 (99%)

Epoch:  4
loss=0.056115299463272095 batch_id=1874: 100%|██████████| 1875/1875 [00:26<00:00, 70.78it/s]

Test set: Average loss: 0.0231, Accuracy: 9926/10000 (99%)

Epoch:  5
loss=0.026963435113430023 batch_id=1874: 100%|██████████| 1875/1875 [00:26<00:00, 70.55it/s]

Test set: Average loss: 0.0277, Accuracy: 9894/10000 (99%)

Epoch:  6
loss=0.0020414951723068953 batch_id=1874: 100%|██████████| 1875/1875 [00:26<00:00, 70.38it/s]

Test set: Average loss: 0.0222, Accuracy: 9927/10000 (99%)

Epoch:  7
loss=0.0024074241518974304 batch_id=1874: 100%|██████████| 1875/1875 [00:26<00:00, 70.80it/s]

Test set: Average loss: 0.0197, Accuracy: 9933/10000 (99%)

Epoch:  8
loss=0.018343215808272362 batch_id=1874: 100%|██████████| 1875/1875 [00:26<00:00, 69.68it/s]

Test set: Average loss: 0.0214, Accuracy: 9931/10000 (99%)

Epoch:  9
loss=0.024688148871064186 batch_id=1874: 100%|██████████| 1875/1875 [00:26<00:00, 70.49it/s]

Test set: Average loss: 0.0258, Accuracy: 9922/10000 (99%)

Epoch:  10
loss=0.018841002136468887 batch_id=1874: 100%|██████████| 1875/1875 [00:26<00:00, 70.33it/s]

Test set: Average loss: 0.0190, Accuracy: 9938/10000 (99%)

Epoch:  11
loss=0.016844671219587326 batch_id=1874: 100%|██████████| 1875/1875 [00:26<00:00, 70.29it/s]

Test set: Average loss: 0.0199, Accuracy: 9941/10000 (99%)

Epoch:  12
loss=0.0022426238283514977 batch_id=1874: 100%|██████████| 1875/1875 [00:26<00:00, 70.46it/s]

Test set: Average loss: 0.0209, Accuracy: 9939/10000 (99%)

Epoch:  13
loss=0.003916250541806221 batch_id=1874: 100%|██████████| 1875/1875 [00:26<00:00, 70.80it/s]

Test set: Average loss: 0.0175, Accuracy: 9948/10000 (99%)

Epoch:  14
loss=0.005738690495491028 batch_id=1874: 100%|██████████| 1875/1875 [00:26<00:00, 69.85it/s]

Test set: Average loss: 0.0196, Accuracy: 9939/10000 (99%)

Epoch:  15
loss=0.004773998167365789 batch_id=1874: 100%|██████████| 1875/1875 [00:27<00:00, 67.55it/s]

Test set: Average loss: 0.0193, Accuracy: 9941/10000 (99%)

Epoch:  16
loss=0.008134204894304276 batch_id=1874: 100%|██████████| 1875/1875 [00:27<00:00, 67.96it/s]

Test set: Average loss: 0.0235, Accuracy: 9930/10000 (99%)

Epoch:  17
loss=0.0012909744400531054 batch_id=1874: 100%|██████████| 1875/1875 [00:26<00:00, 71.14it/s]

Test set: Average loss: 0.0168, Accuracy: 9947/10000 (99%)

Epoch:  18
loss=0.002956618554890156 batch_id=1874: 100%|██████████| 1875/1875 [00:26<00:00, 70.20it/s]

Test set: Average loss: 0.0190, Accuracy: 9940/10000 (99%)

'''

```
## Model

```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 16, 28, 28]             160
       BatchNorm2d-2           [-1, 16, 28, 28]              32
         MaxPool2d-3           [-1, 16, 14, 14]               0
           Dropout-4           [-1, 16, 14, 14]               0
            Conv2d-5           [-1, 32, 14, 14]           4,640
       BatchNorm2d-6           [-1, 32, 14, 14]              64
         MaxPool2d-7             [-1, 32, 7, 7]               0
           Dropout-8             [-1, 32, 7, 7]               0
            Conv2d-9             [-1, 32, 5, 5]           9,248
      BatchNorm2d-10             [-1, 32, 5, 5]              64
          Dropout-11             [-1, 32, 5, 5]               0
           Conv2d-12             [-1, 10, 3, 3]           2,890
AdaptiveAvgPool2d-13             [-1, 10, 1, 1]               0
================================================================
Total params: 17,098
Trainable params: 17,098
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.00
Forward/backward pass size (MB): 0.38
Params size (MB): 0.07
Estimated Total Size (MB): 0.45
----------------------------------------------------------------
```

