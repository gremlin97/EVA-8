## EVA Assignment 4 (S4)

## Goal

To train a neural network model to predict digits from 0-9 using the MNIST Digits Dataset. The model needs to achieve a consistent 99.4% accuracy (for the last few epochs) with a 3-step approach wherein each step incrementally improves the model architecture in-turn increasing the model accuracy.

### Constraints
* Less than 10k params
* Less than 15 Epochs
* Minimum Accuracy of 99.4 % on test data continously for the last few epochs

## Results
* Max Accuracy of 99.20% achieved on the test data
* 7,386 Parameters

## Final Results After Step 3
* Parameters: 7,386
* Best Train Accuracy: 99.07
* Best Test Accuracy: 99.20

## Step 1

**Target:**

Add Dropout, BatchNorm such that conv block stucture is: Conv2D-BatchNorm-Maxpool2D-Dropout. Added GAP Layer.

**Results:**

* Parameters: 17.09k
* Best Test Accuracy: 99.47

**Analysis:**

I am able to cross 99.4% accuracy but the results are not consitent for multiple epochs. Consistent accuracy in 99.3% but the number of parameters are large.

**Colab Link:** https://colab.research.google.com/github/gremlin97/EVA-8/blob/main/S5/Eva3_Step1.ipynb

## Step 2

**Target:**

Added LR Schdeuler with gamma=0.1 and step=6, reduced channel size by reducing the number of kernels for each convolution block. Pushed parms below 10k. Increased learning rate to increasing learning for epochs below 15 and to offset the regularization. Removed Padding=1 to reduce feature map size faster. Added random rotation to image of 7 degrees. Added a Fully Connected Layer after GAP Layer to increase model capacity.

**Results**:

* Parameters: 9,866
* Best Train Accuracy: 98.68
* Best Test Accuracy: 99.23

**Analysis:**
I was able to reduce the model parameters below 10k by reducing the number of filter and maintaining the number of out channels as 16 after each channel. The train accuracy was lower by the test accuracy by around 1% indicating that my model can learn more and achieve higher accuracy. The learning has become harder to to the multiple form of regularizations (dropout, random rotations). Somehow I need to increase the learning.

**Colab Link:** https://colab.research.google.com/github/gremlin97/EVA-8/blob/main/S5/Eva3_Step2.ipynb

## Step 3

**Target:**

Readded padding=1 back to convolutions to reduce the rapid decrease in feature maps resolution. Removed the extra output channels in the last layer and removed the maxpooling in the last block. Added 1 Convolution Layer to increase model capacity.

**Results:**

* Parameters: 7,386
* Best Train Accuracy: 99.07
* Best Test Accuracy: 99.20

**Analysis:**

Difference between training and test accuracy reduces substantially. Testing accuracy is more consistent in the last epochs. Training accuracy increases faster in a linear fashion along with test accuracy. Model params are reduced below 8k.

**Colab Link:** https://colab.research.google.com/github/gremlin97/EVA-8/blob/main/S5/Eva3_Step3.ipynb
