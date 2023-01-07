# Assignment 2.5

## Data Representation
* The custom dataloader returns the sum of the MNIST Data and the random number in decimal format. 
* The trained multibranch neural network returns the sum and MNIST predicted digit in one-hot encoded format. We use argmax to return the value of the sum and MNIST digit in decimal format.

## Data Generation Stratergy

* The Random MNIST class creates an MNIST dataset which uses the original uncompressed MNIST Files, converts them to Train and Test Numpy arrays  respectively and converts them to appropriate tensors. 
* It also randomly generates a random number as a class variable for each image.

```
__getitem__: Returns the image, corresponding label, random number and sum between the random number and label (As Tensors)

__item__: Returns the length of the dataset

```

#### Attributes:

* arr: Numpy array of dataset (Train/Test)
* labels_arr: Corresponding Labels array of arr

#### Arguments:

dir: Working Directory
train: (Bool) True to generate train, False to generate Test Dataset
transform: Dataset Transform


## Model Architecture
Two branches are created to predict both the MNIST Digit and the sum of the label and random number.The first branch applies a 
convolutional  block till a receptive field of 14 is reached (>11), those features are flattened to a size of 128, and relevant features 
are extracted from the images. The other branch accepts a random number from the batch in the forward function, is passed through a set of
linear layers till we obtain logits in a flattened array of size 128. The first branch is continued through linear layers and an output of
size 10 is obtained which corresponds to the output size. In the second branch the logits of size 128 are concatenated with the 128 
logits/features and passed though another layer to obtain a class output of 18 neurons (max sum of the output of sum and label)

#### Attributes:
* random: Random number for the image in the batch
* image: Image from the batch
* label: Digit prediction corresponding to the image
* sum: random + label

#### Output and Output Shape:
* out: Predicted MNIST Digit, shape: [batch_size,10]
* rand: Predicted sum of the label and random number: [batch_size,18]

### Loss Function
* I make use of Cross Entropy Loss as the problem is a classification task in terms of predicting 0-9 digits and sum from 0-18. 
* The outputs are probablistic in nature and hence we use log-loss to make sure the loss is very high in case of an incorrect label and very low in case of a correct label

#### Results

* We calculate the total loss (Sum + MNIST Digit) and individual losses (Sum/MNIST Digit), total accuracy and individual accuracies (Image Accuracy/ Sum Accuracy)
* Training Logs:
```
# Saved training logs of Random Net

    Epoch: 0 Total Correct: 25197 Total Correct Images: 58230 Loss: 1.0580025911331177 Sum Accuracy: 0.41995 Image Accuracy: 0.9705
    Epoch: 1 Total Correct: 34460 Total Correct Images: 59254 Loss: 1.1102770864963531 Sum Accuracy: 0.5743333333333334 Image Accuracy: 0.9875666666666667
    Epoch: 2 Total Correct: 37668 Total Correct Images: 59439 Loss: 0.9038642928935587 Sum Accuracy: 0.6278 Image Accuracy: 0.99065
    Epoch: 3 Total Correct: 39881 Total Correct Images: 59549 Loss: 1.588500349316746 Sum Accuracy: 0.6646833333333333 Image Accuracy: 0.9924833333333334
    Epoch: 4 Total Correct: 41397 Total Correct Images: 59593 Loss: 0.8106887647882104 Sum Accuracy: 0.68995 Image Accuracy: 0.9932166666666666
    Epoch: 5 Total Correct: 48281 Total Correct Images: 59860 Loss: 0.38063156968564726 Sum Accuracy: 0.8046833333333333 Image Accuracy: 0.9976666666666667
    Epoch: 6 Total Correct: 49149 Total Correct Images: 59905 Loss: 0.3384809614290134 Sum Accuracy: 0.81915 Image Accuracy: 0.9984166666666666
    Epoch: 7 Total Correct: 49466 Total Correct Images: 59929 Loss: 0.6633256559725851 Sum Accuracy: 0.8244333333333334 Image Accuracy: 0.9988166666666667
    Epoch: 8 Total Correct: 49553 Total Correct Images: 59937 Loss: 0.5482223332801368 Sum Accuracy: 0.8258833333333333 Image Accuracy: 0.99895
    Epoch: 9 Total Correct: 49630 Total Correct Images: 59950 Loss: 0.6610703746628133 Sum Accuracy: 0.8271666666666667 Image Accuracy: 0.9991666666666666
    Epoch: 10 Total Correct: 50468 Total Correct Images: 59972 Loss: 0.8481923531508073 Sum Accuracy: 0.8411333333333333 Image Accuracy: 0.9995333333333334
    Epoch: 11 Total Correct: 50407 Total Correct Images: 59973 Loss: 0.5151217428501695 Sum Accuracy: 0.8401166666666666 Image Accuracy: 0.99955
    Epoch: 12 Total Correct: 50378 Total Correct Images: 59974 Loss: 0.4910212531685829 Sum Accuracy: 0.8396333333333333 Image Accuracy: 0.9995666666666667
    Epoch: 13 Total Correct: 50407 Total Correct Images: 59974 Loss: 0.47970792889100267 Sum Accuracy: 0.8401166666666666 Image Accuracy: 0.9995666666666667
    Epoch: 14 Total Correct: 50537 Total Correct Images: 59975 Loss: 1.2680460136143665 Sum Accuracy: 0.8422833333333334 Image Accuracy: 0.9995833333333334
    Epoch: 15 Total Correct: 50494 Total Correct Images: 59979 Loss: 0.3732307219179347 Sum Accuracy: 0.8415666666666667 Image Accuracy: 0.99965
    Epoch: 16 Total Correct: 50661 Total Correct Images: 59979 Loss: 0.4400448156229686 Sum Accuracy: 0.84435 Image Accuracy: 0.99965
    Epoch: 17 Total Correct: 50614 Total Correct Images: 59979 Loss: 0.7322045542550768 Sum Accuracy: 0.8435666666666667 Image Accuracy: 0.99965
    Epoch: 18 Total Correct: 50629 Total Correct Images: 59979 Loss: 1.1139443628489971 Sum Accuracy: 0.8438166666666667 Image Accuracy: 0.99965
    Epoch: 19 Total Correct: 50519 Total Correct Images: 59979 Loss: 1.1734270639717579 Sum Accuracy: 0.8419833333333333 Image Accuracy: 0.99965


    Epoch: 0 Total Correct: 24994 Total Correct Images: 58009 Loss: 1.6786536276340485 Sum Accuracy: 0.41656666666666664 Image Accuracy: 0.9668166666666667
    Epoch: 1 Total Correct: 33653 Total Correct Images: 59227 Loss: 0.7367602166486904 Sum Accuracy: 0.5608833333333333 Image Accuracy: 0.9871166666666666
    Epoch: 2 Total Correct: 36920 Total Correct Images: 59441 Loss: 2.4037311747670174 Sum Accuracy: 0.6153333333333333 Image Accuracy: 0.9906833333333334
    Epoch: 3 Total Correct: 38875 Total Correct Images: 59542 Loss: 0.8679653368890285 Sum Accuracy: 0.6479166666666667 Image Accuracy: 0.9923666666666666
    Epoch: 4 Total Correct: 40520 Total Correct Images: 59613 Loss: 0.6755946689772827 Sum Accuracy: 0.6753333333333333 Image Accuracy: 0.99355
    Epoch: 5 Total Correct: 47574 Total Correct Images: 59878 Loss: 0.7885043825954199 Sum Accuracy: 0.7929 Image Accuracy: 0.9979666666666667
    Epoch: 6 Total Correct: 48626 Total Correct Images: 59929 Loss: 0.67600476564985 Sum Accuracy: 0.8104333333333333 Image Accuracy: 0.9988166666666667
    Epoch: 7 Total Correct: 48838 Total Correct Images: 59943 Loss: 0.7899998782668263 Sum Accuracy: 0.8139666666666666 Image Accuracy: 0.99905
    Epoch: 8 Total Correct: 49084 Total Correct Images: 59958 Loss: 0.6330844123876886 Sum Accuracy: 0.8180666666666667 Image Accuracy: 0.9993
    Epoch: 9 Total Correct: 49330 Total Correct Images: 59968 Loss: 0.39457018786561093 Sum Accuracy: 0.8221666666666667 Image Accuracy: 0.9994666666666666
    Epoch: 10 Total Correct: 50073 Total Correct Images: 59983 Loss: 0.9593964765081182 Sum Accuracy: 0.83455 Image Accuracy: 0.9997166666666667
    Epoch: 11 Total Correct: 50152 Total Correct Images: 59983 Loss: 0.3564134389453102 Sum Accuracy: 0.8358666666666666 Image Accuracy: 0.9997166666666667
    Epoch: 12 Total Correct: 50175 Total Correct Images: 59982 Loss: 0.2667074873538695 Sum Accuracy: 0.83625 Image Accuracy: 0.9997
    Epoch: 13 Total Correct: 50236 Total Correct Images: 59981 Loss: 0.4812454587809043 Sum Accuracy: 0.8372666666666667 Image Accuracy: 0.9996833333333334
    Epoch: 14 Total Correct: 50319 Total Correct Images: 59983 Loss: 0.27486886792576115 Sum Accuracy: 0.83865 Image Accuracy: 0.9997166666666667
    Epoch: 15 Total Correct: 50390 Total Correct Images: 59984 Loss: 0.3603895604597369 Sum Accuracy: 0.8398333333333333 Image Accuracy: 0.9997333333333334
    Epoch: 16 Total Correct: 50242 Total Correct Images: 59984 Loss: 0.4148408840137563 Sum Accuracy: 0.8373666666666667 Image Accuracy: 0.9997333333333334
    Epoch: 17 Total Correct: 50283 Total Correct Images: 59984 Loss: 0.5959093793844659 Sum Accuracy: 0.83805 Image Accuracy: 0.9997333333333334
    Epoch: 18 Total Correct: 50349 Total Correct Images: 59984 Loss: 0.3229017183139149 Sum Accuracy: 0.83915 Image Accuracy: 0.9997333333333334
    Epoch: 19 Total Correct: 50334 Total Correct Images: 59984 Loss: 0.32100466638780745 Sum Accuracy: 0.8389 Image Accuracy: 0.9997333333333334
```
### Final Results: 
* Best Training Sum Accuracy: 84.19833333333333 %
* Best Training Image Accuracy: 99.97333333333334 %
