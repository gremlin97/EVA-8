## Goal

Make use of the best model architecture I have created to predict MNIST digits and swap out the default Batch Normalization with:
* Network with *Group Normalization*
* Network with *Layer Normalization*
* Network with *L1 + BN*

## Description of model.py neural network (Net) class

  ### Summary:
  * In order to optimize the Number of Params and make the model edge deployable, a squeeze and expand
  model architecture is used. Initially we apply convolutions with 16 output channels tilla RF=5 is reached.
  Then a single maxpool is used at this point to add invariance. Later we expand upto 32 channels then use a
  transition block to squeeze back to 10 channels with a pointwise convolution(mix channels) and then expand
  back to 16 with a default 3x3 convolutions. Later we add some more layers to increase the capacity, add a GAP
  layer to preservere spatial knowledge and obtain amplitudes of classes. Finally a FCL is added to match the output
  shape.
  * In between the model makes use of a control flow to decide which type of normalization stratergy is to be used
  based on the use input (BN+L1, LN, GN). Depending on the num and norm we implement this in the network.
  * Functional programming is use via torch.nn.functional to simplify the code and reduce overhead.
  
  ### Attributes:
  * **norm**: Type of Normalization used Eg:"BN": L1+BatchNorm, "LN":LayerNorm, "GP": Group Norm
  * **x**: Image from the batch
  * **num**: Number of groups for Group Norm 
  * **device**: Default arg is 'cuda:0', i.e gpu. It can be changed to CPU.
  
  ### Output and Output Shape:
  * **out**: Predicted MNIST Digit, shape: [batch_size,10]

## How is Normalization done?
A functional programming based approach is used wherein the we define all the layers except the normalizing layers normally in `model.py`. However for the Normalizing layers depending upon the params supplied during model initialization we will construct the normalizing layers in the forward function with their corresponding functional apis. 

* Snippet to apply BatchNorm in the forward function:
```
if self.norm == 'BN':
          n_chans = x.shape[1]
          # Running mean and variance is calulated by the model by setting training=True
          # Hence we dont need to provide these params ourselves
          running_mean = torch.zeros(n_chans).to(self.device) 
          running_std = torch.ones(n_chans).to(self.device)
          x = F.batch_norm(x, running_mean, running_std, training=True, momentum=0.9)
```

* Snippet to apply LayerNorm in the forward function:
```
else:
      x = F.layer_norm(x, [16, 26, 26]) # We need to input the normalized shape
      x = F.relu(self.conv3(self.drop1(x)))
```

* Snippet to apply GroupNorm in the forward function:
```
elif self.norm == 'GN':
          x = F.group_norm(x,self.num) #num is the number of groups
          x = F.relu(self.conv3(self.drop1(x)))
```

## Accuracy and Loss Graph for all Normalizations

### Accuracy Graph

![Accuracy](https://github.com/gremlin97/EVA-8/blob/main/S5_Norm/Images/acc.png)

### Loss Graph

![Loss](https://github.com/gremlin97/EVA-8/blob/main/S5_Norm/Images/loss.png)

## Misclassified Images for all the Normalizations
The images are displayed in a 5x2 grid and each image has its corresponding actual ground-truth label.

### Group Normalization

![GN](https://github.com/gremlin97/EVA-8/blob/main/S5_Norm/Images/b1.png)

### Layer Normalization

![Loss](https://github.com/gremlin97/EVA-8/blob/main/S5_Norm/Images/b2.png)

### BatchNorm + L1

![BNL1](https://github.com/gremlin97/EVA-8/blob/main/S5_Norm/Images/b3.png)




