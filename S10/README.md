## Goal

* Implemented a fully-convolutional ViT (Vision Transformer) from scratch by referring to the ViT code from the original paper. Implemented fully-convolutional Embedding layer, Transformer layer with Multi-Head Self Attention, PreNorm, class token-based classification using Adaptive Average Pooling and MLP Head.

* Trained the Model on the CIFAR-10 classification task for 24 Epochs and achieved around 65% test accuracy for achieving decent accuracy on an LLM we need to train for more iterations and generally on a larger dataset.

* Apply GradCAM (>5px) on these 10 aforementioned images and understand which regions in these class activations are responsible for these outputs.

## Implementation

* **Embedding:** a module that extracts patches from the input image using a CNN with in_channels input channels, and maps each patch to an embedding_dim-dimensional embedding vector. The module also includes learnable positional encodings.

* **MhSelfAttention:** a module that implements multi-head self-attention over the input patch embeddings. The module first linearly projects the embeddings into out_channels feature maps, which are then split into heads parallel channels. For each channel, the module computes queries, keys, and values using 1x1 convolutions, and applies dot-product attention to compute a weighted sum of the values, using the queries and keys as the attention scores. The resulting weighted sum is then concatenated across the heads and projected back to out_channels feature maps using another 1x1 convolution.

* **FeedForward:** a module that applies a two-layer MLP with GELU activation to each patch embedding separately.

* **PreNormChannels:** a module that applies layer normalization to the patch embeddings before passing them through a given function fn.

* **Transformer:** a module that stacks multiple PreNormChannels blocks, each consisting of a self-attention module followed by a feedforward module. The module applies residual connections around each block.

* **ViT:** the main model module that combines the Embedding, Transformer, and MLP Head modules together to form the full ViT model. The module takes an input image of size image_size x image_size with in_channels input channels, and produces a classification output with 10 classes. The module also includes a layer normalization step before the final output.

## Model Summary

```
Layer (type (var_name))                                 Input Shape          Output Shape         Param #              Trainable
=======================================================================================================================================
ViT (ViT)                                               [1, 3, 32, 32]       [1, 10]              --                   True
├─Embedding (embedding)                                 [1, 3, 32, 32]       [1, 32, 16, 16]      8,192                True
│    └─Conv2d (patch)                                   [1, 3, 32, 32]       [1, 32, 16, 16]      416                  True
├─Transformer (transformer)                             [1, 32, 16, 16]      [1, 32, 16, 16]      --                   True
│    └─ModuleList (layers)                              --                   --                   --                   True
│    │    └─ModuleList (0)                              --                   --                   12,704               True
│    │    └─ModuleList (1)                              --                   --                   12,704               True
│    │    └─ModuleList (2)                              --                   --                   12,704               True
│    │    └─ModuleList (3)                              --                   --                   12,704               True
├─LayerNorm (norm)                                      [1, 16, 16, 32]      [1, 16, 16, 32]      64                   True
├─Sequential (mlp_head)                                 [1, 32, 16, 16]      [1, 10]              --                   True
│    └─GELU (0)                                         [1, 32, 16, 16]      [1, 32, 16, 16]      --                   --
│    └─AdaptiveAvgPool2d (1)                            [1, 32, 16, 16]      [1, 32, 1, 1]        --                   --
│    └─Flatten (2)                                      [1, 32, 1, 1]        [1, 32]              --                   --
│    └─Linear (3)                                       [1, 32]              [1, 10]              330                  True
=======================================================================================================================================
Total params: 59,818
Trainable params: 59,818
Non-trainable params: 0
Total mult-adds (M): 12.99
=======================================================================================================================================
Input size (MB): 0.01
Forward/backward pass size (MB): 3.01
Params size (MB): 0.21
Estimated Total Size (MB): 3.23
============================================================================================================================

```

### Multi-Head Self-Attention

```

class MhSelfAttention(nn.Module):
  def __init__(self, in_channels, out_channels=512, head_channels=8):
    super().__init__()
    self.head_channels = head_channels
    self.heads = out_channels//head_channels
    self.scale = head_channels**-0.5

    self.k = nn.Conv2d(in_channels,out_channels,1)
    self.q = nn.Conv2d(in_channels,out_channels,1)
    self.v = nn.Conv2d(in_channels,out_channels,1)

    self.unify_heads = nn.Conv2d(out_channels,out_channels,1)

  def forward(self, x):

    b,_,h,w = x.shape
    keys = self.k(x).view(b,self.heads,self.head_channels,-1) # bx8x64x196
    queries = self.q(x).view(b,self.heads,self.head_channels,-1)
    values = self.v(x).view(b,self.heads,self.head_channels,-1) 

    attend = queries @ keys.transpose(-2,-1)*self.scale
    attention = F.softmax(attend,dim=-2) # Softmax along the channels (0-1)

    out = attention @ values
    out = out.view(b,-1,h,w)
    out = self.unify_heads(out)

    return out

```

### ViT

```
class ViT(nn.Module):
  def __init__(self,image_size,patch_size,in_channels,embedding_dim,depth,head_channels):
    super().__init__()
    reduced_size = image_size//patch_size

    image_height = image_size
    image_width = image_size

    num_patches = (image_height*image_width)//(patch_size*patch_size)

    self.embedding = Embedding(embedding_dim,num_patches,reduced_size,in_channels,patch_size)
    self.transformer = Transformer(depth,head_channels,embedding_dim,embedding_dim)
    self.norm =  nn.LayerNorm(embedding_dim)
    self.mlp_head = nn.Sequential(
          nn.GELU(),
          nn.AdaptiveAvgPool2d(1),
          nn.Flatten(),
          nn.Linear(embedding_dim, 10)
    )

  def forward(self,x):
    x = self.embedding(x)
    x = self.transformer(x)
    x = x.transpose(1,-1)
    x = self.norm(x)
    x = x.transpose(-1,1)
    x = self.mlp_head(x)
    return x

```

### Training Logs

```
ViT: Epoch: 0 | Train Acc: 0.2049, Test Acc: 0.2904, Time: 78.6, lr: 0.001000
ViT: Epoch: 1 | Train Acc: 0.2459, Test Acc: 0.2719, Time: 72.1, lr: 0.002000
ViT: Epoch: 2 | Train Acc: 0.2628, Test Acc: 0.3221, Time: 68.4, lr: 0.003000
ViT: Epoch: 3 | Train Acc: 0.2956, Test Acc: 0.3350, Time: 68.1, lr: 0.004000
ViT: Epoch: 4 | Train Acc: 0.3384, Test Acc: 0.4035, Time: 70.2, lr: 0.005000
ViT: Epoch: 5 | Train Acc: 0.3870, Test Acc: 0.4614, Time: 68.7, lr: 0.006000
ViT: Epoch: 6 | Train Acc: 0.4248, Test Acc: 0.5008, Time: 69.1, lr: 0.007000
ViT: Epoch: 7 | Train Acc: 0.4421, Test Acc: 0.4991, Time: 70.3, lr: 0.008000
ViT: Epoch: 8 | Train Acc: 0.4646, Test Acc: 0.5187, Time: 68.8, lr: 0.009000
ViT: Epoch: 9 | Train Acc: 0.4723, Test Acc: 0.5259, Time: 68.0, lr: 0.010000
ViT: Epoch: 10 | Train Acc: 0.4879, Test Acc: 0.5434, Time: 66.4, lr: 0.009050
ViT: Epoch: 11 | Train Acc: 0.5014, Test Acc: 0.5681, Time: 66.5, lr: 0.008100
ViT: Epoch: 12 | Train Acc: 0.5158, Test Acc: 0.5588, Time: 64.5, lr: 0.007150
ViT: Epoch: 13 | Train Acc: 0.5273, Test Acc: 0.5836, Time: 64.4, lr: 0.006200
ViT: Epoch: 14 | Train Acc: 0.5413, Test Acc: 0.5816, Time: 64.2, lr: 0.005250
ViT: Epoch: 15 | Train Acc: 0.5524, Test Acc: 0.5995, Time: 68.0, lr: 0.004300
ViT: Epoch: 16 | Train Acc: 0.5685, Test Acc: 0.5987, Time: 68.0, lr: 0.003350
ViT: Epoch: 17 | Train Acc: 0.5797, Test Acc: 0.6161, Time: 65.6, lr: 0.002400
ViT: Epoch: 18 | Train Acc: 0.5951, Test Acc: 0.6267, Time: 66.2, lr: 0.001450
ViT: Epoch: 19 | Train Acc: 0.6038, Test Acc: 0.6462, Time: 65.6, lr: 0.000500
ViT: Epoch: 20 | Train Acc: 0.6136, Test Acc: 0.6458, Time: 67.9, lr: 0.000400
ViT: Epoch: 21 | Train Acc: 0.6133, Test Acc: 0.6501, Time: 68.0, lr: 0.000300
ViT: Epoch: 22 | Train Acc: 0.6188, Test Acc: 0.6471, Time: 67.3, lr: 0.000200
ViT: Epoch: 23 | Train Acc: 0.6197, Test Acc: 0.6479, Time: 66.6, lr: 0.000100
ViT: Epoch: 24 | Train Acc: 0.6232, Test Acc: 0.6506, Time: 65.1, lr: 0.000000

```

#### GradCAM Outputs for Misclassified Images (For their actual corresponding class)

<p float="left">
  <img src="https://github.com/gremlin97/EVA-8/blob/main/S10/Images/1.png" width="200" />
  <img src="https://github.com/gremlin97/EVA-8/blob/main/S10/Images/2.png" width="200" /> 
  <img src="https://github.com/gremlin97/EVA-8/blob/main/S10/Images/3.png" width="200" />
  <img src="https://github.com/gremlin97/EVA-8/blob/main/S10/Images/4.png" width="200" />
  <img src="https://github.com/gremlin97/EVA-8/blob/main/S10/Images/5.png" width="200" /> 
  <img src="https://github.com/gremlin97/EVA-8/blob/main/S10/Images/6.png" width="200" />
  <img src="https://github.com/gremlin97/EVA-8/blob/main/S10/Images/7.png" width="200" />
  <img src="https://github.com/gremlin97/EVA-8/blob/main/S10/Images/8.png" width="200" />
  <img src="https://github.com/gremlin97/EVA-8/blob/main/S10/Images/9.png" width="200" /> 
  <img src="https://github.com/gremlin97/EVA-8/blob/main/S10/Images/10.png" width="200" />
</p>

## Further Resources
* The [Implementation](https://github.com/gremlin97/EVA-8/tree/main/S10/Implementations) folder consists of the implementations of ConvMixer and Attention calculation using Numpy!
