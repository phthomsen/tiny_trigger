import torch
import torch.nn as nn
import torch.nn.functional as F

"""

  Here's the layout of the graph:

  (fingerprint_input)
          v
      [Conv2D]<-(weights)
          v
      [BiasAdd]<-(bias)
          v
        [Relu]
          v
      [MatMul]<-(weights)
          v
      [BiasAdd]<-(bias)
          v
output shape of the model:
[number of classes x 1]
we just get the logits as output.

input must be the 4d fingerprint
"""
class tiny_conv(nn.Module):
    def __init__(self, bs, classes, is_training):
        self.bs = bs
        self.classes = classes
        self.is_training = is_training
        super(tiny_conv).__init__()
        # nSamples x nChannels x Height x Width
        self.conv1 = nn.Conv2d(in_channels=1,
                                out_channels=1, 
                                kernel_size=(8,10), 
                                padding="same")
        self.fc2 = nn.Linear()

    def forward(self, x):
        # input data comes with shape (1,49,40,1)
        x.reshape(self.bs, 1, 49, 40)
        # pool ??
        # dropout ??
        x = F.relu(self.conv1(x))
        if self.is_training: 
            x = F.dropout(x, p=0.5)
        x = F.softmax(x)

        
        return x
    

class load_and_convert_model():
    def __init__(self):
        pass
    def _load():
        pass
    def _convert():
    
    def _store


def 