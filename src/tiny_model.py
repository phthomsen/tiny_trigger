import torch
import torch.nn as nn
import torch.nn.functional as F

"""

  (fingerprint_input)
          v
      [Conv2D]<-(weights)
          v
      [BiasAdd]<-(bias)
          v
        [Relu]
          v
      [Conv2D]<-(weights)
          v
      [BiasAdd]<-(bias)
          v
        [Relu]
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
    def __init__(self):
        super(tiny_conv).__init__()
        # nSamples x nChannels x Height x Width
        self.conv1 = nn.Conv2d()
        self.conv2 = nn.Conv2d()
        self.conv3 = nn.Conv2d()
        self.fc = nn.Linear()

    def forward(self, x):
        # pool ??
        x = F.relu(self.conv1(x))

        
        return x
    

class load_and_convert_model():
    def __init__(self):
        pass
    def _load():
        pass
    def _convert():
    
    def _store


def 