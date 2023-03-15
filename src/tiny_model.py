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
        x = self.bla

        return x