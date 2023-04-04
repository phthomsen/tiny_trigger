"""
Train a model that has the same input and output as the tensorflow model 'tiny_conv', which looks like this:

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
      
"""