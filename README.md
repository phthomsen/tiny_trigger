# tiny_trigger

## Environment
I'm developing locally on a Macbook Air M1.
MacOs: Ventura 13.1
python version: 3.9.13

A trigger word or wake word detection tiny enough to run on an MCU. The board used here, is the [Arduino Nano 33 BLE](https://store.arduino.cc/products/arduino-nano-33-ble).

# Structure
I have forked the [tflite-micro-arduino-examples repo]( https://github.com/tensorflow/tflite-micro-arduino-examples.git) into this repository. This repo holds the micro library and makes deployment to Arduino very easy. 
# Model
The model is a very simple neural network with a convolutional first layer taking a tensor of heavily pre processed audio data. There are models that are capable of running infrerence on raw audio input, but those are bigger in size and possibly slower in inference which is not what we are looking for here.
## Trainingdata
Training is done on the [Google speech_commands dataset version 0.02](https://arxiv.org/pdf/1804.03209.pdf). The vocabulary is limited, but a good starting point to train a small model.
## Preprocessing
The preprocessing is done using the TensorFlowLiteMicro library.

To individualize the model, I recorded voice samples of friends and me saying the words that should be detected: `Hey tiny`.
## Model transformation
# Application
# Deployment


# Debugging hints
If you want to connect to your board and the correct serial port doesn't show up, you might be using a micro usb cabel that doesn't support data transfer. Sounds simple, but cost me an hour :D
