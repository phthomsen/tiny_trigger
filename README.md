# tiny_trigger
python version: ...

A trigger word or wake word detection tiny enough to run on an MCU. The board used here, is the [Arduino Nano 33 BLE](https://store.arduino.cc/products/arduino-nano-33-ble).

# Structure
I have forked the [tflite-micro]( https://github.com/tensorflow/tflite-micro-arduino-examples.git) into this repository. Operations needed to run models on the device.
# Model
## Training
## Data
For training the model, I used the [Google speech_commands dataset version 0.02](https://arxiv.org/pdf/1804.03209.pdf). The vocabulary is limited, but a good starting point to train a small model.

To individualize the model, I recorded voice samples of friends and me saying the words that should be detected: `Hey tiny`.
## Model transformation
# Application
# Deployment
