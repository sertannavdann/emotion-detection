# Facial Expression Recognition

This project implements a facial expression recognition system using Convolutional Neural Networks (CNN) and Visual Transformers (Multi-Head Attention). The FER2013 dataset is augmented using various image processing techniques to improve the model's performance. The final model is exported as an ONNX file for deployment.

## Description
This facial expression recognition system combines the strengths of Convolutional Neural Networks (CNN) and Visual Transformers (Multi-Head Attention). The FER2013 dataset is used as the basis for training and testing, with multiple augmentation techniques applied to improve the model's generalization capabilities.

<p align="center">
  <img src="https://i0.wp.com/developersbreach.com/wp-content/uploads/2020/08/cnn_banner.png?fit=1200%2C564&ssl=1" width="400" alt="Image">
  <img src="https://miro.medium.com/v2/resize:fit:1400/1*inc9Sty8xMFNNYlNVn9iBQ.png" width="400" alt="Image">
</p>

### Live Deployment

The live deployment takes the ONNX file and applies it to a live video stream. Faces are detected using Haar cascades, resized to match the model's input dimensions, and then passed through the model. The output is the class (emotion) with the highest probability.

<p align="center">
  <img src="Accuracy&LossMaps/take1.png" alt="Powell Project Outline" width="800" />
</p>

### CPP Version

A C++ version of the project compiles everything into a 4 MB executable. It generates a graph of the detected emotions and saves it in JSON format.

<p align="center">
  <img src="Accuracy&LossMaps/elon.png" alt="Elon Musk" width="400" />
  <img src="Accuracy&LossMaps/jim.png" alt="Jim Carrey" width="400" />
  <img src="Accuracy&LossMaps/trump.png" alt="Donald Trump" width="400" />
</p>

### Accuracies & Architecture

<p align="center">
  <img src="Accuracy&LossMaps/accuracy_graph_Adam.png" alt="MobileNet2 accuracy" width="800" />
</p>
<p align="center">
  <img src="Accuracy&LossMaps/loss_graph_Adam.png" alt="MobileNet2 loss" width="800" />
</p>

<p align="center">
  <img src="Accuracy&LossMaps/YoloV5.jpg" alt="YoloV5" width="500" />
</p>
<p align="center">
  <img src="Accuracy&LossMaps/YoloV5-modeldef.jpg" alt="YoloV5-modeldef" width="500" />
</p>
