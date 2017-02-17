# ANN_CNN_fromScratch
This is the code sample that implements the full connected and convolutional neural network models for deep learning totally from scratch.

## Getting started
This code is written in python and the libraries needed are listed in the file requirement.txt.

## Framework
Following a modular design, when constructing a neural network for deep learning computation, it is divided into two steps. First of all, a neural network model is constructed, where the network structure is detailedly defined. Secondly, a certain kind of solver is called to update the learnable parameters of the model build previously and output the evaluated loss and accuracy results periodically.

## How to use
### Define the Model
There are two kinds of neural network models available here: full connected neural network and convolution neural network.

#### Full connected neural network
The structure of the full connected neural network is specified as:

 *{affine - [batch normalization] - relu - [dropout]} x (L - 1) - affine - softmax*
  
Here the number of the hidden layers is L-1. The activation function for the hidden layer is ReLu and for the output layer is softmax. Batch normalization can be turned on for better convergence. The dropout parameter specifies the probability that a neuron in the hidden layers is dropped, so if it is set as 0, then no dropout will be applied.
#### Convolution neural network
The structure of the convolution neural network is specified as:

 *{conv - [spatialbatchnorm] - relu - [max pool]} x N - {affine - [batchnorm] - relu - [dropout]} x M - affine - softmax*
 
 Here the number of the convolution layers is N and the full connected hidden layers is M. Most parameters are similar to the full connected neural network except that the maximum pooling layer can be selected to use or not right after the activation function of the convolution layers.

### Select the solver
The solver takes the responsibility of updating all the learnable parameters in the predefined model, most importantly, the weights and biases. There are four kinds of first-order update methods available: Vanilla SGD, SGD Momentum, RMSprop and Adam. In practice, Adam method usually works the best.Several important parameters to be specified are: initial learning rate, batch size and total number of training epochs. 

## Demo
To illustrate how to use this code, two demo samples are provided. Specifically, both full connected neural network and convolution neural network models are individually created to classify the CIFAR10 dataset which is stored in the 'Dataset' folder. You will need to download the CIFAR-10 dataset. To do so, please run the following command from the Dataset directory:

./get_datasets.sh

Before using the convolution neural network model, please run the following command from the Model directory:

python setup.py build_ext --inplace

## Author
- **Di Jin** -- jindi15@mit.edu
