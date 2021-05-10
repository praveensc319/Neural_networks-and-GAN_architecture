# Neural_networks-and-GAN_architecture

The repository includes the following topics:

* MNIST 10 class classifier
* CIFAR10 10 class classfier
* Convolutional Autoencoder on MNIST data
* Face Generation using GAN architecture

## MNIST 10 class classifier

### About Dataset

It is a dataset of 60,000 small square 28×28 pixel grayscale images of handwritten single digits between 0 and 9. The task is to classify a given image of a handwritten digit into one of 10 classes representing integer values from 0 to 9, inclusively

### Building a Convolutional Neural Network

We will build our model by using high-level Keras API which uses either TensorFlow or Theano on the backend. There are several high-level TensorFlow APIs such as Layers, Keras, and Estimators which helps us create neural networks with high-level knowledge. However, this may lead to confusion since they all vary in their implementation structure. Therefore, if you see completely different codes for the same neural network although they all use TensorFlow. The most straightforward API which is Keras. Import the Sequential Model from Keras and add Conv2D, MaxPooling, Flatten, Dropout, and Dense layers. In addition, Dropout layers fight with the overfitting by disregarding some of the neurons while training while Flatten layers flatten 2D arrays to 1D arrays before building the fully connected layers.

### Optimizer

Adam optimizer is a replacement optimization algorithm for stochastic gradient descent for training deep learning models. Adam combines the best properties of the AdaGrad and RMSProp algorithms to provide an optimization algorithm that can handle sparse gradients on noisy problems.

### Loss

sparse categorical crossentropy when the classes are mutually exclusive (e.g. when each sample belongs exactly to one class) and categorical crossentropy when one sample can have multiple classes or labels are soft probabilities (like [0.5, 0.3, 0.2]).

<p align="center">
  <img src="https://user-images.githubusercontent.com/46927252/117635994-feef2780-b19d-11eb-9f67-8ba952110619.png" />
</p>

This allows to conserve time and memory. Consider case of 10000 classes when they are mutually exclusive - just 1 log instead of summing up 10000 for each sample, just one integer instead of 10000 floats.

### Classification Accuracy

Classification Accuracy is what we usually mean, when we use the term accuracy. It is the ratio of number of correct predictions to the total number of input samples.
Classification test Accuracy of above model is: 97.90 percent 

### Confusion Matrix

Confusion Matrix as the name suggests gives us a matrix as output and describes the complete performance of the model.

## CIFAR10 10 class classifier

### About Dataset

The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images

### Building a Neural Network

We will build our model by using high-level Keras API which uses either TensorFlow or Theano on the backend. There are several high-level TensorFlow APIs such as Layers, Keras, and Estimators which helps us create neural networks with high-level knowledge. However, this may lead to confusion since they all vary in their implementation structure. Therefore, if you see completely different codes for the same neural network although they all use TensorFlow. The most straightforward API which is Keras. Import the Sequential Model from Keras and add Conv2D, MaxPooling, Flatten, Dropout, and Dense layers. In addition, Dropout layers fight with the overfitting by disregarding some of the neurons while training while Flatten layers flatten 2D arrays to 1D arrays before building the fully connected layers.

### Activation Functions

An activation function in a neural network defines how the weighted sum of the input is transformed into an output from a node or nodes in a layer of the network.

Sometimes the activation function is called a “transfer function.” If the output range of the activation function is limited, then it may be called a “squashing function.” Many activation functions are nonlinear and may be referred to as the “nonlinearity” in the layer or the network design.

The choice of activation function has a large impact on the capability and performance of the neural network, and different activation functions may be used in different parts of the model.

#### ReLU 

<p align="center">
<img src="https://user-images.githubusercontent.com/46927252/117638120-2646f400-b1a0-11eb-9955-e0f05635f647.png" data-canonical-src="https://gyazo.com/eb5c5741b6a9a16c692170a41a49c858.png" width="200" height="200" />
</p>

The rectified linear activation function, or ReLU activation function, is perhaps the most common function used for hidden layers.

It is common because it is both simple to implement and effective at overcoming the limitations of other previously popular activation functions, such as Sigmoid and Tanh. Specifically, it is less susceptible to vanishing gradients that prevent deep models from being trained, although it can suffer from other problems like saturated or “dead” units.

#### Softmax

Softmax is a mathematical function that converts a vector of numbers into a vector of probabilities, where the probabilities of each value are proportional to the relative scale of each value in the vector.

### Classification Accuracy

Classification training Accuracy of above model is: 84.76 percent

Classification training Accuracy of above model is: 70.38 percent

The accuracy can significantly increased by increasing the number of Epochs.

## Convolutional Autoencoder on MNIST data

Convolutional Autoencoder is a variant of Convolutional Neural Networks that are used as the tools for unsupervised learning of convolution filters. They are generally applied in the task of image reconstruction to minimize reconstruction errors by learning the optimal filters. Once they are trained in this task, they can be applied to any input in order to extract features. Convolutional Autoencoders are general-purpose feature extractors differently from general autoencoders that completely ignore the 2D image structure. In autoencoders, the image must be unrolled into a single vector and the network must be built following the constraint on the number of inputs.


<p align="center">
  <img src="https://user-images.githubusercontent.com/46927252/117638652-b1c08500-b1a0-11eb-866b-637c5a1efb44.png" />
</p>

<p align="center">
  
[Image Source](https://www.researchgate.net/profile/Xifeng-Guo/publication/320658590/figure/fig1/AS:614154637418504@1523437284408/The-structure-of-proposed-Convolutional-AutoEncoders-CAE-for-MNIST-In-the-middle-there.png)

</p>

## Face Generation using GAN architecture

The GAN is made up of two different neural networks: the discriminator and the generator. The generator generates the images, while the discriminator detects if a face is real or was generated. These two neural networks work as shown below GAN-EVAL:

<p align="center">
  <img src="https://user-images.githubusercontent.com/46927252/117641314-a15dd980-b1a3-11eb-9d52-bc69813db7a7.png" />
</p>

The discriminator accepts an image as its input and produces number that is the probability of the input image being real. The generator accepts a random seed vector and generates an image from that random vector seed. An unlimited number of new images can be created by providing additional seeds.

### Training the Discriminator

<p align="center">
  <img src="https://user-images.githubusercontent.com/46927252/117641414-bd617b00-b1a3-11eb-8386-1ddddc37f75a.png" />
</p>


Here a training set is generated with an equal number of real and fake images. The real images are randomly sampled (chosen) from the training data. An equal number of random images are generated from random seeds. For the discriminator training set, the  x  contains the input images and the  y  contains a value of 1 for real images and 0 for generated ones.

### Training the Generator

<p align="center">
  <img src="https://user-images.githubusercontent.com/46927252/117641579-f13ca080-b1a3-11eb-9357-a51fdb201ae6.png" />
</p>


For the generator training set, the  x  contains the random seeds to generate images and the  y  always contains the value of 1, because the optimal is for the generator to have generated such good images that the discriminiator was fooled into assigning them a probability near 1.

Both the generator and discriminator use Adam and the same learning rate and momentum.

### Training Time

Training Time on MX750 GPU for 400 Epochs is about 6 hrs.

Better result can be obtained by incresing the number of Epochs.

