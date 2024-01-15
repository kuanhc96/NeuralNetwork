# NeuralNetwork
# This is an exploration of how Neural Networks can be implemented by hand
1. nn/pyimagesearch/nn/perceptron.py: this is an implementation of a "neural network" with only one input layer and one
   output layer without any hidden layers. This is a simple "linear" classifier that can correctly classify the output
   of OR and AND datasets, but not XOR
2. nn/pyimagesearch/nn/neuralnetwork.py: this is an implementation of a neural network that can contain an arbitray
   number of input nodes, output nodes, and hidden layers. The math derivation notes can be found in the "Neural Network
   notes.pdf" document. This is used as a non-linear classifier that can correctly classify XOR data, as well as a
   subset of the MNIST dataset
3. nn_tensorflow: this is an implementation of a simple neural network using the tensorflow.keras library. It is used to
   train the full MNIST dataset and CIFAR10 dataset, with plots of the training and validation results indluded. While
   training results for MNIST are decent (~92 accuracy), training results for CIFAR10 are considerably worse (~57
   accuracy) with apparent overfitting
