# usage: python nn_mnist.py
from pyimagesearch.nn.neuralnetwork import NeuralNetwork
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn import datasets

digits = datasets.load_digits()
# Here is how this data is structured:
# `digits` is a dictionary containing data that represents 1797 records of mnist, handwritten number.
# the data contains the following fields:
# data: this contains 1797 records of "flattened" (1-D) data, representing the images of mnist, handwritten numbers
# target: a 1-D array that represents the "ground truth" value of each image in the dataset (1797 records)
# images: a list (1797 records) of 2-D arrays that represent the actual images of the hand-written images. All of these images have a size of 8x8
data = digits['data'].astype("float")
data = (data - data.min()) / (data.max() - data.min()) # scale the data so that it only ranges between [0, 1]
print(f"samples: {data.shape[0]}, dim: {data.shape[1]}")

(trainX, testX, trainY, testY) = train_test_split(data, digits.target, test_size=0.25)

# one-hot encoding: converting labels (0 - 9, integers) to "verctors"
# meaning, for instance, if there are 10 possible outcomes, as it is in this case,
# a vector of 0s of len 10 will be instantiated: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
# if the correct output is 8, then the vector will be represented as [0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
trainY = LabelBinarizer().fit_transform(trainY)

ground_truth = testY
testY = LabelBinarizer().fit_transform(testY)

# the 0th layer (input layer) of the neural network should have `trainX.shape[1]` nodes, or 64 nodes,
# which represent the 64 pixels that are flattened out in the `data` variable
# the output layer of the neural network should have 10 nodes, which represent the 10 possible outcomes (0 - 9)
# that the neural network would try to predict
nn = NeuralNetwork([ 
    trainX.shape[1], # 64
    32, 
    16, 
    10 
])
nn.fit(trainX, trainY, iterations=5000)

predictions = nn.predict(testX)
# the output of the sigmoid function will return a "confidence" score for each potential output
# The label with the highest confidence score will then correspond to the properly predicted output
predictions = predictions.argmax(axis=1)

for (pred, y) in zip(predictions, ground_truth): 
    print(f"prediction: {pred}, ground-truth: {y}, correct prediction: {int(pred) == int(y)}")

print(classification_report(testY.argmax(axis=1), predictions))