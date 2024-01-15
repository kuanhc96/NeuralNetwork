from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.datasets import mnist
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
import numpy as np
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", required=True, help="path to the output loss/accuracy plot")
args = vars(ap.parse_args())

# keras.mnist dataset: 70000 records of mnist, handwritten data images
# each image has a dimension of 28x28
# trainX.shape = (60000, 28, 28)
# trainX[0].shape = (28, 28)
((trainX, trainY), (testX, testY)) = mnist.load_data()

# reshape the dataset so that each record is "flattened", suitable for neural networks
trainX = trainX.reshape((trainX.shape[0], 28*28*1))
testX = testX.reshape((testX.shape[0], 28*28*1))

trainX = trainX.astype("float32") / 255.0
testX = testX.astype("float32") / 255.0

# encode mnist labels:
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.fit_transform(testY)

# network architecture:
# layer 0 (input layer) will have 784 (28 * 28 * 1) nodes, representing the pixels of the input images
# layer 3 (output layer) will have 10 nodes, representing the 10 potential outcomes (0 - 9) of the classification
model = Sequential()
model.add(Dense(256, input_shape=(784,), activation='sigmoid'))
model.add(Dense(128, activation='sigmoid'))
model.add(Dense(10, activation='softmax'))
# sigmoid functions are generally used for binary classifications;
# softmax functions are generally used for multi-class classifications

sgd = SGD(0.01)
model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"])
H = model.fit(trainX, trainY, validation_data=(testX, testY), epochs=100, batch_size=128)

predictions = model.predict(testX, batch_size=128)
# argmax returns the index of the array that contains the greatest value
print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=[str(x) for x in lb.classes_]))

plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 100), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, 100), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, 100), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, 100), H.history["val_accuracy"], label="val_acc")
plt.title("training loss and accuracy")
plt.xlabel("epoch #")
plt.ylabel("loss/accuracy")
plt.legend()
plt.savefig(args["output"])