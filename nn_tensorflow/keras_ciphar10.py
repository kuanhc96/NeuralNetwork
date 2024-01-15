from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.datasets import cifar10
import matplotlib.pyplot as plt
import numpy as np
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", required=True, help="path to the output loss/accuracy plot")
args = vars(ap.parse_args())

# keras.cifar10 dataset: 60000 records of images of 10 classes, e.g., airplane, automobile, bird, cat, etc.
# each image has a dimension of 32x32x3 (RGB)
# trainX.shape = (50000, 32, 32, 3)
# trainX[0].shape = (32, 32, 3)
((trainX, trainY), (testX, testY)) = cifar10.load_data()

# reshape the dataset so that each record is "flattened", suitable for neural networks
trainX = trainX.reshape((trainX.shape[0], 32*32*3))
testX = testX.reshape((testX.shape[0], 32*32*3))

trainX = trainX.astype("float32") / 255.0
testX = testX.astype("float32") / 255.0

# encode mnist labels:
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.fit_transform(testY)

label_names = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",

]

# network architecture:
# layer 0 (input layer) will have 784 (28 * 28 * 1) nodes, representing the pixels of the input images
# layer 3 (output layer) will have 10 nodes, representing the 10 potential outcomes (0 - 9) of the classification
model = Sequential()
model.add(Dense(1024, input_shape=(3072,), activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(10, activation='softmax'))
# sigmoid functions are generally used for binary classifications;
# softmax functions are generally used for multi-class classifications

sgd = SGD(0.01)
model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"])
H = model.fit(trainX, trainY, validation_data=(testX, testY), epochs=100, batch_size=32)

predictions = model.predict(testX, batch_size=32)
# argmax returns the index of the array that contains the greatest value
print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=label_names))

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
