from pyimagesearch.nn.neuralnetwork import NeuralNetwork
import numpy as np

X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])
y = np.array([
    [0],
    [1],
    [1],
    [0]
])

nn = NeuralNetwork([2, 2, 1], alpha=0.5)
nn.fit(X, y, iterations=20000)

for (x, target) in zip(X, y):
    pred = nn.predict(x)[0][0]
    step = 1 if pred > 0.5 else 0
    print(f"data={x}, ground-truth={target[0]}, pred={pred}, step={step}")

nn2 = NeuralNetwork([2, 2, 1], alpha=0.5)
nn2.fit(X, y, iterations=1000)

for (x, target) in zip(X, y):
    pred = nn2.predict(x)[0][0]
    step = 1 if pred > 0.5 else 0
    print(f"data={x}, ground-truth={target[0]}, pred={pred}, step={step}")

nn3 = NeuralNetwork([2, 1], alpha=0.5)
nn3.fit(X, y, iterations=20000)

for (x, target) in zip(X, y):
    pred = nn3.predict(x)[0][0]
    step = 1 if pred > 0.5 else 0
    print(f"data={x}, ground-truth={target[0]}, pred={pred}, step={step}")