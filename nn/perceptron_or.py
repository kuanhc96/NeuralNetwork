from pyimagesearch.nn.perceptron import Perceptron
import numpy as np

# construct OR dataset:
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
    [1]
])

p = Perceptron(N=X.shape[1], alpha=0.1)
p.fit(X, y, iterations=20)

for (x, target) in zip(X, y):
    pred = p.predict(x)
    print(f"data={x}, ground-truth={target[0]}, prediction={pred}")