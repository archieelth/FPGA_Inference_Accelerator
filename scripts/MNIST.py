import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# Load and prepare data
data = pd.read_csv('data/mnist_train.csv')
test = pd.read_csv('data/mnist_train.csv')
data = np.array(data)
m, n = data.shape
np.random.shuffle(data)  # shuffle before splitting into dev and training sets

data_dev = data[0:1000].T
Y_dev = data_dev[0]
X_dev = data_dev[1:n]
X_dev = X_dev / 255.

data_train = data[1000:m].T
Y_train = data_train[0]
X_train = data_train[1:n]
X_train = X_train / 255.
_, m_train = X_train.shape

# Initialization
def init_params():
    W1 = np.random.rand(10, 784) - 0.5
    b1 = np.random.rand(10, 1) - 0.5
    W2 = np.random.rand(10, 10) - 0.5
    b2 = np.random.rand(10, 1) - 0.5
    return W1, b1, W2, b2

# Activations
def ReLU(Z):
    return np.maximum(Z, 0)

def softmax(Z):
    expZ = np.exp(Z - np.max(Z, axis=0, keepdims=True))  # Numerical stability
    return expZ / np.sum(expZ, axis=0, keepdims=True)

# Forward propagation
def forward_prop(W1, b1, W2, b2, X):
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2

# Derivatives
def ReLU_deriv(Z):
    return Z > 0

def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y

# Backward propagation
def backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y):
    one_hot_Y = one_hot(Y)
    dZ2 = A2 - one_hot_Y
    dW2 = (1 / m_train) * dZ2.dot(A1.T)  
    db2 = (1 / m_train) * np.sum(dZ2, axis=1, keepdims=True)
    dZ1 = W2.T.dot(dZ2) * ReLU_deriv(Z1)
    dW1 = (1 / m_train) * dZ1.dot(X.T)
    db1 = (1 / m_train) * np.sum(dZ1, axis=1, keepdims=True)
    return dW1, db1, dW2, db2

# Parameter update
def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1
    W2 = W2 - alpha * dW2
    b2 = b2 - alpha * db2
    return W1, b1, W2, b2

# Helpers
def get_predictions(A2):
    return np.argmax(A2, 0)

def get_accuracy(predictions, Y):
    return np.sum(predictions == Y) / Y.size

# Training loop
def gradient_descent(X, Y, alpha, iterations):
    W1, b1, W2, b2 = init_params()
    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
        if i % 10 == 0:
            predictions = get_predictions(A2)
            print(f"Iteration: {i}, Accuracy: {get_accuracy(predictions, Y):.4f}")
    return W1, b1, W2, b2

# Train the model (example: alpha=0.15, iterations=601)
W1, b1, W2, b2 = gradient_descent(X_train, Y_train, alpha=0.15, iterations=1001)

# Fixed-point conversion (assuming signed 16-bit Q2.13 format)
DATAWIDTH = 16
FRAC = 13
SCALE = 1 << FRAC  # Use 1 << FRAC for exact power of 2

def to_signed_fixed_16(x, frac_bits=FRAC):
    scale = 1 << frac_bits
    val = int(np.round(x * scale))
    val = np.clip(val, - (1 << (DATAWIDTH - 1)), (1 << (DATAWIDTH - 1)) - 1)
    return val & ((1 << DATAWIDTH) - 1)  # As uint16 for hex

# Export Layer 1 (hidden layer)
for n in range(10):
    # Weights
    with open(f"weights_L1_N{n}.hex", "w") as f:
        for w in W1[n]:
            f.write(f"{to_signed_fixed_16(w):04x}\n")
    
    # Bias
    with open(f"bias_L1_N{n}.hex", "w") as f:
        f.write(f"{to_signed_fixed_16(b1[n, 0]):04x}\n")

# Export Layer 2 (output layer)
for n in range(10):
    # Weights
    with open(f"weights_L2_N{n}.hex", "w") as f:
        for w in W2[n]:
            f.write(f"{to_signed_fixed_16(w):04x}\n")
    
    # Bias
    with open(f"bias_L2_N{n}.hex", "w") as f:
        f.write(f"{to_signed_fixed_16(b2[n, 0]):04x}\n")