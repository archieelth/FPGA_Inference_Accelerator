import numpy as np
import pandas as pd
from PIL import Image


DATAWIDTH = 16
FRAC = 13
SCALE = 1 << FRAC


def from_signed_fixed_16(hex_str, frac_bits=FRAC):
    """Convert hex string (uint16) to float"""
    val = int(hex_str, 16)
    if val & (1 << (DATAWIDTH - 1)):  # negative
        val -= (1 << DATAWIDTH)
    return val / (1 << frac_bits)

def load_vector_hex(filename):
    with open(filename, "r") as f:
        return np.array([from_signed_fixed_16(line.strip()) for line in f])


def load_model():
    # Layer 1
    W1 = np.zeros((10, 784))
    b1 = np.zeros((10, 1))
    for n in range(10):
        W1[n] = load_vector_hex(f"weights_L1_N{n}.hex")
        b1[n, 0] = load_vector_hex(f"bias_L1_N{n}.hex")[0]

    # Layer 2
    W2 = np.zeros((10, 10))
    b2 = np.zeros((10, 1))
    for n in range(10):
        W2[n] = load_vector_hex(f"weights_L2_N{n}.hex")
        b2[n, 0] = load_vector_hex(f"bias_L2_N{n}.hex")[0]

    return W1, b1, W2, b2

def ReLU(Z):
    return np.maximum(Z, 0)


def float_to_fixed(x, frac_bits=FRAC):
    val = int(np.round(x * (1 << frac_bits)))
    val = np.clip(val, -(1 << (DATAWIDTH - 1)), (1 << (DATAWIDTH - 1)) - 1)
    return val

def softmax(Z):
    expZ = np.exp(Z - np.max(Z, axis=0, keepdims=True))
    return expZ / np.sum(expZ, axis=0, keepdims=True)

def forward_prop_fixed(W1, b1, W2, b2, X):
    """
    Bit-accurate fixed-point forward pass
    Matches SystemVerilog behavior
    """

    # ---- Quantize inputs ----
    X_q = np.array([float_to_fixed(x) for x in X[:, 0]], dtype=np.int32)

    A1_q = np.zeros(10, dtype=np.int32)

    # ---- Layer 1 ----
    for i in range(10):
        acc = 0  # int32 accumulator

        for j in range(784):
            w_q = float_to_fixed(W1[i, j])
            acc += w_q * X_q[j]        # Q26

        acc >>= FRAC                  # back to Q13
        acc += float_to_fixed(b1[i, 0])

        # ReLU
        if acc < 0:
            acc = 0

        # Saturate to int16
        acc = max(min(acc, 0x7FFF), -0x8000)

        A1_q[i] = acc

        # Debug print (MATCHES SV)
        print(
            f"A1[{i}] = {acc / SCALE:.6f} -> "
            f"{format(acc & 0xFFFF, '016b')} "
            f"(0x{acc & 0xFFFF:04X})"
        )

    # ---- Layer 2 (same idea, but using A1_q) ----
    A2_q = np.zeros(10, dtype=np.int32)

    for i in range(10):
        acc = 0
        for j in range(10):
            w_q = float_to_fixed(W2[i, j])
            acc += w_q * A1_q[j]

        acc >>= FRAC


def get_predictions(A2):
    return np.argmax(A2, axis=0)

def get_accuracy(predictions, Y):
    return np.mean(predictions == Y)

if __name__ == "__main__":
    # Load MNIST
    idx = 1  # change this to test different images
    data = pd.read_csv("data/mnist_train.csv").values

    row = data[idx]
    # Remove label
    pixels = np.array(row[1:], dtype=np.uint8)

    # Reshape to 28x28
    image_array = pixels.reshape((28, 28))

    # Create image
    image = Image.fromarray(image_array, mode="L")
    image.show()

    # Select ONE sample

    Y = data[idx, 0]
    X = data[idx, 1:].reshape(-1, 1) / 255.0  # shape (784, 1)

    # Load model
    W1, b1, W2, b2 = load_model()

    print (W1)
    
    # Forward pass
    A2 = forward_prop_fixed(W1, b1, W2, b2, X)
    prediction = get_predictions(A2)[0]


    print("True label:", Y)
    print("Predicted :", prediction)
    print("Probabilities:", A2.flatten())

