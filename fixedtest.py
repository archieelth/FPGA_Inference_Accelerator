import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# Load and prepare data
data = pd.read_csv('data/mnist_train.csv')
data = np.array(data)
m, n = data.shape
np.random.shuffle(data)

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
    W1 = np.random.rand(HIDDEN_SIZE, 784) - 0.5
    b1 = np.random.rand(HIDDEN_SIZE, 1) - 0.5
    W2 = np.random.rand(10, HIDDEN_SIZE) - 0.5
    b2 = np.random.rand(10, 1) - 0.5
    return W1, b1, W2, b2

# Activations
def ReLU(Z):
    return np.maximum(Z, 0)

def softmax(Z):
    expZ = np.exp(Z - np.max(Z, axis=0, keepdims=True))
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
    m = X.shape[1]
    one_hot_Y = one_hot(Y)
    dZ2 = A2 - one_hot_Y
    dW2 = (1 / m) * dZ2.dot(A1.T)  
    db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)
    dZ1 = W2.T.dot(dZ2) * ReLU_deriv(Z1)
    dW1 = (1 / m) * dZ1.dot(X.T)
    db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)
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

# ============================================================
# CONFIGURATION — set hidden layer size here before training.
# After training, change HIDDEN_NODES in network.sv to match,
# then recompile Verilator and set MODEL_DIR in testbench.py.
# ============================================================
HIDDEN_SIZE = 64   # ← e.g. 10, 32, 64, 128
MODEL_DIR   = f"models/hidden{HIDDEN_SIZE}"
# ============================================================

# Train the model
W1, b1, W2, b2 = gradient_descent(X_train, Y_train, alpha=0.15, iterations=601)

# ============================================================
# FIXED-POINT SIMULATION - Match hardware behavior exactly
# ============================================================

DATAWIDTH = 16
FRAC = 13
SCALE = 1 << FRAC

def to_fixed_16(x):
    """Convert float to fixed-point, return as signed int16"""
    val = int(np.round(x * SCALE))
    val = np.clip(val, -32768, 32767)
    return np.int16(val)

def from_fixed_16(fixed_val):
    """Convert fixed-point back to float"""
    return float(fixed_val) / SCALE

def fixed_multiply(a_fixed, b_fixed):
    """Simulate hardware multiplication: 16x16->32, then shift right 13"""
    # Convert to int32 for multiplication
    result_32 = np.int32(a_fixed) * np.int32(b_fixed)
    # Arithmetic right shift by 13 (keeps sign)
    result_shifted = result_32 >> 13
    # Saturate back to 16-bit
    return np.clip(result_shifted, -32768, 32767).astype(np.int16)

def fixed_relu(x_fixed):
    """ReLU in fixed-point: clip negative to 0, positive saturate at 32767"""
    if x_fixed < 0:
        return np.int16(0)
    elif x_fixed > 32767:
        return np.int16(32767)
    else:
        return np.int16(x_fixed)

def fixed_forward_neuron(weights_fixed, bias_fixed, inputs_fixed):
    """Simulate one neuron in fixed-point (matches hardware)"""
    # Start with bias (sign-extended to 32-bit)
    sum_32 = np.int32(bias_fixed)
    
    # Accumulate: sum += weight[i] * input[i] >> 13
    for w, x in zip(weights_fixed, inputs_fixed):
        product = fixed_multiply(w, x)
        sum_32 += np.int32(product)
    
    # Saturate to 16-bit
    sum_16 = np.clip(sum_32, -32768, 32767).astype(np.int16)
    
    # Apply ReLU
    return fixed_relu(sum_16)

def fixed_forward_layer(W_fixed, b_fixed, X_fixed):
    """Forward pass for entire layer in fixed-point"""
    num_neurons = W_fixed.shape[0]
    outputs = np.zeros(num_neurons, dtype=np.int16)
    
    for i in range(num_neurons):
        outputs[i] = fixed_forward_neuron(W_fixed[i], b_fixed[i], X_fixed)
    
    return outputs

def fixed_inference(W1_fixed, b1_fixed, W2_fixed, b2_fixed, X_fixed):
    """Complete inference in fixed-point"""
    # Layer 1
    A1_fixed = fixed_forward_layer(W1_fixed, b1_fixed, X_fixed)
    
    # Layer 2
    A2_fixed = fixed_forward_layer(W2_fixed, b2_fixed, A1_fixed)
    
    # Find max (argmax doesn't need to convert back to float)
    prediction = np.argmax(A2_fixed)
    
    return prediction, A1_fixed, A2_fixed

# ============================================================
# Convert weights and biases to fixed-point
# ============================================================

W1_fixed = np.array([[to_fixed_16(w) for w in neuron] for neuron in W1], dtype=np.int16)
b1_fixed = np.array([to_fixed_16(b[0]) for b in b1], dtype=np.int16)
W2_fixed = np.array([[to_fixed_16(w) for w in neuron] for neuron in W2], dtype=np.int16)
b2_fixed = np.array([to_fixed_16(b[0]) for b in b2], dtype=np.int16)

# ============================================================
# Test and compare: Float vs Fixed-point
# ============================================================

print("\n" + "="*60)
print("TESTING: Float-point vs Fixed-point Inference")
print("="*60)

num_tests = 10
matches = 0

for test_idx in range(num_tests):
    # Get test image
    X_test = X_dev[:, test_idx]
    true_label = Y_dev[test_idx]
    
    # Float-point inference (Python reference)
    Z1_float, A1_float, Z2_float, A2_float = forward_prop(
        W1, b1, W2, b2, X_test.reshape(-1, 1)
    )
    pred_float = get_predictions(A2_float)[0]
    
    # Convert input to fixed-point
    X_test_fixed = np.array([to_fixed_16(x) for x in X_test], dtype=np.int16)
    
    # Fixed-point inference (hardware simulation)
    pred_fixed, A1_fixed, A2_fixed = fixed_inference(
        W1_fixed, b1_fixed, W2_fixed, b2_fixed, X_test_fixed
    )
    
    # Compare
    match = "✓" if pred_float == pred_fixed else "✗"
    if pred_float == pred_fixed:
        matches += 1
    
    print(f"\nTest {test_idx}:")
    print(f"  True label:       {true_label}")
    print(f"  Float prediction: {pred_float}")
    print(f"  Fixed prediction: {pred_fixed}")
    print(f"  Match: {match}")
    
    # Show intermediate values for first test
    if test_idx == 0:
        print(f"\n  Layer 1 outputs (first 5 neurons):")
        print(f"    Float: {A1_float[:5, 0]}")
        print(f"    Fixed: {[from_fixed_16(x) for x in A1_fixed[:5]]}")
        print(f"  Layer 2 outputs (all):")
        print(f"    Float: {A2_float[:, 0]}")
        print(f"    Fixed: {[from_fixed_16(x) for x in A2_fixed]}")

print(f"\n" + "="*60)
print(f"Agreement: {matches}/{num_tests} ({100*matches/num_tests:.1f}%)")
print("="*60)

# ============================================================
# Analyze quantization errors
# ============================================================

print("\n" + "="*60)
print("QUANTIZATION ERROR ANALYSIS")
print("="*60)

# Check weight ranges
print(f"\nWeight Statistics:")
print(f"  Layer 1: min={W1.min():.4f}, max={W1.max():.4f}")
print(f"  Layer 2: min={W2.min():.4f}, max={W2.max():.4f}")

# Check if any weights are clipping
W1_clipped = np.sum(np.abs(W1 * SCALE) > 32767)
W2_clipped = np.sum(np.abs(W2 * SCALE) > 32767)
print(f"\nClipped weights:")
print(f"  Layer 1: {W1_clipped}/{W1.size} ({100*W1_clipped/W1.size:.2f}%)")
print(f"  Layer 2: {W2_clipped}/{W2.size} ({100*W2_clipped/W2.size:.2f}%)")

if W1_clipped > 0 or W2_clipped > 0:
    print(".    WARNING: Weights are clipping! Consider:")
    print("   - Using more fractional bits (Q2.13 -> Q1.14)")
    print("   - Training with weight regularization")
    print("   - Scaling weights down")

# ============================================================
# Export files for hardware
# ============================================================

print("\n" + "="*60)
print("EXPORTING FOR HARDWARE")
print("="*60)

# Export weights and biases to MODEL_DIR
import os
os.makedirs(MODEL_DIR, exist_ok=True)
print(f"Exporting to {MODEL_DIR}/")

for n in range(HIDDEN_SIZE):
    with open(f"{MODEL_DIR}/weights_L1_N{n}.hex", "w") as f:
        for w in W1_fixed[n]:
            f.write(f"{int(w) & 0xFFFF:04x}\n")
    with open(f"{MODEL_DIR}/bias_L1_N{n}.hex", "w") as f:
        f.write(f"{int(b1_fixed[n]) & 0xFFFF:04x}\n")

for n in range(10):
    with open(f"{MODEL_DIR}/weights_L2_N{n}.hex", "w") as f:
        for w in W2_fixed[n]:
            f.write(f"{int(w) & 0xFFFF:04x}\n")
    with open(f"{MODEL_DIR}/bias_L2_N{n}.hex", "w") as f:
        f.write(f"{int(b2_fixed[n]) & 0xFFFF:04x}\n")

# Export test images
import os
os.makedirs("test_images", exist_ok=True)

for idx in range(10):
    X_test = X_dev[:, idx]
    X_test_fixed = np.array([to_fixed_16(x) for x in X_test], dtype=np.int16)
    
    with open(f"test_images/image_{idx}.hex", "w") as f:
        for pixel in X_test_fixed:
            unsigned_val = int(pixel) & 0xFFFF
            f.write(f"{unsigned_val:04x}\n")
    
    with open(f"test_images/label_{idx}.txt", "w") as f:
        f.write(f"{Y_dev[idx]}\n")
    
    # Also get expected fixed-point prediction
    pred_fixed, _, _ = fixed_inference(
        W1_fixed, b1_fixed, W2_fixed, b2_fixed, X_test_fixed
    )
    
    with open(f"test_images/expected_{idx}.txt", "w") as f:
        f.write(f"True label: {Y_dev[idx]}\n")
        f.write(f"Fixed-point prediction: {pred_fixed}\n")

print("✓ Exported weights, biases, and test images")
print(f"✓ Created {num_tests} test images in test_images/")

# ============================================================
# Generate detailed trace for debugging (first image only)
# ============================================================

print("\n" + "="*60)
print("GENERATING DEBUG TRACE FOR HARDWARE VERIFICATION")
print("="*60)

X_test = X_dev[:, 0]
X_test_fixed = np.array([to_fixed_16(x) for x in X_test], dtype=np.int16)

with open("debug_trace.txt", "w") as f:
    f.write("="*60 + "\n")
    f.write("DETAILED INFERENCE TRACE - Image 0\n")
    f.write("="*60 + "\n\n")
    
    f.write(f"True label: {Y_dev[0]}\n\n")
    
    # Layer 1
    f.write("LAYER 1 (Hidden Layer)\n")
    f.write("-"*60 + "\n")
    A1_fixed = np.zeros(10, dtype=np.int16)
    
    for neuron_idx in range(10):
        f.write(f"\nNeuron {neuron_idx}:\n")
        f.write(f"  Bias: {b1_fixed[neuron_idx]} ({from_fixed_16(b1_fixed[neuron_idx]):.6f})\n")
        
        sum_32 = np.int32(b1_fixed[neuron_idx])
        f.write(f"  Initial sum: {sum_32}\n")
        
        # Show first 5 weight*input computations
        for i in range(min(5, 784)):
            w = W1_fixed[neuron_idx, i]
            x = X_test_fixed[i]
            prod = fixed_multiply(w, x)
            sum_32 += np.int32(prod)
            f.write(f"    [{i}] w={w} x={x} -> prod={prod}, sum={sum_32}\n")
        
        if 784 > 5:
            f.write(f"    ... ({784-5} more multiplications)\n")
            # Complete the sum
            for i in range(5, 784):
                prod = fixed_multiply(W1_fixed[neuron_idx, i], X_test_fixed[i])
                sum_32 += np.int32(prod)
        
        sum_16 = np.clip(sum_32, -32768, 32767).astype(np.int16)
        output = fixed_relu(sum_16)
        A1_fixed[neuron_idx] = output
        
        f.write(f"  Final sum: {sum_32} -> {sum_16} (16-bit)\n")
        f.write(f"  After ReLU: {output} ({from_fixed_16(output):.6f})\n")
    
    # Layer 2
    f.write("\n" + "="*60 + "\n")
    f.write("LAYER 2 (Output Layer)\n")
    f.write("-"*60 + "\n")
    A2_fixed = np.zeros(10, dtype=np.int16)
    
    for neuron_idx in range(10):
        f.write(f"\nNeuron {neuron_idx}:\n")
        f.write(f"  Bias: {b2_fixed[neuron_idx]} ({from_fixed_16(b2_fixed[neuron_idx]):.6f})\n")
        
        sum_32 = np.int32(b2_fixed[neuron_idx])
        
        for i in range(10):
            w = W2_fixed[neuron_idx, i]
            x = A1_fixed[i]
            prod = fixed_multiply(w, x)
            sum_32 += np.int32(prod)
            f.write(f"    [{i}] w={w} x={x} -> prod={prod}, sum={sum_32}\n")
        
        sum_16 = np.clip(sum_32, -32768, 32767).astype(np.int16)
        output = fixed_relu(sum_16)
        A2_fixed[neuron_idx] = output
        
        f.write(f"  Final sum: {sum_32} -> {sum_16} (16-bit)\n")
        f.write(f"  After ReLU: {output} ({from_fixed_16(output):.6f})\n")
    
    # Final prediction
    prediction = np.argmax(A2_fixed)
    f.write("\n" + "="*60 + "\n")
    f.write("FINAL PREDICTION\n")
    f.write("-"*60 + "\n")
    f.write(f"Layer 2 outputs: {A2_fixed}\n")
    f.write(f"Prediction: {prediction}\n")
    f.write(f"True label: {Y_dev[0]}\n")
    f.write(f"Correct: {'YES' if prediction == Y_dev[0] else 'NO'}\n")
