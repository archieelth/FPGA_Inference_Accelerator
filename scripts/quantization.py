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

# ============================================================
# FIXED-POINT CONFIGURATION
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
    result_32 = np.int32(a_fixed) * np.int32(b_fixed)
    result_shifted = result_32 >> 13
    return np.clip(result_shifted, -32768, 32767).astype(np.int16)

def fixed_relu(x_fixed):
    """ReLU in fixed-point: clip negative to 0, positive saturate at 32767"""
    if x_fixed < 0:
        return np.int16(0)
    elif x_fixed > 32767:
        return np.int16(32767)
    else:
        return np.int16(x_fixed)

# ============================================================
# TRAINING WITH FIXED-POINT AWARE QUANTIZATION
# ============================================================

def init_params():
    """Initialize with smaller weights to avoid clipping"""
    W1 = (np.random.rand(10, 784) - 0.5) * 0.5  # Reduced range
    b1 = (np.random.rand(10, 1) - 0.5) * 0.5
    W2 = (np.random.rand(10, 10) - 0.5) * 0.5
    b2 = (np.random.rand(10, 1) - 0.5) * 0.5
    return W1, b1, W2, b2

def quantize_weights(W1, b1, W2, b2):
    """Quantize weights to fixed-point and back (simulates quantization during training)"""
    W1_q = np.array([[from_fixed_16(to_fixed_16(w)) for w in neuron] for neuron in W1])
    b1_q = np.array([[from_fixed_16(to_fixed_16(b))] for b in b1.flatten()])
    W2_q = np.array([[from_fixed_16(to_fixed_16(w)) for w in neuron] for neuron in W2])
    b2_q = np.array([[from_fixed_16(to_fixed_16(b))] for b in b2.flatten()])
    return W1_q, b1_q, W2_q, b2_q

def ReLU(Z):
    return np.maximum(Z, 0)

def softmax(Z):
    expZ = np.exp(Z - np.max(Z, axis=0, keepdims=True))
    return expZ / np.sum(expZ, axis=0, keepdims=True)

def forward_prop(W1, b1, W2, b2, X):
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2

def ReLU_deriv(Z):
    return Z > 0

def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y

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

def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1
    W2 = W2 - alpha * dW2
    b2 = b2 - alpha * db2
    return W1, b1, W2, b2

def get_predictions(A2):
    return np.argmax(A2, 0)

def get_accuracy(predictions, Y):
    return np.sum(predictions == Y) / Y.size

def gradient_descent_quantized(X, Y, alpha, iterations, quantize_every=10):
    """Training with periodic quantization to fixed-point"""
    W1, b1, W2, b2 = init_params()
    
    print("\n" + "="*60)
    print("TRAINING WITH FIXED-POINT AWARE QUANTIZATION")
    print("="*60)
    print(f"Quantizing weights every {quantize_every} iterations\n")
    
    for i in range(iterations):
        # Quantize weights periodically to simulate fixed-point constraints
        if i % quantize_every == 0 and i > 0:
            W1, b1, W2, b2 = quantize_weights(W1, b1, W2, b2)
        
        # Normal forward/backward pass
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
        
        if i % 10 == 0:
            predictions = get_predictions(A2)
            acc = get_accuracy(predictions, Y)
            
            # Check for weight clipping
            W1_clip = np.sum(np.abs(W1 * SCALE) > 32767)
            W2_clip = np.sum(np.abs(W2 * SCALE) > 32767)
            clip_pct = 100 * (W1_clip + W2_clip) / (W1.size + W2.size)
            
            print(f"Iter {i:3d}: Acc={acc:.4f}, Clipped={clip_pct:.2f}%")
    
    # Final quantization
    W1, b1, W2, b2 = quantize_weights(W1, b1, W2, b2)
    
    return W1, b1, W2, b2

# ============================================================
# FIXED-POINT INFERENCE (Hardware Simulation)
# ============================================================

def fixed_forward_neuron(weights_fixed, bias_fixed, inputs_fixed):
    """Simulate one neuron in fixed-point (matches hardware)"""
    sum_32 = np.int32(bias_fixed)
    
    for w, x in zip(weights_fixed, inputs_fixed):
        product = fixed_multiply(w, x)
        sum_32 += np.int32(product)
    
    sum_16 = np.clip(sum_32, -32768, 32767).astype(np.int16)
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
    A1_fixed = fixed_forward_layer(W1_fixed, b1_fixed, X_fixed)
    A2_fixed = fixed_forward_layer(W2_fixed, b2_fixed, A1_fixed)
    prediction = np.argmax(A2_fixed)
    return prediction, A1_fixed, A2_fixed

# ============================================================
# TRAIN THE MODEL
# ============================================================

W1, b1, W2, b2 = gradient_descent_quantized(
    X_train, Y_train, 
    alpha=0.10,  # Slightly lower learning rate for stability
    iterations=601,
    quantize_every=10
)

# ============================================================
# Convert to fixed-point for hardware
# ============================================================

W1_fixed = np.array([[to_fixed_16(w) for w in neuron] for neuron in W1], dtype=np.int16)
b1_fixed = np.array([to_fixed_16(b[0]) for b in b1], dtype=np.int16)
W2_fixed = np.array([[to_fixed_16(w) for w in neuron] for neuron in W2], dtype=np.int16)
b2_fixed = np.array([to_fixed_16(b[0]) for b in b2], dtype=np.int16)

# ============================================================
# TEST: Float vs Fixed-point
# ============================================================

print("\n" + "="*60)
print("TESTING: Float-point vs Fixed-point Inference")
print("="*60)

num_tests = 20
matches = 0
float_correct = 0
fixed_correct = 0

for test_idx in range(num_tests):
    X_test = X_dev[:, test_idx]
    true_label = Y_dev[test_idx]
    
    # Float-point inference
    Z1_float, A1_float, Z2_float, A2_float = forward_prop(
        W1, b1, W2, b2, X_test.reshape(-1, 1)
    )
    pred_float = get_predictions(A2_float)[0]
    
    # Fixed-point inference
    X_test_fixed = np.array([to_fixed_16(x) for x in X_test], dtype=np.int16)
    pred_fixed, A1_fixed, A2_fixed = fixed_inference(
        W1_fixed, b1_fixed, W2_fixed, b2_fixed, X_test_fixed
    )
    
    # Compare
    match = pred_float == pred_fixed
    if match:
        matches += 1
    if pred_float == true_label:
        float_correct += 1
    if pred_fixed == true_label:
        fixed_correct += 1
    
    if test_idx < 10:  # Show first 10
        match_sym = "✓" if match else "✗"
        print(f"Test {test_idx:2d}: True={true_label} Float={pred_float} Fixed={pred_fixed} {match_sym}")

print(f"\n" + "-"*60)
print(f"Float vs Fixed Agreement: {matches}/{num_tests} ({100*matches/num_tests:.1f}%)")
print(f"Float Accuracy: {float_correct}/{num_tests} ({100*float_correct/num_tests:.1f}%)")
print(f"Fixed Accuracy: {fixed_correct}/{num_tests} ({100*fixed_correct/num_tests:.1f}%)")
print("="*60)

# ============================================================
# ANALYZE QUANTIZATION
# ============================================================

print("\n" + "="*60)
print("QUANTIZATION ERROR ANALYSIS")
print("="*60)

print(f"\nWeight Statistics:")
print(f"  Layer 1: min={W1.min():.4f}, max={W1.max():.4f}")
print(f"  Layer 2: min={W2.min():.4f}, max={W2.max():.4f}")

W1_clipped = np.sum(np.abs(W1 * SCALE) > 32767)
W2_clipped = np.sum(np.abs(W2 * SCALE) > 32767)
print(f"\nClipped weights:")
print(f"  Layer 1: {W1_clipped}/{W1.size} ({100*W1_clipped/W1.size:.2f}%)")
print(f"  Layer 2: {W2_clipped}/{W2.size} ({100*W2_clipped/W2.size:.2f}%)")

if W1_clipped > 0 or W2_clipped > 0:
    print("\n⚠️  WARNING: Some weights are still clipping!")
    print("   Consider: lower learning rate, more training, or Q1.14 format")

# ============================================================
# EXPORT FOR HARDWARE
# ============================================================

print("\n" + "="*60)
print("EXPORTING FOR HARDWARE")
print("="*60)

# Export weights and biases
for n in range(10):
    with open(f"weights_L1_N{n}.hex", "w") as f:
        for w in W1_fixed[n]:
            unsigned_val = int(w) & 0xFFFF
            f.write(f"{unsigned_val:04x}\n")
    
    with open(f"bias_L1_N{n}.hex", "w") as f:
        unsigned_val = int(b1_fixed[n]) & 0xFFFF
        f.write(f"{unsigned_val:04x}\n")

for n in range(10):
    with open(f"weights_L2_N{n}.hex", "w") as f:
        for w in W2_fixed[n]:
            unsigned_val = int(w) & 0xFFFF
            f.write(f"{unsigned_val:04x}\n")
    
    with open(f"bias_L2_N{n}.hex", "w") as f:
        unsigned_val = int(b2_fixed[n]) & 0xFFFF
        f.write(f"{unsigned_val:04x}\n")

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
    
    # Get predictions for this image
    pred_fixed, _, _ = fixed_inference(
        W1_fixed, b1_fixed, W2_fixed, b2_fixed, X_test_fixed
    )
    
    Z1_float, A1_float, Z2_float, A2_float = forward_prop(
        W1, b1, W2, b2, X_test.reshape(-1, 1)
    )
    pred_float = get_predictions(A2_float)[0]
    
    with open(f"test_images/expected_{idx}.txt", "w") as f:
        f.write(f"True label: {Y_dev[idx]}\n")
        f.write(f"Float prediction: {pred_float}\n")
        f.write(f"Fixed-point prediction: {pred_fixed}\n")
        f.write(f"Match: {'YES' if pred_float == pred_fixed else 'NO'}\n")

print("✓ Exported weights, biases, and test images")
print(f"✓ Created test images in test_images/")

# ============================================================
# DETAILED DEBUG TRACE
# ============================================================

print("\n" + "="*60)
print("GENERATING DEBUG TRACE")
print("="*60)

X_test = X_dev[:, 0]
X_test_fixed = np.array([to_fixed_16(x) for x in X_test], dtype=np.int16)

with open("debug_trace.txt", "w") as f:
    f.write("="*60 + "\n")
    f.write("DETAILED INFERENCE TRACE - Image 0\n")
    f.write(f"Format: Q2.13 (1 sign bit, 2 integer bits, 13 fractional bits)\n")
    f.write(f"Scale factor: {SCALE}\n")
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
        
        # Show first 5 computations
        for i in range(min(5, 784)):
            w = W1_fixed[neuron_idx, i]
            x = X_test_fixed[i]
            prod = fixed_multiply(w, x)
            sum_32 += np.int32(prod)
            f.write(f"    [{i}] w={w:6d} ({from_fixed_16(w):7.4f}) * "
                   f"x={x:6d} ({from_fixed_16(x):7.4f}) = "
                   f"prod={prod:6d} ({from_fixed_16(prod):7.4f}), "
                   f"sum={sum_32}\n")
        
        if 784 > 5:
            f.write(f"    ... ({784-5} more multiplications)\n")
            for i in range(5, 784):
                prod = fixed_multiply(W1_fixed[neuron_idx, i], X_test_fixed[i])
                sum_32 += np.int32(prod)
        
        sum_16 = np.clip(sum_32, -32768, 32767).astype(np.int16)
        output = fixed_relu(sum_16)
        A1_fixed[neuron_idx] = output
        
        f.write(f"  Final sum (32-bit): {sum_32}\n")
        f.write(f"  Saturated (16-bit): {sum_16} ({from_fixed_16(sum_16):.6f})\n")
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
            f.write(f"    [{i}] w={w:6d} ({from_fixed_16(w):7.4f}) * "
                   f"x={x:6d} ({from_fixed_16(x):7.4f}) = "
                   f"prod={prod:6d} ({from_fixed_16(prod):7.4f}), "
                   f"sum={sum_32}\n")
        
        sum_16 = np.clip(sum_32, -32768, 32767).astype(np.int16)
        output = fixed_relu(sum_16)
        A2_fixed[neuron_idx] = output
        
        f.write(f"  Final sum (32-bit): {sum_32}\n")
        f.write(f"  Saturated (16-bit): {sum_16} ({from_fixed_16(sum_16):.6f})\n")
        f.write(f"  After ReLU: {output} ({from_fixed_16(output):.6f})\n")
    
    # Final
    prediction = np.argmax(A2_fixed)
    f.write("\n" + "="*60 + "\n")
    f.write("FINAL PREDICTION\n")
    f.write("-"*60 + "\n")
    f.write("Layer 2 outputs (fixed-point):\n")
    for i, val in enumerate(A2_fixed):
        f.write(f"  Class {i}: {val:6d} ({from_fixed_16(val):7.4f})\n")
    f.write(f"\nPrediction: {prediction}\n")
    f.write(f"True label: {Y_dev[0]}\n")
    f.write(f"Correct: {'YES' if prediction == Y_dev[0] else 'NO'}\n")

print("✓ Generated debug_trace.txt")
print("\n" + "="*60)
print("READY FOR HARDWARE TESTING!")
print("="*60)
print("\nNext steps:")
print("  1. Run your hardware simulation")
print("  2. Compare with debug_trace.txt")
print("  3. Check that intermediate sums match")
print("  4. Verify multiply-accumulate sequence")