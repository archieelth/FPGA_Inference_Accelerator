import subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
import time


# ============================================================
MODEL_DIR   = "models/hidden64"   #  path to weights & biases
HIDDEN_SIZE = 64                  #  hidden neurons in this model
NUM_IMAGES  = 20                  #  how many images to test
# ============================================================

#  Fixed-point config (must match SV) 

DATAWIDTH = 16
FRAC      = 13
SCALE     = 1 << FRAC
MAX_INT16 =  0x7FFF
MIN_INT16 = -0x8000

def float_to_fixed(x):
    val = int(np.round(x * SCALE))
    return int(np.clip(val, MIN_INT16, MAX_INT16))

def load_hex_as_int(filename):
    """Load a hex file as raw signed Q2.13 integers (same values hardware sees)."""
    vals = []
    with open(filename) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            v = int(line, 16)
            if v & (1 << (DATAWIDTH - 1)):
                v -= (1 << DATAWIDTH)
            vals.append(v)
    return np.array(vals, dtype=np.int64)

def pixels_to_hex_file(pixels_uint8, path):
    """Convert raw uint8 pixels to Q2.13 hex and write for the simulator."""
    with open(path, "w") as f:
        for p in pixels_uint8:
            val = float_to_fixed(int(p) / 255.0) & 0xFFFF
            f.write(f"{val:04x}\n")

#  Load weights from MODEL_DIR 
def load_model(model_dir, hidden_size):
    W1_q = np.zeros((hidden_size, 784), dtype=np.int64)
    b1_q = np.zeros(hidden_size,        dtype=np.int64)
    for n in range(hidden_size):
        W1_q[n] = load_hex_as_int(f"{model_dir}/weights_L1_N{n}.hex")
        b1_q[n] = load_hex_as_int(f"{model_dir}/bias_L1_N{n}.hex")[0]

    W2_q = np.zeros((10, hidden_size), dtype=np.int64)
    b2_q = np.zeros(10,               dtype=np.int64)
    for n in range(10):
        W2_q[n] = load_hex_as_int(f"{model_dir}/weights_L2_N{n}.hex")
        b2_q[n] = load_hex_as_int(f"{model_dir}/bias_L2_N{n}.hex")[0]

    return W1_q, b1_q, W2_q, b2_q

#  Bit-accurate fixed-point inference (mirrors SV exactly) 
def infer_fixed(W1_q, b1_q, W2_q, b2_q, X_q):
    hidden_size = W1_q.shape[0]

    A1 = np.zeros(hidden_size, dtype=np.int64)
    for i in range(hidden_size):
        acc = np.int64(0)
        for j in range(784):
            acc += W1_q[i, j] * X_q[j]
        acc >>= FRAC
        acc += b1_q[i]
        acc = max(np.int64(0), acc)
        A1[i] = int(np.clip(acc, MIN_INT16, MAX_INT16))

    A2 = np.zeros(10, dtype=np.int64)
    for i in range(10):
        acc = np.int64(0)
        for j in range(hidden_size):
            acc += W2_q[i, j] * A1[j]
        acc >>= FRAC
        acc += b2_q[i]
        acc = max(np.int64(0), acc)
        A2[i] = int(np.clip(acc, MIN_INT16, MAX_INT16))

    return int(np.argmax(A2))

# Run hardware simulation for one image 
def run_hw(image_hex_path, model_dir):
    result = subprocess.run(
        ["./obj_dir/Vnetwork",
         f"+IMAGEFILE={image_hex_path}",
         f"+WEIGHTSDIR={model_dir}"],
        capture_output=True, text=True
    )
    for line in result.stdout.splitlines():
        if line.startswith("HW_RESULT:"):
            return int(line.split(":")[1].strip())
    return None

#  Get image & label for a given index 
GEN_DIR = "generated_images"

def get_image(idx, csv_data):
    """
    For idx 0-9: use pre-generated test_images/ (from training).
    For idx >= 10: generate on the fly from the CSV.
    Returns (hex_path, X_q, label, pixels_uint8).
    """
    if idx < 10:
        hex_path = f"test_images/image_{idx}.hex"
        X_q      = load_hex_as_int(hex_path)
        with open(f"test_images/expected_{idx}.txt") as f:
            for line in f:
                if line.startswith("True label:"):
                    label = int(line.split(":")[1].strip())
        pixels = np.clip(np.round(X_q / SCALE * 255), 0, 255).astype(np.uint8)
    else:
        os.makedirs(GEN_DIR, exist_ok=True)
        row    = csv_data[idx]
        label  = int(row[0])
        pixels = row[1:].astype(np.uint8)
        hex_path = f"{GEN_DIR}/image_{idx}.hex"
        pixels_to_hex_file(pixels, hex_path)
        X_q = load_hex_as_int(hex_path)

    return hex_path, X_q, label, pixels

#  Visualization 
def plot_results(results, model_dir, output_path="results.png"):
    n     = len(results)
    ncols = min(n, 5)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 2.4, nrows * 2.8),
                             facecolor="#1a1a2e")
    fig.suptitle(f"FPGA Neural Network — MNIST Inference ({model_dir})",
                 color="white", fontsize=12, fontweight="bold", y=1.01)

    axes_flat = np.array(axes).flatten() if n > 1 else [axes]

    for ax, res in zip(axes_flat, results):
        pixels, label, hw_pred, py_pred = res
        ax.imshow(pixels.reshape(28, 28), cmap="gray", interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])

        hw_correct = hw_pred == label
        colour = "#00e676" if hw_correct else "#ff1744"
        for spine in ax.spines.values():
            spine.set_edgecolor(colour)
            spine.set_linewidth(3)

        ax.set_title(f"Label: {label}", color="white", fontsize=9, fontweight="bold", pad=4)
        ax.set_xlabel(
            f"HW: {hw_pred} {'✓' if hw_correct else '✗'}   "
            f"PY: {py_pred} {'✓' if py_pred == label else '✗'}",
            color=colour, fontsize=8, labelpad=4
        )
        ax.set_facecolor("#0d0d1a")

    for ax in axes_flat[len(results):]:
        ax.set_visible(False)

    correct_patch = mpatches.Patch(color="#00e676", label="HW correct")
    wrong_patch   = mpatches.Patch(color="#ff1744", label="HW wrong")
    fig.legend(handles=[correct_patch, wrong_patch],
               loc="lower center", ncol=2, framealpha=0.2,
               labelcolor="white", fontsize=9, bbox_to_anchor=(0.5, -0.03))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"\nSaved figure → {output_path}")
    plt.show()

# ── Main testbench 
if __name__ == "__main__":
    print(f"Model : {MODEL_DIR}  (hidden={HIDDEN_SIZE})")
    print("Loading weights...")
    W1_q, b1_q, W2_q, b2_q = load_model(MODEL_DIR, HIDDEN_SIZE)

    csv_data = None
    if NUM_IMAGES > 10:
        print("Loading MNIST CSV for extra images...")
        csv_data = pd.read_csv("data/mnist_train.csv").values

    print(f"\nRunning {NUM_IMAGES} images...\n")
    print(f"{'Idx':>4}  {'Label':>5}  {'HW':>4}  {'PY':>4}  {'HW==Label':>10}  {'PY==Label':>10}  {'HW==PY':>7}")
    print("-" * 62)

    hw_correct  = 0
    py_correct  = 0
    hw_py_match = 0
    results     = []
    hw_total    = 0.0
    py_total    = 0.0

    for idx in range(NUM_IMAGES):
        hex_path, X_q, label, pixels = get_image(idx, csv_data)

        t0 = time.perf_counter()
        hw_pred = run_hw(hex_path, MODEL_DIR)
        hw_time = time.perf_counter() - t0

        t0 = time.perf_counter()
        py_pred = infer_fixed(W1_q, b1_q, W2_q, b2_q, X_q)
        py_time = time.perf_counter() - t0

        hw_total += hw_time
        py_total += py_time

        hw_ok = hw_pred == label
        py_ok = py_pred == label
        match = hw_pred == py_pred

        hw_correct  += hw_ok
        py_correct  += py_ok
        hw_py_match += match
        results.append((pixels, label, hw_pred, py_pred))

        hw_str = str(hw_pred) if hw_pred is not None else "TIMEOUT"
        print(f"{idx:>4}  {label:>5}  {hw_str:>4}  {py_pred:>4}  "
              f"{'YES' if hw_ok else 'NO':>10}  {'YES' if py_ok else 'NO':>10}  "
              f"{'YES' if match else 'NO':>7}  "
              f"HW:{hw_time*1000:6.1f}ms  PY:{py_time*1000:6.2f}ms")

    print("-" * 62)
    print(f"\nSummary over {NUM_IMAGES} images ({MODEL_DIR}):")
    print(f"  HW  accuracy : {hw_correct}/{NUM_IMAGES} ({100*hw_correct/NUM_IMAGES:.1f}%)")
    print(f"  PY  accuracy : {py_correct}/{NUM_IMAGES} ({100*py_correct/NUM_IMAGES:.1f}%)")
    print(f"  HW == PY     : {hw_py_match}/{NUM_IMAGES} ({100*hw_py_match/NUM_IMAGES:.1f}%)")
    print(f"\nTiming over {NUM_IMAGES} images:")
    print(f"  HW avg (sim + process spawn) : {hw_total/NUM_IMAGES*1000:.1f} ms")
    print(f"  PY avg (fixed-point Python)  : {py_total/NUM_IMAGES*1000:.2f} ms")
    print(f"  Note: HW time includes OS process overhead (~10-50ms per run)")

    model_name = MODEL_DIR.replace("/", "_").replace("models_", "")
    plot_results(results, MODEL_DIR, output_path=f"results_{model_name}.png")
