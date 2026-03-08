# NeuralNet-for-FPGA

A fully parameterisable fixed-point neural network implemented in SystemVerilog and simulated with Verilator, trained on the MNIST handwritten digit dataset. The hardware implementation is bit-exact with the Python reference model.

---

## 1. Background

In a neural network we create a series of neurons, structured in layers, to map inputs to outputs. Each neuron computes a weighted sum of its inputs plus a bias:

$$n_j = \sum^I_{i=0} w_{ji} x_i + b_j$$

Where $I$ is the number of nodes in the previous layer and $j$ is the node in the current layer.

Without an activation function this would be a purely linear system. To introduce non-linearity, each neuron applies ReLU after the accumulation:

$$\mathrm{ReLU}(x) = \begin{cases} x & \text{if } x > 0 \\ 0 & \text{if } x \le 0 \end{cases}$$

The network is trained on the MNIST dataset — 28×28 greyscale images of handwritten digits (0–9), giving 784 inputs per image.

![MNIST Network](/assets/MNISTexmple.png)

---

## 2. Python Implementation

Training is done in floating-point Python (`fixedtest.py`) using gradient descent with backpropagation and a softmax output layer. The trained weights are then quantised to fixed-point and exported as `.hex` files for the hardware.

A bit-accurate fixed-point inference function mirrors the hardware MAC behaviour exactly, allowing direct comparison between the Python model and the simulation output.

---

## 3. Fixed-Point Representation

The hardware uses **Q2.13 signed 16-bit fixed point** — 1 sign bit, 2 integer bits, 13 fractional bits. This gives a range of approximately ±4 with a precision of $2^{-13} \approx 0.000122$.

The MAC operation accumulates products in a **48-bit Q4.26 accumulator** (the result of two Q2.13 multiplications), then right-shifts by 13 at the end to return to Q2.13 before adding the bias. This single-shift approach preserves precision across the entire accumulation, unlike per-multiply rounding which introduces cumulative error.

---

## 4. Quantization-Aware Training

Naively training in float and then quantising the weights introduces **quantization noise** — small rounding errors that compound across neurons and layers. This degrades accuracy compared to the float model.

Quantization-aware training (QAT) addresses this by periodically snapping weights to their fixed-point representation *during* training, so the model learns to be robust to that rounding. Implemented in `scripts/quantization.py`, this:

- Rounds weights to Q2.13 every N iterations during the training loop
- Forces the gradient updates to account for the precision limits of the hardware
- Results in weights that are already near their quantised values at export time, minimising the accuracy gap between the float and fixed-point models

The benefit is that the exported weights behave consistently in hardware — the Python fixed-point inference and the hardware simulation agree bit-for-bit on every prediction.

---

## 5. Hardware Architecture

The SystemVerilog design (`rtl/`) implements a two-layer feedforward network:

```
784 inputs → [Layer 1: HIDDEN_NODES neurons] → [Layer 2: 10 neurons] → argmax → digit
```

Key modules:

| Module | Description |
|---|---|
| `network.sv` | Top-level FSM, coordinates all modules |
| `layer.sv` | Generates N neurons in parallel via `generate` |
| `neuron.sv` | Single neuron: MAC accumulator + ReLU |
| `pixel_stream.sv` | Streams image pixels to Layer 1 |
| `neuron_loader.sv` | Streams Layer 1 outputs to Layer 2 |
| `result_finder.sv` | Argmax over Layer 2 outputs |

The network is **fully parameterisable** — all widths and depths are driven by four parameters at the top of `rtl/network.sv`:

| Parameter | Default | Description |
|---|---|---|
| `HIDDEN_NODES` | `64` | Hidden layer neuron count — main accuracy knob |
| `NUMWEIGHTL1` | `784` | Input size (pixels per image) |
| `DATAWIDTH` | `16` | Bit width of all data and weights |
| `NEURONNODES` | `10` | Output classes (digits 0–9) |

Counter widths throughout use `$clog2` so they automatically resize when parameters change. To change the hidden layer size, only `HIDDEN_NODES` needs updating — all downstream signal widths and instantiation parameters derive from it.

---

## 6. Results

All results are from the Verilator simulation. Python fixed-point inference is bit-exact with the hardware on every test.

| Model | Hidden Neurons | Float Accuracy | HW Accuracy | HW == PY |
|---|---|---|---|---|
| `models/hidden10` | 10 | ~82% | 80% | 100% |
| `models/hidden64` | 64 | ~91% | 95% | 100% |

The jump from 10 to 64 hidden neurons gives a ~15 percentage point improvement in hardware accuracy. The 100% HW==PY agreement across all tests confirms the fixed-point Python model is a reliable predictor of hardware behaviour before synthesis.

![Results hidden64](/results_hidden64.png)

---

## 7. Project Structure

```
rtl/              SystemVerilog source + Verilator C++ driver
models/           Trained weights — one subfolder per configuration
data/             MNIST CSV dataset
test_images/      Pre-generated test images (exported by fixedtest.py)
scripts/          Utility scripts (quantization, standalone inference, etc.)
fixedtest.py      Train a model and export weights to models/
testbench.py      Run hardware simulation + Python comparison + plot
Makefile          Build system
COMMANDS.txt      Quick reference for all commands
```

---

## 8. Usage

**Train a model**
```bash
# Set HIDDEN_SIZE in fixedtest.py, then:
python3 fixedtest.py
```

**Build the simulation**
```bash
# Set HIDDEN_NODES in rtl/network.sv to match, then:
make build
```

**Run the testbench**
```bash
# Set MODEL_DIR + HIDDEN_SIZE in testbench.py, then:
python3 testbench.py
```

**Run a single image manually**
```bash
./obj_dir/Vnetwork +IMAGEFILE=test_images/image_0.hex +WEIGHTSDIR=models/hidden64
```

**View waveforms**
```bash
gtkwave wave.vcd
```

See `COMMANDS.txt` for the full reference.

---

Thanks for viewing my project, I hope you found it interesting.
