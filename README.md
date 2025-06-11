# 🌀 torch-quantum

*Quantum-native deep-learning layers that slot directly into PyTorch and run on NVIDIA CUDA-Q simulators or real QPUs.*

[![PyPI](https://img.shields.io/pypi/v/torch-qu.svg?logo=pypi)](https://pypi.org/project/torch-qu/) [![Python](https://img.shields.io/pypi/pyversions/torch-qu.svg?logo=python)](https://pypi.org/project/torch-qu) [![License](https://img.shields.io/github/license/SeroviICAI/torch-quantum.svg)](LICENSE)

---

### Why use torch-quantum?  
Current quantum SDKs expose powerful primitives but leave many “deep-learning conveniences” to the user. **Torch-Q** fills that gap by giving PyTorch practitioners a familiar module interface, autograd-compatible parameter-shift gradients and ready-made feature-map / ansatz building blocks. You build models exactly as you do with classic layers, choose a CUDA-Q target (CPU, GPU or cloud QPU) and start iterating.

| Classical DL | torch-quantum | Notes |
|--------------|---------------|-------|
| `nn.Linear`  | `quantum.QuantumLayer` | drop-in layer that outputs class probabilities |
| GPU accel    | cuStateVec / cuTensorNet | automatic, single or multi-GPU |
| Autograd     | parameter-shift implemented in pure PyTorch | works with all optimizers |

---

### Features

* **Drop-in PyTorch modules** (`QuantumLayer`, `QNN`, `HybridQNN`) that behave like any other `nn.Module` and are compatible with `torch.optim`.
* **Flexible circuit construction**: pick standard feature maps (`Z`, `ZZ`) or supply custom kernels; choose from `RealAmplitudes`, `EfficientSU2`, `PauliTwoDesign` ansätze or roll your own.
* **Hybrid workflow**: seamlessly chain classical layers before or after a quantum block, making it easy to build quantum-enhanced CNNs, MLPs or Transformers.
* **Multiple back-ends**: one line (`cudaq.set_target(...)`) switches from local CPU simulation to GPU acceleration or a cloud device (IonQ, Quantinuum, OQC, Infleqtion, Pasqal, QuEra…).
* **Exact gradients** via the parameter-shift rule, automatically invoked by `loss.backward()`; no manual circuit plumbing required.
* **Research-friendly utilities**: empirical Fisher information, layer-wise kernel drawers, circuit depth counters and parameter initialisation helpers.

---

### Installation

```bash
python -m pip install --upgrade pip
pip install cudaq
git clone https://github.com/SeroviICAI/cuda-quantum.git
python - <<'PY'
import importlib.util, pathlib, os, shutil
src = pathlib.Path("cuda-quantum/python/cudaq/kernel/quake_value.py").resolve()
dst = pathlib.Path(importlib.util.find_spec("cudaq.kernel.quake_value").origin)
dst.unlink()
os.symlink(src, dst)
PY
pip install torch-qu
```

### Quick example (4-qubit classifier)
```python
import torch, torch.nn as nn, torch.optim as optim
from torchq.models import QNN
import cudaq

# Use fast GPU simulator if available
cudaq.set_target("nvidia", option="fp32")

model = QNN(
    in_features = 4,        # qubits / input dimension
    out_features = 3,       # number of classes
    num_layers = 2,         # ansatz depth
    shots = 1024,           # measurement shots per forward pass
    feature_map = "zz",     # entangling data embedding
    var_form = "efficientSU2",
    reupload = False
)

x = torch.randn(16, 4)              # mini-batch
y = torch.randint(0, 3, (16,))      # labels
opt = optim.Adam(model.parameters(), lr=0.02)
criterion = nn.CrossEntropyLoss()

for step in range(50):
    opt.zero_grad()
    logits = model(x)               # parameter-shift handled automatically
    loss = criterion(logits, y)
    loss.backward()
    opt.step()
print("final loss:", loss.item())
```

Change back-end to a real device with one line:

```python
# Add CREDENTIALS here
cudaq.set_target("ionq", qpu="qpu.aria-1")  # 25-qubit trapped-ion hardware
```

All circuits are now queued to the cloud without further code changes.

### Documentation & book

The open-access book “Toward a Quantum Advantage in Deep Learning Architectures” lives in `docs/` and is rendered online at https://SeroviICAI.github.io/torch-quantum/book. It introduces quantum mechanics, CUDA-Q programming, variational circuits and information-geometric capacity in detail and Torch-Q code examples.

### Citation

```bibtex
@software{torchquantum2025,
  author  = {Sergio Rodríguez Vidal},
  title   = {torch-quantum: Quantum-ready layers for PyTorch},
  year    = {2025},
  url     = {https://github.com/SeroviICAI/torch-quantum},
  license = {Apache-2.0}
}
```

### License
Apache 2.0 — free to use, modify and distribute, with permissive terms.

> **Happy hacking — and welcome to quantum-enhanced deep learning!**
