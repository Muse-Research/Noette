# Noette

> **Zero-Token Latent Reasoning for Small Language Models (SLMs)**

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-3776ab.svg)](https://www.python.org/)

**Noette** (*noh-ETT*, from *Noetic*) is a compact neural architecture that replaces verbose Chain-of-Thought (CoT) text tokens with **adaptive, intra-token recurrent pondering in continuous latent space**.

---

## Key Highlights

* **Zero-Token Multi-Step Reasoning:** Executes multi-hop deductions inside continuous representation space before emitting a single answer token.
* **Adaptive Compute:** Allocates 1 step to routine tokens and scales up to $K$ ponder loops on high-entropy reasoning bottlenecks.
* **Edge-Optimized:** Eliminates KV-cache bloat and memory-bandwidth bottlenecks on 100M–350M parameter models.
* **Attractor Stability:** Integrates latent velocity momentum to guide recurrent trajectories toward stable fixed-point attractors.

---

## 📐 How It Works

```
Token Input ──► Embedding ──► [ Universal Recurrent Block ] ──► Halting Gate (p_t)
                                        ▲            │               │
                                        └── Ponder ──┘ (p_t < tau)   ▼ (p_t >= tau)
                                                                 Next Token
```

1. **Intra-Token Pondering:** Hidden states iterate recurrently through a universal shared core ($t = 1 \dots K$).
2. **Epistemic Halting ($H_\theta$):** A learned head dynamically predicts thought convergence probability $p_t$.
3. **Geometric Prior Regularization:** Model is optimized with cross-entropy weighted across all ponder steps and regularized against a Geometric compute prior:
   $$\mathcal{L} = \sum_{t=1}^K q_t \, \mathcal{L}_{\text{CE}}(\hat{y}_t, y) + \beta \, D_{\text{KL}}(q \,\|\, \text{Geom}(\lambda))$$

---

## 🚀 Quickstart

```bash
# Clone and install
git clone https://github.com/Muse-Research/Noette.git
cd Noette
pip install -e .

# Run automated tests
python test_noette.py

# Run live interactive ponder heatmap demo
python demo.py
```

---

## 📊 Minimal Example

```python
import torch
from noette import NoetteConfig, NoetteLM

# Initialize lightweight 200M-scale reasoning core
config = NoetteConfig(
    d_model=1024,
    n_heads=8,
    d_ff=4096,
    max_ponder_steps=6,
    ponder_lambda=0.25
)
model = NoetteLM(config)

# Input tokens dynamically ponder in latent space
inputs = torch.randint(0, config.vocab_size, (1, 16))
outputs = model(inputs)

print("Logits shape:", outputs["logits"].shape)
print("Avg ponder depth per token:", outputs["expected_steps"])
```

---

## 📄 License

Licensed under the [Apache 2.0](LICENSE) License.
