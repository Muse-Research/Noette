# Noette

**Zero-Token Latent Reasoning for Small Language Models**

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-3776ab.svg)](https://www.python.org/)

Noette is a neural architecture for Small Language Models (SLMs) that replaces external Chain-of-Thought (CoT) token generation with adaptive, intra-token recurrent computation in continuous latent space.

---

## Motivation

Standard autoregressive Transformers apply uniform compute across all token positions, allocating identical layer depth to routine syntax and complex deductions. Generating multi-step scratchpads externally introduces significant key-value cache overhead and memory-bandwidth bottlenecks on resource-constrained devices.

Noette addresses this by decoupling token emission from computational depth. Difficult tokens trigger internal recurrent attractor dynamics in latent space, resolving multi-step deductions before collapsing to the next output token.

---

## Architecture

```
Input Tokens -> Embedding -> [ Recurrent Core Block ] -> Halting Gate (p_t)
                                   ^            |              |
                                   +-- Ponder --+ (p_t < tau)  v (p_t >= tau)
                                                             Output Token
```

### 1. Intra-Token Recurrent Latent Pondering
At any sequence position, the hidden state iterates through a shared recurrent core:

$$h_t = \text{AttractorMomentum}\Big(\text{TransformerBlock}(h_{t-1}), \, h_{t-1}, \, v_{t-1}\Big)$$

where $t \in [1, K]$ denotes the internal ponder step, and $v_t$ tracks latent velocity to stabilize trajectory convergence.

### 2. Epistemic Halting Head
An epistemic projection head estimates thought convergence probability $p_t$:

$$p_t = \sigma(W_h \cdot \text{RMSNorm}(h_t) + b_h)$$

The halting probability mass assigned to step $t$ is:

$$q_t = p_t \prod_{j=1}^{t-1} (1 - p_j)$$

### 3. Ponder Loss Formulation
The objective function optimizes task loss across all intermediate compute steps while penalizing deviation from a Geometric compute prior $\text{Geom}(\lambda)$:

$$\mathcal{L} = \sum_{t=1}^K q_t \, \mathcal{L}_{\text{CE}}(\hat{y}_t, y) + \beta \, D_{\text{KL}}\big(q_{1:K} \,\|\, \text{Geom}(\lambda)\big)$$

---

## Installation & Usage

```bash
git clone https://github.com/Muse-Research/Noette.git
cd Noette
pip install -e .
```

### Python API

```python
import torch
from noette import NoetteConfig, NoetteLM

config = NoetteConfig(
    d_model=1024,
    n_heads=8,
    d_ff=4096,
    max_ponder_steps=6,
    ponder_lambda=0.25,
    halting_threshold=0.95
)

model = NoetteLM(config)

input_ids = torch.randint(0, config.vocab_size, (1, 32))
outputs = model(input_ids)

print("Output Logits:", outputs["logits"].shape)
print("Expected Ponder Depth:", outputs["expected_steps"])
```

### Running Verification & Benchmark

```bash
# Execute unit test suite
python test_noette.py

# Run comparative benchmark
python -m noette.benchmark
```

---

## License

Apache 2.0. See [LICENSE](LICENSE) for details.
