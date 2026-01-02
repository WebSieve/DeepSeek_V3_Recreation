# 🧠 DeepSeek V3 Recreation

A clean, educational PyTorch implementation of DeepSeek V3's core architectural innovations, including Multi-Head Latent Attention (MLA) and Mixture of Experts (MoE).

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 🌟 Features

This implementation recreates the key architectural components of DeepSeek V3:

- **Multi-Head Latent Attention (MLA)**: Low-rank key-value compression for efficient attention computation
- **Mixture of Experts (MoE)**: Top-k routing with auxiliary loss-free load balancing
- **DeepSeek Transformer Block**: Complete block combining MLA attention and MoE feed-forward layers
- **RMS Normalization**: Root Mean Square Layer Normalization for stable training

---

## 📋 Table of Contents

- [Architecture Overview](#-architecture-overview)
- [Installation](#-installation)
- [Usage](#-usage)
- [Components](#-components)
- [Model Architecture](#-model-architecture)
- [References](#-references)

---

## 🏗️ Architecture Overview

### Multi-Head Latent Attention (MLA)

MLA introduces a key innovation: **low-rank compression** of key-value projections, dramatically reducing the memory footprint and KV cache size while maintaining model performance.

**Key Features:**
- LoRA-style decomposition for K/V projections (down → up)
- Separate handling of RoPE and non-RoPE components
- Efficient attention computation with shared compressed representations

### Mixture of Experts (MoE)

The MoE layer implements efficient sparse computation through expert routing:

**Key Features:**
- Top-k expert selection per token
- Shared experts (always activated) + routed experts (sparsely activated)
- Auxiliary loss-free load balancing
- SwiGLU activation for expert networks

---

## 🚀 Installation

1. **Clone the repository:**
```bash
git clone https://github.com/yourusername/DeepSeek_V3_Recreation.git
cd DeepSeek_V3_Recreation
```

2. **Create a virtual environment (recommended):**
```bash
python -m venv .venv
source .venv/bin/activate  # On Linux/Mac
# or
.venv\Scripts\activate  # On Windows
```

3. **Install dependencies:**
```bash
pip install torch
```

---

## 💻 Usage

### Quick Start

```python
import torch
from DS_Block import DS_Block

# Create a DeepSeek V3 transformer block
block = DS_Block(
    hidden_size=512,
    num_heads=8,
    kv_lora_rank=64,
    qk_rope_head_dim=32,
    v_head_dim=64,
    intermediate_dim=1024,
    num_experts=8,
    num_experts_per_token=2,
)

# Forward pass
input_tensor = torch.randn(2, 10, 512)  # (batch, seq_len, hidden_size)
output = block(input_tensor)

print(f"Input shape: {input_tensor.shape}")
print(f"Output shape: {output.shape}")
print(f"Parameters: {sum(p.numel() for p in block.parameters()):,}")
```

### Individual Components

#### Multi-Head Latent Attention

```python
from multi_head_latent_attention import Multi_Head_Latent_Attention

mhla = Multi_Head_Latent_Attention(
    hidden_size=512,
    num_heads=8,
    kv_lora_rank=64,
    qk_rope_head_dim=32,
    v_head_dim=64,
)

input_tensor = torch.randn(2, 10, 512)
output = mhla(input_tensor)
```

#### Mixture of Experts

```python
from Mixture_Of_Experts import Mixture_Of_Experts

moe = Mixture_Of_Experts(
    hidden_size=512,
    intermediate_dim=1024,
    num_experts=8,
    num_experts_per_token=2,
)

input_tensor = torch.randn(2, 10, 512)
output = moe(input_tensor)
```

---

## 🧩 Components

### 1. **DS_Block.py**

The main transformer block that combines attention and MoE layers with residual connections:
- RMS normalization before each sublayer
- Multi-Head Latent Attention for self-attention
- Mixture of Experts for feed-forward computation
- Residual connections around both sublayers

### 2. **multi_head_latent_attention.py**

Implements the novel MLA mechanism:
- **Down projection**: Compresses hidden states to lower rank
- **Up projection**: Expands to multi-head key-value representations
- **RoPE integration**: Separate handling of rotary position embeddings
- **Efficient attention**: Scaled dot-product attention with optional masking

### 3. **Mixture_Of_Experts.py**

Sparse MoE layer with top-k routing:
- **Router network**: Learns expert affinity scores
- **Shared experts**: Always-active base computation
- **Routed experts**: Sparsely activated based on routing
- **SwiGLU activation**: Gated linear units for expert networks

### 4. **RMS_Norm.py**

Root Mean Square Layer Normalization:
- Simpler alternative to LayerNorm
- More stable gradients during training
- Learnable scale parameters

---

## 📊 Model Architecture

```
┌─────────────────────────────────────┐
│         Input (Hidden States)       │
└──────────────┬──────────────────────┘
               │
               ▼
       ┌───────────────┐
       │   RMS Norm    │
       └───────┬───────┘
               │
               ▼
  ┌────────────────────────────┐
  │  Multi-Head Latent Attn    │
  │  - Low-rank KV compression │
  │  - RoPE integration        │
  └────────────┬───────────────┘
               │
               ▼ (+ residual)
               │
               ▼
       ┌───────────────┐
       │   RMS Norm    │
       └───────┬───────┘
               │
               ▼
  ┌────────────────────────────┐
  │   Mixture of Experts       │
  │  - Shared experts          │
  │  - Top-k routed experts    │
  │  - Load balancing          │
  └────────────┬───────────────┘
               │
               ▼ (+ residual)
               │
               ▼
┌──────────────────────────────────────┐
│        Output (Hidden States)        │
└──────────────────────────────────────┘
```

---

## 📖 References

This implementation is inspired by the DeepSeek V3 architecture. For more details, refer to:

- **DeepSeek V3 Paper**: [Link to paper when available]
- **Multi-Head Latent Attention**: Novel attention mechanism with low-rank KV compression
- **Mixture of Experts**: Sparse computation for efficient scaling

---

## 🔧 Configuration Parameters

| Parameter | Description | Example Value |
|-----------|-------------|---------------|
| `hidden_size` | Model embedding dimension | 512, 1024, 4096 |
| `num_heads` | Number of attention heads | 8, 16, 32 |
| `kv_lora_rank` | Rank for KV compression | 64, 128, 256 |
| `qk_rope_head_dim` | Dimension for RoPE in Q/K | 32, 64, 128 |
| `v_head_dim` | Dimension per value head | 64, 128, 256 |
| `intermediate_dim` | MoE hidden dimension | 1024, 2048, 4096 |
| `num_experts` | Total number of experts | 4, 8, 16, 64 |
| `num_experts_per_token` | Active experts per token | 2, 4, 8 |

---

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Report bugs
- Suggest features
- Submit pull requests
- Improve documentation

---

## 📝 License

This project is open source and available under the MIT License.

---

## ⭐ Acknowledgments

This is an educational recreation of DeepSeek V3's architecture for learning and research purposes. Credit goes to the DeepSeek team for their innovative architecture design.

---

**Made with ❤️ for the ML community**