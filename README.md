# centralperkGPT

A decoder-only Transformer built completely from scratch in PyTorch and trained on a custom **Friends-inspired Software Engineering dialogue dataset**.

This project implements the core architecture behind modern GPT-style models — including multi-head causal self-attention, residual connections, feed-forward networks, and autoregressive text generation — without relying on high-level libraries.

The goal of this repository is educational depth and architectural clarity: every component of the model is explicitly constructed to understand how large language models actually work under the hood.

---

## 🚀 Features

* Decoder-only Transformer architecture
* Multi-head causal self-attention
* Pre-LayerNorm blocks
* Residual connections
* Position + token embeddings
* Feed-forward (MLP) layers
* Dropout regularization
* Autoregressive character-level generation
* Custom training loop with validation tracking

---

## 🧠 Architecture Overview

centralperkGPT follows the standard GPT design pattern:

Token Embeddings + Positional Embeddings
→ Stacked Transformer Blocks
→ LayerNorm
→ Linear Language Modeling Head

Each Transformer block contains:

* Multi-head self-attention (causal masked)
* Feed-forward network (4× expansion)
* Residual pathways
* Layer normalization

The model is trained to predict the next character given previous context.

---

## ⚙️ Model Configuration

| Parameter       | Value           |
| --------------- | --------------- |
| Layers          | 6               |
| Attention Heads | 6               |
| Embedding Size  | 384             |
| Context Length  | 256             |
| Dropout         | 0.2             |
| Vocabulary      | 87 characters   |
| Dataset Size    | ~76K characters |

---

## 📉 Training Behavior

The model quickly learns dialogue structure, speaker formatting, and conversational tone.

Due to the relatively small dataset, overfitting begins after extended training — demonstrating the expressive power of even modest transformer architectures.

This repository intentionally preserves that behavior as a learning signal.

---

## ✨ Example Generated Output

```
SCENE: MONICA'S APARTMENT

RACHEL: My laptop is frozen.
ROSS: Have you tried turning it off and on again?
CHANDLER: It says "Good luck."
JOEY: Check the cluster.
PHOEBE: I deployed the chaos monkey.
```

(The occasional glitches are artifacts of character-level sampling.)

---

## 🧩 Why This Project Exists

Most transformer tutorials hide complexity behind frameworks.

This project does the opposite.

It is designed to build intuition for:

* how attention actually mixes information
* why residual connections stabilize deep networks
* how feed-forward layers expand representational capacity
* what causes overfitting in language models
* how autoregressive generation works

Understanding these mechanics is critical before scaling to production LLM systems.

---

## 🛠️ How to Run

```bash
git clone https://github.com/YOUR_USERNAME/centralperkGPT.git
cd centralperkGPT
pip install torch
python train.py
```

Training will automatically begin and periodically report validation loss.

---

## 🔭 Future Improvements

Potential directions for scaling:

* Weight tying between embeddings and LM head
* KV-cache for faster generation
* FlashAttention-style optimization
* Parallel QKV projection
* Larger datasets / BPE tokenization
* Hyperparameter scaling experiments

---

## 📚 Inspiration

This project was inspired by the growing body of research on transformer architectures and the importance of implementing models from first principles.

---

## License

MIT
