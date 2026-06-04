# CNNs and Transformers in PyTorch

Implementing CNN and Transformer architectures from scratch in PyTorch. Covers a custom CNN, ResNet-18 with residual blocks, and a full Transformer built component by component — trained on MiniImageNet and TinyStories.

---

## What's covered

### Part 1 — Custom CNN
- Built a two-layer CNN using PyTorch's `nn.Module` system
- Wrote a custom `Dataset` and `DataLoader` for MiniImageNet (100-class ImageNet subset, 50k training images)
- Applied image transforms (resize, center crop, normalization)
- Trained with AdamW + CrossEntropyLoss for 20 epochs (~50% training accuracy)

### Part 2 — ResNet-18
- Implemented residual blocks with skip connections and optional downsampling
- Assembled the full ResNet-18 architecture: initial 7×7 conv → 4 residual stages → global average pool → linear classifier
- Trained for 50 epochs (~70% training accuracy, ~50% validation accuracy)

### Part 3 — Transformer from Scratch
Built every component individually:
- Numerically stable softmax
- Scaled dot-product attention with causal masking
- Single attention head and multi-head attention
- Encoder layer (self-attention + FFN + residual + LayerNorm)
- Decoder layer (masked self-attention + cross-attention + FFN)
- Sinusoidal positional encoding
- Full `TransformerEncoder`, `TransformerDecoder`, and `Transformer` (supports encoder-decoder and decoder-only modes)

Trained a decoder-only Transformer on [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) for next-token prediction, with autoregressive text generation from a prompt.

---

## Stack
Python, PyTorch, Hugging Face `datasets`, MiniImageNet, TinyStories

## File
`cp_cnn_transformers.ipynb` — full implementation notebook
