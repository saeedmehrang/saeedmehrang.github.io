---
title: "Elaborated Automatic Mixed Precision in PyTorch"
date: 2025-10-31
draft: true
author: "Saeed Mehrang"
tags: ["deep-learning", "pytorch", "mixed-precision"]
categories: ["Technical Deep Dives", "Optimization"]
weight: 2
description: "A detailed exploration of PyTorch's torch.amp module, focusing on its core components—autocast and GradScaler—and the numerical stability trade-offs between float16 (FP16) and bfloat16 (BF16) data types for accelerating deep learning model training on GPUs with Tensor Cores."
summary: "PyTorch Automatic Mixed Precision (AMP) accelerates model training by strategically mixing 16-bit and 32-bit floating-point arithmetic. This is achieved via two main tools: `autocast` (which auto-selects the precision for each operation to maximize speed while maintaining stability) and `GradScaler` (which prevents gradient underflow when using lower-precision types like FP16). The key benefits are 1.5x-3x faster training and reduced GPU memory usage, with BF16 offering greater numerical range than FP16."
cover:
    image: cover.png
showtoc: true
math: true
TocOpen: true
disableAnchoredHeadings: false
---

## 1\. Automatic Mixed Precision (AMP): The Technical Deep Dive

Automatic Mixed Precision is a powerful technique that significantly reduces the training time and memory footprint of deep neural networks without compromising model convergence. It achieves this by selectively using lower-precision floating-point formats, primarily **FP16** (half-precision, 16-bit), alongside the default **FP32** (single-precision, 32-bit).

### 1\. The Necessity of Mixed Precision

The foundation of modern GPU acceleration lies in specialized hardware, like NVIDIA's **Tensor Cores**, which are exceptionally fast at performing matrix multiplications with FP16 inputs, often yielding $\mathbf{2\times}$ to $\mathbf{8\times}$ the throughput of FP32.

#### A. Data Representation and Instability

The core challenge is the numerical stability of FP16.

  * **FP32 (Single-Precision):** Uses 1 sign bit, 8 exponent bits, and 23 mantissa bits. This provides a wide dynamic range (smallest positive normalized number $\approx 1.18 \times 10^{-38}$, largest $\approx 3.4 \times 10^{38}$) and high precision.
  * **FP16 (Half-Precision):** Uses 1 sign bit, 5 exponent bits, and 10 mantissa bits.
      * **Limited Dynamic Range:** Its maximum positive value is $\mathbf{65,504}$. The smallest positive **normalized** number is $\approx \mathbf{6.10 \times 10^{-5}}$.
      * **The Underflow Problem:** Gradients, especially in deeper layers, often have magnitudes far smaller than $6.10 \times 10^{-5}$. When these small gradients are represented in FP16, they "flush to zero" ($\mathbf{0}$), a phenomenon called **underflow**. If $\nabla_{\theta} \mathcal{L} < 6.10 \times 10^{-5}$, the computed $\nabla_{\theta} \mathcal{L}_{\text{FP16}} = 0$, halting learning for that parameter.
      * **The Overflow Problem:** Conversely, intermediate computations, like large matrix products or sums, can easily exceed $65,504$ and result in $\mathbf{\text{Inf}}$ (infinity) or $\mathbf{\text{NaN}}$ (Not a Number), leading to training divergence.

#### B. The PyTorch Solution: $\text{torch.cuda.amp}$

PyTorch's AMP module addresses these issues using two key components:

1.  $\mathbf{\text{torch.cuda.amp.autocast}}$: Manages precision of the forward pass.
2.  $\mathbf{\text{torch.cuda.amp.GradScaler}}$: Prevents gradient underflow during the backward pass.

-----

## 2\. $\mathbf{\text{torch.cuda.amp.autocast}}$: Dynamic Type Casting

The $\text{autocast}$ context manager wraps the forward pass and loss calculation. It intelligently determines the optimal datatype for each operation based on an internal whitelist/blacklist:

  * **FP16/BF16 Whitelist:** Operations that are computationally intensive and numerically safe to run in lower precision, such as $\mathbf{\text{nn.Linear}}$ and $\mathbf{\text{nn.Conv2d}}$. This is where the bulk of the speedup from Tensor Cores comes from.
  * **FP32 Blacklist:** Operations that require the full dynamic range and precision of FP32 for numerical stability, such as **reductions** (like summation in the loss function $\mathcal{L}$) and **exponential/logarithmic functions** (e.g., $\mathbf{\text{torch.exp}}$, $\mathbf{\text{torch.log}}$, Softmax).

$$\mathbf{y = \text{Model}(\mathbf{x}) \implies \mathbf{y}_{\text{FP16/FP32}} = \text{Autocast}(\mathbf{x}_{\text{FP32}})}$$

The key is that $\text{autocast}$ keeps the model weights in **FP32 (Master Weights)** but casts them to FP16 just before an eligible operation. The output of an FP16 operation is also FP16, which becomes the input to the next layer.

-----

## 3\. $\mathbf{\text{torch.cuda.amp.GradScaler}}$: Mathematical Basis for Stability

The $\text{GradScaler}$ component is crucial for solving the **gradient underflow** problem.

### A. The Loss Scaling Principle

Before the backward pass, the loss ($\mathcal{L}$) is multiplied by a large constant factor, $S$ (the scale factor).

$$\mathbf{\mathcal{L}_{\text{scaled}} = S \cdot \mathcal{L}}$$

Since the backpropagation chain rule dictates that gradients are proportional to the loss, the scaled loss produces correspondingly **scaled gradients**:

$$\mathbf{\nabla_{\theta} \mathcal{L}_{\text{scaled}} = \nabla_{\theta} (S \cdot \mathcal{L}) = S \cdot (\nabla_{\theta} \mathcal{L})}$$

By choosing a sufficiently large $S$ (e.g., $S \approx 2^{16} = 65536$), the small, non-zero FP32 gradients $\nabla_{\theta} \mathcal{L}$ are amplified to a magnitude that is safely representable in FP16, preventing underflow.

### B. The Optimization Step

The optimization step—where the weights are actually updated—must occur with **unscaled** gradients. If $\mathbf{\theta}$ is the weight vector, the update rule for Stochastic Gradient Descent (SGD) is:

$$\mathbf{\theta_{k+1} = \theta_k - \eta \cdot \nabla_{\theta} \mathcal{L}}$$

Where $\mathbf{\eta}$ is the learning rate.

If we were to use the scaled gradient $\nabla_{\theta} \mathcal{L}_{\text{scaled}}$, the update would be:

$$\mathbf{\theta_{k+1} = \theta_k - \eta \cdot (S \cdot \nabla_{\theta} \mathcal{L})}$$

This results in a $\mathbf{S \times}$ larger update, effectively using a learning rate of $\mathbf{S \cdot \eta}$, which is guaranteed to cause divergence.

Therefore, $\text{GradScaler}$ performs the following steps inside $\mathbf{\text{scaler.step}(\text{optimizer})}$:

1.  **Unscaling:** Divides the accumulated FP16 gradients by $S$:
    $$\mathbf{\nabla_{\theta} \mathcal{L}_{\text{unscaled}} = \frac{1}{S} \cdot \nabla_{\theta} \mathcal{L}_{\text{scaled}} = \frac{1}{S} \cdot (S \cdot \nabla_{\theta} \mathcal{L}) = \nabla_{\theta} \mathcal{L}}$$
2.  **Safety Check:** Checks the unscaled gradients for $\text{Inf}$ or $\text{NaN}$ values (which indicate an overflow during the scaled backward pass).
3.  **Weight Update:**
      * If no overflow is detected, it applies the unscaled gradient to the FP32 master weights via the optimizer's $\mathbf{\text{step()}}$ method.
      * If overflow is detected, the step is **skipped**, and the accumulated gradients are discarded.

### C. Dynamic Scale Adjustment

To maximize performance, the scale factor $S$ must be as large as possible without causing overflow. $\text{GradScaler}$ implements a **dynamic loss scaling** mechanism:

  * If $G$ consecutive steps complete without an overflow, $S$ is increased (e.g., $S \leftarrow S \cdot 2$).
  * If an overflow occurs, $S$ is reduced (e.g., $S \leftarrow S / 2$), the weight update is skipped, and the count $G$ is reset.

The $\mathbf{\text{scaler.update()}}$ call handles this logic.

-----

## 4\. Real-World PyTorch AMP Implementation (Large Language Model)

We'll apply AMP to a simplified Transformer-based sequence-to-sequence model—a common, memory-intensive architecture in Natural Language Processing (NLP).

### A. The Model Architecture

A single layer of a Transformer encoder uses complex operations like multi-head attention and layer normalization, which are ideal candidates for mixed precision.

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
import time

# --- Setup for a typical NLP scenario (e.g., large transformer block) ---
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
D_MODEL = 1024  # Large model dimension
N_HEADS = 16    # Number of attention heads
SEQ_LEN = 512   # Sequence length
BATCH_SIZE = 16 # Batch size
NUM_BATCHES = 200

# 1. Define a complex, memory-intensive model layer (Transformer Encoder Layer)
class LargeTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        # Multi-Head Attention (highly optimized for FP16/Tensor Cores)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        # Feed-Forward Network (FFN)
        self.linear1 = nn.Linear(d_model, d_model * 4)
        # Activation (e.g., ReLU or GELU - often unstable in pure FP16)
        self.activation = nn.GELU()
        self.linear2 = nn.Linear(d_model * 4, d_model)
        
        # LayerNorm (numerically sensitive, often better in FP32)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        # Sublayer 1: Multi-Head Attention
        # (autocast will run nn.MultiheadAttention in FP16)
        attn_output, _ = self.attn(x, x, x, need_weights=False)
        x = x + self.dropout(attn_output)
        x = self.norm1(x) # LayerNorm is often forced to FP32 for stability
        
        # Sublayer 2: FFN
        # (autocast will run Linear layers in FP16)
        ffn_output = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = x + self.dropout(ffn_output)
        x = self.norm2(x)
        return x

model = LargeTransformerEncoderLayer(D_MODEL, N_HEADS).to(DEVICE)
optimizer = optim.AdamW(model.parameters(), lr=1e-4)
criterion = nn.MSELoss().to(DEVICE)
```

### B. AMP Training Loop

The integration of $\text{autocast}$ and $\text{GradScaler}$ is critical.

```python
# 2. Instantiate the GradScaler for dynamic loss scaling
scaler = GradScaler()

print(f"Starting AMP Training on {DEVICE}...")
start_time = time.time()
total_loss = 0.0

for batch_idx in range(NUM_BATCHES):
    # Dummy data (representing a batch of sequence embeddings)
    inputs = torch.randn(BATCH_SIZE, SEQ_LEN, D_MODEL, device=DEVICE)
    # Dummy targets (next token predictions or reconstruction)
    targets = torch.randn(BATCH_SIZE, SEQ_LEN, D_MODEL, device=DEVICE)
    
    optimizer.zero_grad()
    
    # 3. Apply torch.cuda.amp.autocast for the forward pass and loss calculation
    # This enables mixed precision, casting ops to FP16 where stable/fast.
    with autocast(device_type=DEVICE, dtype=torch.float16):
        outputs = model(inputs)
        loss = criterion(outputs, targets)

    # 4. Use scaler.scale(loss).backward() instead of loss.backward()
    # This scales the loss by S before the backward pass to prevent gradient underflow.
    scaler.scale(loss).backward()

    # 5. Use scaler.step(optimizer) instead of optimizer.step()
    # This performs:
    # a) Unscaling of gradients.
    # b) Checks for Inf/NaN (overflow).
    # c) Applies unscaled gradients to FP32 master weights if safe, or skips the step.
    scaler.step(optimizer)
    
    # 6. Update the scale factor S for the next iteration
    # This dynamically increases S if no recent overflows occurred, or decreases it on overflow.
    scaler.update()

    total_loss += loss.item()
    if (batch_idx + 1) % 50 == 0:
        print(f"Batch {batch_idx+1}/{NUM_BATCHES} - Avg Loss: {total_loss/50:.4f}")
        total_loss = 0.0

end_time = time.time()
print(f"\nTraining Complete. Total time for {NUM_BATCHES} batches: {end_time - start_time:.2f} seconds.")
```

### C. Advanced API Usage: State Management and Gradient Accumulation

In a real-world scenario, you must save and load the $\text{GradScaler}$ state, and you might use gradient accumulation.

#### 1\. Checkpointing $\text{GradScaler}$

The scale factor $S$ is part of the training state, and is essential for stable continued training. It must be saved and loaded alongside the model and optimizer state.

```python
# To Save Checkpoint
checkpoint = {
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scaler_state_dict': scaler.state_dict(), # <<< Essential part
    'epoch': epoch,
}
torch.save(checkpoint, 'amp_model_checkpoint.pt')

# To Load Checkpoint
checkpoint = torch.load('amp_model_checkpoint.pt')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

# Initialize the scaler and load its state
scaler = GradScaler()
scaler.load_state_dict(checkpoint['scaler_state_dict']) # <<< Essential part
```

#### 2\. Gradient Accumulation with AMP

Gradient accumulation is used to simulate a larger batch size ($\text{Effective Batch Size} = \text{Batch Size} \times \text{Accumulation Steps}$). The key is to **only unscale the gradients** on the final accumulation step before calling $\mathbf{\text{scaler.step}(\text{optimizer})}$.

```python
ACCUM_STEPS = 4

for batch_idx in range(NUM_BATCHES):
    # Data loading (inputs, targets) ...
    
    with autocast(device_type=DEVICE, dtype=torch.float16):
        outputs = model(inputs)
        # Scale loss by accumulation steps to get the average loss per sample
        loss = criterion(outputs, targets) / ACCUM_STEPS
    
    # Scale the loss *before* backward pass
    scaler.scale(loss).backward() 

    if (batch_idx + 1) % ACCUM_STEPS == 0:
        # On the last step, apply the accumulated, scaled gradients
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad() # Clear gradients after weight update
```

This advanced implementation provides a robust, high-performance training workflow for large-scale models, demonstrating the practical application of PyTorch's AMP API in a numerically stable manner.
