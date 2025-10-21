---
title: "SwinUNETR: A Vision Transformer for Medical Image Segmentation"
date: 2025-10-20
draft: false
tags: ["computer-vision", "deep-learning", "transformers", "medical-imaging", "segmentation"]
categories: ["Computer Vision"]
description: "An in-depth look at SwinUNETR, combining Swin Transformers with U-Net architecture for 3D medical image segmentation"
math: true
---

## Introduction

Medical image segmentation is a critical task in healthcare, enabling precise delineation of organs, tumors, and other anatomical structures in 3D scans. **SwinUNETR** represents a significant advancement in this domain by combining the hierarchical representation power of **Swin Transformers** with the proven encoder-decoder architecture of **U-Net**.

Traditional CNNs excel at capturing local features but struggle with long-range dependencies. Vision Transformers address this limitation through self-attention mechanisms, but standard ViTs process images as fixed-size patches without hierarchical feature learning. SwinUNETR bridges this gap by using **shifted window attention** to efficiently model both local and global contexts in 3D medical volumes.

### Why SwinUNETR?

- **Hierarchical Feature Learning**: Multi-scale representations through progressive patch merging
- **Efficient Self-Attention**: Window-based attention reduces computational complexity from $O(N^2)$ to $O(N)$
- **3D Medical Imaging**: Native support for volumetric data (CT, MRI scans)
- **Skip Connections**: U-Net-style connections preserve fine-grained spatial details

For a complete minimal implementation, see [swinunetr.py](../swinunetr.py).

---

## Architecture Overview

SwinUNETR consists of two main components:

1. **Swin Transformer Encoder**: Hierarchical feature extraction with shifted window attention
2. **CNN Decoder**: Progressive upsampling with skip connections from encoder stages

The architecture processes 3D volumes through four encoder stages, each progressively reducing spatial dimensions while increasing feature channels, followed by symmetric decoder stages that recover the original resolution.

---

## Key Building Block 1: The Swin Transformer Backbone

The core of SwinUNETR's encoder is the **Swin Transformer**, which processes 3D medical volumes through a hierarchy of Swin Transformer blocks.

### A. 3D Patch Embedding and Merging (Tokenization)

The first step converts the raw 3D input volume into a sequence of embedded tokens. Unlike standard ViTs that use fixed-size patches throughout, SwinUNETR employs hierarchical tokenization.

**Purpose**: Transform volumetric input $(B, C, D, H, W)$ into token sequence $(B, N, C_{embed})$ where $N = \frac{D \times H \times W}{P^3}$ and $P$ is the patch size.

| PyTorch Component | Purpose in SwinUNETR |
| :--- | :--- |
| `PatchEmbed3D` | Converts the input 3D volume (e.g., $1\times D\times H\times W$) into a sequence of 3D tokens (e.g., $N\times C$). Implemented using a 3D convolutional layer with kernel size and stride equal to the patch size. |

**Implementation (from [swinunetr.py](../swinunetr.py)):**

```python
class PatchEmbed3D(nn.Module):
    def __init__(self, in_channels=1, embed_dim=96, patch_size=4):
        super().__init__()
        self.patch_size = patch_size
        # 3D Conv with stride=patch_size performs the embedding
        self.proj = nn.Conv3d(in_channels, embed_dim,
                             kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # (B, C, D, H, W) -> (B, embed_dim, D/p, H/p, W/p)
        x = self.proj(x)
        B, C, D, H, W = x.shape
        # Flatten and transpose to (B, N_tokens, embed_dim)
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, (D, H, W)
```

The patch embedding layer uses a 3D convolution with stride equal to patch size, effectively partitioning the volume into non-overlapping cubic patches. Each patch is then projected to the embedding dimension.

### B. The Swin Transformer Block (Shifted Window Attention)

The Swin Transformer Block is the fundamental computational unit, featuring **window-based** and **shifted-window** self-attention mechanisms. This design enables efficient modeling of both local patterns and cross-window interactions.

**Key Innovation**: Instead of computing attention across all tokens (quadratic complexity), attention is computed within local windows. To enable cross-window connections, consecutive blocks alternate between regular and shifted window partitions.

| PyTorch Component | Purpose in SwinUNETR |
| :--- | :--- |
| `SwinTransformerBlock` | Encapsulates the core computation: LayerNorm, W-MSA (Window Multi-head Self-Attention) or SW-MSA (Shifted Window MSA), and the MLP. This is where local and shifted attention is calculated. |
| `WindowAttention3D` | Calculates 3D self-attention within fixed window sizes. Handles window partitioning and reverse-partitioning of the tensor. |

**Implementation (from [swinunetr.py](../swinunetr.py)):**

```python
class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, window_size=7, shift_size=0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention3D(dim, num_heads, window_size)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Linear(4 * dim, dim)
        )

    def forward(self, x, dims):
        D, H, W = dims
        B, N, C = x.shape

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, D, H, W, C)

        # Skip window attention if dimensions are smaller than window size
        if D < self.window_size or H < self.window_size or W < self.window_size:
            x = x.view(B, N, C)
        else:
            # Cyclic shift for shifted window attention
            if self.shift_size > 0:
                shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size, -self.shift_size),
                                      dims=(1, 2, 3))
            else:
                shifted_x = x

            # Partition windows and compute attention
            x_windows = window_partition(shifted_x, self.window_size)
            attn_windows = self.attn(x_windows)
            shifted_x = window_reverse(attn_windows, self.window_size, D, H, W)

            # Reverse cyclic shift
            if self.shift_size > 0:
                x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size, self.shift_size),
                              dims=(1, 2, 3))
            else:
                x = shifted_x

            x = x.view(B, N, C)

        # Residual connection and FFN
        x = shortcut + x
        x = x + self.mlp(self.norm2(x))
        return x
```

**Shifted Window Mechanism**: By shifting the window partition between consecutive blocks, the architecture enables information flow across window boundaries while maintaining computational efficiency. The shift size is typically half the window size.

### C. Patch Merging (Downsampling)

Between encoder stages, **Patch Merging** layers reduce spatial dimensions while increasing channel capacity, similar to pooling in CNNs but learnable.

| PyTorch Component | Purpose in SwinUNETR |
| :--- | :--- |
| `PatchMerging3D` | Halves the spatial dimensions $(D, H, W)$ and doubles the channel dimension. Implemented by concatenating adjacent patches and using a linear projection. |

**Implementation (from [swinunetr.py](../swinunetr.py)):**

```python
class PatchMerging3D(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(8 * dim, 2 * dim, bias=False)
        self.norm = nn.LayerNorm(8 * dim)

    def forward(self, x, dims):
        D, H, W = dims
        B, N, C = x.shape

        x = x.view(B, D, H, W, C)

        # Concatenate 2x2x2 neighbors
        x0 = x[:, 0::2, 0::2, 0::2, :]  # (B, D/2, H/2, W/2, C)
        x1 = x[:, 1::2, 0::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, 0::2, :]
        x3 = x[:, 1::2, 1::2, 0::2, :]
        x4 = x[:, 0::2, 0::2, 1::2, :]
        x5 = x[:, 1::2, 0::2, 1::2, :]
        x6 = x[:, 0::2, 1::2, 1::2, :]
        x7 = x[:, 1::2, 1::2, 1::2, :]

        x = torch.cat([x0, x1, x2, x3, x4, x5, x6, x7], dim=-1)  # (B, D/2, H/2, W/2, 8*C)
        x = x.view(B, -1, 8 * C)

        x = self.norm(x)
        x = self.reduction(x)  # (B, N/8, 2*C)

        new_dims = (D // 2, H // 2, W // 2)
        return x, new_dims
```

This operation is analogous to $2\times$ downsampling in CNNs but operates in the token space, concatenating $2\times2\times2$ neighboring patches.

---

## Key Building Block 2: The U-Net-like Structure (Encoder-Decoder)

While the Swin Transformer provides powerful feature extraction, the U-Net structure enables precise localization through skip connections.

### A. The Encoder (Feature Extraction)

The encoder consists of four stages, each containing multiple Swin Transformer blocks followed by patch merging.

| Stage | Input Dims | Output Dims | Channels | Resolution |
|-------|------------|-------------|----------|------------|
| 1 | $D/4 \times H/4 \times W/4$ | $D/4 \times H/4 \times W/4$ | 96 | $1\times$ |
| 2 | $D/4 \times H/4 \times W/4$ | $D/8 \times H/8 \times W/8$ | 192 | $1/2\times$ |
| 3 | $D/8 \times H/8 \times W/8$ | $D/16 \times H/16 \times W/16$ | 384 | $1/4\times$ |
| 4 | $D/16 \times H/16 \times W/16$ | $D/32 \times H/32 \times W/32$ | 768 | $1/8\times$ |

Each stage progressively builds higher-level semantic representations while features from all stages are preserved for skip connections.

### B. The Decoder and Skip Connections

The decoder mirrors the encoder structure, using transposed convolutions for upsampling and skip connections to integrate multi-scale features.

| PyTorch Component | Purpose in SwinUNETR |
| :--- | :--- |
| `DecoderBlock` | Standard U-Net-style upsampling block. Takes concatenated features from decoder path and skip connection, uses 3D Transposed Convolution for upsampling. |
| `torch.cat()` | Skip connection implementation. Concatenates encoder features with upsampled decoder features along the channel dimension. |

**Implementation (from [swinunetr.py](../swinunetr.py)):**

```python
class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_channels, in_channels // 2,
                                     kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels // 2 + skip_channels, out_channels,
                     kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x_decoder, x_skip):
        # 1. Upsample the decoder input
        x_decoder = self.up(x_decoder)

        # 2. Skip Connection (Concatenation)
        x = torch.cat([x_skip, x_decoder], dim=1)

        # 3. Convolution for feature fusion
        x = self.conv(x)
        return x
```

**Skip Connection Rationale**: While the Swin Transformer encoder captures semantic information, spatial details are lost during downsampling. Skip connections directly pass high-resolution features from the encoder to the decoder, enabling precise boundary delineation.

The decoder progressively recovers spatial resolution:
- Stage 1: $D/32 \times H/32 \times W/32 \rightarrow D/16 \times H/16 \times W/16$
- Stage 2: $D/16 \times H/16 \times W/16 \rightarrow D/8 \times H/8 \times W/8$
- Stage 3: $D/8 \times H/8 \times W/8 \rightarrow D/4 \times H/4 \times W/4$
- Stage 4: $D/4 \times H/4 \times W/4 \rightarrow D \times H \times W$

---

## Complete Architecture Flow

Let's trace a 3D volume through the entire network:

1. **Input**: $(B, 1, 96, 96, 96)$ - A batch of 3D medical scans
2. **Patch Embedding**: $(B, 1, 96, 96, 96) \rightarrow (B, 13824, 96)$ - Tokenization with patch size 4
3. **Swin Stage 1**: $(B, 13824, 96)$ - 2 Swin blocks, no downsampling
4. **Patch Merging 1**: $(B, 13824, 96) \rightarrow (B, 1728, 192)$ - $2\times$ downsample
5. **Swin Stage 2**: $(B, 1728, 192)$ - 2 Swin blocks
6. **Patch Merging 2**: $(B, 1728, 192) \rightarrow (B, 216, 384)$ - $2\times$ downsample
7. **Swin Stage 3**: $(B, 216, 384)$ - 6 Swin blocks (bottleneck)
8. **Patch Merging 3**: $(B, 216, 384) \rightarrow (B, 27, 768)$ - $2\times$ downsample
9. **Swin Stage 4**: $(B, 27, 768)$ - 2 Swin blocks
10. **Decoder**: Progressive upsampling with skip connections back to $(B, C_{classes}, 96, 96, 96)$

---

## Training Considerations

### Loss Functions

For multi-class medical image segmentation, a combination of losses works best:

$$\mathcal{L}_{total} = \mathcal{L}_{CE} + \lambda \mathcal{L}_{Dice}$$

where:
- $\mathcal{L}_{CE}$: Cross-entropy loss for pixel-wise classification
- $\mathcal{L}_{Dice}$: Dice loss to handle class imbalance
- $\lambda$: Weighting factor (typically 1.0)

### Data Augmentation

3D medical volumes benefit from:
- Random affine transformations (rotation, scaling, translation)
- Elastic deformations
- Intensity normalization and shifts
- Random cropping to fixed sizes

### Computational Requirements

SwinUNETR is computationally intensive due to 3D operations:
- **Memory**: Typically requires 16-32GB GPU VRAM for training
- **Input Size**: Common patch sizes are $96^3$ or $128^3$
- **Batch Size**: Usually 1-2 samples per GPU
- **Training Time**: Several days on multi-GPU setups for large datasets

---

## Conclusion: Why SwinUNETR Excels

SwinUNETR represents a successful marriage of two powerful architectural paradigms:

1. **Swin Transformers** provide hierarchical feature learning with linear complexity through shifted window attention
2. **U-Net architecture** enables precise localization through multi-scale skip connections

The key advantages are:

- **Better Long-Range Modeling**: Self-attention captures dependencies across the entire volume
- **Hierarchical Representations**: Multi-scale features from patch merging
- **Computational Efficiency**: Window-based attention reduces complexity from $O(N^2)$ to $O(N)$
- **State-of-the-Art Performance**: Achieves leading results on medical segmentation benchmarks (BTCV, MSD, etc.)

For medical imaging tasks requiring precise 3D segmentation, SwinUNETR offers a compelling alternative to pure CNN approaches, particularly when training data is sufficient to leverage the transformer's capacity.

The complete minimal implementation is available in [swinunetr.py](../swinunetr.py), demonstrating all core components in runnable PyTorch code.

---

## References

1. Hatamizadeh, A., et al. "Swin UNETR: Swin Transformers for Semantic Segmentation of Brain Tumors in MRI Images." *MICCAI 2022*.
2. Liu, Z., et al. "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows." *ICCV 2021*.
3. Ronneberger, O., et al. "U-Net: Convolutional Networks for Biomedical Image Segmentation." *MICCAI 2015*.

---
