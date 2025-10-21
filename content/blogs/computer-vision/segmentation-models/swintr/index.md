---
title: "Swin Transformer: Shifting Windows to Build Hierarchical Vision Models"
date: 2025-10-21
draft: false
author: Saeed Mehrang
tags: ["computer-vision", "deep-learning", "transformers", "segmentation"]
categories: ["Computer Vision"]
description: "A brief overview at Swin Transformer"
math: true
summary: "This post provides a minimal PyTorch implementation of Swin Transformer for a simple image classification."
showToc: true
disableAnchoredHeadings: false
cover:
  image: "cover.png"
  image_alt: "SwinTransformer"
---


| Characteristic | Detail |
| :--- | :--- |
| **Estimated Reading Time** | 5-15 minutes |
| **Technical Level** | Intermediate |
| **Prerequisites** | Transformers, Vision Transformers (ViT) |


## Swin Transformer: A Hierarchical Vision Backbone via Shifting Windows

The Swin Transformer's core innovation is to replace the computationally expensive **Global Self-Attention** of the standard Vision Transformer (ViT) with an efficient, localized attention mechanism: **Shifted Window Multi-Head Self-Attention (SW-MSA)**. This achieves a hierarchical feature map design and $\mathbf{O(N)}$ computational complexity, where $N$ is the number of tokens.

The entire architecture is built by stacking four key modules: **Patch Embedding**, the **Swin Transformer Layer** (which alternates between Windowed and Shifted-Window attention), and **Patch Merging**.

In this blog I want to disentangle how Swin Transformer works by rewriting the original code [^2] in a slightly more modularized way and by adding more comments to the code showing the dimensionality of the tensors as they flow through the layers and operations of the model. See the [code here](https://github.com/saeedmehrang/computer-vision-learning/blob/main/swin_transformer.py) in my Github repo.


See the image that compares ViT with Swin Transformer (image adopted from the original article [^1]).


{{< framed_image src="comparison.png" alt="comparison" width="500px" height="550px" >}}
{{< /framed_image >}}
***

## 1. Initial Tokenization: Patch Embedding

The first step, implemented in the `PatchEmbed` class, converts the input image into a sequence of feature tokens, setting up the feature resolution and initial channel dimension.

1.  **Patchification and Projection:** The `self.proj` `nn.Conv2d` layer acts as a combined patching and linear projection. A stride equal to the patch size (e.g., $4$) ensures non-overlapping patches.
2.  **Flattening:** The output feature map (e.g., $B \times 96 \times 56 \times 56$) is reshaped via `x.flatten(2).transpose(1, 2)` into the standard Transformer token format $\mathbf{x}$ (shape $B \times N \times C_{\text{embed}}$), where $N$ is the total number of patches (tokens).
3.  **Normalization:** A `nn.LayerNorm` is applied to stabilize the embedding features before they enter the main Transformer stages.

***

## 2. Core Computation: The Swin Transformer Layer

The `SwinTransformerLayer` is the model's workhorse. It implements the main attention and MLP steps, and its operation is conditional on the $\mathbf{shift\_size}$: **zero for W-MSA** (Windowed MSA), and **non-zero for SW-MSA** (Shifted Window MSA). All layers first restore the input token tensor to its 2D spatial format $\mathbf{x} \in \mathbb{R}^{B \times H \times W \times C}$ using `x.view(B, H, W, C)` inside the `forward` pass.

### Case 1: Windowed Multi-Head Self-Attention (W-MSA)

This is the standard attention step, occurring when the `shift_size` is $\mathbf{0}$ on the even layer indices (e.g., the indices 0, 2, 4, etc) of the block.

| Operation | Code Reference | Description |
| :--- | :--- | :--- |
| **Window Partitioning** | `window_partition(shifted_x, self.window_size)` | This function efficiently divides the 2D feature map into a batch of non-overlapping windows. It outputs a collection of flattened windows $\mathbf{x}_{\text{windows}} \in \mathbb{R}^{B' \times W^2 \times C}$, where $B'$ is the total number of windows across the batch, ready for attention. |
| **Attention & RPB** | `self.attn(x_windows, mask=shift_mask)` in `SwinTransformerLayer.forward` | The `WindowAttention` module computes self-attention *only within* each window. It incorporates a learnable **Relative Position Bias (RPB)** (indexed by `relative_position_index`), which is added to the attention scores to encode relative spatial location. **No mask is used** (`shift_mask=None`). |
| **Window Reversal** | `window_reverse(attn_windows, self.window_size, H, W)` | The processed windows are precisely "folded" back together into the original feature map shape $\mathbf{x}_{\text{out}} \in \mathbb{R}^{B \times H \times W \times C}$. |
| **Final Steps** | `x = x.view(B, H * W, C)` followed by $\mathbf{x} = \mathbf{x}_{\text{shortcut}} + \mathbf{x} + \text{MLP}(\mathbf{x})$ | The feature map is flattened back to a token sequence, followed by a **residual connection** and a standard **MLP** (Multi-Layer Perceptron) step. |

---

### Case 2: Shifted Window Multi-Head Self-Attention (SW-MSA)

This occurs when the `shift_size` is non-zero (typically half the window size, e.g., the second layer of a block) on the odd layer indices (e.g., the indices 1, 3, 5, etc) of the block. This shift is the innovation that enables communication between windows.

| Operation | Code Reference | Description |
| :--- | :--- | :--- |
| **Cyclic Shift** | `torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))` | The 2D feature map is cyclically shifted by $\mathbf{shift\_size}$ pixels. This is crucial for making features from adjacent windows become neighbors within the new, shifted windows, allowing cross-window information flow during attention. |
| **Attention Mask Creation** | `create_mask(H, W, self.window_size, self.shift_size)` | The cyclic shift causes some pixels to wrap around (e.g., from the bottom to the top), resulting in non-adjacent patches being grouped into the same window. This function generates an **attention mask** to prevent attention between these invalid, non-neighboring pairings. |
| **Attention & Masking** | `attn = attn.view(...) + mask.unsqueeze(1).unsqueeze(0)` in `WindowAttention.forward` | The mask is added to the attention logits. The negative values in the mask (e.g., $-100.0$) ensure that non-adjacent tokens resulting from the shift receive an attention weight of zero after the softmax. |
| **Reverse Cyclic Shift** | `torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))` | After window reversal, the feature map must be shifted back to its **original spatial alignment** using the inverse shift. This prepares the tensor for the subsequent residual connection and MLP block. |

See how circular shift is applied to the windows on the 2d spatial plain in the image below (Figure 2 adopted from the original article[^1]). This is nothing but a shifted window to the right and circular padding that is better visualized by the second image below (Figure 4 adopted from the original article[^1]). Remember that padding is needed simply due to the fact that neural network layers apply matrix algebra to their inputs and we have to keep the input size fixed. The trick is to mask those parts that we pad. Masking is nothing but a simple zeroing-out operation that should be placed in an appropriate location in the matrix multiplication chain of the layer. Most of the standard layers in PyTorch and Tensorflow do this for us automatically when we supply them the mask tensor/matrix/vector.


{{< framed_image src="circular_shift.png" alt="Circular Shift" width="500px" height="450px" >}}
{{< /framed_image >}}


When there is circular shift, we must do a masking on those wrapped sections of the image that are artifically placed next to eachother. See the image below for better clarification.


{{< framed_image src="masking.png" alt="masking" width="500px" height="300px" >}}
{{< /framed_image >}}

***

## 3. Creating Hierarchy: Patch Merging

The `PatchMerging` module is applied between stages to build the hierarchical feature pyramid, doubling the channel dimension while halving the spatial resolution (similar to pooling and stride convolutions in a CNN).

| Operation | Code Reference | Description |
| :--- | :--- | :--- |
| **2x2 Sampling** | `x0 = x[:, 0::2, 0::2, :]`, `x1 = x[:, 1::2, 0::2, :]`, etc., within `PatchMerging.forward` | The layer views the input $\mathbf{x} \in \mathbb{R}^{B \times H \times W \times C}$ and samples four distinct non-overlapping $2 \times 2$ regions. This effectively downsamples the spatial dimensions to $H/2, W/2$. |
| **Concatenation and Reduction** | `x = torch.cat([x0, x1, x2, x3], -1)` followed by `x = self.reduction(x)` | Concatenating the four $C$-dimensional samples expands the feature size to $\mathbf{4C}$. This $\mathbf{4C}$ feature token is then normalized and passed through the `self.reduction` `nn.Linear` layer to project it to the new, typically $\mathbf{2C}$ channel dimension. |

# References

[^1]: Liu, Ze, et al. "Swin transformer: Hierarchical vision transformer using shifted windows." Proceedings of the IEEE/CVF international conference on computer vision. 2021.

[^2]: Original Microsoft's implementation of Swin transformer: https://github.com/microsoft/Swin-Transformer/blob/main/models/swin_transformer.py
