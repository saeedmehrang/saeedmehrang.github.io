---

title: "ViTDet: Plain Vision Transformer Backbones for Object Detection"
date: 2025-10-23
draft: false
author: "Saeed Mehrang"
tags: ["Computer Vision", "Deep Learning", "Object Detection", "Vision Transformer", "ViT"]
categories: ["Machine Learning", "AI Models"]
description: "Understanding ViTDet's approach to using plain Vision Transformers for object detection"
summary: "ViTDet demonstrates that plain, non-hierarchical Vision Transformers can compete with hierarchical backbones for object detection through simple adaptations."
math: true
ShowToc: true
TocOpen: true
cover: 
  image: "cover.png"
  alt: "ViTDet"
  caption: "ViTDet: Plain Vision Transformer Backbones for Object Detection"
---

| Characteristic | Detail |
| :--- | :--- |
| **Estimated Reading Time** | 15-20 minutes |
| **Technical Level** | Intermediate to Advanced |
| **Prerequisites** | Vision Transformers (ViT), Object Detection |

---

## 1. Introduction: Breaking the Hierarchical Backbone Paradigm

The ViTDet paper, "Exploring Plain Vision Transformer Backbones for Object Detection" by Li et al. (2022) [^1], challenges a fundamental assumption in modern object detection: the necessity of **hierarchical, multi-scale backbones**. Since the rise of Vision Transformers (ViT) [^2], most detection architectures have adopted hierarchical designs like Swin Transformer [^4] or MViT [^6] to handle the multi-scale nature of object detection. ViTDet takes a different path: it demonstrates that the original **plain ViT architecture** can be highly effective for detection with minimal adaptations applied only during fine-tuning.

**Key Novelty and Innovation**: ViTDet maintains the philosophy of **decoupling pre-training from fine-tuning**. The backbone remains a plain, non-hierarchical ViT during pre-training (using masked autoencoder or MAE [^3]), and only during detection fine-tuning are simple adaptations introduced to handle high-resolution images and multi-scale objects.

**Main Message**: A plain, single-scale ViT backbone, combined with strong self-supervised pre-training and minimal fine-tuning adaptations, can **match or surpass** the performance of complex hierarchical backbones for object detection.

**Real-World Impact**: This approach was notably adopted by Meta's **Segment Anything Model (SAM V1)** [^5], which used ViTDet's window attention strategy to efficiently process $1024 \times 1024$ images for segmentation tasks.

---

## 2. The Motivation: Why Plain Backbones?

### 2.1. The Hierarchical Backbone Trend

Traditional ConvNets (ResNet, VGG) are inherently hierarchical: they progressively downsample feature maps while increasing channel dimensions, naturally producing multi-scale representations. When Vision Transformers emerged, many researchers assumed that to be effective for detection, ViT must also adopt hierarchical designs:

- **Swin Transformer**: Introduces shifted windows and hierarchical multi-scale features
- **MViT**: Uses pooling attention to create pyramid representations
- **PVT**: Builds spatial reduction attention for multi-scale processing

These hierarchical Transformers work well but require **redesigning the architecture for pre-training**, tying the backbone design to downstream task requirements.

### 2.2. The Plain Backbone Philosophy

ViTDet argues for a different approach based on **decoupling**:

1.  **Task-Agnostic Pre-training**: The backbone should remain simple and general-purpose during pre-training, not constrained by detection-specific requirements. **All layers use global self-attention** during this phase.
2.  **Task-Specific Adaptation**: Detection-specific modifications should be introduced **only during fine-tuning**. This includes modifying the attention mechanism and adding a simple feature pyramid.
3.  **Decoupling Benefits**: This separation allows the detection community to leverage advances in ViT pre-training (like MAE) without redesigning detection systems.

**The Central Question**: Can a plain ViT backbone maintaining a single-scale, non-hierarchical structure compete with hierarchical backbones for object detection?

ViTDet answers: **Yes, with surprising effectiveness.** See the adopted Figure 1 from [^1] to see the schematic diagram of the ViTDet proposal.


{{< framed_image src="cover.png" alt="vitdet-proposal" width="800px" height="350px" >}}
{{< /framed_image >}}

---

## 3. The ViTDet Architecture and Innovations

The ViTDet architecture is based on minimal, highly effective adaptations applied to the pre-trained plain ViT.

### 3.1. Simple Feature Pyramid (No FPN!)

ViTDet's novelty lies in generating multi-scale features from a single feature map, avoiding the complexity of traditional Feature Pyramid Networks (FPNs) that rely on combining features from multiple stages.

**Building Multi-Scale Features from a Single Map**:
- Start with the last ViT feature map: **1/16 scale** ($64 \times 64$ for $1024 \times 1024$ input)
- Generate coarser scales via **strided convolutions**:
  - 1/32 scale: $2 \times 2$ max pooling with stride 2
- Generate finer scales via **deconvolutions**:
  - 1/8 scale: One $2 \times 2$ deconvolution with stride 2
  - 1/4 scale: Two $2 \times 2$ deconvolutions with stride 2

This creates a pyramid of scales $\{1/32, 1/16, 1/8, 1/4\}$ from only the final feature map, **without any lateral or top-down FPN connections**.

**Empirical Results**: The simple pyramid achieves the same performance as FPN variants (54.6 $AP^{box}$ vs 54.4 for FPN on ViT-L), demonstrating that **pyramidal feature maps are key, not the FPN connections themselves.**

### 3.2. Window Attention: Handling High-Resolution Inputs

The primary bottleneck for plain ViT in detection is the computational complexity of global self-attention: $O(N^2 \cdot D)$, where $N$ is the number of patches (e.g., $N=4096$ for $1024 \times 1024$ input).

**ViTDet's Solution**: Window attention with sparse cross-window propagation.

#### 3.2.1. Window Attention Mechanism

**During Fine-tuning**:
1. Divide the $64 \times 64$ feature map into non-overlapping **$14 \times 14$ windows**.
2. Compute self-attention **within each window independently**.
3. This achieves a **speedup factor** of approximately $21\times$ per layer compared to global attention, making high-resolution processing feasible.

#### 3.2.2. Cross-Window Propagation Blocks

Window attention isolates information within windows. ViTDet uses a small number of **propagation blocks** to mitigate this:

**Strategy**: Insert only **4 propagation blocks** evenly spaced throughout the backbone (e.g., blocks 6, 12, 18, 24 for ViT-L).

**Two Propagation Approaches**:

1.  **Global Attention Blocks**: These 4 blocks perform full global self-attention across all 4096 patches.
2.  **Convolutional Blocks (Preferred)**: Insert residual convolutional blocks (e.g., $1 \times 1 \times 3 \times 3 \times 1 \times 1$) initialized to identity. Convolutions propagate information to adjacent windows.

**Empirical Finding**: Both strategies provide similar gains (+1.7 to +1.9 $AP^{box}$ over no propagation). Importantly, ViTDet achieves these results **without using shifted windows** like the Swin Transformer.


**Convolutional Blocks (Preferred) Implementation**:

The $4$ cross-window propagation layers, when implemented using the preferred **convolutional blocks**, function as a lightweight mechanism for local information exchange.

##### Numerical Tensor Shape Example (ViT-L)

The input and output shapes of these blocks are determined by the ViT architecture used. For a **ViT-L** model processing a $1024 \times 1024$ input image (which results in $64 \times 64$ patches):

| Metric | Input/Output Tensor Shape | Axes |
| :--- | :--- | :--- |
| **Patch Resolution** | $64 \times 64$ patches | (Height, Width) |
| **Patch Embedding Dim (D)** | 1024 (for ViT-L) | (Channel/Dimension) |
| **Input Shape** | $\mathbf{[B, 4096, 1024]}$ (Sequential Tokens) | (Batch, Patches, Dimension) |
| **Conversion for Conv** | $\mathbf{[B, 1024, 64, 64]}$ (Reshaped Grid) | (Batch, Dimension, Height, Width) |
| **Output Shape** | $\mathbf{[B, 1024, 64, 64]}$ (Reshaped Grid) | (Batch, Dimension, Height, Width) |

*The input, typically a sequence of tokens from the preceding ViT block, must be **reshaped** from $[B, 4096, 1024]$ back into a 2D feature map $[B, 1024, 64, 64]$ before the convolution is applied.*



##### Convolutional Implementation Details

###### Convolutional Layers: $1 \times 1 \times 3 \times 3 \times 1 \times 1$

The notation $1 \times 1 \times 3 \times 3 \times 1 \times 1$ describes the sequence and size of kernels within the residual convolutional block:

1.  **$1 \times 1$ Convolution (Pointwise):** Used to reduce the channel dimension (e.g., $1024 \rightarrow$ smaller dimension).
2.  **$3 \times 3$ Convolution (Spatial):** Used to perform spatial mixing and propagate information locally across the $64 \times 64$ grid.
3.  **$1 \times 1$ Convolution (Pointwise):** Used to restore the channel dimension back to 1024, matching the output of the ViT block.

The layer is a standard residual block inserted between two ViT components (e.g., between the Multi-Head Attention (MHA) and the Feed Forward Network (FFN), or after the FFN).

###### Aggregation and Convolution Axis

| Aspect | Description |
| :--- | :--- |
| **Axis of Convolution** | The convolution is applied spatially across the $\mathbf{64 \times 64}$ height and width axes of the reshaped tensor. |
| **Kind of Aggregation** | The operation performs **local aggregation** (or local mixing). The $3 \times 3$ convolution combines features from a $3 \times 3$ neighborhood of tokens (patches) to create the new feature for the central token. This effectively breaks the window boundary and propagates information to adjacent windows. |
| **Input/Output for $3 \times 3$ Conv**| The $3 \times 3$ convolution acts on the feature map: it takes $C'$ channels in and outputs $C'$ channels out, where $C'$ is the reduced bottleneck dimension. |

The key takeaway is that these blocks are **inserted** and trained to perform the task of cross-window communication that was lost when the global attention in the preceding ViT layers was switched to window attention.




### 3.3. Architectural Summary

**Plain ViT Backbone**:
- Standard ViT-B/L/H architecture (12/24/32 blocks)
- During pre-training: All blocks use **global self-attention**
- During fine-tuning: Adapt to window attention + propagation blocks

**Detection Head**:
- Compatible with standard frameworks: Mask R-CNN, Cascade Mask R-CNN
- Uses the simple feature pyramid described above
- No special modifications needed

### 3.4. Transitioning from Pre-training to Fine-tuning: The Mechanism of Adaptation

The core innovation of ViTDet lies in the efficient and minimal way it adapts the pre-trained, global-attention-based ViT to the detection task, **maximizing the reuse of learned weights** while addressing computational and multi-scale requirements.

#### Reusing Pre-trained Weights

The most crucial detail is the transferability of the attention mechanism's parameters:

* **Projection Layer Transfer**: For every self-attention block in the ViT backbone, the **pre-trained weights for the Query (Q), Key (K), and Value (V) linear projection matrices are transferred directly** to the fine-tuning stage. These weights, having learned robust visual features during MAE pre-training, form the basis of the fine-tuned model.

#### Targeted Adaptations during Fine-tuning

The change from global attention to window attention is an operational change, not a structural insertion of new attention layers:

1.  **Modification of Self-Attention Blocks (Most Layers):**
    * The pre-trained attention module is **reconfigured** from **Global Self-Attention** to the **Window Attention module**.
    * This is achieved by implementing a **restriction/masking** in the attention computation to only calculate relationships within the $14 \times 14$ non-overlapping windows. The underlying **Q, K, and V projection weights remain unchanged** and are fine-tuned for the detection task.

2.  **Insertion of Cross-Window Propagation Blocks:**
    * A few new modules are explicitly **inserted** into the backbone's sequential flow to restore global information exchange.
    * The preferred method is to **insert residual convolutional blocks** (e.g., $1 \times 1 \times 3 \times 3 \times 1 \times 1$) after the attention layer in approximately four positions. These are new layers, which are **initialized randomly** and trained from scratch during fine-tuning, allowing local information to propagate to adjacent windows.

3.  **Attachment of the Simple Feature Pyramid:**
    * The pyramid generation is an **external extension** to the backbone. New layers—specifically the **deconvolutional (transposed convolutional) layers** for upsampling and simple pooling layers—are **appended** to the final $1/16$ scale output of the ViT backbone. These new layers are also trained from scratch.

This mechanism ensures the powerful representations learned by the transferred projection weights are preserved, while the computational bottleneck (Global Attention) is replaced by the efficient Window Attention, and new components (propagation and pyramid) are only added where necessary.

---

## 4. The Training Strategy

### 4.1. MAE Pre-training is Critical

ViTDet's success relies heavily on **Masked Autoencoder (MAE)** pre-training:

**Why MAE Matters**:
- Plain ViT has fewer inductive biases (no built-in scale or translation equivariance like CNNs)
- MAE's self-supervised pre-training on ImageNet-1K teaches robust visual representations
- MAE allows plain ViT to learn scale-equivariant features from data

**Quantitative Impact** (ViT-L on COCO, Table 4 in the paper [^1]):
- Random initialization: 50.0 $AP^{box}$
- IN-1K supervised: 49.6 $AP^{box}$ (worse than random!)
- IN-21K supervised: 50.6 $AP^{box}$ (+0.6)
- **IN-1K MAE**: **54.6 $AP^{box}$ (+4.6)**

The gain from MAE is much larger for plain ViT than for hierarchical backbones, suggesting that MAE compensates for the lack of scale-related inductive biases.

### 4.2. Fine-tuning Details

**Key Hyperparameters**:
- Input resolution: $1024 \times 1024$
- Heavy augmentation: Large-scale jitter (scale range [0.1, 2.0])
- Training epochs: 75-100 (longer training needed due to heavy regularization)
- Optimizer: AdamW with layer-wise learning rate decay

**Window Size**: Default $14 \times 14$ to match MAE pre-training feature map resolution

**Relative Position Bias**: Added during fine-tuning (not pre-training) for fair comparison with Swin/MViT

---

## 5. Experimental Results

### 5.1. Comparison with Hierarchical Backbones (COCO)

**Mask R-CNN Framework**:

| Backbone | Pre-training | $AP^{box}$ | $AP^{mask}$ |
|:---------|:-------------|:------|:-------|
| Swin-B | IN-21K, sup | 51.4 | 45.4 |
| Swin-L | IN-21K, sup | 52.4 | 46.2 |
| MViTv2-B | IN-21K, sup | 53.1 | 47.4 |
| MViTv2-L | IN-21K, sup | 53.6 | 47.5 |
| MViTv2-H | IN-21K, sup | 54.1 | 47.7 |
| **ViT-B** | IN-1K, MAE | 51.6 | 45.9 |
| **ViT-L** | IN-1K, MAE | **55.6** | **49.2** |
| **ViT-H** | IN-1K, MAE | **56.7** | **50.1** |

**Key Observations**:
- ViT-L outperforms all hierarchical competitors despite using only IN-1K (vs IN-21K)
- ViT-H surpasses MViTv2-H by **2.6 $AP^{box}$**
- Plain ViT shows better scaling behavior gains increase with model size

**Cascade Mask R-CNN** (more challenging framework):
- ViT-H: 58.7 $AP^{box}$, 50.9 $AP^{mask}$
- Still outperforms hierarchical backbones

### 5.2. Ablation Studies

**Feature Pyramid Design** (ViT-L):
- No pyramid: 51.2 $AP^{box}$
- FPN (4-stage): 54.4 $AP^{box}$ (+3.2)
- FPN (last map only): 54.6 $AP^{box}$ (+3.4)
- **Simple pyramid**: **54.6 $AP^{box}$ (+3.4)**

$\rightarrow$ Simple pyramid matches FPN without complexity

**Propagation Strategy** (ViT-L):
- No propagation: 52.9 $AP^{box}$
- 4 global blocks: 54.6 $AP^{box}$ (+1.7)
- 4 conv blocks: 54.8 $AP^{box}$ (+1.9)
- Shifted windows (Swin-style): 54.0 $AP^{box}$ (+1.1)

$\rightarrow$ Propagation blocks are more effective than shifted windows

**Number of Propagation Blocks**:
- 0 blocks: 52.9 $AP^{box}$
- 2 blocks: 54.4 $AP^{box}$ (+1.5)
- 4 blocks: 54.6 $AP^{box}$ (+1.7)
- 24 blocks (all global): 55.1 $AP^{box}$ (+2.2, requires memory optimization)

$\rightarrow$ 4 blocks offer the best practical trade-off

---

## 6. Why ViTDet Works: Theoretical Insights

### 6.1. Decoupling Pre-training from Fine-tuning

**The Philosophy**:
- **Pre-training**: Learn general-purpose, task-agnostic features with fewer inductive biases
- **Fine-tuning**: Add task-specific adaptations (window attention, feature pyramids)

This is analogous to NLP: BERT/GPT use simple architectures for pre-training, then adapt to specific tasks.

**Benefits**:
1. Can leverage any ViT pre-training improvements (MAE, better architectures, larger datasets)
2. Don't need to redesign entire system when pre-training methods evolve
3. Detection-specific priors (multi-scale, locality) introduced only when needed

### 6.2. The Role of Positional Embeddings

Plain ViT uses learned positional embeddings to encode spatial information. This may explain why:
- Simple deconvolutions work without lateral connections
- ViT can recover spatial structure from low-resolution feature maps
- Information isn't "lost" despite aggressive downsampling

The high-dimensional embeddings (768 for ViT-B, 1280 for ViT-H) can theoretically preserve all patch information ($16 \times 16 \times 3 = 768$ parameters per patch).

### 6.3. Learning Scale Equivariance from Data

Hierarchical backbones have built-in scale priors (progressive downsampling). Plain ViT must **learn** scale equivariance:
- MAE pre-training teaches robust multi-scale representations
- Self-attention can learn to aggregate information at different scales
- The lack of hard-coded scale assumptions may provide flexibility

This is similar to how ViT learns translation equivariance without convolutions through data and self-attention.

---

## 7. Practical Considerations and Impact

### 7.1. Computational Efficiency

**Memory and Speed** (ViT-L with 4 propagation blocks):

| Strategy | Parameters | Train Memory | Inference Time |
|:---------|:-----------|:-------------|:---------------|
| Window only | $1.00\times$ (331M) | $1.00\times$ (14.6G) | $1.00\times$ (88ms) |
| 4 conv blocks | $1.04\times$ | $1.05\times$ | $1.04\times$ |
| 4 global blocks | $1.00\times$ | $1.39\times$ | $1.16\times$ |
| 24 global (all) | $1.00\times$ | $3.34\times$ | $1.86\times$ |

**Key Takeaway**: Window attention + 4 propagation blocks provides excellent efficiency. Using all global attention is impractical for large models.

### 7.2. When to Use ViTDet

**Advantages**:
- Want to leverage latest ViT/MAE pre-training advances
- Need compatibility with standard ViT architectures
- Prefer simpler, more modular designs
- Working with large-scale pre-training data

**Considerations**:
- Requires good pre-training (MAE recommended)
- Best performance at larger model sizes (ViT-L, ViT-H)
- May need hyperparameter tuning for specific tasks

### 7.3. Connection to SAM V1

The Segment Anything Model (SAM V1) adopted ViTDet's architectural innovations for its image encoder:

**SAM's Use of ViTDet**:
1. **Base Architecture**: Plain ViT-H/L/B with MAE pre-training
2. **Window Attention**: Uses $14 \times 14$ windows during segmentation fine-tuning
3. **Global Propagation**: Inserts 4 global attention blocks evenly across the 32 layers
4. **High-Resolution Processing**: Processes $1024 \times 1024$ images efficiently with this strategy

This demonstrates ViTDet's real-world impact enabling SAM to handle high-resolution segmentation at scale while maintaining compatibility with standard ViT pre-trained weights.

---

## 8. Key Takeaways

1. **Plain backbones can compete**: Non-hierarchical ViT matches or exceeds hierarchical Transformers for detection
2. **Simple adaptations suffice**: Window attention + 4 propagation blocks + simple feature pyramid
3. **MAE is critical**: Self-supervised pre-training compensates for fewer inductive biases
4. **Decoupling matters**: Separating pre-training from fine-tuning architecture enables flexibility
5. **FPN isn't necessary**: Multi-scale detection works with simple parallel conv/deconv from single map
6. **Better scaling**: Plain ViT shows superior performance gains at larger model sizes

---

## 9. Conclusion

ViTDet challenges the assumption that object detection requires hierarchical backbones. By maintaining the simplicity of plain Vision Transformers and introducing minimal, well-motivated adaptations during fine-tuning, it achieves state-of-the-art results while preserving the decoupling between pre-training and downstream tasks.

This work exemplifies a broader principle: sometimes, simpler architectures with strong pre-training can outperform more complex, task-specific designs. As pre-training methods continue to advance (MAE, CLIP, and beyond), ViTDet's philosophy of plain backbones becomes increasingly valuable.

The adoption of ViTDet's approach by subsequent models like SAM V1 validates its practical effectiveness and suggests that plain-backbone detection is not just an academic curiosity but a viable path forward for the field.

---


## References

[^1]: Li, Y., Mao, H., Girshick, R., & He, K. (2022, October). Exploring plain vision transformer backbones for object detection. In European conference on computer vision (pp. 280-296). Cham: Springer Nature Switzerland.

[^2]: Dosovitskiy, A. (2020). An image is worth 16x16 words: Transformers for image recognition at scale. arXiv preprint arXiv:2010.11929.

[^3]: He, K., Chen, X., Xie, S., Li, Y., Dollár, P., & Girshick, R. (2022). Masked autoencoders are scalable vision learners. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition (pp. 16000-16009).

[^4]: Liu, Z., Lin, Y., Cao, Y., Hu, H., Wei, Y., Zhang, Z., ... & Guo, B. (2021). Swin transformer: Hierarchical vision transformer using shifted windows. In Proceedings of the IEEE/CVF international conference on computer vision (pp. 10012-10022).

[^5]: Kirillov, A., Mintun, E., Ravi, N., Mao, H., Rolland, C., Gustafson, L., ... & Girshick, R. (2023). Segment anything. In Proceedings of the IEEE/CVF international conference on computer vision (pp. 4015-4026).

[^6]: Fan, H., Xiong, B., Mangalam, K., Li, Y., Yan, Z., Malik, J., & Feichtenhofer, C. (2021). Multiscale vision transformers. In Proceedings of the IEEE/CVF international conference on computer vision (pp. 6824-6835).

---

**Note**: For implementation details and code, see the official ViTDet repository at [detectron2/projects/ViTDet](https://github.com/facebookresearch/detectron2/tree/main/projects/ViTDet).