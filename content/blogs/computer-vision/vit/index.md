---
title: "The Image is a Sequence: Dissecting the Vision Transformer (ViT)"
date: 2025-10-22
author: "Saeed Mehrang"
tags: ["Deep Learning", "Vision Transformer", "ViT", "Attention", "Computer Vision", "AI"]
categories: ["Research"]
summary: "An in-depth look at 'An Image is Worth 16x16 Words,' the paper that introduced the pure Vision Transformer, its architecture, novelty, limitations, and how modern models like Swin Transformer evolved from it."
ShowToc: true
TocOpen: true
cover:
    image: "cover.png"
    alt: "ViT"
    caption: "Vision Transformer (ViT)"
math: true
---

| Characteristic | Detail |
| :--- | :--- |
| **Estimated Reading Time** | 15-20 minutes |
| **Technical Level** | Intermediate |
| **Prerequisites** | Transformers, Deep Learning |



# The Image is a Sequence: Dissecting the Vision Transformer (ViT)

The ICLR 2021 paper, "**An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale**," [^1] did more than just introduce a new model; it initiated a paradigm shift in computer vision. It demonstrated that the **Transformer**, an architecture previously dominant only in Natural Language Processing (NLP), could successfully tackle image recognition tasks **without any reliance on Convolutional Neural Networks (CNNs)**.

Let's dive into the core of the Vision Transformer (ViT), its theory, and its lasting impact on the field.

## I. The Novelty: From Pixels to Patches

The fundamental theoretical novelty of ViT is conceptual simplicity: it treats an image exactly like a sentence of words, allowing it to leverage the proven global reasoning power of the Transformer architecture.

### 1. Architectural Choice: The Pure Transformer

ViT rejects the traditional inductive biases of CNNs (like locality and translation equivariance). Instead, it adopts the standard **Transformer Encoder** from the original 2017 "Attention Is All You Need" paper. This Encoder is composed of alternating layers of **Multi-Head Self-Attention (MSA)** and a **Multi-Layer Perceptron (MLP)**. This choice emphasizes a highly flexible, general-purpose architecture that can theoretically model complex, long-range relationships in the data, a capability often limited in early CNN layers.

### 2. The Core Mechanism: Image Patch Embeddings

How do you turn an image into a sequence of "words" (tokens)? This is where ViT's ingenious yet simple patching mechanism comes in. 

1.  **Image Segmentation:** A standard $H \times W \times C$ image is segmented into non-overlapping flat patches of fixed size, typically $16 \times 16$. The authors of the ViT paper found $16 \times 16$ to be an effective balance between computational cost and feature granularity. If the input image is $224 \times 224$ and the patch size is $16 \times 16$, the total number of patches, $N$, is $$(224/16) \times (224/16) = 14 \times 14 = 196$$ tokens. The choice of patch size is critical; smaller patches increase $N$ and the resulting computational cost, while larger patches lose fine-grained spatial information.
2.  **Linear Projection and Patch Embedding:** Each $16 \times 16 \times 3$ patch (containing $768$ pixel values for RGB) is flattened into a one-dimensional vector, which is then linearly projected (embedded) into a constant $D$-dimensional vector space (e.g., $D=768$ for ViT-Base). This collection of $N$ vectors forms the core sequence input to the Transformer, essentially mapping the visual input into the vocabulary space of the Transformer.
3.  **The Class Token ($[CLS]$):** Inspired by BERT in NLP, a special learnable classification token, $\mathbf{x}_{\text{class}}$, is prepended to the sequence of embedded patches. This token does not correspond to a visual patch; rather, its purpose is to aggregate the global information processed by the self-attention layers across the entire image. The final state of this token, after passing through the Transformer Encoder, serves as the aggregate image representation used for the final classification head.
4.  **Positional Embeddings:** Since the Transformer architecture is **permutation-invariant**—it processes the input tokens without an inherent understanding of their order or relative position—ViT adds a learnable, one-dimensional **Positional Embedding** to each patch token. The model learns to encode the relative location of the image patches in the original 2D space, which is critical for reconstructing the image's structure and preventing the loss of spatial context. The ViT paper experimented with 2D-aware positional embeddings but found that simple 1D embeddings performed nearly as well, simplifying the architecture.

The sequence fed into the Transformer Encoder is therefore:
$$
\mathbf{z}_{0} = [\mathbf{x}_{\text{class}}; \mathbf{E}_{1}; \mathbf{E}_{2}; \dots; \mathbf{E}_{N}] + \mathbf{P}
$$
where $\mathbf{E}_i$ is the $i$-th patch embedding and $\mathbf{P}$ is the Positional Embedding matrix.

## II. The Architecture and Contrast to CNNs

### The ViT Block and Self-Attention

The core block of the ViT Encoder is structurally identical to the original Transformer block, adhering to the "residual connection followed by LayerNorm" structure.

* **Multi-Head Self-Attention (MSA):** This is the heart of the Transformer. It computes attention scores between every patch token and every other patch token (and the $CLS$ token) across multiple parallel "heads." This mechanism allows a token corresponding to one part of the image (e.g., an eye) to directly attend to and integrate information from a distant part of the image (e.g., a hand or background detail) in the very first layer. This is a key departure from CNNs, which only mix distant information after many sequential layers of local convolution and pooling.
    $$
    \text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{Softmax}\left(\frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d_k}}\right) \mathbf{V}
    $$
* **MLP Block (Feed-Forward):** A two-layer Feed-Forward Network (FFN) with a GELU non-linearity is applied independently to each token vector. This acts as a per-token processing module, enabling the model to combine the information gathered during the attention step.

### ViT vs. CNN: A Fundamental Shift

The contrast between ViT and CNNs highlights the profound change in approach to computer vision modeling.

| Feature | CNN (e.g., ResNet) | Vision Transformer (ViT) |
| :--- | :--- | :--- |
| **Input Processing** | Local kernels (filters) applied hierarchically to pixels. Features extracted are highly local and spatially constrained. | Global self-attention applied to non-overlapping patches. Relationships are modeled globally from the outset. |
| **Inductive Bias** | Strong (Locality, Translation Equivariance). This inherent bias makes them efficient on smaller datasets. | Weak (Only MLP/Attention structure). The model must learn locality and structure entirely from the data. |
| **Computational Complexity** | Linear with respect to image size $O(H W)$. Highly scalable with increasing image resolution. | **Quadratic** with respect to the number of patches $O(N^2)$, which is $O((HW/P^2)^2)$ where $P$ is patch size. This is the primary bottleneck for high-resolution images. |
| **Data Requirements** | Data-efficient; performs well on mid-sized datasets (e.g., ImageNet-1k) due to built-in biases. | **Data-hungry**; ViT requires pre-training on massive datasets (e.g., JFT-300M, over 300 million images) to outperform CNNs, as the model must explicitly learn the underlying visual grammar. |
| **Information Flow** | Local initially, information spreads gradually across the image through deep stacking and pooling layers. | **Global from the very first layer** via full self-attention, resulting in powerful long-range dependency modeling. |

The crucial finding in the ViT paper is that the lack of vision-specific inductive biases in the Transformer architecture can be perfectly compensated for by **scaling the training dataset**. This confirmed the "scaling hypothesis" in vision: given enough data, a general-purpose, high-capacity model (the Transformer) will outperform specialized models (CNNs). When trained on massive internal datasets like JFT-300M, ViT models achieved state-of-the-art results while often requiring substantially fewer computational resources for *pre-training* compared to highly optimized, large CNN models. In the paper, in Figure 5 (see its copy below), the authors show the superiority of ViT over CNN models and write:

> A few patterns can be observed. First, Vision Transformers dominate ResNets on the
performance/compute trade-off. ViT uses approximately 2 − 4× less compute to attain the same
performance (average over 5 datasets). Second, hybrids slightly outperform ViT at small computational budgets, but the difference vanishes for larger models. This result is somewhat surprising,
since one might expect convolutional local feature processing to assist ViT at any size. Third, Vision
Transformers appear not to saturate within the range tried, motivating future scaling efforts.

As a note on how things evolved after this publication, the hybrid architectures that were shown competitive, have been very popular as they present the same performance levels with better efficiency and computational costs. Also, it was shown by many articles and experiments that long range dependencies do not need to be modeled at every layer of a deep ViT or ViT-variant model. See SwinTransformer [^2] and ViTDet [^3] for more details on this claim, or see my 2 blogs, [SwinTransformer blog](../swintr/) and [ViTDet blog](../vitdet/) for a plain description of each.


{{< framed_image src="perf-comparison.png" alt="performance-comparison" width="900px" height="500px" >}}
{{< /framed_image >}}


## III. Limitations of ViT

While revolutionary, the original ViT architecture presents three primary limitations that fueled the next generation of research:

1.  **Quadratic Complexity and High Cost:** The global self-attention mechanism scales quadratically with the number of input tokens ($N^2$). Consider a high-resolution medical image, where using $16 \times 16$ patches results in thousands of tokens. The computational requirement for $N^2$ attention matrix multiplications quickly becomes prohibitive in terms of both GPU memory consumption and inference latency. This severely limits its practical application in scenarios requiring high-resolution input or real-time processing.
2.  **Severe Data Hunger:** The weak inductive bias means the model enters training as a **blank slate**; it must learn fundamental visual concepts (like edges, textures, and local shapes) purely from the data. This dependency means that training ViT models from scratch on smaller, common datasets like ImageNet-1k often results in sub-par performance compared to modern CNNs.
3.  **Lack of Hierarchy for Dense Prediction:** ViT maintains a single, fixed resolution throughout all its encoder layers. All tokens remain $16 \times 16$ patches, regardless of the layer depth. This fixed-scale representation is fundamentally sub-optimal for downstream computer vision tasks like **object detection** and **semantic segmentation**, which require feature maps at multiple granularities to accurately localize objects of varying sizes.

## IV. ViT's Evolution: Comparison to Swin Transformer

The **Swin Transformer** [^2] (Shifted Window Transformer) was explicitly designed to address ViT's three primary limitations, particularly the quadratic complexity and the missing multi-scale hierarchy.

| Feature | Vision Transformer (ViT) | Swin Transformer (Shifted Window) |
| :--- | :--- | :--- |
| **Attention Scope** | **Global** Self-Attention across all patches. A patch attends to *every* other patch. | **Local** Self-Attention within small, non-overlapping windows. Highly efficient. |
| **Complexity** | **Quadratic** $O(N^2)$, where $N$ is the total number of patches. | **Linear** $O(N)$, significantly improving scalability for high-resolution images. |
| **Structure** | **Flat**, fixed resolution across all layers. Simplistic but inflexible. | **Hierarchical** structure, mirroring the feature pyramid of a CNN (Stage 1, 2, 3, 4). |
| **Resolution Handling** | Fixed patch size; poor for dense prediction tasks. | Successively **merges** patches in deeper layers (e.g., $4 \times 4$ patches merge into a single token), reducing the number of tokens and increasing the receptive field. |
| **Novel Mechanism** | Standard full self-attention. | **Shifted Window Attention** is introduced to enable communication between adjacent windows in consecutive layers. This ensures the model retains global connectivity without sacrificing the efficiency of local attention. |

The Swin Transformer's ability to create feature maps at multiple resolutions (a feature pyramid) is what allowed it to achieve state-of-the-art results in dense prediction tasks like object detection and segmentation, something the original ViT struggled with due to its flat design.

## V. Potential Improvements for Efficiency

The limitations of ViT have spurred extensive research focused on making the architecture more efficient, less data-hungry, and more robust in practical scenarios.

1.  **Data-Efficiency (DeiT, MoCo-v3):** The **Data-efficient Image Transformer (DeiT)** demonstrated a breakthrough by achieving competitive performance on ImageNet-1k **without massive external pre-training data**. It achieved this using a **distillation strategy**, where a highly performing CNN (like a RegNet) acts as a 'teacher' model, guiding the ViT (the 'student') through a specialized distillation loss. This method effectively transfers the strong inductive biases of the CNN to the ViT, greatly reducing the data dependence and training time. Furthermore, self-supervised learning methods like **MoCo-v3** adapted for ViT also allowed for effective training on ImageNet-1k alone.
2.  **Linearizing Attention (PaCa-ViT, Perceiver IO):** To tackle the quadratic complexity bottleneck, newer architectures introduce sparse or linear attention mechanisms. Research like **Patch-to-Cluster Attention (PaCa-ViT)** replaces the full patch-to-patch attention with attention focused on a much smaller set of **learned cluster tokens** (or prototypes). By attending to $K$ clusters instead of $N$ patches, the complexity is reduced from $O(N^2)$ to $O(N \cdot K)$, where $K \ll N$. This innovation makes ViT backbones vastly more scalable and computationally efficient for high-resolution images.
3.  **Model Compression and Hardware-Aware Optimization:** For deployment on resource-constrained edge devices, direct ViT deployment is impractical. To overcome this, techniques like **quantization** (reducing the bit precision of weights, e.g., from 32-bit floating point to 8-bit integer) and **pruning** (removing redundant weights and connections) are critical. Frameworks like **UP-ViTs** (Unified Pruning) and **As-ViT** (Auto-scaling) focus on automating these optimizations to maintain accuracy while drastically reducing the model's footprint and computational demands. 
4.  **Hybrid Architectures:** A simple yet effective improvement involves mixing the best of both worlds. Many models now use a lightweight CNN (e.g., a simple stack of convolutional layers) for the initial patch embedding or the first few blocks. This "hybrid" approach re-introduces the benefits of locality and translation invariance at the start, making the overall model more robust and data-efficient, while retaining the powerful global reasoning capability of the Transformer layers deeper in the network.

The Vision Transformer was the starting gun for the attention-based revolution in computer vision. While subsequent models like Swin Transformer addressed its practical limitations, ViT's core contribution remains its elegance and its powerful proof that attention, truly, is all you need, provided you have the scale.

## References

[^1]: Dosovitskiy, A. (2020). An image is worth 16x16 words: Transformers for image recognition at scale. arXiv preprint arXiv:2010.11929.

[^2]: Liu, Z., Lin, Y., Cao, Y., Hu, H., Wei, Y., Zhang, Z., ... & Guo, B. (2021). Swin transformer: Hierarchical vision transformer using shifted windows. In Proceedings of the IEEE/CVF international conference on computer vision (pp. 10012-10022).

[^3]: Li, Y., Mao, H., Girshick, R., & He, K. (2022, October). Exploring plain vision transformer backbones for object detection. In European conference on computer vision (pp. 280-296). Cham: Springer Nature Switzerland.