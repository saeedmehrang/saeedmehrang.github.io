---
title: "Taming the Transformer: How Perceiver IO and PaCa-ViT Conquer Quadratic Complexity"
date: 2025-10-23
author: "Saeed Mehrang"
tags: ["AI", "Machine Learning", "Transformers", "Efficiency", "Computer Vision", "Perceiver IO", "PaCa-ViT", "Attention Mechanisms"]
summary: "A deep dive into two novel architectures, Perceiver IO and PaCa-ViT, that break the O(N^2) barrier in Transformers, enabling them to process massive inputs efficiently."
cover:
    image: "cover.png"
    alt: "Diagram showing efficient transformer architectures"
    caption: "The future of Transformers is linear, not quadratic."
showToc: true
TocOpen: true
---

## The $\mathbf{O(N^2)}$ Wall: Why Standard Transformers Can't Scale

The **Transformer** architecture[^1] revolutionized deep learning when it was introduced in 2017, providing a powerful mechanism for modeling relationships in sequential data through its self-attention mechanism. At the heart of the Transformer is a simple but computationally expensive operation: every token must attend to every other token in the sequence. This all-to-all interaction enables the model to capture long-range dependencies, but it comes at a steep price.

**The Math:** The complexity of standard self-attention is $\mathbf{O(N^2)}$, where $N$ is the sequence length. For a sequence of length $N$, computing attention requires comparing each of $N$ query tokens with all $N$ key tokens, resulting in $\mathbf{N \times N}$ pairwise interactions. Both computational cost (**FLOPs**) and memory scale **quadratically** with sequence length.


## The "Sequence Length" Trap

The implications of this quadratic scaling vary dramatically across domains, though the industry has successfully pushed the boundaries far beyond previous limits:

- **Text (NLP):** This domain has witnessed the most dramatic scaling. While models like BERT still use short contexts (512 tokens) for many fine-tuning tasks, state-of-the-art **Large Language Models (LLMs)** now routinely operate with massive context windows:
    - Standard context lengths have jumped to $\mathbf{128\text{K}}$ to $\mathbf{200\text{K}}$ tokens (e.g., Llama 3.1, Claude 3.5 Sonnet).
    - Flagship models like **Gemini 2.5 Pro** and **Llama 4 Maverick** are production-ready at $\mathbf{1\text{M}}$ to $\mathbf{2\text{M}}$ tokens, enabling processing of entire books, large codebases, or years of corporate documentation in a single context.
    - This is achieved using highly efficient attention variants (e.g., FlashAttention) and architectural innovations that bypass full self-attention (e.g., **Mamba** and **Hyena** architectures).
    - **The cost of all-to-all attention for these massive contexts remains quadratic** ($\mathbf{O(N^2)}$), but these models employ sparse, linear, or state-space mechanisms to effectively handle the bulk of the work with near-linear scaling.

- **Vision (Images):** The problem of quadratic scaling for high-resolution imagery remains significant, though models are now deployed to handle it:
    - **Vision Transformers (ViTs)** and diffusion models have become dominant, operating on high-resolution images up to $\mathbf{1024 \times 1024}$ pixels, yielding $\mathbf{4,096}$ patches, or even $\mathbf{2\text{K}}$ resolution (4 megapixels) for models like **FLUX1.1 Pro Ultra**.
    - For dense prediction tasks like segmentation, the patch count can easily exceed $\mathbf{10,000}$.
    - Efficiency is managed by techniques like **PaCa-ViT** (Patch-to-Cluster attention, $\mathbf{O(N \times M)}$ complexity), **Swin Transformers** (local window attention), and the use of Transformer blocks within cascaded diffusion pipelines to process different image scales efficiently.

- **Other Domains (Audio/Video):** These high-bandwidth domains are now being directly modeled by Transformers, albeit often after heavy pre-processing:
    - **Audio:** Raw audio sampled at 16,000 Hz still generates 160,000 elements for a 10-second clip. Models now use efficient backbones (like those in **Continuous Audio Language Models (CALM)**) to capture long-term context over hours of audio, achieving effective processing of sequences well over $\mathbf{1\text{M}}$ tokens.
    - **Video:** Video and multi-modal tasks (combining raw audio, video frames, and text) present the greatest challenge, requiring inputs of **over $\mathbf{800,000}$ to $\mathbf{2.8\text{M}}$ elements (tokens)** for a single input sequence.
    - **Perceiver IO's** cross-attention bottleneck ($\mathbf{O(M \times N)}$ encoder complexity where $M$ is the large input size) is a canonical solution for handling these diverse, astronomically large, raw inputs by compressing them into a small latent state ($N$). This ability to process long, raw sequences without intermediate domain-specific tokenization is now a key factor in advancing multi-modal AI.


**The Thesis:** The $\mathbf{O(N^2)}$ cost is the biggest bottleneck to applying Transformers to high-bandwidth data. To unlock Transformers for these domains, we must break the $N \times N$ interaction. This post explores two groundbreaking architectures that do exactly that: **Perceiver IO**[^2] and **PaCa-ViT**[^3]. Both achieve linear scaling, but through fundamentally different philosophies.

***

## Solution 1: Perceiver IO - The Generalist's Bottleneck

### The Core Idea: Decouple Processing from Input Size

**Perceiver IO** is a *general-purpose architecture* designed to handle arbitrary inputs and outputs across any domain—vision, language, audio, multi-modal data, or symbolic representations. Its fundamental insight is to decouple the computational cost from the size of the input and output arrays.

**Analogy:** Imagine a busy CEO (the Processor) who cannot possibly read millions of employee reports (the Input). Instead, the CEO has a small team of expert assistants (the **Latent Array**) who read all the reports, extract the essential information, discuss it deeply among themselves, and then provide targeted answers to specific questions. The CEO never directly processes the overwhelming raw input—everything is mediated through the compact team of assistants.

### The Architecture: A Three-Step "Read-Process-Write" Pipeline

Perceiver IO implements a clean separation of concerns through three distinct stages:

**1. The Input/Output Arrays:**
- **Input Array (Size $M$):** The raw, potentially massive data. This could be 50,176 pixels in an image, 160,000 audio samples, or 2,048 UTF-8 bytes in a text sequence.
- **Output Query Array (Size $O$):** Queries that specify what outputs you want. This could be a single class label ($O=1$), 182,528 optical flow vectors for dense prediction, or 800,000+ elements for multi-modal autoencoding.

**2. The "Magical" Latent Array (Size $N$):**
This is the secret sauce—a *small, fixed-size* array of latent vectors (e.g., N=256, 512, or 2,048).

**Crucially: $N \ll M$ and often $N \ll O$.** The latent array acts as an information bottleneck, forcing the model to compress the input into a compact, semantically rich representation.

**3. The Three-Stage Mechanism:**

- **Stage 1: Encode (Read)** A **cross-attention** layer where the small **Latent Array** ($N$ queries) attends to the massive **Input Array** ($M$ keys/values). The latents "read" and "summarize" the input. The computational complexity of this step is $\mathbf{O(N \times M)}$.

- **Stage 2: Process** A deep stack of **self-attention** layers (a standard Transformer) that operates *only* on the small **Latent Array** ($N$). This is where the heavy computation happens, but it's computationally cheap because $N$ is small. If there are $L$ layers, the total complexity is $\mathbf{O(L \times N^2)}$.

- **Stage 3: Decode (Write)** Another **cross-attention** layer where the **Output Query Array** ($O$ queries) attends to the processed **Latent Array** ($N$ keys/values) to produce the final output. The computational complexity of this step is $\mathbf{O(O \times N)}$.

### The Efficiency Win: From $\mathbf{O(M^2)}$ to $\mathbf{O(M+O)}$

Let's break down the computational complexity:

| **Component** | **Complexity** |
|:---|:---|
| Vanilla Transformer (on raw input) | $\mathbf{O(M^2)}$ |
| Perceiver IO Encoder | $\mathbf{O(M \times N)}$ |
| Perceiver IO Processor ($L$ layers) | $\mathbf{O(L \times N^2)}$ |
| Perceiver IO Decoder | $\mathbf{O(O \times N)}$ |
| **Total Perceiver IO** | $\mathbf{O(M \times N + L \times N^2 + O \times N)}$ |

**The Punchline:** Because $N$ is a small, fixed constant (independent of $M$ and $O$), the effective complexity becomes $\mathbf{O(M + O)}$, which is **linear** in both input and output size! The quadratic term $\mathbf{O(L \times N^2)}$ is fixed and small because $N$ is small.

**Intuition:** If we consider $N$ and $L$ as constants, the most expensive part scales as $\mathbf{O(M)}$ (encoding) and $\mathbf{O(O)}$ (decoding). This allows Perceiver IO to process:
- Images with 365,056 patches for optical flow[^2]
- UTF-8 byte sequences of 2,048 characters (4$\times$ longer than BERT)[^2]
- Multi-modal Kinetics data with over 800,000 combined audio, video, and label elements[^2]
- All with tractable computational costs

**Decoding Flexibility:** The output query mechanism is exceptionally flexible. Queries are constructed by combining position encodings, task embeddings, modality embeddings, or input features—whatever is needed to specify the desired output semantics. For classification, a single learned query suffices. For optical flow, each output query combines spatial position with input pixel features. For multi-modal autoencoding, queries combine position encodings with modality-specific embeddings.

***

## Solution 2: PaCa-ViT - The Vision Specialist's Bottleneck

### The Core Idea: Attend to Clusters, Not Every Patch

While Perceiver IO is a general architecture for any domain, **PaCa-ViT** (Patch-to-Cluster Attention Vision Transformer)[^3] is a specialized, efficient *attention mechanism* designed specifically for **Vision Transformers (ViTs)**.

**The Problem with Standard ViT:** Vision Transformers[^4] apply patch-to-patch self-attention, which is still $\mathbf{O(N^2)}$ where $N$ is the number of image patches. For a $224 \times 224$ image with $16 \times 16$ patches, that's $N=196$ patches—manageable but expensive. For higher resolutions or dense prediction tasks, the cost explodes.

**Analogy:** In a 1,000-person lecture hall (the Patches), you don't need to have a conversation with all 999 other students to understand the material. You just need to engage with the 10 teaching assistants (the **Clusters**) who represent the main concepts. The TAs aggregate information from groups of students and provide unified explanations. Information flows efficiently through this hierarchy.

### The Architecture: Patch-to-Cluster Attention (PaCa)

PaCa-ViT replaces standard patch-to-patch self-attention with a novel **asymmetric attention mechanism**:

**1. The Inputs:**
- **Patches (Queries, Size $N$):** The standard $N$ patch embeddings from the image (e.g., $N=196$ for $224 \times 224$ images with $16 \times 16$ patches).

**2. The "Magical" Clusters (Keys/Values, Size $M$):**
- A *small, fixed number* of cluster "tokens" or "centroids" (e.g., $M=64$ or 100).
- These clusters are *learnable parameters* trained end-to-end via backpropagation.
- **Crucially: $M \ll N$.** The clusters act as a learned, compressed representation of visual concepts.

**3. The Mechanism:**

- **Step 1: Cluster Assignment** A lightweight clustering module (implemented via learnable cluster embeddings) groups the $N$ patches into $M$ semantic clusters. This can be done via soft assignment (attention-based) or through learnable cluster centers.

- **Step 2: Patch-to-Cluster Attention** The $N$ patches (Queries) attend to the $M$ cluster centers (Keys and Values). Each patch aggregates information from the compressed set of clusters rather than from all other patches.

- **Information Flow:** $\text{Patches} \to \text{Clusters} \to \text{Patches}$. Information is bottlenecked *through* the clusters within each attention layer. The clusters serve as a learned "communication hub" that captures the most important visual patterns.

### The Efficiency Win: From $\mathbf{O(N^2)}$ to $\mathbf{O(N)}$

The computational savings are dramatic:

| **Attention Type** | **Complexity** |
|:---|:---|
| Standard ViT Self-Attention | $\mathbf{O(N^2)}$ |
| PaCa-ViT Patch-to-Cluster | $\mathbf{O(N \times M)}$ |

**The Punchline:** Because $M$ is a small constant (independent of $N$), the complexity of each attention layer becomes $\mathbf{O(N)}$, which is **linear** in the number of patches!

For a $224 \times 224$ image with $N=196$ patches and $M=64$ clusters:
- Standard attention: $196 \times 196 = 38,416$ interactions
- PaCa attention: $196 \times 64 = 12,544$ interactions
- **Speedup: $\sim 3\times$** with even greater gains at higher resolutions

**Bonus: Interpretability** The $M$ learned clusters are semantically meaningful and interpretable. Visualizations reveal that different clusters specialize in different visual concepts—one might capture edges, another textures, another object boundaries. This provides insight into what the model has learned, something standard ViTs lack.

**Implementation Details:** PaCa-ViT typically uses:
- Learned cluster embeddings initialized randomly or via k-means
- Soft assignment where each patch can attend to all clusters with learned weights
- Standard FFN layers and residual connections
- Can be inserted as a drop-in replacement in any ViT architecture

***

## Comparative Analysis: Two Philosophies for Linear Scaling

Both Perceiver IO and PaCa-ViT replace an $\mathbf{O(N^2)}$ operation with a linear one by introducing a small bottleneck, but their approach, scope, and philosophy differ fundamentally.

### Table: Perceiver IO vs. PaCa-ViT

| **Feature** | **Perceiver IO** | **PaCa-ViT** |
|:---|:---|:---|
| **Main Goal** | General architecture for any domain (vision, audio, text, multi-modal, symbolic) | Efficient attention module specifically for Vision Transformers |
| **Scope** | End-to-end architecture | Attention layer replacement within ViT |
| **What is $N$?** | Size of the **latent array** (hyperparameter, e.g., 256-2,048) | Number of **input patches** (depends on image size, e.g., 196) |
| **What is $M$?** | Size of the **raw input array** (e.g., 50,176 pixels, 160,000 audio samples) | Number of **clusters** (hyperparameter, e.g., 64-100) |
| **The Bottleneck** | Fixed-size **latent array** that mediates all input/output | Fixed-size set of **cluster centers** inside attention layers |
| **Information Flow** | $\text{Input}(M) \to \text{Latent}(N) \to \text{Process}(N) \to \text{Output}(O)$ (3-stage pipeline) | $\text{Patches}(N) \to \text{Clusters}(M) \to \text{Patches}(N)$ (per layer) |
| **Complexity** | $\mathbf{O(MN + LN^2 + ON)}$ total | $\mathbf{O(N \times M)}$ per attention layer |
| **Linear In...** | Input size ($M$) and Output size ($O$) | Number of patches ($N$) |
| **Architecture Changes** | Complete replacement of standard Transformer | Drop-in replacement for attention in ViT blocks |
| **Processing Depth** | Deep latent processing (e.g., 26-40 layers) on small $N$ | Standard ViT depth with modified attention |
| **Domain Assumptions** | Minimal—works on raw bytes, pixels, audio samples | Vision-specific—assumes patch-based image input |
| **Key Advantage** | Unprecedented generality. Handles massive, diverse, multi-modal I/O | Simple, efficient ViT upgrade. Interpretable clusters |
| **Trained Bottleneck** | Latent array initialization (sometimes learned via processing) | Explicit learned cluster embeddings |
| **Output Mechanism** | Flexible query-based cross-attention decoding | Standard classification head or dense prediction |

### Deeper Dive: When to Use Each Architecture?

**When to use Perceiver IO:**
- **Massive inputs:** High-resolution images (e.g., $1024 \times 1024$ or larger), video sequences, raw audio waveforms, long text sequences, scientific data
- **Multi-modal fusion:** Jointly processing video + audio + text, or any combination of heterogeneous modalities with different spatial/temporal structures
- **Domain-agnostic requirements:** Building a single unified model that can handle multiple data types without modality-specific preprocessing
- **Flexible outputs:** Dense prediction tasks with varying output sizes (e.g., optical flow, segmentation masks, autoencoding)
- **No tokenization:** Processing raw bytes in NLP without handcrafted vocabularies

**Example applications:**
- Optical flow estimation from raw pixels (state-of-the-art on Sintel benchmark)[^2]
- Multi-modal autoencoding on Kinetics video+audio+labels with $800\text{K}+$ outputs[^2]
- UTF-8 byte-level language modeling matching BERT performance[^2]
- Replacing Transformers in AlphaStar for StarCraft II with $3.5\times$ FLOPs reduction[^2]

**When to use PaCa-ViT:**
- **Upgrading existing ViTs:** Making pre-trained or custom ViTs more efficient without complete architectural redesign
- **High-resolution vision:** Scaling ViTs to higher resolutions for classification, detection, or segmentation
- **Model interpretability:** Understanding what visual concepts your model learns through cluster visualization
- **Resource constraints:** Deploying ViTs on edge devices or in real-time applications where compute is limited
- **Vision-specific optimizations:** Leveraging the spatial structure and patch-based nature of images

**Example applications:**
- Efficient image classification with $3\times$ speedup on attention computation[^3]
- High-resolution dense prediction (semantic segmentation, depth estimation)[^3]
- Video understanding where temporal dimension increases patch count[^3]
- Interpretable vision models for medical imaging or safety-critical applications[^3]

***

## Implementation Insights and Performance

### Perceiver IO: Empirical Results

Perceiver IO demonstrates strong performance across diverse domains:

| **Task** | **Dataset** | **Metric** | **Performance** |
|:---|:---|:---|:---|
| Language (MLM) | GLUE | Avg. Score | 81.8 (matches BERT)[^2] |
| Optical Flow | Sintel Final | EPE | 1.81 (state-of-the-art)[^2] |
| Multi-modal Autoencoding | Kinetics-700 | Video PSNR | 24.37 dB[^2] |
| Image Classification | ImageNet | Top-1 Acc. | 84.5% (with JFT pretraining)[^2] |
| Symbolic Reasoning | StarCraft II | Win Rate | 87% (matches Transformer)[^2] |
| Audio-Video Classification | AudioSet | mAP | 44.9%[^2] |

**Key architectural choices:**
- Latent array size: $N=256$ to 2,048 depending on task complexity
- Depth: 26-40 processing layers (enabled by small latent size)
- Training: Subsamples outputs during training for very large output spaces (e.g., only 512 of $800\text{K}$ elements in Kinetics)
- Inference: Generates full outputs in batches using parallel decoding

### PaCa-ViT: Efficiency Gains

While detailed performance metrics vary by implementation, PaCa-ViT typically achieves:
- **Computational savings:** $3-4\times$ reduction in attention FLOPs[^3]
- **Memory reduction:** Proportional to cluster count vs. patch count
- **Accuracy:** Minimal degradation ($<1-2\%$) compared to standard ViT[^3]
- **Scalability:** Linear scaling to higher resolutions vs. quadratic for standard ViT

**Cluster insights:**
- Learned clusters exhibit semantic specialization (edges, textures, object parts)[^3]
- Optimal cluster count: Typically $M=64$ to 100 for $224 \times 224$ images[^3]
- Clusters are shared across attention heads but specialized per layer[^3]

***

## The Broader Context: Efficient Attention Mechanisms

Perceiver IO and PaCa-ViT are part of a broader research effort to make Transformers practical for real-world data:

**Other efficient attention approaches:**
- **Sparse attention:** Longformer[^5], BigBird restrict attention to local windows plus global tokens
- **Low-rank approximations:** Linformer[^6], Performer approximate attention matrix with lower-rank factorizations
- **Kernel methods:** Linear Transformers replace softmax attention with kernel functions
- **Recurrent mechanisms:** Transformer-XL[^7], Compressive Transformers reuse representations across segments

**What makes Perceiver IO and PaCa-ViT unique:**
1. **Generality:** Perceiver IO works across all domains without modification
2. **Simplicity:** PaCa-ViT is a simple drop-in replacement requiring minimal changes
3. **Strong empirical results:** Both achieve competitive or state-of-the-art performance
4. **Principled design:** Information bottlenecks force efficient representations

***

## Conclusion: The Future of Transformers is Asymmetric

The original Transformer's $N \times N$ "all-to-all" attention is powerful but incredibly wasteful. Most tokens don't need to directly interact with most other tokens. Both Perceiver IO and PaCa-ViT demonstrate the power of **asymmetric** or **bottlenecked** attention.

**Perceiver IO's Philosophy:** Processing power is precious. Dedicate a small, fixed-size processor (the latent array) to deep computation, and force the large, messy world (inputs and outputs) to conform to it via efficient cross-attention. The latent array acts as a universal information hub—a "cognitive workspace" that mediates all interactions.

**PaCa-ViT's Philosophy:** The world (especially images) is redundant. We don't need to compare every tiny piece to every other tiny piece. Instead, we can learn a compressed set of semantic concepts (clusters) and compare pieces to concepts. This achieves the same representational power far more efficiently.

**Commonalities:**
- Both introduce a **learned bottleneck** ($N$ latents or $M$ clusters)
- Both achieve **linear complexity** (in input/output size or patch count)
- Both rely on **cross-attention** between different-sized arrays
- Both learn **compressed representations** that capture essential information

**Differences:**
- **Scope:** Perceiver IO is a complete architecture; PaCa-ViT is a layer modification
- **Generality:** Perceiver IO is domain-agnostic; PaCa-ViT is vision-specific
- **Where bottleneck lives:** Perceiver IO has a persistent latent state; PaCa-ViT has per-layer clusters

**Final Thought:** The path to scaling Transformers to real-world, high-bandwidth data is clear: we must abandon homogeneous, all-to-all attention. Whether by creating a central information hub (Perceiver IO) or by learning a compressed set of concepts (PaCa-ViT), the future of efficient Transformers is **linear, not quadratic**. As we push AI to process increasingly rich, multi-modal data—4K video, high-resolution medical scans, long-context documents, real-time sensor streams—architectures that break the $\mathbf{O(N^2)}$ barrier won't just be nice to have. They'll be essential.

***

## References

[^1]: Vaswani, A., et al. (2017). "Attention is All You Need." *Advances in Neural Information Processing Systems (NeurIPS)*.

[^2]: Jaegle, A., Borgeaud, S., Alayrac, J. B., Doersch, C., Ionescu, C., Ding, D., ... & Carreira, J. (2021). Perceiver io: A general architecture for structured inputs & outputs. arXiv preprint arXiv:2107.14795.

[^3]: Grainger, R., Paniagua, T., Song, X., Cuntoor, N., Lee, M. W., & Wu, T. (2023). Paca-vit: learning patch-to-cluster attention in vision transformers. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (pp. 18568-18578).

[^4]: Dosovitskiy, A., et al. (2021). "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale." *International Conference on Learning Representations (ICLR)*.

[^5]: Beltagy, I., Peters, M. E., & Cohan, A. (2020). "Longformer: The Long-Document Transformer." *arXiv preprint arXiv:2004.05150*.

[^6]: Wang, S., et al. (2020). "Linformer: Self-Attention with Linear Complexity." *arXiv preprint arXiv:2006.04768*.

[^7]: Dai, Z., et al. (2019). "Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context." *Association for Computational Linguistics (ACL)*.