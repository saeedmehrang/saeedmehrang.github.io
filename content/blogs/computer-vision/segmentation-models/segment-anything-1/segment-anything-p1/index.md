---
title: "The Segment Anything Model Version 1 Overview (Part 1/3)"
date: 2025-10-22
draft: false
author: "Saeed Mehrang"
tags: ["Computer Vision", "Deep Learning", "Image Segmentation", "SAM", "Meta AI"]
categories: ["Machine Learning", "AI Models"]
description: "A brief guide to understanding Meta's Segment Anything Model (SAM 1)"
summary: "Meta's Segment Anything Model (SAM 1) delivers a wide variety of predictsion, detections, and segmentations with a remarkable accuracy. Part 1 from 3."
cover:
    image: "cover.png"
    alt: "Segment Anything Model"
    caption: "Meta's groundbreaking zero-shot segmentation model"
math: true
ShowToc: true
TocOpen: true
---


| Characteristic | Detail |
| :--- | :--- |
| **Estimated Reading Time** | 30-40 minutes |
| **Technical Level** | Intermediate |
| **Prerequisites** | Transformers, Vision Transformers (ViT) |


**Note:** This technical deep dive is split into three parts. This part (**Part 1**) covers the foundational innovations of SAM: the promptable segmentation task formulation, the three-module architecture (image encoder, prompt encoder, and mask decoder), and the training objectives including the combined Focal+Dice loss and multi-mask ambiguity handling. [**Part 2**](../segment-anything-p2/) explores the data engine methodology that enabled the SA-1B dataset, implementation details, theoretical foundations, comprehensive zero-shot transfer experiments across five tasks, ablation studies, and limitations with future directions. **Part 3** shows the API usage and code examples in a Google Colab.

---

## 1. Introduction: A Foundation Model for Segmentation

The Segment Anything Model (SAM) [^1], introduced by Meta AI Research [^2] in April 2023, represents a fundamental paradigm shift in how we approach image segmentation. Before SAM, segmentation models were predominantly trained for specific object categories on carefully curated datasets like COCO or LVIS. These models excelled within their training domains but struggled to generalize beyond the semantic categories they were explicitly taught. SAM broke this mold by treating segmentation not as a category-specific classification problem, but as a promptable task—a universal interface for identifying and delineating *any* object in *any* image. This is an open research with the code available in their Github [^3].

### 1.1. The Foundation Model Paradigm

The success of foundation models in natural language processing—models like GPT-3 and BERT that are pre-trained on massive text corpora and fine-tuned for specific tasks—inspired a fundamental question: Could computer vision achieve similar generalization capabilities? While models like CLIP demonstrated that vision-language alignment could enable zero-shot classification, segmentation remained a significantly more challenging problem. Segmentation requires dense, pixel-level predictions rather than image-level labels, and the lack of web-scale mask annotations meant that the data abundance enjoyed by NLP foundation models simply didn't exist for segmentation.

SAM's creators posed three interconnected questions that would determine the project's success:

1. **What task enables zero-shot generalization?** The answer was the promptable segmentation task—a flexible formulation where the model must produce valid masks given any type of prompt (points, boxes, masks, or text).

2. **What is the corresponding model architecture?** The solution was a three-component design: a heavyweight Vision Transformer [^4] image encoder that processes images once, a lightweight prompt encoder that handles various prompt types, and a fast mask decoder that combines these embeddings to predict masks in real-time.

3. **What data can power this task and model?** Since web-scale mask data doesn't exist, they created a "data engine"—an iterative human-in-the-loop annotation system powered by increasingly capable versions of SAM itself.

### 1.2. Why Segmentation Needed This Approach

Traditional segmentation approaches fall into several categories, each with inherent limitations:

- **Semantic segmentation** assigns class labels to every pixel but doesn't distinguish between different instances of the same class.
- **Instance segmentation** identifies individual objects but requires pre-defined categories and extensive labeled training data for each category.
- **Interactive segmentation** allows users to refine masks through iterative clicks, but these models are typically trained for the iterative refinement task rather than general segmentation.
- **Panoptic segmentation** combines semantic and instance segmentation but still requires category-specific training.

What was missing was a general-purpose segmentation model that could:
1. Segment objects from any visual domain without fine-tuning
2. Handle ambiguous prompts where multiple valid interpretations exist
3. Integrate seamlessly into larger systems as a composable component
4. Process prompts in real-time to enable interactive applications

SAM was designed from first principles to satisfy all these requirements, making it not just another segmentation model but a foundation for building segmentation capabilities into diverse applications.

---

## 2. The Novelty: The Promptable Segmentation Task (PST)

The single most significant innovation of SAM is its reframing of segmentation as a **promptable task**. This seemingly simple conceptual shift has profound implications for how we think about, train, and deploy segmentation models.

### 2.1. The Conceptual Breakthrough

In natural language processing, the "next token prediction" task serves dual purposes: it provides a scalable pre-training objective that requires no manual labels, and it creates an interface for solving downstream tasks through prompt engineering. SAM's creators recognized that segmentation needed an analogous task—one that could leverage diverse training data and enable zero-shot transfer to new problems.

The promptable segmentation task can be formally stated as follows:

**Given an image $I$ and a prompt $P$ (which may be a set of points, a bounding box, a coarse mask, text, or any combination thereof), predict a valid segmentation mask $M$ that corresponds to the prompt.**

The crucial word here is "valid." Unlike traditional segmentation, which seeks to predict *the* ground truth mask, promptable segmentation acknowledges that prompts can be inherently ambiguous. A single point on a person's shirt could validly refer to:
- The shirt itself
- The person wearing the shirt
- The person as part of a group
- The torso of the person

Rather than forcing the model to choose one "correct" interpretation, SAM is designed to predict multiple plausible masks and rank them by confidence. This ambiguity-awareness is not a limitation but a feature—it reflects the reality that segmentation is fundamentally an under-specified problem when given minimal prompts.

### 2.2. Why Prompting Enables Zero-Shot Transfer

The power of the promptable segmentation task lies in its composability. Once trained to respond appropriately to any prompt, SAM can solve downstream tasks through prompt engineering—designing appropriate prompts that convert the downstream problem into the promptable segmentation format.

**Example 1: Instance Segmentation**
Given an object detector that outputs bounding boxes, instance segmentation becomes trivial: simply prompt SAM with each detected box. The detector provides object locations and categories; SAM provides precise masks. This composability means SAM doesn't need to be trained on detection—it operates as a pure segmentation module.

**Example 2: Interactive Segmentation**
Traditional interactive segmentation models are trained with iterative user clicks that progressively refine a mask. SAM handles this naturally because it was trained with simulated interactive prompts—starting with sparse points and progressively adding clicks where predictions are uncertain. No special training is needed; the interactive capability emerges from the promptable task formulation.

**Example 3: Automatic Segmentation**
By prompting SAM with a regular grid of points across an image and using its ambiguity-aware multiple mask predictions, we can automatically discover and segment all objects without any prior knowledge of what objects are present. This is exactly how SAM generated 1.1 billion masks for its SA-1B dataset.

### 2.3. Types of Prompts and Their Design Rationale

SAM was designed to handle both **sparse** and **dense** prompts:

#### Sparse Prompts
1. **Points**: The most basic prompt, where each point can be labeled as foreground (click on the object) or background (click outside the object). Points are efficient for human annotation and simulate the natural interaction pattern of clicking on objects of interest.

2. **Bounding Boxes**: Rectangular regions that roughly enclose an object. Boxes are the output format of most object detectors, making them crucial for composability with detection systems.

3. **Text**: Free-form text descriptions like "a cat" or "person wearing red shirt." While SAM's text capabilities are limited compared to specialized vision-language models, the architecture supports text prompts by encoding them through CLIP's text encoder.

#### Dense Prompts
1. **Masks**: Coarse or incomplete segmentation masks that can be refined. This enables iterative segmentation where an initial prediction is progressively improved, or where one model's output is refined by another.

The design supports composing multiple prompt types simultaneously—for example, a box to localize a region plus a point to disambiguate which object within the box to segment, plus text to specify semantic constraints.

### 2.4. The Pre-Training Strategy

The promptable segmentation task naturally suggests a pre-training algorithm: For each training mask, simulate a sequence of prompts and train the model to predict valid masks given each prompt. The key insight is that unlike interactive segmentation—which aims to eventually predict a perfect mask after enough user input—SAM is trained to predict valid masks even from ambiguous or minimal prompts.

This training strategy is implemented through simulated interactive annotation:
1. Sample a ground truth mask from the training data
2. Randomly select a prompt strategy (e.g., random points, center point, box, etc.)
3. Sample an initial prompt according to that strategy
4. Predict a mask and compare it to the ground truth
5. If needed, sample additional prompts based on prediction errors (clicking on false negatives or false positives)
6. Repeat for multiple rounds (SAM uses 11 rounds per mask)

This simulated interaction teaches SAM to handle diverse prompt types and progressively refine predictions, while the ambiguity-aware loss (described in Section 4) ensures it learns to produce multiple valid interpretations when prompts are under-specified.

### 2.5. Task Generalization vs. Multi-Task Learning

It's important to distinguish SAM's approach from traditional multi-task learning. In multi-task systems, a single model is trained to perform multiple pre-defined tasks simultaneously—for example, a model might jointly perform semantic, instance, and panoptic segmentation. However, the set of tasks is fixed at training time, and the model cannot perform new tasks without retraining.

SAM exhibits *task generalization*: it can perform tasks at inference time that were not explicitly defined during training. The model learns a general-purpose segmentation capability that can be directed toward specific tasks through appropriate prompting. This is analogous to how CLIP, trained only on image-text alignment, can be composed with other systems to enable text-to-image generation (DALL·E) or other applications.

---

## 3. The SAM Architecture: Decoupling and Efficiency

SAM's architecture is a masterclass in efficiency through decoupling. The three-component design—image encoder, prompt encoder, and mask decoder—is motivated by a key computational insight: image encoding is expensive but needs to happen only once per image, while prompt encoding and mask prediction must be fast enough for real-time interaction. See the image below for a simple schematic illustration of the architecture (image adopted from the original article [^1])

{{< framed_image src="arch-simple.png" alt="simple-arch" width="900px" height="300px" >}}
{{< /framed_image >}}

### 3.1. Image Encoder: The Vision Transformer Backbone

The image encoder is the computational heavyweight of SAM's architecture, responsible for transforming high-resolution input images into rich, semantically meaningful embeddings.

#### 3.1.1. Architecture Specification

SAM uses a **Vision Transformer (ViT)** architecture, specifically adapted for high-resolution processing:

- **Model Variant**: ViT-Huge (ViT-H/16)
- **Patch Size**: 16×16 pixels
- **Input Resolution**: 1024×1024×3 (RGB)
- **Output Embedding**: 64×64×256 (spatial downsampling by 16×, with 256-dimensional features per position)
  - Internal hidden dimension: 1280 (ViT-H), 1024 (ViT-L), 768 (ViT-B)
  - Output features are projected down to 256 dimensions for the decoder
- **Parameters**: 636 million in ViT-H (308M in ViT-L, 91M in ViT-B)
- **Architecture**: 32 Transformer blocks (ViT-H), 24 blocks (ViT-L), 12 blocks (ViT-B)

The ViT-H architecture processes images by:
1. Dividing the 1024×1024 input into a grid of 64×64 patches (each 16×16 pixels, total 4096 patches)
2. Linearly projecting each patch into a 1280-dimensional embedding (the hidden dimension of ViT-H)
3. Adding learnable positional embeddings to retain spatial information
4. Processing the sequence of patch embeddings through 32 Transformer blocks (each with multi-head self-attention and MLP)
5. Projecting the final 64×64×1280 feature map down to 64×64×256 for compatibility with the lightweight decoder

#### 3.1.2. Masked Autoencoder (MAE) Pre-training

A critical architectural decision was to initialize the ViT encoder with weights from **Masked Autoencoder (MAE)** pre-training. MAE is a self-supervised learning method that:

1. Randomly masks out a large proportion (typically 75%) of image patches
2. Trains the model to reconstruct the missing patches from the visible context
3. Forces the encoder to learn rich, generalizable visual representations without requiring labeled data

MAE pre-training provides SAM with several advantages:
- **Robust features**: The encoder learns to understand object structure, textures, and relationships from massive unlabeled image datasets
- **Initialization**: Starting from strong pre-trained weights accelerates training and improves final performance
- **Generalization**: MAE's self-supervised nature means the encoder isn't biased toward specific object categories

The specific MAE pre-training used for SAM employed the ViT-H architecture on large-scale image datasets, providing a powerful starting point before fine-tuning on the promptable segmentation task.

#### 3.1.3. Pre-training vs Fine-tuning Architecture

The ViT image encoder architecture differs between pre-training and fine-tuning phases:

**During MAE Pre-training:**
- Uses standard **global self-attention** across all patches in all 32 Transformer blocks (for ViT-H)
- Processes images at lower resolution (typically 224×224 or 448×448)
- This is the standard ViT architecture without hierarchical modifications

**During SAM Fine-tuning (adapted from ViTDet):**
To handle high-resolution 1024×1024 images efficiently, SAM adapts the pre-trained backbone:

1. **Window Attention**: Divides the 64×64 feature map into non-overlapping 14×14 windows. Self-attention is computed within each window independently, reducing complexity from $O(N^2)$ to $O(N \cdot W^2)$ where $N = 4096$ patches and $W = 196$ patches per window.

2. **Cross-Window Propagation**: Inserts **4 global attention blocks** (or convolutional blocks) evenly spaced throughout the 32-block backbone to enable information flow across windows. These propagation blocks process the entire 64×64 spatial grid.

3. **Efficient Implementation**: Uses optimized attention kernels and mixed-precision training (FP16/FP32) to maximize throughput.

This adaptation strategy—maintaining global attention during pre-training but using windowed attention with sparse global blocks during fine-tuning—allows SAM to leverage standard ViT/MAE pre-trained weights while processing 1024×1024 images efficiently. The windowed attention adaptations are applied **only during fine-tuning** and do not require retraining the MAE pre-trained model.

#### 3.1.4. The One-Time Cost Advantage

The key architectural insight is that the image encoder runs **only once per image**. Its output—the 64×64×256 embedding—is cached and reused for all subsequent prompts on that image. This amortization is crucial for interactive applications:

- **Initial encoding**: ~200-300ms on GPU (one-time cost)
- **Prompt encoding + mask decoding**: ~50ms per prompt (real-time)

For interactive segmentation, where users may issue dozens of prompts on a single image, this design enables seamless real-time experiences that would be impossible if the encoder ran for every prompt.

### 3.2. Prompt Encoder: Flexible and Lightweight

The prompt encoder is designed to be fast, flexible, and capable of handling diverse prompt types through a unified embedding space.

#### 3.2.1. Sparse Prompt Encoding

For **points and boxes**, SAM uses positional encodings summed with learned embeddings:

1. **Positional Encoding**: Each spatial coordinate $(x, y)$ is mapped to a high-dimensional vector using Fourier features (sinusoidal position encodings), similar to those used in the original Transformer. This encoding allows the network to learn smooth functions over spatial coordinates.

2. **Type Embeddings**: Learned embeddings distinguish between different prompt types:
   - Foreground point vs. background point
   - Box corner types (top-left, top-right, bottom-left, bottom-right)
   - Click round (first click, second click, etc., for simulated interactive annotation)

3. **Combination**: The final sparse prompt embedding is the sum of the positional encoding and the appropriate type embedding.

Formally, for a point prompt $(x, y)$ of type $t$:
$$\mathbf{e}_{\text{point}} = \text{PE}(x, y) + \mathbf{e}_t$$

where $\text{PE}(\cdot)$ is the positional encoding function and $\mathbf{e}_t$ is a learned type embedding vector.

For a bounding box prompt with corners $(x_1, y_1, x_2, y_2)$, we create embeddings for each corner and combine them:
$$\mathbf{e}_{\text{box}} = \text{concat}(\text{PE}(x_1, y_1) + \mathbf{e}_{\text{TL}}, \text{PE}(x_2, y_2) + \mathbf{e}_{\text{BR}})$$

#### 3.2.2. Dense Prompt Encoding

For **mask prompts**, SAM uses a small convolutional network:
1. Input: A coarse binary mask (or soft probability map) at the same 1024×1024 resolution as the image
2. Architecture: Multiple convolutional layers with stride and downsampling
3. Output: A 64×64×256 embedding that matches the spatial dimensions of the image embedding

The mask encoder uses:
- 2D convolutions with 4×4 kernels and stride 2 (for downsampling)
- GELU activation functions
- Layer normalization
- Output channels matching the image embedding dimension (256)

The mask prompt embedding is added element-wise to the image embedding, allowing the mask to spatially modulate which regions the decoder should focus on.

#### 3.2.3. Text Prompt Encoding

For **text prompts**, SAM leverages CLIP's text encoder:
1. Input: Free-form text description (e.g., "a red car")
2. Encoder: Off-the-shelf CLIP text encoder (frozen weights)
3. Output: A fixed-dimensional text embedding

During training, SAM uses a clever trick to learn text-aware segmentation without requiring text annotations:
- For each training mask, extract the CLIP *image* embedding of the region
- Prompt SAM with this image embedding during training
- At inference time, substitute CLIP *text* embeddings for prompts

This works because CLIP is trained to align image and text embeddings in the same space. However, SAM's text capabilities are limited—it handles simple object descriptions better than complex phrases or spatial relationships.

#### 3.2.4. Multi-Prompt Composition

SAM can handle multiple prompts simultaneously by combining their embeddings:
- Multiple points: concatenate all point embeddings into a sequence
- Points + box: combine point and box embeddings
- Points + mask: add mask embedding to image embedding, provide point embeddings to decoder

This composability allows rich prompt specifications that disambiguate and refine segmentation requests.

### 3.3. Lightweight Mask Decoder: Speed Through Design

The mask decoder is the third and final component, designed to be extremely fast while producing high-quality, ambiguity-aware predictions. See the decoder architecture in the image below (Figure 14 adopted from the original article [^1]).

{{< framed_image src="decoder.png" alt="simple-decoder-arch" width="500px" height="450px" >}}
{{< /framed_image >}}

#### 3.3.1. Overall Architecture

The decoder takes three inputs:
1. **Image embedding**: 64×64×256 from the image encoder
2. **Prompt embeddings**: Variable-length sequence from the prompt encoder
3. **Output tokens**: Learned query embeddings that will become mask predictions

And produces:
1. **Mask predictions**: Three different masks (for ambiguity handling)
2. **IoU scores**: Confidence estimates for each predicted mask

#### 3.3.2. Two-Way Transformer Decoder

The core innovation is the **two-way attention mechanism**, inspired by DETR but with a crucial modification. Traditional transformer decoders use unidirectional cross-attention: queries attend to keys, but keys don't attend back. SAM's decoder uses bidirectional attention:

**Block Structure** (repeated 2 times):
1. **Self-attention on prompts**: Prompt embeddings attend to each other
   $$\mathbf{P}' = \text{SelfAttn}(\mathbf{P})$$

2. **Cross-attention: Prompts → Image**: Updated prompt embeddings attend to image embeddings
   $$\mathbf{P}'' = \text{CrossAttn}_{\text{P2I}}(\text{Q}=\mathbf{P}', \text{KV}=\mathbf{I})$$

3. **Cross-attention: Image → Prompts**: Image embeddings attend to prompt embeddings (the reverse direction!)
   $$\mathbf{I}' = \text{CrossAttn}_{\text{I2P}}(\text{Q}=\mathbf{I}, \text{KV}=\mathbf{P}'')$$

4. **MLP layers**: Both updated prompt and image embeddings pass through MLPs

This two-way attention allows:
- Prompts to gather relevant visual information from the image (prompt-to-image)
- The image embedding to be spatially modulated by prompt information (image-to-prompt)
- Both representations to be jointly refined based on their interaction

The bidirectionality is critical for ambiguity handling—the image embedding is updated based on prompt specificity, allowing subsequent processing to focus on prompt-relevant regions.

#### 3.3.3. Attention Configuration

- **Number of heads**: 8 attention heads per attention layer
- **Embedding dimension**: 256
- **MLP hidden dimension**: 2048 (8× expansion)
- **Number of decoder blocks**: 2 (stacking more didn't significantly help)

The relatively shallow decoder (only 2 blocks) is possible because the heavy lifting of visual understanding is done by the image encoder.

#### 3.3.4. Dynamic Mask Prediction Head

After the two-way transformer blocks, SAM upsamples and predicts masks:

1. **Upsampling**: The 64×64×256 image embedding is upsampled through transpose convolutions to 256×256

2. **Dynamic Linear Classifier**: Instead of using fixed convolutional layers, SAM uses output tokens processed by an MLP to generate dynamic classifier weights. These weights are applied to each spatial location's features to compute mask probabilities.

Formally, for output token $\mathbf{o}_i$ (where $i \in \{1, 2, 3\}$ for the three masks):
$$\mathbf{w}_i = \text{MLP}(\mathbf{o}_i)$$
$$M_i(x, y) = \sigma(\mathbf{w}_i^T \mathbf{f}(x, y))$$

where $\mathbf{f}(x, y)$ is the upsampled feature vector at spatial location $(x, y)$ and $\sigma$ is the sigmoid function.

This dynamic prediction allows each mask to specialize based on the prompt and image content, rather than using shared static convolutional weights.

3. **Final Upsampling**: Masks are upsampled from 256×256 to the original 1024×1024 resolution using bilinear interpolation.

#### 3.3.5. IoU Prediction Head

Alongside mask predictions, SAM predicts the Intersection-over-Union (IoU) quality score for each mask:

1. Each output token $\mathbf{o}_i$ is processed by a small MLP
2. The MLP outputs a scalar value in [0, 1] estimating the IoU between the predicted mask and the (unknown) ground truth

$$\hat{S}_{\text{IoU}}^{(i)} = \text{MLP}_{\text{IoU}}(\mathbf{o}_i)$$

This predicted IoU serves multiple purposes:
- **Ranking**: When SAM produces three masks, the one with highest predicted IoU is typically returned
- **Confidence**: The score indicates how confident SAM is in its prediction
- **Quality Control**: In automatic mask generation, low-scoring masks are filtered out

The IoU head is trained with Mean Squared Error loss between predicted and actual IoU:
$$\mathcal{L}_{\text{IoU}} = (\hat{S}_{\text{IoU}} - S_{\text{IoU}})^2$$

where $S_{\text{IoU}}$ is computed by comparing the predicted mask to the ground truth mask during training.

---

## 4. Training Objectives: Losses and Multi-Mask Ambiguity

SAM's training methodology is carefully designed to handle the unique challenges of promptable, ambiguity-aware segmentation.

### 4.1. The Combined Loss Function

SAM's mask prediction is supervised using a linear combination of **Focal Loss** and **Dice Loss**, following successful prior work in segmentation (particularly DETR and Mask2Former):

$$\mathcal{L}_{\text{mask}} = \lambda_{\text{focal}} \mathcal{L}_{\text{focal}} + \lambda_{\text{dice}} \mathcal{L}_{\text{dice}}$$

where $\lambda_{\text{focal}} = 20$ and $\lambda_{\text{dice}} = 1$, giving a **20:1 ratio** heavily favoring Focal Loss.

#### 4.1.1. Focal Loss: Handling Class Imbalance

Focal Loss addresses the fundamental class imbalance in segmentation: most pixels are background (or outside the target mask), while only a small fraction belong to the object. Standard cross-entropy loss is dominated by easy-to-classify background pixels, providing little learning signal for the challenging boundary regions.

Focal Loss down-weights easy examples and focuses learning on hard examples:

$$\mathcal{L}_{\text{focal}} = -\frac{1}{N} \sum_{i=1}^{N} \alpha_i (1 - p_t)^{\gamma} \log(p_t)$$

where:
- $N$ is the number of pixels
- $p_t$ is the confidence for pixel $i$'s ground truth class. Confidence is $p$ for positive class and $1-p$ for the negative class.
- $\alpha_i$ is a class-balancing weight (typically $\alpha = 0.25$ for positive class)
- $\gamma$ is the focusing parameter (typically $\gamma = 2$)

The key term is $(1 - p_t)^{\gamma}$:
- When $p_t \approx 1$ (easy example, correct prediction), $(1 - p_t)^{\gamma} \approx 0$ → loss contribution is minimal
- When $p_t \approx 0$ (hard example, incorrect prediction), $(1 - p_t)^{\gamma} \approx 1$ → full loss contribution

This focusing effect forces the model to improve on challenging pixels—typically object boundaries, small structures, and ambiguous regions—rather than getting distracted by easy background classifications.

#### 4.1.2. Dice Loss: Optimizing Overlap

Dice Loss (also called Soft Dice Loss or F1 Loss) directly optimizes the segmentation quality metric most relevant for evaluation: overlap between prediction and ground truth.

$$\mathcal{L}_{\text{dice}} = 1 - \frac{2 \sum_{i=1}^{N} p_i g_i + \epsilon}{\sum_{i=1}^{N} p_i + \sum_{i=1}^{N} g_i + \epsilon}$$

where:
- $p_i$ is the predicted probability for pixel $i$ (continuous value in [0,1])
- $g_i$ is the ground truth binary label for pixel $i$ (0 or 1)
- $\epsilon$ is a small smoothing constant (typically $\epsilon = 1$) to avoid division by zero
- $N$ is the number of pixels

The numerator $2 \sum p_i g_i$ represents twice the intersection (predicted positives that are actually positive), while the denominator $\sum p_i + \sum g_i$ represents the sum of predicted and actual positives. This is exactly the Sørensen–Dice coefficient, a measure of overlap.

Dice Loss has several advantages:
- **Class-imbalance robustness**: Unlike cross-entropy, Dice Loss isn't dominated by the majority class because it only considers positive predictions and labels
- **Direct optimization**: Optimizes IoU-like metrics directly rather than as a downstream effect of pixel classification
- **Smooth gradients**: Provides meaningful gradients even when the predicted mask is far from the ground truth

#### 4.1.3. Why the 20:1 Ratio?

The 20:1 weighting between Focal Loss and Dice Loss is an empirically determined balance:

- **Focal Loss** provides strong per-pixel gradients and handles classification, especially at object boundaries where pixel-level precision matters
- **Dice Loss** provides global shape information and ensures overall mask quality

The heavy weighting toward Focal Loss (20:1) suggests that per-pixel classification quality is more important than global overlap for SAM's learning dynamics. This makes sense given SAM's diverse training data—Focal Loss is more robust to annotation noise and varying object scales.

### 4.2. Handling Ambiguity: Multiple Mask Outputs

SAM predicts **three distinct masks** for each prompt, addressing the fundamental ambiguity in the promptable segmentation task.

#### 4.2.1. The Ambiguity Problem

Consider a point clicked on a person's face in a group photo. Valid segmentations include:
1. The face (subpart)
2. The person (part)
3. The group of people (whole)

Traditional segmentation models trained with single-output losses would average these possibilities or arbitrarily choose one, leading to inconsistent predictions. SAM explicitly models this ambiguity through multiple outputs.

#### 4.2.2. The Minimum Loss Matching Strategy

During training, SAM computes the loss for all three predicted masks against the single ground truth mask:

$$\mathcal{L}_{\text{total}} = \min_{i \in \{1,2,3\}} \left( \mathcal{L}_{\text{mask}}^{(i)} + \mathcal{L}_{\text{IoU}}^{(i)} \right)$$

where:
- $\mathcal{L}_{\text{mask}}^{(i)}$ is the mask loss (Focal + Dice) for prediction $i$
- $\mathcal{L}_{\text{IoU}}^{(i)}$ is the IoU prediction loss for prediction $i$

**Crucially, gradients are only backpropagated from the mask with the minimum loss.** This "winner-takes-all" strategy, borrowed from multiple-hypothesis learning, encourages the three prediction heads to specialize:

- One head learns to predict conservative, high-confidence masks (often the smallest valid mask)
- Another learns to predict larger, more inclusive masks
- The third learns intermediate or alternative interpretations

This specialization emerges naturally from the minimum loss matching—during training, whichever head is closest to the ground truth receives gradients to improve further, while the others are free to explore alternative valid interpretations without being penalized.

#### 4.2.3. Nested Mask Representation

In practice, SAM's three masks often form a nested hierarchy capturing different granularities:

1. **Subpart**: The most specific mask (e.g., a person's hand if the click is on the hand)
2. **Part**: The intermediate mask (e.g., the person)
3. **Whole**: The most inclusive mask (e.g., the group of people)

This nesting isn't explicitly enforced but emerges from the training dynamics. The paper notes that three masks are sufficient to capture most real-world ambiguity—nested masks are rarely more than three levels deep in typical images.

#### 4.2.4. Inference-Time Ranking

At inference time, SAM uses the predicted IoU scores to rank the three masks:
- The mask with the highest predicted IoU is returned by default
- All three masks can be retrieved if the application needs alternatives
- The IoU scores provide confidence estimates for downstream decision-making

This ranking mechanism is essential for zero-shot transfer—without ground truth, the predicted IoU is the only signal for which mask is most appropriate.

### 4.3. Simulated Interactive Training

SAM is trained with a sophisticated interactive simulation strategy that prepares it for diverse prompting scenarios.

#### 4.3.1. The 11-Round Protocol

For each ground truth mask in the training data, SAM simulates an interactive annotation session with **11 rounds of prompting**:

**Round 1**: Sample an initial prompt (e.g., a random point on the mask, or the mask's center point, or a random box containing the mask)

**Rounds 2-11**: Iteratively sample additional prompts based on the current prediction error:
- Randomly choose between clicking on a false negative pixel (inside ground truth but outside prediction) or a false positive pixel (inside prediction but outside ground truth)
- Sample the point location either randomly or from regions with highest error
- Add the new point to the accumulated prompt set
- Predict a new mask given all accumulated prompts

This protocol teaches SAM to:
- Make reasonable predictions from sparse initial prompts
- Progressively refine predictions as more information is provided
- Handle both positive and negative point prompts (indicating what to include vs. exclude)

#### 4.3.2. Prompt Sampling Strategies

SAM uses diverse prompt sampling to ensure robustness:

1. **Point sampling**:
   - Center point (at maximal distance transform value)
   - Random point inside mask
   - Grid points

2. **Box sampling**:
   - Tight bounding box around mask
   - Noisy bounding box (perturbed coordinates)
   - Loose bounding box (expanded by random margin)

3. **Mask sampling**:
   - Dilated/eroded versions of ground truth
   - Predictions from previous training iterations
   - Random partial masks

4. **Mixed sampling**: Combinations of the above (e.g., box + point)

This diversity ensures SAM doesn't overfit to any particular prompting pattern and can handle the varied inputs it will encounter during zero-shot transfer.

#### 4.3.3. Oversampling Strategy

The SA-1B dataset is dominated by automatically generated masks (99.1%), with only a small fraction from manual and semi-automatic annotation stages. To maintain representation of high-quality manually annotated data, SAM uses a **10× oversampling** strategy for manual and semi-automatic masks during training.

However, the paper notes that training on *only* the automatic masks (without oversampling) yields performance within ~0.5 mIoU of the full training setup. This validates the quality of SAM's automatically generated masks and simplifies the training pipeline.

---

This was the first part of the 3 part series on Segment Anything Model V1. See the part 2 [here](../segment-anything-p2/) that explores how SAM was trained at scale, its zero-shot capabilities, and technical analysis. Part 3 will cover the api and code examples in python and Google Colab.

---

## References

[^1]: Kirillov, Alexander, et al. "Segment anything." Proceedings of the IEEE/CVF international conference on computer vision. 2023.

[^2]: Meta AI Research: https://segment-anything.com/

[^3]: GitHub Repository: https://github.com/facebookresearch/segment-anything

[^4]: Dosovitskiy, A., et al. (2020). "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale."
