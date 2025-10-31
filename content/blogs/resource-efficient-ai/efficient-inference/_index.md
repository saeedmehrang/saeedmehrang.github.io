---
title: "Methods for Neural Networks Post-training Optimization"
description: "A practical guide to methods aiming for **deployment efficieny** after the model is trained."
cover:
    # image: "efficient-ai-cover.png"
    alt: "Diagram illustrating optimized computational flow"
    relative: true
---

The challenge of modern AI lies in scaling models without skyrocketing **operational costs (Inference)**. This section explores breakthroughs in **model efficiency** that focus on optimization after (or during) the training process, specifically targeting deployment.

We categorize methods by their phase of implementation (e.g., *Applied During Training* vs. *Applied Post-Training*) and their primary goal (reducing **Training Time** vs. **Inference Time**). Discover techniques that decouple performance from prohibitive resource consumption, making large, high-performing AI models practical for everything from mobile devices to large-scale data center serving. See the list of methods below,


| Category (Implementation) | Method | Primary Benefit | Rationale / Key Trade-off |
| :--- | :--- | :--- | :--- |
| **Applied Post-Training / Inference** | **Post-Training Quantization (PTQ)** | **Inference Time** | Weights are reduced in precision *after* training to immediately reduce model size and speed up prediction. |
| **Applied Post-Training / Inference** | **Pruning** | **Inference Time** | Removes redundant structure *after* the full model is trained to achieve a smaller, faster deployment model. |
| **Applied Post-Training / Inference** | **Low-Rank Factorization (LRF)** | **Inference Time** | Decomposes weight matrices post-training to reduce parameters and FLOPs for deployment. |
| **Applied Post-Training / Inference** | **Model Compilers (TVM, XLA)** | **Inference Time** | Software-level optimization of the computational graph tailored for specific deployment hardware. |
| **Applied Post-Training / Inference** | **Neural Architecture Search (NAS)** | **Inference Time** (Net) | **Trade-off:** NAS significantly **increases** the total **training time** (search cost) to find an architecture that maximizes **inference speed**. |


