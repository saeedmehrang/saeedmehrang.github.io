---
title: "Unpacking Mixture-of-Experts (MoE) in LLMs: A Foundational Dive" 
date: 2025-10-09
tags: ["LLM", "MoE", "Deep Learning", "Sparse Models"] 
author: "Saeed Mehrang" 
description: "Explore the foundational concepts of Mixture-of-Experts (MoE) in Large Language Models, from its origins to a simple PyTorch implementation of the original architecture." 
summary: "This blog post demystifies Mixture-of-Experts (MoE) layers, a key innovation for scaling Large Language Models efficiently. We'll trace its origins, delve into the mathematical underpinnings, and build a foundational MoE block in PyTorch, mirroring the architecture from its initial conception."
cover:
    image: "moe.png"
    alt: "The What,When, and Why of Mixture-of-Experts (MoE)"
    relative: true
showToc: true
disableAnchoredHeadings: false
---


## The "What," "When," and "Why" of Mixture-of-Experts (MoE)

The relentless pursuit of larger and more capable Large Language Models (LLMs) often hits a wall: computational cost. It is widely reported and generally believed that many, if not most, of the leading LLM providers and the AI startups building services on top of them are currently running at a loss due to the immense computational costs. As models grow, so does their appetite for compute, both during training and inference. The cost scales with the total number of parameters, making training and serving trillion-parameter models prohibitively expensive. Enter **Mixture-of-Experts (MoE)** layers, a paradigm shift designed to tackle this challenge by introducing a sparsity-inducing mechanism. This technique allows for a dramatic increase in a model's total parameter count while maintaining a significantly lower computational budget for any given input.


**What is MoE?**  At its core, an MoE layer replaces a dense feed-forward network (FFN) with several smaller, specialized "expert" networks. For each input, a "gating network" (or router) decides which expert(s) should process it, effectively activating only a subset of the model's parameters for a given computation.


**When did it emerge?**  The concept of Mixture-of-Experts isn't new; it has roots dating back to the early 1990s. The seminal work by **Jacobs et al. (1991), "Adaptive Mixtures of Local Experts,"** laid the theoretical groundwork. However, its resurgence and practical applicability in deep learning, particularly with Transformers and LLMs, gained significant traction with works like "Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer" by Shazeer et al. (2017) at Google Brain.


A significant recent milestone demonstrating the practical power of MoE in the open-source community is the Mixtral 8x7B model, detailed in the paper "Mixtral of Experts" by Mistral AI (2024). Mixtral uses a **Sparse Mixture of Experts (SMoE)** architecture where each Transformer layer has 8 experts, but a router network selects only two experts to process each token. This design gives the model access to ≈47 billion total parameters, while only ≈13 billion are actively used per token. This low active parameter count enables Mixtral to achieve faster inference speeds and higher throughput than dense models like Llama 2 70B, all while matching or outperforming them (and even GPT-3.5) across major language, code generation, and mathematics benchmarks.


**Why does it matter for LLMs?** In the context of LLMs, MoE offers a tantalizing advantage: it allows for a massive increase in model capacity (more experts mean more parameters) without a proportional increase in computational cost. By activating only a few experts per token, an MoE model can have trillions of parameters while still maintaining manageable training and inference FLOPs, enabling the creation of "outrageously large" yet efficient models. This blog post will focus on the foundational principles, drawing inspiration from the original concepts to build a simple, illustrative implementation.



## Core Technical and Mathematical Foundation

To understand MoE, let's first establish the context of a standard Feed-Forward Network (FFN), which MoE layers often replace within architectures like the Transformer. If you are not familiar with Transformer architecture, see my other blog post where I briefly explain what the vanilla Transformer architecture does and how.


### The Standard Feed-Forward Network (FFN)

In a typical Transformer block, the architecture is modular, consisting of two main sub-layers: the **Multi-Head Attention (MHA)** sub-layer and the **Feed-Forward Network (FFN)** sub-layer. The FFN is the **second and final main sub-layer** of a Transformer block.

After the self-attention mechanism processes the relationships between all tokens, the data usually flows into the FFN. Importantly, in the standard Transformer architecture, **both** the MHA and FFN sub-layers are wrapped by a combination of a **Residual Connection** (or **Add**) and a **Layer Normalization** (or **Norm**).

Therefore, the sequential flow of data in the FFN sub-layer is:

1.  **Input:** The output from the previous sub-layer (or the output of the MHA, followed by its Add & Norm operation) serves as the input $\mathbf{x}$.
2.  **FFN Computation:** The data then flows through the FFN, which is a **two-layer Multi-Layer Perceptron (MLP)**. This FFN is **position-wise**, meaning it applies the exact same set of weights to every single token's representation independently.
3.  **Add & Norm:** The output of the FFN is then added back to its input ($\mathbf{x}$), followed by another Layer Normalization operation. This is represented as $\text{LayerNorm}(\mathbf{x} + \text{FFN}(\mathbf{x}))$.

For an input vector $\mathbf{x} \in \mathbb{R}^{d_{model}}$ (where $\mathbf{x}$ is a token representation), a standard FFN block can be described as:

$$\text{FFN}(\mathbf{x}) = \max(0, \mathbf{x}W_1 + b_1)W_2 + b_2$$

where:

* $\mathbf{x}$ is the input from the attention sub-layer's Add & Norm.
* $W_1 \in \mathbb{R}^{d_{model} \times d_{ff}}$ and $b_1 \in \mathbb{R}^{d_{ff}}$ are the weights and biases of the **first linear layer**. This layer is responsible for **expanding** the dimensionality, as $d_{ff}$ is often $4 \times d_{model}$.
* $\max(0, \cdot)$ is the non-linear activation function (such as ReLU or the more common **GELU** in modern LLMs).
* $W_2 \in \mathbb{R}^{d_{ff} \times d_{model}}$ and $b_2 \in \mathbb{R}^{d_{model}}$ are the weights and biases of the **second linear layer**. This layer **contracts** the dimension back to the original $d_{model}$.

This FFN processes every input $\mathbf{x}$ through all its parameters $W_1, b_1, W_2, b_2$. The core purpose of the FFN is to allow the model to perform per-token, non-linear computations on the features that have been contextually enriched by the attention mechanism. **In the context of scaling LLMs, this two-layer dense structure is the primary component that the Mixture-of-Experts layer is designed to replace, as it accounts for the vast majority of the trainable parameters and computational cost in a dense Transformer block.** Note that there is another popular variant of the FFN block, the SwiGLU block, that has been used in many LLM architectures. SwiGLU does not have sparsity as MOE has. Read my [other blog]() about SwiGLU if you want to learn what that is, also read [the summary table](#summary-table-ffn-vs-swiglu-vs-simple-moe
) toward the end of this blog.

### The Mixture-of-Experts (MoE) Layer: The Key Innovation

The Mixture-of-Experts (MoE) layer is an evolution of the FFN designed to decouple a model's **capacity** (total parameters) from its **computational cost** (active parameters per token). It achieves **sparsity**—a critical concept where only a small subset of the total model is used for any single input.

Instead of a single, dense FFN (which processes all tokens using all parameters), an MoE layer comprises a massively parallel structure:

1.  **Multiple "Expert" Networks ($E_i$):** These are typically small, independent Feed-Forward Networks (FFNs). Each expert specializes in processing different kinds of information or subtasks. A typical MoE layer may have $N_e = 8$ or more experts.
2.  **A "Gating Network" (or Router $G$):** This is the brain of the MoE layer. It learns to determine which expert(s) are best suited for a given input.

The MoE's primary goal is **conditional computation**: for any given input $\mathbf{x}$, it only activates a limited number of experts ($k$, typically $k=1$ or $k=2$).

***

#### The Gating Network and Sparse Activation (Top-$k$ Routing)

The router is what converts the dense input into a sparse computation. For an input vector $\mathbf{x}$, the router calculates an affinity score for every expert:

$$\mathbf{S} = \mathbf{x}W_g \in \mathbb{R}^{N_e}$$

where $W_g$ is the router's weight matrix.

The core of modern MoE models, such as Mixtral, is the **Top-$k$ Gating** mechanism, which enables the crucial computational efficiency through **sparse activation**:

1.  **Select Indices:** The router calculates the scores $\mathbf{S}$ and finds the indices of the $k$ experts with the highest scores. The computation is **sparse** because the input is only ever passed through these $k$ selected experts, thereby skipping the computation for the remaining $N_e - k$ experts.
    $$\text{Indices} = \text{Top}k(\mathbf{S})$$

2.  **Softmax and Normalization (Gating Scores $g'$):** The scores of the $k$ selected experts are converted into probabilities using a softmax function and potentially re-normalized. This provides the final gating weights $g'$ for integration.
    $$\mathbf{g}' = \text{softmax}(\mathbf{S}[\text{Indices}])$$
    The scores for all non-selected experts are implicitly set to zero, achieving the computational saving.

3.  **Load Balancing:** During training, an auxiliary loss is added to encourage a balanced distribution of tokens across all experts, preventing "expert collapse" where only a few experts are ever utilized.
4.  **Capacity Constraint:** Each expert has a maximum number of tokens it can process in a batch to manage memory and computational bottlenecks.

***

#### The Experts and Final Output Integration

The input $\mathbf{x}$ is then passed **only** to the $k$ selected experts $E_i$, avoiding the majority of the computational cost.

The final output $\mathbf{y}$ of the MoE layer is the **weighted sum** of the outputs from the active experts.

$$\mathbf{y} = \sum_{i \in \text{Top-}k} g'_i \cdot E_i(\mathbf{x})$$

This mechanism enables the construction of models with **hundreds of billions of total parameters (high capacity)**, while ensuring that the **FLOPs per token (inference cost) remain equivalent to a much smaller, dense model**. The MoE layer is essentially a powerful form of **structural sparsity** that has been vital to scaling LLMs efficiently.

Let's illustrate the data flow:

1.  An input $\mathbf{x}$ (e.g., a token embedding) enters the MoE block.
2.  The gating network computes scores for each expert.
3.  Based on these scores, usually, the top-k scoring experts are selected.
4.  The input $\mathbf{x}$ is passed to *only* these selected experts.
5.  Each selected expert computes its output $E_i(\mathbf{x})$.
6.  The outputs of the selected experts are multiplied by their respective gating scores $g'_i(\mathbf{x})$.
7.  These weighted outputs are summed to produce the final output of the MoE layer.

## Analysis, Rationale, and Proof of Superiority

### Why MoE Works: Theoretical Rationale

The power of MoE comes from several key theoretical advantages:

  * **Increased Capacity with Reduced Computational Cost:** This is the most significant benefit. By having many experts, the model effectively has vastly more parameters. However, since only a small fraction ($k$) of these experts are activated for any given input, the computational cost (FLOPs) during inference and training remains relatively low, scaling with $k$ rather than $N_e$. This allows for models with "trillions of parameters" to be trained and run efficiently.
  * **Specialization:** Different experts can specialize in different sub-problems or patterns within the data. For example, some experts might become adept at handling grammatical structures, while others focus on factual knowledge, or even specific languages. The gating network learns to route tokens to the most appropriate expert.
  * **Mitigation of Catastrophic Forgetting (in some contexts):** While not the primary goal, by allowing different parts of the model to activate for different tasks or data modalities, MoE can potentially offer some resilience against catastrophic forgetting, though this is more nuanced in LLMs.

### Empirical Findings

The empirical evidence for MoE's effectiveness is compelling. Shazeer et al. (2017) demonstrated that sparsely-gated MoE layers could train models with over a trillion parameters, achieving **significantly faster training speeds** (e.g., 4x speedup) while maintaining or even improving model quality on various tasks like machine translation and language modeling. Subsequent works, such as Google's Switch Transformers and Microsoft's DeepSpeed-MoE, and of course the DeepSeek V3/R1 have further validated these findings, showcasing substantial improvements in training efficiency and model performance at scale. 

### Practical Intuition

Imagine you have a team of highly specialized doctors. Instead of one general practitioner trying to diagnose and treat every ailment, you have a router (a receptionist) who directs you to the exact specialist (expert) you need based on your symptoms. A brain surgeon doesn't need to be involved in a common cold diagnosis. Similarly, an LLM might have experts specialized in "Python code generation," "historical facts," or "creative writing." The router learns to send a prompt about coding to the "Python expert" and a prompt about ancient Rome to the "historical facts expert," avoiding unnecessary computation through all other experts. This selective activation makes the overall system more efficient and potentially more performant as each expert can be highly optimized for its niche.

## Practical Implementation and Architectural Integration

Let's build a simple MoE layer in PyTorch, focusing on the foundational implementation that reflects the early ideas, rather than the most advanced load-balancing or router designs found in modern MoE LLMs. This implementation will showcase a "top-1" gating mechanism for simplicity, meaning each input token is routed to only one expert.

### Code Evolution: From FFN to MoE

First, let's establish a basic FFN structure.
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List

class FeedForward(nn.Module):
    """
    A standard Feed-Forward Network (FFN) block, typically used in Transformer models.

    Args:
        d_model (int): The dimensionality of the input and output features.
        d_ff (int): The dimensionality of the inner hidden layer.
        activation (nn.Module): The activation function to use (e.g., nn.ReLU, nn.GELU).
    """
    def __init__(self, d_model: int, d_ff: int, activation: nn.Module = nn.ReLU()):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the FeedForward network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, d_model).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, sequence_length, d_model).
        """
        return self.w_2(self.activation(self.w_1(x)))
```

Now, let's create our basic MoE layer. This version will use a simple gating network and route each input to a single expert.

### Architectural Detail: The Simple MoE Block

Our `SimpleMoELayer` will consist of:

  * A `gate` (gating network) which is a linear layer mapping `d_model` to `num_experts`.
  * A list of `experts`, where each expert is an instance of our `FeedForward` network.



```python
class SimpleMoELayer(nn.Module):
    """
    A foundational Mixture-of-Experts (MoE) layer with a top-1 gating mechanism.
    Each input token is routed to exactly one expert.

    Args:
        d_model (int): The dimensionality of the input and output features for each expert.
        d_ff (int): The dimensionality of the inner hidden layer for each expert.
        num_experts (int): The total number of expert networks in the layer.
        activation (nn.Module): The activation function for each expert's FFN.
    """
    def __init__(self, d_model: int, d_ff: int, num_experts: int, activation: nn.Module = nn.ReLU()):
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts

        # Gating network: maps input to logits for each expert
        # Weights W_g from the formula: x * W_g
        self.gate = nn.Linear(d_model, num_experts)

        # List of expert networks
        self.experts = nn.ModuleList([
            FeedForward(d_model, d_ff, activation) for _ in range(num_experts)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the SimpleMoELayer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, d_model).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, sequence_length, d_model).
        """
        batch_size, seq_len, _ = x.shape

        # Flatten the input to process tokens independently
        # Shape: (batch_size * seq_len, d_model)
        flat_x = x.view(-1, self.d_model)

        # Compute expert logits using the gating network
        # Shape: (batch_size * seq_len, num_experts)
        gate_logits = self.gate(flat_x)

        # Get probabilities using softmax and identify the top-1 expert for each token
        # gate_probs: (batch_size * seq_len, num_experts)
        # expert_indices: (batch_size * seq_len) - contains index of chosen expert for each token
        gate_probs = F.softmax(gate_logits, dim=-1)
        expert_weights, expert_indices = torch.max(gate_probs, dim=-1)

        # Initialize output tensor
        # Shape: (batch_size * seq_len, d_model)
        output = torch.zeros_like(flat_x)

        # Iterate through experts to process their assigned tokens
        for i, expert in enumerate(self.experts):
            # Find tokens assigned to the current expert
            # assigned_tokens_mask: (batch_size * seq_len) - boolean mask
            assigned_tokens_mask = (expert_indices == i)

            # If no tokens are assigned to this expert, skip
            if not assigned_tokens_mask.any():
                continue

            # Select tokens for this expert
            # Shape: (num_assigned_tokens, d_model)
            tokens_for_expert = flat_x[assigned_tokens_mask]

            # Get the gating weights for these specific tokens
            # Shape: (num_assigned_tokens)
            weights_for_expert = expert_weights[assigned_tokens_mask].unsqueeze(1) # unsqueeze for broadcasting

            # Process tokens with the expert and apply gating weights
            # Shape: (num_assigned_tokens, d_model)
            expert_output = expert(tokens_for_expert)

            # Apply the gating weight to the expert's output
            weighted_expert_output = expert_output * weights_for_expert

            # Scatter the weighted expert outputs back into the main output tensor
            output[assigned_tokens_mask] = weighted_expert_output

        # Reshape the output back to the original (batch_size, sequence_length, d_model)
        return output.view(batch_size, seq_len, self.d_model)

```

Let's walk through the `forward` pass in detail:

1.  **Flatten Input:** `x.view(-1, self.d_model)` transforms the input from `(batch_size, sequence_length, d_model)` to `(num_tokens, d_model)`. This allows us to treat each token independently when routing to experts, as is typical in MoE layers.
2.  **Gating Logits:** `self.gate(flat_x)` computes raw scores (logits) for each expert for every token. For `N` tokens and `num_experts` experts, this results in an `(N, num_experts)` tensor.
3.  **Expert Probabilities & Selection:**
      * `F.softmax(gate_logits, dim=-1)` converts logits into probabilities (`gate_probs`) indicating how likely each expert is to be chosen for each token.
      * `torch.max(gate_probs, dim=-1)` identifies the top-1 expert. It returns two tensors: `expert_weights` (the probability of the chosen expert) and `expert_indices` (the index of the chosen expert).

**Thus far, you should have noticed another nuance! each and every token has the freedom to pass through a specific set of top-k experts.**

4.  **Initialize Output:** `output = torch.zeros_like(flat_x)` creates an empty tensor to accumulate the weighted outputs from the experts.
5.  **Expert Processing Loop:** The code iterates through each expert:
      * `assigned_tokens_mask = (expert_indices == i)` creates a boolean mask to find all tokens that were assigned to the current `expert_i`.
      * `tokens_for_expert = flat_x[assigned_tokens_mask]` extracts only those tokens assigned to `expert_i`.
      * `weights_for_expert = expert_weights[assigned_tokens_mask].unsqueeze(1)` retrieves the gating probability (weight) for each of these assigned tokens. `unsqueeze(1)` is crucial to allow broadcasting when multiplying with the expert's output.
      * `expert_output = expert(tokens_for_expert)`: The selected tokens are passed through `expert_i`.
      * `weighted_expert_output = expert_output * weights_for_expert`: The output of the expert is scaled by the gating probability. This implements the $g_i(\mathbf{x}) \cdot E_i(\mathbf{x})$ part of the formula.
      * `output[assigned_tokens_mask] = weighted_expert_output`: The weighted output is placed back into the correct positions in the `output` tensor.
6.  **Reshape Output:** Finally, `output.view(batch_size, seq_len, self.d_model)` reshapes the output back to its original `(batch_size, sequence_length, d_model)` form.

This simple implementation demonstrates the core mechanism: routing inputs to specific experts and weighting their outputs. Modern MoE implementations often include more sophisticated load balancing objectives to ensure experts are utilized evenly and `top-k` routing (e.g., top-2) for better performance and robustness.

### Parameter/Efficiency Analysis

Let's compare the parameter count and computational cost.

  * **Standard FFN:**

      * Parameters: $d_{model} \times d_{ff} + d_{ff} \times d_{model} = 2 \times d_{model} \times d_{ff}$ (ignoring biases). If $d_{ff} = 4 \times d_{model}$, then $8 \times d_{model}^2$.
      * FLOPs (per token): $2 \times d_{model} \times d_{ff}$.

  * **Simple MoE Layer (Top-1, $N_e$ experts):**

      * **Parameters:**
          * Gating Network: $d_{model} \times N_e$
          * Experts: $N_e \times (2 \times d_{model} \times d_{ff\_expert})$ (assuming each expert is a smaller FFN).
          * Total: $d_{model} \times N_e + N_e \times 2 \times d_{model} \times d_{ff\_expert}$.
          * Crucially, if we want to maintain *parameter parity* with a dense FFN, a common strategy is to make each expert's `d_ff_expert` significantly smaller, e.g., $d_{ff\_expert} = d_{ff} / N_e$. Or, as is often the case with MoE, we *embrace* the higher parameter count for higher capacity, as FLOPs are the bottleneck.
      * **FLOPs (per token):**
          * Gating Network: $d_{model} \times N_e$
          * One Expert: $2 \times d_{model} \times d_{ff\_expert}$ (since only one is active).
          * Total: $d_{model} \times N_e$ (for gating) + $2 \times d_{model} \times d_{ff\_expert}$ (for one expert).
          * If we set $d_{ff\_expert} = d_{ff} / N_e$ to keep FLOPs similar to a dense FFN, the compute per token for experts would be $2 \times d_{model} \times (d_{ff} / N_e)$. However, the real advantage comes from keeping $d_{ff\_expert}$ similar to a dense FFN's $d_{ff}$ but only activating a few experts.
          * The key is that the FLOPs scale with $k$ (the number of active experts) instead of $N_e$. For `top-1` routing, $k=1$, so the compute is roughly equivalent to a single FFN plus the gating cost, despite having $N_e$ times the FFN parameters.

**Example for FLOPs comparison:**
Let $d_{model}=1024$, $d_{ff}=4096$.

  * Dense FFN FLOPs: $2 \times 1024 \times 4096 \approx 8.4 \times 10^6$
    Let $num\_experts=8$, $d_{ff\_expert}=4096$ (i.e., each expert is as large as the original FFN).
  * MoE FLOPs:
      * Gating: $1024 \times 8 = 8192$
      * One expert: $2 \times 1024 \times 4096 \approx 8.4 \times 10^6$
      * Total $\approx 8.4 \times 10^6$
      * Here, we have $8 \times (8.4 \times 10^6)$ parameters in total (ignoring gating network params), but the FLOPs per token are effectively the same as a single dense FFN\! This is the core efficiency gain.

### Architectural Context: A Drop-in Replacement

The `SimpleMoELayer` is designed to be a "drop-in replacement" for a standard FFN within a Transformer block. Where a typical Transformer block might have:

```
x = x + self.attention(self.norm1(x))
x = x + self.feed_forward(self.norm2(x))
```

You could replace `self.feed_forward` with `self.moe_layer`:

```
x = x + self.attention(self.norm1(x))
x = x + self.moe_layer(self.norm2(x)) # MoE instead of FFN
```

This structural compatibility makes MoE layers easy to integrate into existing Transformer architectures, leading to models like the "Sparsely-Gated Mixture-of-Experts Transformer."

## Conclusion and Resources

Mixture-of-Experts represents a crucial architectural innovation, enabling the construction of truly massive language models that remain computationally efficient. By selectively activating specialized "experts," MoE layers allow models to scale in capacity without incurring prohibitive increases in training and inference costs. This foundation has paved the way for the "trillion-parameter" era of LLMs.

### Real-World Impact

| Model / Framework | Year | Developer | Total Params / Activated Params | Key Innovation / Use |
| :--- | :--- | :--- | :--- | :--- |
| **Grok-1** | 2024 | xAI | 314 Billion (25% active) | A high-profile MoE model demonstrating its use in competitive, general-purpose LLMs. |
| **Mixtral 8x7B** | 2023 | Mistral AI | 47 Billion (13 Billion active) | Demonstrated that MoE is not just for hyper-scale models, offering state-of-the-art performance in an accessible open-source package. |
| **DeepSeek-V3** | 2025 | DeepSeek-AI | 671 Billion (37 Billion active) | A powerful open-weight MoE model known for its **extreme training efficiency** and competitive performance, often matching or exceeding much larger models. Pioneers **auxiliary-loss-free** load balancing. |
| **Qwen2.5-Max** | 2025 | Alibaba Cloud | Large MoE | An extremely large-scale MoE model pre-trained on over 20 trillion tokens, demonstrating the continued push toward ultra-scale MoE for commercial applications. |


### Summary Table: FFN vs. SwiGLU vs. Simple MoE



| Feature | Standard FFN (ReLU/GELU) | SwiGLU FFN (Dense LLMs like LLaMA) | Mixture-of-Experts (MoE LLMs like DeepSeek) |
| :--- | :--- | :--- | :--- |
| **Architectural Role** | The **sole** FFN sub-layer in the block (Original Transformer). | The **sole** FFN sub-layer in the block (Modern Dense LLMs). | **Replaces** the single FFN sub-layer in the block (Sparse LLMs). |
| **Structure** | A single set of **two** weight matrices/layers per block. | A single set of **three** weight matrices/layers per block (Gated). | A **Router** + $N$ independent FFNs (**Experts**). |
| **Activation Function** | **ReLU** or **GELU** (non-gated). | **SwiGLU** (a Gated Linear Unit variant). | **Each Expert** typically uses **SwiGLU** or GELU. |
| **Parameter Count** | **Lowest** total parameters. | **Low** total parameters (slightly higher than Standard FFN for equivalent size). | **Highest** total parameters (can be 10x larger than a dense model). |
| **Compute (FLOPs)** | **Fixed, High** per token (all parameters activated). | **Fixed, High** per token (all parameters activated). | **Fixed, Low** per token (only $K$ experts activated, e.g., $K=2$). |
| **Information Flow** | **Dense:** Every token flows through the same parameters. | **Dense/Gated:** Every token flows through the same parameters with input-dependent filtering (gating). | **Sparse/Conditional:** Tokens are **routed** to different, specialized experts. |
| **Complexity** | Simple, well-understood. | Medium complexity due to the three-matrix, gated structure. | High complexity due to the routing mechanism and load balancing. |
| **Typical Use** | Original **Transformer** (e.g., Vaswani et al., 2017), **BERT**, and older deep learning models. | State-of-the-art **Dense** models like **LLaMA 1 & 2** and **Gemma**. | Highly scalable **Sparse** models like **Mixtral, Grok-1**, and **DeepSeek-MoE**. |


### References

  * Jacobs, R. A., Jordan, M. I., Nowlan, S. J., & Hinton, G. E. (1991). Adaptive mixtures of local experts. *Neural Computation, 3*(1), 79-87.
  * Shazeer, N., Mirhoseini, A., Maziarz, K., Davis, A., Le, Q. V., Hinton, G., & Dean, J. (2017). Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer. *arXiv preprint arXiv:1701.06538*.
  * Fedus, W., Zoph, B., & Shazeer, N. (2022). Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity. *Journal of Machine Learning Research, 23*, 1-36.
  * Jiang, A. Q., Sablayrolles, A., Roux, A., Mensch, A., Savary, B., Bamford, C., ... & Sayed, W. E. (2024). Mixtral of experts. arXiv preprint arXiv:2401.04088.