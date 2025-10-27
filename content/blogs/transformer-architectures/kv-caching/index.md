---
title: "KV-Caching in LLMs: The Optimization That Makes Inference Practical"
date: 2025-10-05
tags: ["Deep Learning", "Transformers", "LLM"]
author: "Saeed Mehrang"
description: "A deep dive into Key-Value caching in transformer models - the critical optimization that transforms O(n²) autoregressive generation into O(n), enabling real-time LLM inference at scale."
summary: "Learn how KV-caching makes ChatGPT respond in seconds instead of minutes. This comprehensive guide explains the quadratic complexity problem in transformers, how caching Keys and Values solves it with 10-100x speedups, and the memory trade-offs - complete with full PyTorch implementations, benchmarks, and interactive visualizations."
cover:
    image: "cover.svg"
    alt: "KV Caching Simply"
    relative: true
showToc: true
disableAnchoredHeadings: false
---

## Introduction

If you've ever wondered how ChatGPT, Gemini, or Claude generate responses so quickly, or how language models can maintain long conversations without grinding to a halt, KV-caching is a big part of the answer. This optimization technique is one of the most critical innovations that makes modern LLM inference practical.

In this post, we'll dive deep into what KV-caching is, why it's necessary, and how it's implemented in transformer-based language models.

For a simple but complete python implementation in PyTorch, please see [this python script](https://github.com/saeedmehrang/llm-learning/blob/main/kv_caching.py) in my GitHub.


See the process in the diagram below,


{{< framed_image src="cover.svg" alt="KV Caching Simplified" width="900px" height="900px" >}}
{{< /framed_image >}}



## The Problem: Quadratic Complexity in Autoregressive Generation

### How Transformers Generate Text

Transformers generate text **autoregressively** - one token at a time. At each step, the model needs to:

1. Process the new token
2. Attend to all previous tokens in the sequence
3. Predict the next token
4. Repeat

Here's the catch: without optimization, each generation step requires reprocessing the **entire sequence** from scratch.

### The Computational Bottleneck

Let's visualize what happens without caching:

```
Step 1: Process [token_1]
Step 2: Process [token_1, token_2]         (recomputes token_1)
Step 3: Process [token_1, token_2, token_3] (recomputes token_1, token_2)
Step 4: Process [token_1, ..., token_4]    (recomputes everything)
...
```

For a sequence of length `n`, the total number of token operations is:

```
1 + 2 + 3 + ... + n = n(n+1)/2 ≈ O(n²)
```

**This quadratic complexity is disastrous for long sequences.** Generating 1,000 tokens would require processing roughly 500,000 token operations - with massive redundancy since you're recomputing the same representations over and over.

## The Solution: KV-Caching

### Core Insight

The key observation is that in the attention mechanism, the **Keys (K)** and **Values (V)** for previous tokens never change once computed. Only the **Query (Q)** for the new token is fresh.

Recall the attention formula:

```
Attention(Q, K, V) = softmax(QK^T / √d_k) × V
```

Where:
- **Q (Query)**: "What am I looking for?" - only needed for the current token
- **K (Keys)**: "What do I offer?" - can be cached for all past tokens
- **V (Values)**: "What information do I provide?" - can be cached for all past tokens

### How KV-Caching Works

Instead of recomputing K and V for all tokens at each step, we:

1. **Compute** K and V for the new token only
2. **Retrieve** cached K and V from all previous tokens
3. **Concatenate** new K, V with cached K, V
4. **Compute** attention using the new Q against all K
5. **Update** the cache with the new K, V for next iteration

This reduces complexity from **O(n²) to O(n)** - a massive improvement!

### Why We Don't Cache Q Tensors in Autoregressive Models?!

Short answer, because we no longer need them! The fundamental reason we cache K and V but not Q is that they play asymmetric roles in autoregressive generation. When generating token `t`, the attention computation is `Attention(Q_t, K_{1:t}, V_{1:t}) = softmax(Q_t @ K_{1:t}^T / sqrt(d)) @ V_{1:t}`, where Q_t comes only from the current token at position t, while K_{1:t} and V_{1:t} come from all previous tokens (positions 1 to t). At each generation step, we compute a new query (q_1, then q_2, then q_3, etc.) that attends to an accumulating set of keys and values—for example, q_3 attends to [k_1, k_2, k_3] and [v_1, v_2, v_3]. The critical observation is that we never reuse q_1 or q_2 after their respective steps, but we always reuse k_1, k_2, v_1, v_2 for all subsequent tokens. This happens because of the causal constraint in autoregressive models: each new token's query looks back at all previous keys and values, but once a token is generated, its query is never needed again.

## Interactive Visualization of Memory Accumulation

To better understand how memory accumulation in KV-caching works in practice, I've created an interactive visualization that shows the generation process step-by-step. Click "Start Prefill Phase" to see how the model first processes the entire prompt and populates the initial cache. Then watch as the model generates tokens one at a time, with the cache growing incrementally at each step.

The visualization clearly demonstrates the two-phase nature of KV-cached generation: the **prefill phase** where we process all prompt tokens at once, and the **generation phase** where we process one token at a time while reusing cached computations.


{{< include-html "kv_cache_viz.html" >}}


## Implementation Details

Understanding how KV-caching is implemented reveals both its elegance and its practical considerations. Let's walk through the complete implementation from data structures to the two-phase generation process.

### Cache Structure

At its core, the KV-cache is a straightforward data structure that stores Key and Value tensors for each layer in the transformer. The cache is organized as a list of dictionaries, with one entry per layer:

```python
# Model configuration
config = {
    'num_layers': L,           # Number of transformer layers
    'num_heads': H,            # Number of attention heads
    'head_dim': D_h,           # Dimension per head
    'hidden_dim': D_model      # Total model dimension
}

# Initialize empty cache structure (one entry per layer)
kv_cache = [
    {
        'keys': None,     # Shape: [batch, num_heads, seq_len, head_dim]
        'values': None,   # Shape: [batch, num_heads, seq_len, head_dim]
    }
    for _ in range(config['num_layers'])
]
```

Each cache entry stores two tensors with shape `[batch_size, num_heads, sequence_length, head_dim]`. The `sequence_length` dimension grows incrementally during generation as we add each new token's Keys and Values.

### Two-Phase Generation

Modern LLM inference separates generation into two distinct phases, each optimized for different computational characteristics:

#### Phase 1: Prefill (Prompt Processing)

The prefill phase processes the entire input prompt at once. While this phase still has O(n²) complexity due to computing full attention over all prompt tokens, it only happens once and benefits from parallel processing:

```python
# Starting tokens from user prompt
# Shape: [batch_size, initial_seq_len]
input_ids = tokenize(prompt)
position = 0

# Compute embeddings for all prompt tokens at once
hidden_states = embed(input_ids)  # [batch, initial_seq_len, D_model]

for layer_idx in range(config['num_layers']):
    attention = model.layers[layer_idx].attention
    
    # Compute Q, K, V for ALL prompt tokens simultaneously
    # Each shape: [batch, num_heads, initial_seq_len, head_dim]
    Q = attention.project_queries(hidden_states)
    K = attention.project_keys(hidden_states)
    V = attention.project_values(hidden_states)
    
    # Store K and V in cache (first time initialization)
    kv_cache[layer_idx]['keys'] = K
    kv_cache[layer_idx]['values'] = V
    
    # Compute attention over all prompt tokens
    # attention_scores shape: [batch, num_heads, initial_seq_len, initial_seq_len]
    attention_scores = (Q @ K.transpose(-2, -1)) / sqrt(config['head_dim'])
    attention_scores = apply_causal_mask(attention_scores)  # Prevent looking ahead
    attention_weights = softmax(attention_scores, dim=-1)
    
    # Apply attention to values
    # attention_output shape: [batch, num_heads, initial_seq_len, head_dim]
    attention_output = attention_weights @ V
    
    # Continue through rest of layer (feedforward network, etc.)
    hidden_states = layer.forward(attention_output)

# Update position to end of prompt
position = initial_seq_len

# Generate logits and sample first generated token
logits = model.lm_head(hidden_states[:, -1, :])  # Only use last position
next_token_id = sample(logits)  # Shape: [batch, 1]
```

After prefill completes, the cache contains Keys and Values for all prompt tokens, and we're ready to begin autoregressive generation.

#### Phase 2: Autoregressive Generation

This is where KV-caching shines. Instead of reprocessing the entire sequence, we only compute representations for the single new token and concatenate with the cache:

```python
generated_tokens = [next_token_id]

for step in range(max_new_tokens - 1):
    # Embed only the NEW token (not the entire sequence!)
    # Shape: [batch, 1, D_model]
    hidden_states = embed(next_token_id)
    
    for layer_idx in range(config['num_layers']):
        attention = model.layers[layer_idx].attention
        
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # KEY OPTIMIZATION: Only compute Q, K, V for NEW token
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        
        # Compute projections only for the new token
        # Each shape: [batch, num_heads, 1, head_dim]
        Q_new = attention.project_queries(hidden_states)
        K_new = attention.project_keys(hidden_states)
        V_new = attention.project_values(hidden_states)
        
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # Retrieve cached K, V and concatenate with new values
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        
        # Retrieve cached K and V from all previous tokens
        # Shape: [batch, num_heads, position, head_dim]
        K_cached = kv_cache[layer_idx]['keys']
        V_cached = kv_cache[layer_idx]['values']
        
        # Concatenate along sequence dimension (dim=2)
        # New shapes: [batch, num_heads, position+1, head_dim]
        K_all = concatenate([K_cached, K_new], dim=2)
        V_all = concatenate([V_cached, V_new], dim=2)
        
        # Update cache with new concatenated tensors for next iteration
        kv_cache[layer_idx]['keys'] = K_all
        kv_cache[layer_idx]['values'] = V_all
        
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # Compute attention (Q is only for the new token)
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        
        # Q_new:  [batch, num_heads, 1, head_dim]
        # K_all:  [batch, num_heads, position+1, head_dim]
        # Result: [batch, num_heads, 1, position+1]
        attention_scores = (Q_new @ K_all.transpose(-2, -1)) / sqrt(config['head_dim'])
        
        # No causal mask needed - we only attend to past + current
        attention_weights = softmax(attention_scores, dim=-1)
        
        # attention_weights: [batch, num_heads, 1, position+1]
        # V_all:            [batch, num_heads, position+1, head_dim]
        # Result:           [batch, num_heads, 1, head_dim]
        attention_output = attention_weights @ V_all
        
        # Continue through rest of layer (feedforward network, residuals, etc.)
        hidden_states = layer.forward(attention_output)
    
    # Update position counter
    position += 1
    
    # Generate next token from last hidden state
    logits = model.lm_head(hidden_states[:, -1, :])
    next_token_id = sample(logits)
    generated_tokens.append(next_token_id)
    
    # Check for end-of-sequence token
    if next_token_id == EOS_TOKEN:
        break

return generated_tokens
```

Notice the key difference: in the generation phase, we compute Q, K, V only for **one token** per step, not the entire sequence. This single optimization transforms the complexity from O(n²) to O(n).

### Memory Considerations

While KV-caching provides massive speed improvements, it introduces significant memory overhead that must be carefully managed in production systems.

#### Memory Calculation

The memory required for KV-cache grows linearly with sequence length:

**Memory per token per layer** = `2 × num_heads × head_dim × sizeof(dtype)`

Or equivalently:

**Memory per token per layer** = `2 × d_model × sizeof(dtype)`

For a concrete example with GPT-3 scale parameters:
- `d_model = 12,288`
- `num_layers = 96`
- `dtype = float16` (2 bytes per element)

**Per token:** 2 × 12,288 × 2 bytes = **49,152 bytes ≈ 48 KB per layer**

**Full model:** 48 KB × 96 layers = **4.6 MB per token**

**Long sequence:** 4.6 MB × 10,000 tokens = **46 GB of cache!**

### Practical Implications and Memory Management

The substantial memory footprint of KV-caching creates important trade-offs that shape how LLMs are deployed in production. Understanding these implications is crucial for effective system design.

#### The Memory-Performance Trade-off

While KV-caching delivers 10-100x speedups (with longer sequences seeing greater gains), this performance comes at a cost. The technique exchanges compute for memory—instead of recomputing attention repeatedly, we store intermediate results that grow linearly with sequence length.

This fundamental trade-off manifests in several ways:

**Batch Size Limitations**: With limited GPU memory, you must choose between serving more users simultaneously (larger batch sizes) or supporting longer conversations (longer sequences). A single 10,000-token conversation can consume as much memory as 50 shorter interactions, forcing difficult capacity planning decisions.

**Context Window Costs**: Extending context windows has multiplicative effects. Moving from 4K to 32K tokens increases cache memory by 8x, which is why long-context models like Claude or GPT-4 with 100K+ token windows require specialized memory management and more expensive hardware.

**Hardware Requirements**: Production LLM serving demands careful capacity planning. A single user with an extended conversation can consume gigabytes of GPU memory, limiting how many concurrent users a server can handle. This directly impacts the economics of LLM deployment.

#### Advanced Optimization Techniques

The memory pressure from KV-caching has driven significant innovation in transformer architectures and serving systems:

**PagedAttention (vLLM)**: The most impactful recent advancement treats KV-cache like virtual memory. By splitting the cache into fixed-size "pages," it allows non-contiguous memory allocation and enables cache sharing across requests. This can improve GPU memory utilization by 2-4x, dramatically increasing serving throughput.

**Multi-Query Attention (MQA)**: Instead of having separate Keys and Values for each attention head, MQA shares them across all heads. This reduces cache size by a factor of `num_heads` (often 32-96), cutting memory requirements by 97-99% with minimal quality impact. Used in models like PaLM and StarCoder.

**Grouped-Query Attention (GQA)**: A middle ground that groups multiple heads to share K, V. This balances MQA's memory efficiency with standard attention's quality. LLaMA 2 and similar models use GQA to achieve 4-8x cache reduction while maintaining performance.

**Cache Quantization**: Storing cached values in int8 or even lower precision can halve or quarter memory usage with careful implementation, though this requires validation to ensure quality isn't degraded.



## References and Further Reading

### Foundational Papers

1. **Shazeer, N. (2019).** ["Fast Transformer Decoding: One Write-Head is All You Need"](https://arxiv.org/abs/1911.02150). *arXiv preprint*.  
   Introduces Multi-Query Attention (MQA), which reduces KV-cache memory requirements by sharing Keys and Values across attention heads—a key innovation for making caching more memory-efficient in production systems.

2. **Kwon, W., et al. (2023).** ["Efficient Memory Management for Large Language Model Serving with PagedAttention"](https://arxiv.org/abs/2309.06180). *SOSP 2023*.  
   The vLLM paper that revolutionizes KV-cache management by treating it like virtual memory with paging. This work enables much higher throughput and better memory utilization in production LLM serving systems.

### Implementation and Practice

3. **Pope, R., et al. (2022).** ["Efficiently Scaling Transformer Inference"](https://arxiv.org/abs/2211.05102). *arXiv preprint*.  
   Comprehensive analysis of inference optimization techniques for transformers, including detailed discussions of KV-caching, quantization, and hardware considerations for production deployment.

4. **Ainslie, J., et al. (2023).** ["GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints"](https://arxiv.org/abs/2305.13245). *EMNLP 2023*.  
   Introduces Grouped-Query Attention (GQA), a middle ground between standard attention and MQA that's used in models like LLaMA 2. Essential reading for understanding modern approaches to balancing cache efficiency with model quality.

### Additional Resources

- **[Hugging Face Transformers Documentation](https://huggingface.co/docs/transformers/main_classes/text_generation#transformers.GenerationConfig.use_cache)** - Practical guide to using KV-caching in production with the Transformers library
- **[vLLM GitHub Repository](https://github.com/vllm-project/vllm)** - Production-grade implementation of PagedAttention and advanced KV-cache management
- **[Flash Attention Repository](https://github.com/Dao-AILab/flash-attention)** - Combines KV-caching with memory-efficient attention computation for even better performance
