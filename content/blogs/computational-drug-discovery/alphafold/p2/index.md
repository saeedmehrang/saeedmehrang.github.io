---
title: "Computational Drug Discovery Part 3 (Subpart 2/3): AlphaFold's Evoformer Block Disassembled, A Matrix-Level Deep Dive into AlphaFold2's Core"
date: 2025-10-24
draft: false
author: Saeed Mehrang
description: "A detailed mathematical breakdown of AlphaFold2's Evoformer block, explaining each operation with concrete matrix algebra and dimensions - Subpart 2/3"
summary: "A detailed mathematical breakdown of AlphaFold2's Evoformer block, explaining each operation with concrete matrix algebra and dimensions - Subpart 2/3"
tags: ["AlphaFold2", "Protein Structure", "Deep Learning", "Bioinformatics", "Machine Learning"]
categories: ["Computational Biology", "AI"]
math: true
showToc: true
disableAnchoredHeadings: false
cover:
  image: "cover.png"
  image_alt: "Alphafold Evoformer Block"
---

| Characteristic | Detail |
| :--- | :--- |
| **Estimated Reading Time** | 20-30 minutes |
| **Technical Level** | Advanced (requires understanding of deep learning, basic chemistry) |
| **Prerequisites** | [Subpart 1](../p1) in this miniseries required |


This is the Subpart 2 of the miniseries on Alphafold2.


The Evoformer is the beating heart of AlphaFold2, the revolutionary protein structure prediction system. While [Subpart 1](../p1) explanations stayed at a high level, this post takes you inside the machinery—showing you exactly what happens at each step with concrete matrix operations.

If you've ever wondered "what actually happens when the MSA and Pair representations communicate?"—this is for you. See the schematic diagram below adopted from the source article [^1], 



{{< framed_image src="cover.png" alt="Evoformer Architecture" width="800px" height="850px" >}}
a, Evoformer block. Arrows show the information flow. The shape of the arrays is shown in parentheses. b, The pair representation interpreted as directed edges in a graph. c, Triangle multiplicative update and triangle self-attention. The circles represent residues. Entries in the pair representation are illustrated as directed edges and in each diagram, the edge being updated is ij. d, Structure module including Invariant point attention (IPA) module. The single representation is a copy of the first row of the MSA representation. e, Residue gas: a representation of each residue as one free-floating rigid body for the backbone (blue triangles) and χ angles for the side chains (green circles). The corresponding atomic structure is shown below. f, Frame aligned point error (FAPE). Green, predicted structure; grey, true structure; (Rk, tk), frames; xi, atom positions.
{{< /framed_image >}}



## The Two Core Representations

Before we dissect the Evoformer's operations, we need to understand the two tensors it operates on—think of these as the network's "working memory" for evolutionary and structural information.

### MSA Representation: The Evolutionary Memory

**Shape**: `[N_seq, L, c_msa]`

The MSA (Multiple Sequence Alignment) tensor stores information about homologous sequences—proteins from different organisms that share a common ancestor with your target protein. Each of the `N_seq` sequences (typically ~1000) represents how evolution has explored different amino acid combinations while maintaining protein function.

- **Rows** (`N_seq`): Different species or homologs (typically ~1000 sequences)
- **Columns** (`L`): Positions along the protein sequence (say 200 amino acids)
- **Depth** (`c_msa`): Learned feature vectors (typically 256 dimensions) encoding amino acid identity, conservation patterns, and evolutionary context

When position 50 is highly conserved across all sequences, that signals functional importance. When positions 20 and 100 co-vary together across evolution (both change simultaneously), that hints at a physical interaction.

### Pair Representation: The Structural Hypothesis

**Shape**: `[L, L, c_pair]`

The Pair tensor represents the network's evolving beliefs about relationships between every pair of residues. For a protein of length `L`, this is an `L × L` grid where entry `(i, j)` describes the relationship between residues i and j.

- **Rows & Columns**: Every residue pair `(i, j)`
- **Depth** (`c_pair`): Feature vectors (typically 128 dimensions) encoding distance distributions, orientations, contact probabilities, and geometric constraints

This isn't just a simple contact map—it's a rich, learned representation that captures uncertainty, geometry, and spatial relationships. Initially populated with basic features (sequence separation, positional encodings), it gets progressively refined through the Evoformer blocks into a detailed structural hypothesis.

These two representations talk to each other throughout the 48 Evoformer blocks, with evolutionary signals informing structural predictions and structural hypotheses guiding evolutionary interpretation.


---

## The Evolutionary Track: MSA Representation Refinement

The MSA track captures evolutionary information—patterns of conservation and co-variation across millions of years of evolution.

### MSA Row-wise Self-Attention

**Purpose**: Enable positions within a single sequence to communicate, capturing long-range dependencies.

**Input**: MSA tensor `M` with shape `[N_seq, L, c_msa]`

**Operation**:

For each sequence `s` (each row), we extract that sequence:
```
M_s = M[s, :, :]  # Shape: [L, c_msa]
```

We then create three projection matrices:
```
Q = M_s @ W_q  # Query: [L, d_k]
K = M_s @ W_k  # Key: [L, d_k]
V = M_s @ W_v  # Value: [L, d_v]
```

Where:
- `W_q, W_k, W_v` are learned weight matrices
- `d_k` = dimension of queries/keys (typically 64)
- `d_v` = dimension of values (typically 256)

**Attention Computation**:
```
Attention_weights = softmax(Q @ K^T / sqrt(d_k))  # Shape: [L, L]
```

This creates an `L × L` matrix where entry `(i,j)` represents: "How much should position i attend to position j?"

**Output** for sequence s:
```
M_s_updated = Attention_weights @ V  # Shape: [L, d_v]
```

This operation runs independently for all N_seq sequences.

**Biological Intuition**: If residue 10 needs to interact with residue 200 to form a disulfide bond or salt bridge, this mechanism lets them "communicate" despite being distant in the primary sequence.

---

### MSA Column-wise Gated Self-Attention

**Purpose**: Allow different sequences (representing different species or homologs) to compare their amino acids at the same position, revealing evolutionary patterns.

**Input**: MSA tensor `M` with shape `[N_seq, L, c_msa]`

**Operation**:

For each position `i` (each column), extract all sequences at that position:
```
M_i = M[:, i, :]  # Shape: [N_seq, c_msa]
```

**Gated Attention Mechanism**:

First, compute gate values to modulate information flow:
```
G = sigmoid(M_i @ W_g)  # Shape: [N_seq, c_msa]
```

Then perform standard attention:
```
Q = M_i @ W_q  # Shape: [N_seq, d_k]
K = M_i @ W_k  # Shape: [N_seq, d_k]
V = M_i @ W_v  # Shape: [N_seq, d_v]

Attention_weights = softmax(Q @ K^T / sqrt(d_k))  # Shape: [N_seq, N_seq]
Attended = Attention_weights @ V  # Shape: [N_seq, d_v]
```

**Apply Gate**:
```
M_i_updated = G ⊙ Attended  # Element-wise (Hadamard) product
```

The `⊙` operator denotes element-wise multiplication.

**Why Gating?**: The gate acts as a learned filter, deciding "how much of this attended information should influence the output?" It's analogous to a volume control that the network learns to adjust.

**Biological Significance**: If position 50 is conserved as Glycine across 800 species but varies in 200 species, this attention mechanism captures that conservation pattern. The gate then modulates how strongly this pattern should influence downstream predictions—perhaps downweighting positions with ambiguous conservation signals.

---

### Communication: Pair → MSA (Gated Extractor)

**Purpose**: Inject structural hypotheses from the Pair representation into the evolutionary MSA representation.

**Inputs**: 
- MSA: `[N_seq, L, c_msa]`
- Pair: `[L, L, c_pair]`

**Operation**:

The Pair representation encodes the network's current beliefs about which residues are spatially proximate. We want to use this structural information to contextualize the MSA.

**For each sequence s and position i**:

Extract relevant pair information involving position i:
```
# Get all pair features where i is involved
Pair_i = Pair[i, :, :]  # Shape: [L, c_pair]
```

**Compute gating signal**:
```
# Project pair features to MSA feature space
Gate_signal = Pair_i @ W_pair_to_msa  # Shape: [L, c_msa]

# Aggregate across all positions (via averaging or attention)
Gate_i = mean(Gate_signal, axis=0)  # Shape: [c_msa]

# Apply sigmoid activation
Gate_i = sigmoid(Gate_i)  # Shape: [c_msa]
```

**Update MSA**:
```
M[s, i, :] = M[s, i, :] ⊙ Gate_i
```

**Architectural Intuition**: If the Pair representation strongly indicates that residues i and j are close in 3D space, this mechanism can emphasize MSA sequences that support this hypothesis. For example, it might upweight sequences where both positions contain amino acids with compatible chemical properties for forming a contact (e.g., both hydrophobic, or oppositely charged).

This creates a feedback loop: structural predictions guide evolutionary interpretation.

---

## The Structural Track: Pair Representation Refinement

The Pair track maintains hypotheses about residue-residue relationships in 3D space.

### Communication: MSA → Pair (Outer Product Mean)

**Purpose**: Extract co-evolutionary signals from the MSA to update structural hypotheses in the Pair representation.

**Input**: MSA `M` with shape `[N_seq, L, c_msa]`

**Operation**:

For every pair of positions `(i, j)`:

**Step 1: Extract columns**:
```
M_i = M[:, i, :]  # All sequences at position i: [N_seq, c_msa]
M_j = M[:, j, :]  # All sequences at position j: [N_seq, c_msa]
```

**Step 2: Compute outer product for each sequence**:

For sequence `s`:
```
Outer_s = M_i[s, :].reshape(c_msa, 1) @ M_j[s, :].reshape(1, c_msa)
# Shape: [c_msa, c_msa]
```

This creates a matrix where entry `(a, b)` captures the co-occurrence of feature `a` at position `i` with feature `b` at position `j` in sequence `s`.

**Step 3: Average across all sequences**:
```
Outer_mean = (1/N_seq) * sum_over_s(Outer_s)  # Shape: [c_msa, c_msa]
```

**Step 4: Flatten and project to pair feature space**:
```
Outer_flat = flatten(Outer_mean)  # Shape: [c_msa * c_msa]
Update_ij = Outer_flat @ W_outer  # Shape: [c_pair]
```

**Step 5: Update pair representation**:
```
Pair[i, j, :] = Pair[i, j, :] + Update_ij
```

**Biological Foundation**: This operation implements the core principle of **correlated mutations analysis**. If two positions co-vary across evolution (when one changes, the other changes in a correlated manner), they likely interact physically in 3D space.

**Example**: Consider position i often being Lysine (positively charged) when position j is Glutamate (negatively charged), and when they swap charges together across evolutionary time. This correlated variation strongly suggests they form a salt bridge contact in the folded structure.

---

### Triangle Multiplicative Updates: The Geometric Constraint

**Purpose**: Enforce geometric consistency in predicted residue-residue relationships using the triangle inequality from 3D Euclidean space.

**Input**: Pair representation `P` with shape `[L, L, c_pair]`

**Mathematical Foundation - The Triangle Inequality**: 

In 3D space, for any three points i, j, k, the distance d between any two points has a lower and an upper bound as follows:
```
|d(i,k) - d(i,j)| ≤ d(j,k) ≤ d(i,k) + d(i,j)
```

This constraint ensures geometric plausibility.

**AlphaFold2's Implementation**:

For every triple of residues (i, j, k):

**Step 1: Extract relevant edges**:
```
P_ij = P[i, j, :]  # Edge from i to j: [c_pair]
P_jk = P[j, k, :]  # Edge from j to k: [c_pair]
P_ik = P[i, k, :]  # Edge from i to k: [c_pair]
```

**Step 2: Multiplicative Update ("Outgoing" variant)**:

Project to intermediate hidden dimension:
```
A_ij = P_ij @ W_a  # Shape: [c_hidden]
B_jk = P_jk @ W_b  # Shape: [c_hidden]
```

**Element-wise multiplication** (the key "multiplicative" operation):
```
G_ijk = A_ij ⊙ B_jk  # Shape: [c_hidden]
```

Apply learned gate:
```
Gate_ijk = sigmoid(P_ik @ W_g)  # Shape: [c_hidden]
Update_ijk = Gate_ijk ⊙ G_ijk  # Shape: [c_hidden]
```

**Step 3: Aggregate over all intermediate nodes j**:
```
Update_ik = sum_over_j(Update_ijk)  # Sum over all possible intermediate nodes
Update_ik = Update_ik @ W_o  # Project back to pair space: [c_pair]
```

**Step 4: Update edge**:
```
P[i, k, :] = P[i, k, :] + Update_ik
```

**Efficient Matrix Formulation**:

Let's express this for all triples simultaneously:
```
A = P @ W_a  # Shape: [L, L, c_hidden]
B = P @ W_b  # Shape: [L, L, c_hidden]
G = sigmoid(P @ W_g)  # Shape: [L, L, c_hidden]
```

For "outgoing" edges (information propagates from i through j to k):
```
# For each feature channel c:
Update[:, :, c] = G[:, :, c] ⊙ (A[:, j, c] @ B[j, :, c])
```

Where `@` denotes matrix multiplication over the intermediate j dimension.

**Why Multiplication Enforces Geometry**:

- If edges (i,j) and (j,k) both have strong features indicating small distances, their element-wise product produces large values
- This large product drives the update to edge (i,k) toward also indicating a small distance
- The learned gate modulates this update based on current confidence about (i,k)
- **Transitivity**: If i and j are close, AND j and k are close, THEN i and k must be close. The multiplication operation naturally enforces this logical constraint.

This prevents the network from predicting geometrically impossible configurations (e.g., i near j, j near k, but i far from k).

---

### Triangle Attention (Complementary): Geometry via Attention Mechanism

**Purpose**: Enforce triangle consistency using an attention mechanism instead of multiplication—providing an alternative geometric constraint.

**Input**: Pair representation `P` with shape `[L, L, c_pair]`

**Operation**:

For edge (i, k), we update it by attending to all edges that share a common node.

**Step 1: Create queries, keys, values**:

For "around starting node" variant (edges sharing node i):
```
Q_ik = P[i, k, :] @ W_q  # Query for target edge (i,k): [c_pair] → [d_k]
```

For all edges emanating from i:
```
K_ij = P[i, :, :] @ W_k  # Keys for all j: [L, c_pair] → [L, d_k]
V_ij = P[i, :, :] @ W_v  # Values for all j: [L, c_pair] → [L, d_v]
```

**Step 2: Compute attention with geometric bias**:
```
# For edge (i,k), compute attention over all edges (i,j)
Attention_scores = Q_ik @ K_ij^T  # Shape: [d_k] @ [L, d_k]^T = [L]

# Apply bias term that encodes information about the third edge (j,k)
Bias_ijk = f(P[j, k, :])  # Some learned function of the closing edge
Attention_scores = Attention_scores + Bias_ijk

# Normalize
Attention_weights = softmax(Attention_scores / sqrt(d_k))  # Shape: [L]
```

**Step 3: Apply attention and update**:
```
Update_ik = Attention_weights @ V_ij  # [L] @ [L, d_v] = [d_v]
P[i, k, :] = P[i, k, :] + (Update_ik @ W_o)
```

**Geometric Intuition**: Edge (i,k) attends to all edges (i,j), with attention weights influenced by the relevance of the third edge (j,k). This creates an attention mechanism that's constrained by triangle geometry—edges that form more consistent triangles receive higher attention weights.

This mechanism is complementary to Triangle Multiplicative Updates, providing the network with multiple ways to enforce spatial consistency.

---

## Final Transition: Non-linear Processing

**Purpose**: Apply additional non-linear transformations to increase representational capacity after all the sophisticated geometric and evolutionary updates.

**Input**: Updated MSA and Pair representations

**Operation**:

Standard two-layer MLP (Multi-Layer Perceptron) applied point-wise to each element.

**For MSA**:
```
For each position M[s, i, :]:
  H = ReLU(M[s, i, :] @ W1 + b1)  # Shape: [c_hidden]
  M_new[s, i, :] = H @ W2 + b2      # Shape: [c_msa]
```

**For Pair**:
```
For each pair P[i, j, :]:
  H = ReLU(P[i, j, :] @ W1 + b1)  # Shape: [c_hidden]
  P_new[i, j, :] = H @ W2 + b2      # Shape: [c_pair]
```

**Purpose**: After all the specialized attention and geometric operations, this simple feedforward network provides additional capacity to:
- Mix features in non-linear ways
- Increase the expressiveness of the representation
- Allow the network to learn complex, non-geometric relationships

It's the "standard deep learning" component after all the domain-specific innovations.

---

## The Power of Iteration: Why 48 Blocks?

Each Evoformer block executes a complete cycle:

1. **Refine evolutionary understanding** (MSA row and column attention)
2. **Refine structural understanding** (Pair operations)
3. **Bidirectional information exchange** (MSA ↔ Pair communication)
4. **Enforce geometric constraints** (Triangle operations)
5. **Non-linear processing** (Transition layers)

**After 48 iterations**:
- Weak evolutionary signals get progressively amplified through repeated refinement
- Geometric constraints become increasingly tight and self-consistent
- The MSA and Pair representations converge toward a coherent, geometrically plausible 3D structure

**Analogy**: Think of it like iteratively solving a complex jigsaw puzzle:
- The MSA provides clues about which pieces fit together (co-evolutionary couplings)
- Triangle updates ensure the assembled pieces form valid 3D shapes (no geometric contradictions)
- Row attention connects distant pieces along the sequence
- Column attention identifies patterns across different versions of the puzzle (homologs)
- After 48 passes through this process, you emerge with a complete, self-consistent structural picture

---

## From Evoformer to Structure

The output of the 48 Evoformer blocks is a highly refined Pair representation that encodes:
- Distance distributions between all residue pairs
- Orientation relationships
- Contact probabilities
- Geometric constraints

This representation feeds into AlphaFold2's Structure Module, which uses it to predict precise 3D atomic coordinates. But that's a story for the upcoming subpart blog.

---

## Impact on Drug Discovery

With AlphaFold2's ability to predict protein structures with atomic accuracy:

1. **Identify binding pockets** where drug molecules can fit
2. **Enable structure-based drug design** for previously intractable targets
3. **Virtual screening** of millions of compounds becomes feasible
4. **Accelerate target validation** and hit-to-lead optimization

This is revolutionary: AlphaFold2 has made experimentally inaccessible proteins—roughly 40% of the human proteome—available as potential drug targets for the first time.

The Evoformer is the engine that makes this possible, transforming raw evolutionary sequences into precise 3D structures through an elegant interplay of attention, geometry, and iteration.

---

## Conclusion

The Evoformer represents a masterful integration of biological insight and deep learning innovation:

- **MSA operations** capture millions of years of evolutionary information
- **Pair operations** maintain and refine structural hypotheses
- **Triangle updates** enforce physical plausibility
- **Iterative refinement** allows complex patterns to emerge

By understanding these operations at the matrix level, we can appreciate not just *what* AlphaFold2 does, but *how* it achieves its remarkable accuracy—and why each component is essential to the whole.

The devil, as always, is in the details. And in this case, the details are beautifully elegant.

## References

[^1]: Jumper, J., Evans, R., Pritzel, A., Green, T., Figurnov, M., Ronneberger, O., ... & Hassabis, D. (2021). Highly accurate protein structure prediction with AlphaFold. nature, 596(7873), 583-589.