---
title: "Computational Drug Discovery Part 3 (Subpart 3/3): From Abstract Features to Atoms, AlphaFold2's Structure Module and Training"
date: 2025-10-24
draft: false
author: Saeed Mehrang
summary: "AlphaFold2's Structure Module uses Invariant Point Attention to convert abstract Evoformer predictions into 3D atomic coordinates through iterative refinement, while multi-objective loss functions guide training - Subpart 3/3"
description: "A detailed exploration of how AlphaFold2 converts Evoformer predictions into 3D coordinates using Invariant Point Attention, and the multi-objective loss functions that guide training"
tags: ["AlphaFold2", "Protein Structure", "Deep Learning", "Invariant Point Attention", "Loss Functions"]
categories: ["Computational Biology", "AI"]
math: true
cover:
    image: "cover.png"
    image_alt: "Alphafold Structure Module and Training"
showToc: true
disableAnchoredHeadings: false
---

In our [Subpart 2](../p2), we dissected the Evoformer—AlphaFold2's engine for refining evolutionary and geometric information. But the Evoformer produces abstract numerical representations, not actual 3D structures. How does AlphaFold2 bridge this gap?

This post explores the Structure Module, which translates high-dimensional embeddings into precise atomic coordinates, and the sophisticated loss functions that guide the entire system during training.


## The Structure Module: From Abstract to Physical

### The Translation Problem

**Input**: Evoformer's Pair Representation with shape `[L, L, c_pair]`
- Contains learned features encoding distance distributions, orientations, and geometric relationships
- These are abstract embeddings—numbers in high-dimensional space

**Output**: 3D Cartesian Coordinates with shape `[N_atoms, 3]`
- Precise (x, y, z) positions for every atom in the protein
- Must obey the laws of physics and geometry

This translation is non-trivial. A naive neural network would struggle because protein structures have inherent symmetries that must be respected.

---

## The Challenge of 3D Symmetry

### SE(3) Equivariance: Why It Matters

Protein structures possess **SE(3) symmetry** (Special Euclidean group in 3 dimensions):

**Rotational Invariance**: 
- Rotating the entire protein in space doesn't change its physical structure
- The relative positions of atoms remain identical

**Translational Invariance**:
- Moving the protein across space doesn't change its structure
- Only internal geometry matters, not global position

**The Problem**: Standard neural networks are not naturally equivariant. A regular network might learn that "the protein should point upward" or "residue 1 should be at the origin"—arbitrary choices that have nothing to do with actual protein structure.

**The Solution**: We need architecture that guarantees: *if you rotate/translate the input features, the output coordinates rotate/translate by exactly the same amount.*

---

## Invariant Point Attention (IPA): Geometry-Aware Processing

IPA is the core innovation that makes geometry-aware structure prediction possible.

### The Architecture

**Input Representations**:
- Single representation: `[L, c_single]` - per-residue features
- Pair representation: `[L, L, c_pair]` - pairwise features
- Current coordinates: `[L, 3]` - current 3D positions (updated iteratively)
- Local frames: `[L, 4, 4]` - transformation matrices defining local coordinate systems

### Local Reference Frames: The Key Insight

Each residue i has its own local coordinate system defined by its backbone atoms:

```
Frame_i = [origin, x_axis, y_axis, z_axis]
```

**Construction**:
```
# Origin at C_alpha atom
origin_i = position of C_alpha

# x-axis points from C_alpha to C (carbonyl carbon)
x_axis_i = normalize(C_position - C_alpha_position)

# y-axis from C_alpha to N, orthogonalized to x
y_temp = N_position - C_alpha_position
y_axis_i = normalize(y_temp - (y_temp · x_axis_i) * x_axis_i)

# z-axis from cross product
z_axis_i = x_axis_i × y_axis_i

# Combine into 4x4 transformation matrix
Frame_i = [
    [x_axis_i, y_axis_i, z_axis_i, origin_i],
    [0,        0,        0,        1        ]
]  # Shape: [4, 4]
```

**Why Local Frames?**: By describing neighboring residues relative to each residue's own frame, we eliminate dependence on global position/orientation.

### IPA Operation: Step by Step

**Step 1: Query, Key, Value Projections**

Split features into abstract and geometric components:

```
# Abstract features
Q_abstract = Single @ W_q_abstract  # [L, h, d_qk]
K_abstract = Single @ W_k_abstract  # [L, h, d_qk]
V_abstract = Single @ W_v_abstract  # [L, h, d_v]

# Geometric features (points in 3D space)
Q_points = Single @ W_q_points  # [L, h, n_points, 3]
K_points = Single @ W_k_points  # [L, h, n_points, 3]
V_points = Single @ W_v_points  # [L, h, n_points, 3]
```

Where:
- `h` = number of attention heads (typically 12)
- `d_qk` = dimension of query/key vectors (typically 16)
- `d_v` = dimension of value vectors (typically 16)
- `n_points` = number of geometric query/key points (typically 4)

**Step 2: Transform Points to Local Frames**

Transform query and key points into residues' local frames:

```
# For each residue i and each head
for i in range(L):
    for head in range(h):
        # Transform query points of residue i into i's local frame
        Q_points_local[i, head] = Frame_i^(-1) @ Q_points[i, head]
        
        # Transform all key points into i's local frame
        for j in range(L):
            K_points_local[i, j, head] = Frame_i^(-1) @ K_points[j, head]
```

Shape: `Q_points_local[i]`: `[h, n_points, 3]`  
Shape: `K_points_local[i, j]`: `[h, n_points, 3]`

**Step 3: Compute Attention Weights**

Combine abstract and geometric similarity:

```
# Abstract attention (standard dot product)
attn_abstract = (Q_abstract @ K_abstract^T) / sqrt(d_qk)  # [L, L, h]

# Geometric attention (based on 3D distances in local frames)
for i in range(L):
    for j in range(L):
        # Squared distance between query and key points in local frame
        dist_sq = ||Q_points_local[i] - K_points_local[i, j]||^2
        attn_geometric[i, j] = -sum(dist_sq) / 2  # [L, L, h]

# Combine with learned weights
attn_logits = attn_abstract + w_geometric * attn_geometric + bias_pair

# Normalize
attn_weights = softmax(attn_logits, dim=1)  # [L, L, h]
```

**Step 4: Aggregate Values**

Update both abstract features and geometric positions:

```
# Abstract update (standard attention)
V_aggregated_abstract = attn_weights @ V_abstract  # [L, h, d_v]

# Geometric update (aggregate geometric value points)
for i in range(L):
    V_points_global[i] = 0
    for j in range(L):
        # Transform value points from j's frame to global frame
        V_j_global = Frame_j @ V_points[j]
        
        # Weight by attention and accumulate
        V_points_global[i] += attn_weights[i, j] * V_j_global

    # Transform aggregated points into i's local frame
    V_points_local[i] = Frame_i^(-1) @ V_points_global[i]
```

**Step 5: Update Coordinates and Features**

```
# Update abstract single representation
Single_new = Single + MLP(concat(V_aggregated_abstract, V_points_local))

# Update 3D coordinates (in global frame)
Coordinate_delta = V_points_global @ W_coord  # Project to 3D offset
Coordinates_new = Coordinates + Coordinate_delta
```

### Why IPA Works

**Equivariance Guarantee**: Because all geometric computations happen in local frames:
- If you rotate/translate all input coordinates → local frames rotate/translate identically
- Geometric attention scores remain unchanged (computed in local frames)
- Output updates transform by the same rotation/translation

**Physics-Aware**: The network doesn't need to "learn" that rotating a protein doesn't change it—this is built into the architecture.

---

## Iterative Refinement: Building Structure Layer by Layer

### Inner Refinement Loop (8 Iterations)

The Structure Module doesn't predict the final structure in one shot. It refines iteratively:

**Initialization**:
```
# Start with rough backbone from Pair representation
Coordinates = initial_guess_from_pair(Pair)  # [L, 3] (C_alpha positions)
Single = single_representation_from_evoformer  # [L, c_single]
```

**Refinement Loop** (8 iterations):
```
for iteration in range(8):
    # 1. Update frames from current coordinates
    Frames = compute_frames_from_backbone(Coordinates)
    
    # 2. Apply IPA to refine features and coordinates
    Single, Coordinates = IPA(Single, Pair, Coordinates, Frames)
    
    # 3. Predict backbone torsion angles
    Angles = angle_prediction_network(Single)  # [L, 3] (phi, psi, omega)
    
    # 4. Predict sidechain torsion angles
    Sidechain_angles = sidechain_network(Single, Pair)  # [L, max_chi]
    
    # 5. Build all-atom structure from backbone + angles
    All_atom_coords = build_structure(Coordinates, Angles, Sidechain_angles)
    # Shape: [L, 14, 3] (up to 14 atoms per residue)
    
    # 6. Apply all-atom refinement
    Coordinates = All_atom_coords[:, CA_index, :]  # Update C_alpha positions
```

**Progressive Refinement**:
- Early iterations: Establish global fold
- Middle iterations: Refine backbone geometry
- Final iterations: Optimize sidechain conformations and fine details

### Outer Recycling Loop (3 Cycles)

After the complete network processes the input once, AlphaFold2 feeds the output back as input:

**Recycling Process**:
```
# Initial forward pass
MSA_0, Pair_0 = initial_features(sequence, msa_input)

for recycle in range(3):
    # Run full network
    MSA_out, Pair_out, Structure = AlphaFold2(MSA_0, Pair_0)
    
    # Extract structural features from prediction
    Structure_features = extract_features(Structure)  # Distances, angles, etc.
    
    # Augment input for next cycle
    MSA_0 = concat(MSA_0, structure_to_msa_features(Structure_features))
    Pair_0 = Pair_0 + structure_to_pair_features(Structure_features)

# Final prediction uses outputs from last recycle
Final_structure = Structure
```

**Why Recycling Works**:
1. **Better initialization**: Second pass starts with a good structural prior
2. **Error correction**: Network can fix mistakes using self-generated constraints
3. **Iterative refinement**: Mimics experimental structure determination methods
4. **Confidence building**: High-confidence regions stabilize uncertain regions

---

## Loss Functions: Guiding the Learning

AlphaFold2 optimizes a sophisticated multi-objective loss function. Let's examine each component.

### Primary Loss: Frame Aligned Point Error (FAPE)

**Purpose**: Measure geometric accuracy in a rotationally and translationally invariant way.

**Computation**:

For each residue i (acting as alignment reference):

```
# Step 1: Get local frame for residue i
Frame_i = compute_frame(Predicted_coords[i])  # [4, 4]
Frame_i_true = compute_frame(True_coords[i])

# Step 2: Transform all atom positions into i's local frame
for atom in all_atoms:
    # Predicted atom in i's frame
    p_pred_local = Frame_i^(-1) @ Predicted_coords[atom]  # [3]
    
    # True atom in i's true frame
    p_true_local = Frame_i_true^(-1) @ True_coords[atom]  # [3]
    
    # Squared error in local frame
    error[i, atom] = ||p_pred_local - p_true_local||^2

# Step 3: Aggregate across alignment frames
FAPE = (1 / (L * N_atoms)) * sum_over_i(sum_over_atoms(sqrt(error[i, atom])))
```

**In tensor form**:
```
Predicted: [L, 14, 3]  # All atoms for L residues
True: [L, 14, 3]
Frames_pred: [L, 4, 4]  # Transformation matrices
Frames_true: [L, 4, 4]

# Transform to local frames (broadcasting over atoms)
P_local_pred = Frames_pred^(-1) @ Predicted  # [L, L, 14, 3]
P_local_true = Frames_true^(-1) @ True        # [L, L, 14, 3]

# Compute errors
Errors = ||P_local_pred - P_local_true||_2  # [L, L, 14]

# Average
FAPE = mean(Errors)
```

**Key property**: FAPE is SE(3) invariant—rotating/translating the entire structure doesn't change the loss.

**Weighting**:
```
FAPE_loss = w_backbone * FAPE(backbone_atoms) + w_sidechain * FAPE(sidechain_atoms)
```

Typically: `w_backbone = 0.5`, `w_sidechain = 0.5`

---

### Auxiliary Loss 1: Distogram

**Purpose**: Guide the Pair representation to encode accurate distance information early.

**Target**: Binned distance distribution between C_alpha atoms.

**Distance bins**: Typically 64 bins covering 2.3Å to 21.6Å

**Computation**:

```
# Extract C_alpha positions
CA_coords_pred = Structure[:, CA_index, :]  # [L, 3]
CA_coords_true = True_structure[:, CA_index, :]  # [L, 3]

# Compute pairwise distances
Dist_pred = ||CA_coords_pred[i] - CA_coords_pred[j]||_2  # [L, L]
Dist_true = ||CA_coords_true[i] - CA_coords_true[j]||_2  # [L, L]

# Get distance logits from Pair representation
Logits = Pair @ W_distogram  # [L, L, n_bins]

# Compute true bin indices
Bin_true = digitize(Dist_true, bin_edges)  # [L, L]

# Cross-entropy loss
Distogram_loss = CrossEntropy(Logits, Bin_true)
```

**In detail**:
```
# For each residue pair (i, j)
Distogram_loss = -(1 / L^2) * sum_i(sum_j(log(softmax(Logits[i,j])[Bin_true[i,j]])))
```

**Why distributions?**: Predicting a distribution over bins captures uncertainty better than predicting a single distance value.

---

### Auxiliary Loss 2: Masked MSA Prediction

**Purpose**: Force the network to learn evolutionary patterns and co-evolution signals.

**Mechanism**: Randomly mask 15% of MSA positions and predict the masked amino acids.

**Computation**:

```
# Input MSA (with some positions masked)
MSA_input: [N_seq, L, 21]  # One-hot encoded, 21 amino acid types + gap

# Network predictions
MSA_logits = MSA_representation @ W_aa_prediction  # [N_seq, L, 21]

# Mask matrix (1 where masked, 0 otherwise)
Mask: [N_seq, L]

# True amino acids at masked positions
True_AA: [N_seq, L]

# Cross-entropy on masked positions only
Masked_MSA_loss = -(1 / sum(Mask)) * sum_masked_positions(
    log(softmax(MSA_logits[masked_position])[True_AA[masked_position]])
)
```

**Why it matters**: Similar to BERT in NLP, this self-supervised task forces the model to understand sequence context and evolutionary relationships.

---

### Auxiliary Loss 3: Predicted LDDT (pLDDT)

**Purpose**: Train the model to predict its own confidence for each residue.

**LDDT** (Local Distance Difference Test): Measures how well local atomic distances are preserved.

**Target computation**:

```
# For residue i, find all atoms within 15Å
Neighbors_i = atoms within 15Å of residue i

# For each neighbor pair (j, k) in Neighbors_i:
d_pred_jk = ||Predicted[j] - Predicted[k]||
d_true_jk = ||True[j] - True[k]||

# Count conserved distances (4 thresholds: 0.5Å, 1Å, 2Å, 4Å)
conserved = 0
for threshold in [0.5, 1.0, 2.0, 4.0]:
    if |d_pred_jk - d_true_jk| < threshold:
        conserved += 1

# LDDT score for residue i
LDDT_i = conserved / (4 * |neighbor_pairs|)  # Range: [0, 1]
```

**Network prediction**:
```
# Network outputs confidence logits
pLDDT_logits = Single_representation @ W_lddt  # [L, n_bins]

# Bin LDDT scores (typically 50 bins from 0 to 1)
LDDT_binned = digitize(LDDT_i, lddt_bins)  # [L]

# Cross-entropy loss
pLDDT_loss = CrossEntropy(pLDDT_logits, LDDT_binned)
```

**Output interpretation**: At inference, `pLDDT > 90` indicates very high confidence, `pLDDT < 50` indicates low confidence.

---

### Auxiliary Loss 4: Predicted Aligned Error (PAE)

**Purpose**: Predict expected positional error between residue pairs after alignment.

**Definition**: PAE(i, j) = expected error in position of residue j after aligning on residue i.

**Target computation**:

```
# Align predicted and true structures on residue i
for i in range(L):
    # Compute optimal rotation/translation aligning on residue i
    R_i, t_i = align_on_residue(Predicted, True, align_residue=i)
    
    # Transform predicted structure
    Predicted_aligned = R_i @ Predicted + t_i
    
    # Compute error for each residue j
    for j in range(L):
        PAE_true[i, j] = ||Predicted_aligned[j] - True[j]||_2

# Shape: [L, L], symmetric matrix
```

**Network prediction**:

```
# Network outputs PAE logits from Pair representation
PAE_logits = Pair @ W_pae  # [L, L, n_bins]

# Bin true PAE values (typically 64 bins, 0-32Å)
PAE_binned = digitize(PAE_true, pae_bins)  # [L, L]

# Cross-entropy loss
PAE_loss = CrossEntropy(PAE_logits, PAE_binned)
```

**Interpretation**: Low PAE between residues indicates high confidence in their relative positioning. Useful for multi-domain proteins.

---

### Auxiliary Loss 5: Violation Loss

**Purpose**: Penalize physically implausible structures.

**Components**:

**1. Bond length violations**:
```
# Expected bond lengths (from chemistry)
Expected_bonds: dict mapping (atom_i, atom_j) to expected_length

for (i, j) in covalent_bonds:
    d_pred = ||Predicted[i] - Predicted[j]||
    d_expected = Expected_bonds[(i, j)]
    
    # Tolerance: typically 0.02Å
    violation = max(0, |d_pred - d_expected| - tolerance)
    Bond_violation += violation^2
```

**2. Steric clash violations**:
```
# Minimum allowed distance between non-bonded atoms
min_distance = 1.5  # Ångströms

for (i, j) in non_bonded_pairs:
    d_pred = ||Predicted[i] - Predicted[j]||
    
    if d_pred < min_distance:
        Clash_violation += (min_distance - d_pred)^2
```

**3. Chirality violations**:
```
# C_alpha atoms should have L-chirality (with rare exceptions)
for residue in residues:
    # Compute chirality from positions of N, CA, C, CB
    chirality_sign = sign(det([CA-N, CA-C, CA-CB]))
    
    if chirality_sign != expected_chirality:
        Chirality_violation += 1
```

**Total violation loss**:
```
Violation_loss = w_bond * Bond_violation + 
                 w_clash * Clash_violation + 
                 w_chirality * Chirality_violation
```

---

### Combined Loss Function

The total loss is a weighted sum:

```
Total_loss = w_fape * FAPE_loss +
             w_distogram * Distogram_loss +
             w_masked_msa * Masked_MSA_loss +
             w_plddt * pLDDT_loss +
             w_pae * PAE_loss +
             w_violation * Violation_loss
```

**Typical weights** (from AlphaFold2 paper):
```
w_fape = 1.0
w_distogram = 0.3
w_masked_msa = 2.0
w_plddt = 0.01
w_pae = 0.1
w_violation = 0.01
```

**Multi-scale training**: Different loss components operate at different scales:
- FAPE: atomic-level accuracy
- Distogram: residue-pair distances
- Masked MSA: sequence-level evolutionary patterns
- Violations: chemical constraints

This multi-objective approach ensures the model learns from multiple perspectives simultaneously.

---

## Training Strategy

### Training Data

**Primary dataset**: Protein Data Bank (PDB)
- ~170,000 experimentally determined structures (as of 2020)
- X-ray crystallography, NMR, Cryo-EM

**Data augmentation**:
```
# Random crops for long sequences
if L > max_length:
    start = random(0, L - max_length)
    sequence = sequence[start:start + max_length]
    
# MSA depth subsampling
N_seq_sampled = random(N_seq_min, N_seq_max)
msa = random_sample(msa, N_seq_sampled)

# Random MSA deletion
msa = randomly_delete_rows(msa, p=0.15)
```

### Self-Distillation: Learning from Yourself

**Key innovation**: Use the trained model to generate high-confidence predictions for sequences without experimental structures.

**Process**:

```
# Stage 1: Train on PDB data
Model_v1 = train(PDB_structures)

# Stage 2: Generate predictions for large sequence database
for sequence in UniRef90:  # ~100M sequences
    prediction = Model_v1.predict(sequence)
    
    if prediction.pLDDT > 90:  # High confidence only
        Pseudo_labels.add(sequence, prediction.structure)

# Stage 3: Retrain on combined dataset
Model_v2 = train(PDB_structures + Pseudo_labels)
```

**Why it works**:
- Vastly expands training data (PDB: ~170K → Combined: ~100M)
- High-confidence predictions are nearly as good as experimental structures
- Model learns from its own best predictions, reinforcing accurate patterns

**Result**: AlphaFold2 trained with self-distillation significantly outperforms models trained only on PDB.

---

## Putting It All Together

The complete AlphaFold2 training pipeline:

1. **Input processing**: Sequence → MSA search → Feature generation
2. **Evoformer**: 48 blocks of evolutionary and geometric refinement
3. **Structure Module**: 8 iterations of IPA-based coordinate refinement
4. **Recycling**: 3 outer loops feeding predictions back as input
5. **Loss computation**: Multi-objective loss combining 6+ components
6. **Backpropagation**: Gradients flow through entire network
7. **Optimization**: Adam optimizer with learning rate scheduling

**Training time**: 
- ~1 week on 128 TPUv3 cores
- ~11 days of GPU time per model

**The result**: A model that predicts protein structures with experimental accuracy for the majority of the proteome.

---

## From Theory to Practice

Understanding these components reveals why AlphaFold2 works:

- **IPA** ensures geometric consistency without learning physics from scratch
- **Iterative refinement** allows progressive structure building
- **Multi-objective losses** guide learning from complementary signals
- **Self-distillation** dramatically expands effective training data

Each design choice addresses a specific challenge in the protein folding problem. Together, they form a system that doesn't just predict structures—it understands the principles underlying protein geometry.

The beauty lies not in any single innovation, but in how these components work in concert, each amplifying the others to achieve something greater than the sum of its parts.

---

## What's Next?

With AlphaFold2's architecture demystified, future directions include:
- **AlphaFold3**: Extending to protein-ligand complexes and other biomolecules
- **Protein design**: Running the model in reverse to design novel proteins
- **Dynamics prediction**: Moving beyond static structures to protein motion

The foundation AlphaFold2 has laid transforms not just structure prediction, but our entire approach to computational biology.

Understanding how it works is the first step to building what comes next.
