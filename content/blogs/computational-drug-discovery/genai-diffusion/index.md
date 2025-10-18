---
title: "Computational Drug Discovery Part 5 (Part 2/3): Generative Models for De Novo Drug Design - Diffusion Models"
date: 2025-10-18
draft: false
author: Saeed Mehrang
summary: "From prediction to creation (Part 2/3): Understanding diffusion models for molecular generation, with detailed implementation of torsional diffusion for 3D conformation generation."
tags: ["Computational Drug Discovery", "Generative Models", "Molecular Generation", "Deep Learning", "Denoising Diffusion"]
series_order: 5
series: ["Computational Drug Discovery"]
showToc: true
disableAnchoredHeadings: false
cover:
  image: "cover.png"
  image_alt: "Denoising Diffusion"
---

| Characteristic | Detail |
| :--- | :--- |
| **Estimated Reading Time** | 40-50 minutes |
| **Technical Level** | Advanced (requires understanding of deep learning, basic chemistry) |
| **Prerequisites** | [Part 1](../genai-vae-gan-diffusion/) on VAE/GAN recommended |


## 1. Introduction: The Generative Revolution

### 1.1 From Evaluation to Creation

In [Part 1 of this mini-series](../genai-vae-gan-diffusion/), we explored how **Variational Autoencoders (VAEs)** and **Generative Adversarial Networks (GANs)** can generate molecules by learning continuous latent representations and adversarial training, respectively. These approaches opened the door to de novo molecular design.

Now, we turn to **Diffusion Models**—the current state-of-the-art in generative modeling that has revolutionized both image generation (DALL-E 2, Stable Diffusion) and molecular design.

**The paradigm shift:**
- **Discriminative models**: Given a molecule, predict properties → $P(\text{property}|\text{molecule})$
- **Generative models**: Create molecules with desired properties → $P(\text{molecule}|\text{property})$ or $P(\text{molecule})$

Instead of screening millions of compounds, we **design molecules optimized for specific requirements**.

### 1.2 Why Diffusion Models?

Diffusion models offer compelling advantages over VAEs and GANs for molecular generation:

| Aspect | Diffusion Models | VAEs | GANs |
|--------|------------------|------|------|
| **Sample Quality** | Highest | Medium (blurry) | High |
| **Training Stability** | High | High | Low (mode collapse) |
| **Mode Coverage** | Excellent | Good | Poor |
| **3D Generation** | Native support | Requires modification | Requires modification |
| **Interpretability** | Low | High (latent space) | Low |
| **Sampling Speed** | Slow (1000 steps) | Fast | Fast |

**Key innovations for chemistry:**
- **SE(3) equivariance**: Respects rotational/translational symmetry of molecules
- **Torsional diffusion**: Focuses on flexible degrees of freedom (10-100x faster)
- **Conditional generation**: Easy to incorporate protein context, properties
- **Superior diversity**: Captures full distribution without mode collapse

### 1.3 What This Blog Covers

We'll explore diffusion models for molecular generation through two lenses:

1. **Theory** (Sections 2-3): Mathematical foundations of diffusion models
   - Forward and reverse processes
   - Training objectives and loss functions
   - Noise schedules and sampling algorithms

2. **Implementation** (Section 4): Detailed torsional diffusion walkthrough
   - Identifying rotatable bonds in molecules
   - EGNN architecture for SE(3) equivariance
   - Complete implementation from scratch
   - Training and generation on real molecules

By the end, you'll understand both the theory behind diffusion models and how to implement them for 3D molecular conformation generation.

---
This text provides a solid, detailed overview of Diffusion Models. Here is a reviewed and clarified version, incorporating necessary explanations for a beginner while maintaining the mathematical rigor.

***

# 2. Diffusion Models: Core Concepts

Diffusion Models (DMs) are generative models that learn to reverse a gradual process of information loss. They are state-of-the-art in generating high-quality complex data like images and molecular structures.

## 2.1 The Denoising Paradigm

The core idea is to train a neural network to **denoise** data at every possible noise level.

1.  **Forward process (The Destruction):** Gradually corrupt clean data by adding small amounts of noise over many steps (easy, no learning required).
2.  **Reverse process (The Restoration):** Learn a neural network ($\mathbf{\epsilon}_\theta$) to predict and remove the noise added at each step (the hard, learned part).
3.  **Generation (The Sampling or Creation):** Start from pure, random noise and use the learned network to iteratively reverse the forward process, step-by-step, until a clean sample is generated.

This approach contrasts sharply with VAEs (which compress to a small latent space) and GANs (which directly map noise to data). Diffusion models learn to **gradually denoise**, which makes the learning problem easier and more stable to optimize.

***

## 2.2 Forward Process: Gradually Adding Noise

The **forward process** $q$ is a fixed, predefined **Markov chain** (meaning each step $x_t$ depends only on the immediate previous step $x_{t-1}$) that corrupts data by adding Gaussian noise over $T$ steps (typically $T \approx 1000$ to ensure the final state is pure noise).

The transition from one state to the next is defined by the following probability distribution:

$$q(\mathbf{x}_t \mid \mathbf{x}_{t-1}) = \mathcal{N}(\mathbf{x}_t; \sqrt{1-\beta_t} \mathbf{x}_{t-1}, \beta_t \mathbf{I})$$

**Breaking down the notation:**
* $q$: Forward transition probability (fixed, no learning needed).
* $\mathcal{N}$: The **Gaussian (Normal) distribution**.
* **Mean:** $\sqrt{1-\beta_t} \mathbf{x}_{t-1}$. This term ensures the new noisy state $\mathbf{x}_t$ stays close to the previous state $\mathbf{x}_{t-1}$, scaled slightly downwards.
* **Covariance:** $\beta_t \mathbf{I}$. This is the variance of the added noise. $\mathbf{I}$ is the identity matrix, meaning the noise is **isotropic** (same in all dimensions).
* $\beta_t$: The **noise schedule** (a small value, like $0.01$) that determines how much noise is added at step $t$.

**The process is a sequence of transformations:**
$$\text{Clean data } \mathbf{x}_0 \rightarrow \mathbf{x}_1 \rightarrow \mathbf{x}_2 \rightarrow \dots \rightarrow \mathbf{x}_T \approx \mathcal{N}(\mathbf{0}, \mathbf{I}) \text{ (pure noise)}$$

### The "Single Step" Closed-Form Sampling

A crucial mathematical property allows us to skip all intermediate steps and calculate **any noisy state $\mathbf{x}_t$ directly** from the clean data $\mathbf{x}_0$:

$$\mathbf{x}_t = \sqrt{\bar{\alpha}_t} \mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_t} \boldsymbol{\epsilon}$$

* $\bar{\alpha}_t$: A precomputed coefficient derived from the $\beta_t$ schedule, where $\bar{\alpha}_t = \prod_{i=1}^t (1 - \beta_i)$. $\sqrt{\bar{\alpha}_t}$ controls the signal ($\mathbf{x}_0$) strength, and $\sqrt{1 - \bar{\alpha}_t}$ controls the noise ($\boldsymbol{\epsilon}$) strength.
* $\boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$: A sample of standard Gaussian noise.

This closed-form calculation is **critical for efficient training**—we use it to instantly generate a noisy version $\mathbf{x}_t$ to train our denoiser against, without having to run all $t$ steps sequentially, i.e. making jumps in the forward process if we want to.

***

## 2.3 Reverse Process: Learning to Denoise

The goal of the **reverse process** $p_\theta$ is to approximate the true (but intractable) reverse transition $q(\mathbf{x}_{t-1} \mid \mathbf{x}_t)$. We train a neural network (with parameters $\theta$) to learn the parameters of this distribution:

$$p_\theta(\mathbf{x}_{t-1} \mid \mathbf{x}_t) = \mathcal{N}(\mathbf{x}_{t-1}; \boldsymbol{\mu}_\theta(\mathbf{x}_t, t), \boldsymbol{\Sigma}_\theta(\mathbf{x}_t, t))$$

* $p_\theta$: The **learned (or approximated)** reverse transition probability.
* $\boldsymbol{\mu}_\theta(\mathbf{x}_t, t)$: The **learned mean** of the Gaussian distribution, predicting the slightly cleaner $\mathbf{x}_{t-1}$.
* $\boldsymbol{\Sigma}_\theta(\mathbf{x}_t, t)$: The **learned or fixed covariance** (variance) of the step-wise noise to be added.

**The generation process sequence:**
$$\text{Pure noise } \mathbf{x}_T \xrightarrow{\text{learned } p_\theta} \mathbf{x}_{T-1} \xrightarrow{\text{learned } p_\theta} \dots \xrightarrow{\text{learned } p_\theta} \mathbf{x}_0 \text{ (clean data)}$$

### What the Network Predicts

It can be mathematically shown that if the $\beta_t$ values are small, the reverse mean $\boldsymbol{\mu}$ depends on a function that predicts the original noise $\boldsymbol{\epsilon}$. This means that instead of having the neural network directly predict the complex formula for the next denoised state, it only needs to predict the simple noise vector—and the rest of the complicated denoising step can be calculated using that prediction.

In practice, the network $\mathbf{\epsilon}_\theta$ is trained to predict the **noise** $\boldsymbol{\epsilon}$ added to $\mathbf{x}_0$ to produce $\mathbf{x}_t$. The final reverse mean $\boldsymbol{\mu}_\theta$ can then be computed using this prediction:

$$\boldsymbol{\mu}_\theta(\mathbf{x}_t, t) = \frac{1}{\sqrt{\alpha_t}} \left( \mathbf{x}_t - \frac{1-\alpha_t}{\sqrt{1-\bar{\alpha}_t}} \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t) \right)$$

**Key Takeaway:** The network $\boldsymbol{\epsilon}_\theta$ is a **Noise Predictor** that acts as the core engine for the reverse process.

***

# 3. Training and Sampling

## 3.1 Training Objective (The Loss Function)

The objective function is designed to make the network's predicted noise $\boldsymbol{\epsilon}_\theta$ match the actual noise $\boldsymbol{\epsilon}$ used in the forward process. This is done using a simple Mean Squared Error (L2 loss):

$$\mathcal{L} = \mathbb{E}_{t, \mathbf{x}_0, \boldsymbol{\epsilon}} \left[ \Vert \boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t) \Vert^2 \right]$$

* $\mathbb{E}_{t, \mathbf{x}_0, \boldsymbol{\epsilon}} [\dots]$: This is the **Expectation** (average) taken over all possible random choices of time steps $t$, initial data $\mathbf{x}_0$, and noise $\boldsymbol{\epsilon}$.
* $\Vert \mathbf{A} - \mathbf{B} \Vert^2$: The **L2 Loss** (squared distance) between the actual noise $\boldsymbol{\epsilon}$ and the network's prediction $\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)$.

## 3.2 Training Algorithm

The training algorithm leverages the **closed-form** of the forward process (Section 2.2) for extreme efficiency:

**Training pseudo-code for one iteration**

1.  Sample a clean data point $\mathbf{x}_0$ from the dataset.
2.  Sample a random timestep $t \sim \text{Uniform}(1, T)$.
3.  Sample standard Gaussian noise $\boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$.
4.  Compute the noisy state $\mathbf{x}_t$ **in one step** using the closed form.
5.  Predict the noise: $\hat{\mathbf{\epsilon}} = \boldsymbol{\epsilon}_{\theta}(\mathbf{x}_t, t)$.
6.  Compute loss: $L = \Vert \boldsymbol{\epsilon} - \hat{\mathbf{\epsilon}} \Vert^2$.
7.  Update network weights $\theta$ via gradient descent.

**Significance:** By sampling random $t$ values, the network learns to denoise effectively at **all noise levels simultaneously** with a single unified function.

## 3.3 Sampling Algorithm (Generation)

Sampling is the execution of the learned reverse process, beginning with pure noise:

1.  Start with pure Gaussian noise: $\mathbf{x}_T \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$.

2.  **For $t = T$ down to $1$:** (Iterate backward through time)
    1.  **Predict Noise:** $\hat{\mathbf{\epsilon}} = \boldsymbol{\epsilon}_{\theta}(\mathbf{x}_t, t)$.
    2.  **Compute Denoised Mean ($\boldsymbol{\mu}_t$):** Use the network's prediction $\hat{\mathbf{\epsilon}}$ to calculate the deterministic part of the next, cleaner step:
        $$\mathbf{\mu}_t = \frac{1}{\sqrt{\alpha_t}} \left( \mathbf{x}_t - \frac{1-\alpha_t}{\sqrt{1-\bar{\alpha}_t}} \hat{\mathbf{\epsilon}} \right)$$
    3.  **Sample $\mathbf{x}_{t-1}$ (Add Noise):** The new state is a combination of the predicted mean and a bit of sampling noise (unless it's the final step, $t=1$):
        $$\mathbf{x}_{t-1} = \mathbf{\mu}_t + \sigma_t \mathbf{z}, \quad \text{where } \mathbf{z} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$$
        * $\sigma_t$: The variance of the added sampling noise. $\sigma_t=0$ yields a deterministic process (DDIM), while $\sigma_t>0$ yields a stochastic process (DDPM).

3.  Return the final clean sample $\mathbf{x}_0$.

### Sampling Variants

* **DDPM (Denoising Diffusion Probabilistic Models):** Uses the full stochastic sampling (non-zero $\sigma_t$) and typically requires the full $T$ steps (e.g., 1000) for high quality.
* **DDIM (Denoising Diffusion Implicit Models):** Uses a **deterministic** (or less stochastic) reverse process ($\sigma_t$ is zero or very small). This allows for **skip-sampling**, meaning high-quality results can be achieved in far fewer steps (e.g., 50-100 steps), making generation much faster.



***

## Note: The Role of Sampling Noise (DDPM vs. DDIM) in Step 3 of Sampling

The need to occasionally add random noise back during the denoising steps is one of the most confusing, yet crucial, parts of Diffusion Models. It boils down to whether the model is **probabilistic** or **deterministic**.

### Background: Probabilistic vs. Deterministic

The model is trained to reverse a physical process (diffusion) where noise is added randomly. Therefore, the **true reversal is also a stochastic (probabilistic) process**.

1.  **Denoised Mean ($\boldsymbol{\mu}_\theta$):** This is the deterministic, noise-removing component. It represents the **center** of the correct distribution and is calculated from the network's predicted noise ($\hat{\boldsymbol{\epsilon}}$) using the derived sampling formula:
    $$\boldsymbol{\mu}_\theta(\mathbf{x}_t, t) = \frac{1}{\sqrt{\alpha_t}} \left( \mathbf{x}_t - \frac{1-\alpha_t}{\sqrt{1-\bar{\alpha}_t}} \hat{\boldsymbol{\epsilon}} \right)$$

2.  **Sampling Noise ($\sigma_t \mathbf{z}$):** This is the random component. It ensures we sample a point *from the entire distribution*, not just its center. For **DDPM**, the true reverse process is stochastic, so we add this noise ($\sigma_t > 0$) to maintain probabilistic accuracy. For **DDIM**, we set this noise to zero ($\sigma_t = 0$), resulting in a deterministic and faster process.

### Sampling Noise Unlocks Diversity

The sampling noise $\sigma_t \mathbf{z}$ is essential for DDPM to express the full range of data it learned during training:

* **It Prevents Collapse:** The model's training process (using the L2 loss on noise) forces it to learn the full data distribution, preventing "mode collapse" (where the model only generates the most common type of image/molecule).
* **It Selects a Unique Path:** When generating, if we only use the deterministic mean ($\boldsymbol{\mu}_\theta$), we are performing a **deterministic** generation. This is the **DDIM** approach, which is fast but reduces the expressive power of the full probabilistic model.
* **It Ensures Diversity:** By adding the $\sigma_t \mathbf{z}$ noise back (the **DDPM** approach), we introduce a small, unique, random "jitter" at every step. This jitter compounds over the hundreds of steps, ensuring that even if you start with the exact same initial pure noise $\mathbf{x}_T$, you will generate a **different, unique, and diverse** final result $\mathbf{x}_0$. The noise is not canceling the denoising; it is the **key mechanism** for exploring the full learned diversity.


---

## 4. 3D Molecular Diffusion: From Theory to Implementation

### 4.1 The 3D Challenge

Generating valid 3D molecular structures requires respecting fundamental physical constraints:

| Constraint | Description | Example |
| :--- | :--- | :--- |
| **Bond Lengths** | Fixed distances between bonded atoms | C-C: 1.54Å, C=C: 1.34Å |
| **Bond Angles** | Angles between three connected atoms | Tetrahedral: 109.5°, Trigonal: 120° |
| **Torsional Angles** | Rotations around single bonds | Define molecular conformation |
| **SE(3) Equivariance** | Predictions must be invariant to rotation/translation | Model learns physics, not orientation |

### 4.2 Two Approaches to 3D Diffusion

**GeoDiff: Full Coordinate Diffusion**
- Diffuses all 3D atomic coordinates directly
- Uses SE(3)-equivariant GNN (e.g., EGNN) to predict noise
- **Advantage**: Complete flexibility
- **Disadvantage**: High-dimensional (3N dimensions for N atoms)

**Torsional Diffusion: Smart Dimensionality Reduction**
- Key insight: Bond lengths and angles are rigid; only torsions are flexible
- Fix rigid geometry, diffuse only torsional angles
- Reduces dimensionality from 3N to ~N/3

| Aspect | GeoDiff (Full 3D) | Torsional Diffusion |
|--------|-------------------|---------------------|
| **Dimensions** | 3N (all coordinates) | ~N/3 (torsions only) |
| **Sampling speed** | Baseline | 10-100x faster |
| **Sample quality** | High | Higher (fewer dimensions) |
| **Geometry respect** | Learned | Built-in (fixed bonds/angles) |

**Why torsional diffusion wins:**
- Dramatically fewer dimensions to optimize
- Naturally respects chemical constraints
- Faster sampling without quality loss
- Focus learning on what actually varies (flexibility)
---

### 4.3 Understanding Torsional Angles

#### What is a Dihedral (Torsion) Angle?

A **dihedral angle** $\phi$ describes the twist around a bond, defined by 4 atoms in sequence: A-B-C-D

- **Plane 1**: Atoms A, B, C
- **Plane 2**: Atoms B, C, D
- **Rotation axis**: The B-C bond
- **Dihedral angle**: Angle between the two planes (range: -180° to +180°)

{{< framed_image src="dihedral_angle.png" alt="Torsion Angle" width="500px" height="300px" >}}
Dihedral angle in HOOH. Image from Chemistry StackExchange, CC BY-SA 4.0.
{{< /framed_image >}}

#### Rules for Rotatable Bonds

A bond is rotatable if it meets ALL criteria:

| Criterion | Reason |
|-----------|--------|
| ✅ Single bond | Double/triple bonds are rigid |
| ✅ Not in ring | Ring bonds are constrained |
| ✅ Not terminal | Need substituents on both sides |
| ✅ Both atoms have degree ≥ 2 | Need 4 atoms (A-B-C-D) to define dihedral |

**Example: Ibuprofen (C₁₃H₁₈O₂)**
- Benzene ring: 6 bonds → **NOT rotatable** (in ring)
- C=O bond: 1 bond → **NOT rotatable** (double bond)
- C-C chains: ~8 bonds → **ROTATABLE!**

---

### 4.4 Implementation: Torsional Diffusion from Scratch

We'll implement torsional diffusion in 5 key components using Ibuprofen (C₁₃H₁₈O₂) as our example. The full implementation is available in [denoising_diffusion_mol.py](denoising_diffusion_mol.py).

#### Component 1: Molecular Torsion Analyzer

The `MolecularTorsionAnalyzer` class provides three key methods:

```python
class MolecularTorsionAnalyzer:
    @staticmethod
    def identify_rotatable_bonds(mol):
        """Apply 4 rules: single bond, not in ring, not terminal, degree >= 2"""
        # Returns list of (atom_i, atom_j) tuples

    @staticmethod
    def get_dihedral_atoms(mol, bond_atoms):
        """For bond (i, j), find neighbors to form (a, i, j, b)"""
        # Returns 4-atom tuple defining dihedral angle

    @staticmethod
    def calculate_dihedral_angle(coords, atom_indices):
        """Calculate angle between two planes using cross products"""
        # Returns angle in radians [-π, π]
```

**Example: Ibuprofen Analysis**

```python
ibuprofen_smiles = "CC(C)Cc1ccc(cc1)C(C)C(=O)O"
mol = Chem.AddHs(Chem.MolFromSmiles(ibuprofen_smiles))
AllChem.EmbedMolecule(mol, randomSeed=42)

analyzer = MolecularTorsionAnalyzer()
torsion_info, torsion_angles = analyzer.extract_torsion_angles(mol)

# Results:
# Formula: C13H18O2 (33 atoms total)
# Found 8 rotatable bonds
# Extracted 8 torsion angles (e.g., 62.22°, 174.56°, -171.05°, ...)
```

**Efficiency gain**: Ibuprofen has 33 atoms × 3 coordinates = 99 dimensions in full 3D space, but only **8 torsional dimensions** → 12x reduction!

---

#### Component 2: EGNN (SE(3)-Equivariant GNN)

To make our denoiser geometry-aware, we use an E(n) Equivariant Graph Neural Network:

```python
class EGNN_Layer(MessagePassing):
    """
    Key idea: Use distances (invariant) to update features,
    and displacement vectors (equivariant) to update coordinates.
    """
    def __init__(self, in_features, hidden_dim, out_features):
        # Edge MLP: processes [h_i, h_j, ||x_i - x_j||²]
        # Node MLP: updates node features
        # Coord MLP: outputs scalar weights for coordinate updates

    def message(self, h_i, h_j, pos_i, pos_j):
        dist_squared = ||pos_i - pos_j||²  # Invariant
        edge_msg = EdgeMLP([h_i, h_j, dist_squared])
        coord_weight = CoordMLP(edge_msg)  # Scalar (invariant)
        coord_msg = (pos_i - pos_j) * coord_weight  # Equivariant!
        return edge_msg, coord_msg
```

**Why SE(3) equivariance matters**: If you rotate the input molecule, the predicted noise rotates identically. The network learns physics, not arbitrary orientations.

---

#### Component 3: Torsion Denoiser Network

The core neural network that predicts noise in torsion angles:

```python
class TorsionDenoiser(nn.Module):
    def __init__(self, hidden_dim=128):
        # Time embedding: sinusoidal encoding of timestep
        self.time_mlp = MLP(hidden_dim)

        # Node embedding: atomic features
        self.node_embedding = Linear(128, hidden_dim)

        # EGNN: geometry-aware message passing
        self.egnn = EGNN(hidden_dim, num_layers=3)

        # Torsion prediction: sin/cos encoding + 4 atom features
        self.torsion_mlp = MLP(hidden_dim * 4 + 2 → 1)

    def forward(self, torsions_t, t, node_features, pos, edge_index, torsion_to_atoms):
        # 1. Embed timestep and nodes
        time_embed = sinusoidal_encoding(t)
        h = node_embedding(node_features) + time_embed

        # 2. Run EGNN for geometry-aware features
        h, pos_updated = egnn(h, pos, edge_index)

        # 3. For each torsion angle:
        #    - Get features of 4 defining atoms
        #    - Add circular encoding: [sin(angle), cos(angle)]
        #    - Predict noise for this torsion
        return noise_predictions  # Shape: [batch, num_torsions]
```

**Circular encoding**: Torsion angles are periodic (0° = 360°), so we use sin/cos representation to handle wrap-around properly.

---

#### Component 4: Torsional Diffusion Model

The complete diffusion model with forward/reverse processes:

```python
class TorsionalDiffusionModel(nn.Module):
    def __init__(self, hidden_dim=128, num_timesteps=1000):
        # Initialize noise schedule (cosine)
        self.betas = cosine_schedule(num_timesteps)
        self.alphas = 1 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

        # Denoising network
        self.denoiser = TorsionDenoiser(hidden_dim)

    def add_noise_to_torsions(self, torsions_0, t):
        """Add noise and wrap to [-π, π] for circular angles"""
        noise = torch.randn_like(torsions_0)
        α_bar = self.alpha_bars[t]
        torsions_t = √α_bar * torsions_0 + √(1-α_bar) * noise
        torsions_t = atan2(sin(torsions_t), cos(torsions_t))  # Wrap!
        return torsions_t, noise

    def training_step(self, clean_torsions, features):
        """Standard diffusion training: predict noise"""
        t = random_timestep()
        noisy_torsions, true_noise = self.add_noise_to_torsions(clean_torsions, t)
        predicted_noise = self.denoiser(noisy_torsions, t, features)
        loss = MSE(predicted_noise, true_noise)
        return loss

    @torch.no_grad()
    def generate_torsions(self, features, num_torsions):
        """Iterative denoising from random noise"""
        torsions = torch.randn(1, num_torsions) * π
        for t in reversed(range(self.num_timesteps)):
            noise_pred = self.denoiser(torsions, t, features)
            torsions = denoise_step(torsions, noise_pred, t)
            torsions = atan2(sin(torsions), cos(torsions))  # Wrap!
        return torsions
```

**Key implementation details:**
1. **Circular wrapping**: Always use `atan2(sin(θ), cos(θ))` to wrap angles to [-π, π]
2. **Cosine schedule**: Better than linear for torsional diffusion
3. **Sin/cos encoding**: Network sees `[sin(θ), cos(θ)]` instead of raw θ

---

#### Component 5: Training and Generation

Complete pipeline on Ibuprofen:

```python
# 1. Load molecule and extract torsions
mol = Chem.AddHs(Chem.MolFromSmiles("CC(C)Cc1ccc(cc1)C(C)C(=O)O"))
AllChem.EmbedMolecule(mol)
torsion_info, torsion_angles = analyzer.extract_torsion_angles(mol)
# Result: 8 rotatable bonds, 8 torsion angles

# 2. Create model and train
model = TorsionalDiffusionModel(hidden_dim=64, num_timesteps=1000)
optimizer = Adam(model.parameters(), lr=1e-4)

for epoch in range(50):
    loss = model.training_step(torsion_angles, molecular_features)
    loss.backward()
    optimizer.step()

# 3. Generate new conformations
generated_torsions = model.generate_torsions(molecular_features, num_torsions=8)
# Returns 8 new torsion angles → reconstructs 3D conformation
```

**Results**: From 33 atoms × 3 coords = **99 dimensions** down to **8 torsional dimensions** → **12x efficiency gain!**

---

## 5. Applications and Practical Considerations

### 5.1 State-of-the-Art Methods

Recent diffusion models have achieved breakthrough results in molecular generation:

| Method | Innovation | Application | Key Result |
|--------|------------|-------------|------------|
| **GeoDiff** | SE(3)-equivariant 3D diffusion | Conformer generation | 95%+ validity |
| **Torsional Diffusion** | Diffuse only torsion angles | Fast 3D generation | 10-100x speedup |
| **DiffSBDD** | Protein-conditioned generation | Structure-based drug design | 87% binding success |
| **TargetDiff** | Target-aware diffusion | Hit discovery | 92% property match |

**DiffSBDD** (Structure-Based Drug Design) is particularly exciting:
- Generates molecules directly **inside protein binding pockets**
- Conditions on AlphaFold-predicted protein structures
- Ensures spatial and chemical compatibility
- Current state-of-the-art for de novo drug design

### 5.2 When to Use Diffusion Models

**Best for:**
- 3D molecular conformation generation
- Structure-based drug design (with protein context)
- Maximum sample quality and diversity
- Capturing full distribution without mode collapse

**Consider alternatives when:**
- Fast inference is critical (VAEs: 10-100x faster)
- Need interpretable latent space (VAEs better for optimization)
- Limited computational budget (GANs/VAEs cheaper)
- Generating SMILES strings only (Transformers often better)

**Typical resource requirements:**
- **Training**: 10-20 GPU days for 1M molecules
- **Inference**: 10-100 molecules/sec (use DDIM for 50-100 steps instead of 1000)
- **Memory**: 16-32 GB GPU RAM

---

## 6. Summary and Next Steps

### What We Covered

In this blog, we explored diffusion models for molecular generation:

1. **Core concepts**: Forward/reverse processes, noise schedules, training objectives
2. **3D generation**: GeoDiff (full coordinates) vs. Torsional Diffusion (efficient)
3. **Implementation**: Complete torsional diffusion from scratch with EGNN backbone
4. **Applications**: State-of-the-art methods like DiffSBDD for structure-based drug design

**Key takeaways:**
- Diffusion models achieve superior sample quality and diversity
- Torsional diffusion reduces dimensionality by 10-100x (focus on flexible degrees of freedom)
- SE(3) equivariance is crucial for learning physics, not arbitrary orientations
- Circular encoding (sin/cos) handles periodic nature of torsion angles

### Coming Up: Part 3

In the final part of this mini-series, we'll explore **autoregressive and transformer-based molecular generation**:
- How to treat molecules as sequences (SMILES, SELFIES)
- GPT-style models for molecular generation (MolGPT, ChemFormer)
- Reinforcement learning for multi-objective optimization
- Conditional generation guided by desired properties

**The journey**: VAEs/GANs (Part 1) → Diffusion (Part 2) → Transformers (Part 3) → Complete generative AI toolkit for drug discovery!

---

## References

[^1]: Jing, Bowen, et al. "Torsional diffusion for molecular conformer generation." *Advances in Neural Information Processing Systems* 35 (2022): 24240-24253.

[^2]: Dihedral angle visualization: Chemistry StackExchange, "[What is the meaning of the dihedral angle in HOOH?](https://chemistry.stackexchange.com/questions/87067/what-is-the-meaning-of-the-dihedral-angle-in-hooh)" License: CC BY-SA 4.0.

**Additional References:**
- Satorras, Victor Garcia, Emiel Hoogeboom, and Max Welling. "E(n) equivariant graph neural networks." *International Conference on Machine Learning*. PMLR, 2021.
- Ho, Jonathan, Ajay Jain, and Pieter Abbeel. "Denoising diffusion probabilistic models." *Advances in Neural Information Processing Systems* 33 (2020): 6840-6851.
- Schneuing, Arne, et al. "Structure-based drug design with equivariant diffusion models." *arXiv preprint arXiv:2210.13695* (2022).
- Guan, Jiaqi, et al. "3D equivariant diffusion for target-aware molecule generation and affinity prediction." *arXiv preprint arXiv:2303.03543* (2023).