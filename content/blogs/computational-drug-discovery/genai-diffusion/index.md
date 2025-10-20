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

### Note: The Role of Sampling Noise (DDPM vs. DDIM) in Step 3 of Sampling

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


***

## 4. Diffusion Models for 3D Molecule Generation: The Landscape

The field of 3D molecular diffusion is rapidly evolving beyond initial models. The choice of model depends on the task: generating **conformations** (3D shape for a known molecule) or generating **full, novel molecules** (graph structure + 3D coordinates).

Here is a summary of the leading approaches, their purposes, and trade-offs.

| Model | Purpose | Novelty/Limitation | Implementation Complexity |
| :--- | :--- | :--- | :--- |
| **GeoDiff** | **Conformer Generation** | Diffuses all $3N$ coordinates (full Euclidean space). **Limitation:** Struggles to maintain perfect bond lengths/angles. | High (Requires SE(3)-Equivariant GNNs) |
| **Torsional Diffusion** | **Conformer Generation** | Diffuses only $\approx M$ torsion angles (torus space). **Novelty:** Inherently respects rigid chemistry; extremely fast. | Medium (Requires specialized periodic diffusion) |
| **MUDiff** | **Full Molecule Generation** | Jointly generates **2D graph** and **3D coordinates**. **Novelty:** Unifies discrete (graph) and continuous (coordinates) diffusion. | Very High (Requires two coupled diffusion processes) |
| **GeoLDM** | **Full Molecule Generation** | Generates both graph and coordinates in a **low-dimensional latent space**. **Novelty:** Faster sampling, better scalability via latent compression. | Very High (Requires training a geometric autoencoder + latent DM) |
| **GCDM** | **Full Molecule Generation** | Generates graph and coordinates; focuses on local geometric fidelity. **Novelty:** Uses a **Geometry-Complete GNN** for highly accurate local bond geometry. | High (Requires advanced equivariant GNN architecture) |

---

### 4.1 GeoDiff: The Coordinate Baseline

* **Purpose:** Generate diverse 3D **conformations** given a fixed molecular graph.
* **Novelty:** Was one of the first to apply the DDPM framework with an **SE(3)-Equivariant GNN** to 3D coordinates, ensuring predictions are consistent regardless of molecular orientation.
* **Limitation:** By diffusing all $3N$ coordinates, it forces the model to learn the rigid chemical rules (bond lengths, bond angles) from scratch, leading to a high proportion of generated structures with slight geometric inaccuracies.
* **Application:** Conformer generation for small to medium-sized molecules.

---

### 4.2 Torsional Diffusion: The Conformer Specialist

* **Purpose:** Generate high-quality 3D **conformations** by exploiting molecular flexibility.
* **Novelty:** Reduces the problem dimensionality by fixing all rigid components and diffusing **only the $\approx M$ rotatable torsion angles** over the specialized $\mathbb{T}^M$ (torus) space. This provides **10x to 1000x faster sampling** than GeoDiff (since Torsional Diffusion uses 5-20 steps vs GeoDiff's 5000 steps).
* **Limitation:** Cannot generate novel molecular graphs.
* **Application:** Rapid and accurate conformer generation for virtual screening and preparing training data.

---

### 4.3 Models for Full Molecule Generation (Addressing GeoDiff's Scope)

Models like GeoDiff and Torsional Diffusion are limited to generating conformers for *known* molecules. The following models achieve **de novo molecule generation** by producing the graph structure (connectivity and atom types) and 3D coordinates simultaneously.

#### A. MUDiff: Unified Discrete/Continuous Diffusion

* **Purpose:** **De novo generation** of complete molecules (2D graph and 3D coordinates).
* **Novelty:** It is a **unified diffusion model** that couples a **discrete diffusion process** (for bond and atom types) with a **continuous diffusion process** (for 3D coordinates). It uses a specialized Transformer to denoise both features concurrently, leveraging the synergy between topology and geometry.
* **Limitation:** High implementation complexity due to managing two distinct, yet coupled, diffusion processes.
* **Application:** Generates novel, stable 2D/3D molecules from a single latent space.

#### B. GeoLDM: Geometric Latent Diffusion

* **Purpose:** Scalable **de novo generation** of large 3D molecules.
* **Novelty:** Inspired by Stable Diffusion, it uses a geometric **Autoencoder** to compress the high-dimensional molecule representation into a lower-dimensional latent space. Diffusion is performed efficiently in this latent space, significantly reducing training and sampling time compared to working directly in the $3N$ feature space.
* **Limitation:** Requires training a complex two-stage model (Autoencoder + Diffusion Model), where the quality of the Autoencoder is critical.
* **Application:** Fast, high-throughput generation of novel drug-like molecules.

#### C. GCDM: Geometry-Complete Diffusion Model

* **Purpose:** **De novo generation** with extreme geometric fidelity.
* **Novelty:** While also a full molecule generator, its primary innovation is using a **Geometry-Complete GNN** (GCPNet) that explicitly encodes local geometric relationships (like inter-atomic distances and frames) into its message-passing. This is a direct attempt to resolve the issue of subtle bond/angle inaccuracies seen in earlier SE(3)-equivariant models.
* **Limitation:** The advanced GNN architecture increases complexity and computational cost per step.
* **Application:** Generating large, complex molecules where geometric and energetic stability are paramount.

***


## 5. Torsional Diffusion: A Simple Diffusion Model for Conformer Generation

Torsional Diffusion is built on the crucial insight that for molecular structures, the majority of chemical information (bond types, bond lengths, bond angles) is **rigid and fixed**. Only the **torsional angles** allow for molecular flexibility. By restricting diffusion to this small, flexible set of degrees of freedom, the model achieves superior performance in **conformer generation**.

Conformer generation, the process of accurately predicting the various low-energy three-dimensional shapes (conformations) a molecule can adopt, is an **indispensable capability** at the heart of modern drug discovery. Molecules are not rigid; they flex and twist, and their biological activity is directly governed by their shape. Since a drug molecule must physically fit into a specific pocket on a target protein (like a key fitting a lock), knowing the molecule's accessible 3D forms is critical. Generating diverse and high-quality conformers is the foundational first step for techniques such as **molecular docking** and **quantitative structure-activity relationship (QSAR) modeling**. Without a reliable set of conformers, the best drug candidate might be overlooked, as the computational screen would miss the active conformation required for binding.

Torsional Diffusion, and other conformer generation models like GeoDiff, fundamentally require the **initial molecular formula and connectivity (the molecular graph)** to begin the generation process. These models are not chemists; they are computational tools designed to explore the 3D flexibility of a structure that has already been designed or discovered. The scientist must first supply the **SMILES string** or other chemical identifier that defines the atoms and bonds, effectively telling the model, "Here is the molecule; now find all its possible shapes." This dependency underscores the two-stage nature of *de novo* drug discovery: first, using models like MUDiff or VAEs to hypothesize a novel **2D molecular graph**, and second, using conformer-specific models like Torsional Diffusion to predict the crucial **3D geometry** necessary for biological interaction.

In the modern age, the ability of models like Torsional Diffusion to rapidly generate vast libraries of chemically accurate conformers addresses major bottlenecks in high-throughput screening. This computational speed allows researchers to move beyond small, pre-calculated conformation libraries to generate conformations *on-the-fly* for millions of novel drug candidates. This accelerated process is vital for two key reasons: **virtual screening** and **pharmacophore modeling**. Virtual screening relies on docking every possible conformer into a target protein to predict binding strength; a fast, accurate conformer generator drastically improves the hit rate. Furthermore, drug design often involves defining a **pharmacophore**—the essential 3D arrangement of functional groups necessary for activity. Accurate conformer generation validates the existence of molecules that can satisfy this precise 3D arrangement, empowering structure-based drug design and significantly accelerating the path from initial lead compound to clinical trials.

### 5.1 Understanding Molecular Flexibility (Torsional Angles)

#### What is a Dihedral (Torsion) Angle?

A **dihedral angle** $\phi$ describes the twist around a bond, defined by 4 atoms in sequence: A-B-C-D.  The angle is the separation between **Plane 1** (defined by atoms A, B, C) and **Plane 2** (defined by atoms B, C, D), rotated around the central B-C bond (the **rotation axis**). The angle ranges from $-180^\circ$ to $+180^\circ$.

#### Rules for Rotatable Bonds

A bond is rotatable only if its rotation does not violate the molecule's overall chemical structure.

| Criterion | Reason |
|-----------|--------|
| ✅ **Single bond** | Double/triple bonds are rigid (fixed angle). |
| ✅ **Not in ring** | Ring bonds are geometrically constrained. |
| ✅ **Not terminal** | Requires substituents on both sides (A-B-C-D) to define the four-atom sequence. |
| ✅ **Both atoms have degree $\ge 2$** | Need 4 atoms (A-B-C-D) to define dihedral. |

**Example: Ibuprofen ($\text{C}_{13}\text{H}_{18}\text{O}_2$)**
The rigid structure includes the benzene ring and the carbonyl group ($\text{C}=\text{O}$). The flexible parts are the saturated C-C chains, which contain the **rotatable bonds** that the model will diffuse.

***

### 5.2 The Torsional Diffusion Process: $\mathbb{T}^M$

#### A. Core Function: Conformer Generation Only

**Torsional Diffusion is a Conformer Generator, not a full molecule generator.**

* It requires the complete **molecular graph** (atom types and connectivity) as a fixed input.
* It **only** generates the 3D shape (conformation), assuming the rigid bond lengths and angles are fixed constants.
* The model **cannot** generate entirely novel molecular graphs.

#### B. Forward Diffusion on the Hypertorus ($\mathbb{T}^M$)

The model diffuses the vector of $M$ rotatable torsion angles, $\boldsymbol{\phi}_0 \in [-\pi, \pi)^M$, into a random state. Since the data is circular, standard Gaussian noise cannot be used.

1.  **Circular Data Challenge:** Angles are **periodic** (e.g., $181^\circ \equiv -179^\circ$). Adding linear noise can cause small changes to result in large jumps across the boundary.
2.  **The Solution:** The diffusion is performed over the **hypertorus** ($\mathbb{T}^M$), which is equivalent to performing diffusion using a **Wrapped Gaussian distribution**. But, **what is a Hypertorus?**


The term **hypertorus** ($\mathbb{T}^M$) refers to the mathematical space formed by the **Cartesian product of $M$ circles**.

* A **circle** ($\mathbb{S}^1$ or $\mathbb{T}^1$) is a 1-dimensional manifold that represents all possible values for a single periodic variable, like a clock face or a single torsion angle.
* A **hypertorus** is simply the shape created when you combine $M$ such circles. Since a molecule's conformation is defined by $M$ independent torsion angles ($\phi_1, \phi_2, \dots, \phi_M$), the complete space of all possible conformations is this $M$-dimensional hypertorus $\mathbb{T}^M$.

Imagine a molecule with only *two* rotatable bonds ($M=2$): the space of all its conformations is a 2D surface shaped like a **donut** (a standard torus). For a typical drug molecule with $M \approx 8$ rotatable bonds, the space is an 8-dimensional hypertorus.

**The Solution for Circular Data**

Operating diffusion on $\mathbb{T}^M$ is necessary because standard Euclidean diffusion (which assumes the data spans $\mathbb{R}^M$) would break the molecular structure:

* **Problem:** If a torsion angle is at $179^\circ$ and standard Gaussian noise adds $3^\circ$, the result is $182^\circ$. In chemistry, $182^\circ$ is chemically equivalent to **$-178^\circ$** (a jump across the $180^\circ$ boundary). Standard diffusion models interpret this as a massive change, greatly inflating the error.
* **The Solution:** Performing diffusion on the $\mathbb{T}^M$ space means using a **Wrapped Gaussian distribution**. This distribution correctly handles periodicity by mapping any value outside the $[-\pi, \pi)$ range back into it using the **modulo $2\pi$ operation**. This ensures that the noise perturbation is chemically sensible, allowing the model to smoothly and accurately denoise angles, even near the boundary. This focus on the correct geometric space is a key factor in Torsional Diffusion's superior speed and accuracy.


All in all, the noisy torsion $\boldsymbol{\phi}_t$ at time $t$ is calculated by:

$$\boldsymbol{\phi}_t = (\boldsymbol{\phi}_0 \sqrt{\bar{\alpha}_t} + \sqrt{1 - \bar{\alpha}_t} \boldsymbol{\epsilon}) \pmod{2\pi}$$

Where:
* $\boldsymbol{\epsilon}$: A **wrapped** Gaussian noise vector, ensuring periodicity.
* $\pmod{2\pi}$: Ensures the angle remains mathematically within the range $[-\pi, \pi)$.
* The final state $\boldsymbol{\phi}_T$ is a uniform random distribution over the torus.

#### C. The Denoising Network and Equivariance

The key to the model's success lies in its neural network architecture, which must predict the noise in a way that is invariant to global rotations/translations, while still respecting local chemistry.

1.  **Target:** The neural network ($\mathbf{s}_{\theta}$) is trained to predict the **score function**—the gradient of the log probability density—over the torus:
    $$\mathbf{s}_{\theta}(\boldsymbol{\phi}_t, t) \approx \nabla_{\boldsymbol{\phi}_t} \log p(\boldsymbol{\phi}_t)$$

2.  **Architecture:** The model typically uses a **Graph Neural Network (GNN)**, often an **SE(3)-equivariant** or **invariant** variant (like a Torsion GNN), to process the molecule. The GNN takes the atom features and coordinates as input, but **only outputs a prediction for the $M$ torsion angles**, ensuring the model only denoises the flexible degrees of freedom.

#### D. Reverse Process and Fast Sampling

The reverse process is the generation step, starting from the random state $\boldsymbol{\phi}_T$.

1.  **Training Loss:** The network is trained using a generalized **score matching loss** to accurately predict $\mathbf{s}_{\theta}$.
2.  **Sampling:** Generation is typically performed using an **accelerated Ordinary Differential Equation (ODE) solver**, a deterministic approach similar to **DDIM**. Starting from a uniform random sample $\boldsymbol{\phi}_T$, the solver integrates the learned score function $\mathbf{s}_{\theta}$ back toward $\boldsymbol{\phi}_0$:

$$\mathbf{d}\boldsymbol{\phi} = \left[ \mathbf{f}(\boldsymbol{\phi}, t) - \frac{1}{2} g(t)^2 \mathbf{s}_{\theta}(\boldsymbol{\phi}, t) \right] \mathbf{d}t$$

The deterministic, low-dimensional nature of this integration is the reason for the **$10\text{x}$ to $100\text{x}$ speed-up** over full 3D diffusion models like GeoDiff.

#### E. Final 3D Reconstruction

The model's final output is a clean vector of torsion angles $\boldsymbol{\phi}_0$. A separate, deterministic chemical tool (e.g., using bond kinematics from RDKit or OpenBabel) then uses these predicted angles, along with the **fixed bond lengths and fixed bond angles**, to construct the final, complete **3D Cartesian coordinates** $\mathbf{x}_0$. This ensures the final molecule is chemically valid and geometrically precise by design.

---

### 5.3 Implementation: Torsional Diffusion from Scratch

We'll implement torsional diffusion in 5 key components using Ibuprofen (C₁₃H₁₈O₂) as our example. The full implementation is available in my Github repo [computational-drug-discovery-learning](https://github.com/saeedmehrang/computational-drug-discovery-learning/blob/main/torsional_diffusion.py).

### Component 1: Molecular Torsion Analyzer

The `MolecularTorsionAnalyzer` class provides three key methods that fuel the last method `extract_torsion_angles`:

```python
class MolecularTorsionAnalyzer:
    @staticmethod
    def identify_rotatable_bonds(mol: Chem.Mol) -> List[Tuple[int, int]]:
        """Apply 4 rules: single bond, not in ring, not terminal (degree >= 2), not hydrogen"""
        # Returns list of (atom_i, atom_j) tuples

    @staticmethod
    def get_dihedral_atoms(mol: Chem.Mol, bond_atoms: Tuple[int, int]) -> Optional[Tuple[int, int, int, int]]:
        """For bond (i, j), find neighbors to form (a, i, j, b)"""
        # Returns 4-atom tuple defining dihedral angle

    @staticmethod
    def calculate_dihedral_angle(coords: np.ndarray, atom_indices: Tuple[int, int, int, int]) -> float:
        """Calculate angle between two planes using cross products"""
        # Returns angle in radians [-π, π]

    @staticmethod
    def extract_torsion_angles(mol: Chem.Mol) -> List[TorsionInfo]:
        """Extract all torsion angles from a molecule conformation."""
        conf = mol.GetConformer()
        coords = conf.GetPositions()

        # Step 1: Find all rotatable bonds
        rotatable_bonds = MolecularTorsionAnalyzer.identify_rotatable_bonds(mol)

        torsion_info = []

        # Step 2: For each rotatable bond, compute its torsion angle
        for bond in rotatable_bonds:
            # Get the 4 atoms (a-i-j-b) defining the dihedral
            dihedral_atoms = MolecularTorsionAnalyzer.get_dihedral_atoms(mol, bond)

            if dihedral_atoms is None:
                continue

            # Calculate the actual angle value
            angle = MolecularTorsionAnalyzer.calculate_dihedral_angle(coords, dihedral_atoms)

            torsion_info.append(TorsionInfo(
                bond=bond,
                dihedral_atoms=dihedral_atoms,
                angle=angle
            ))

        return torsion_info
```

**Example: Ibuprofen Analysis**

```python
from rdkit import Chem
from rdkit.Chem import AllChem
from torsional_diffusion import MolecularTorsionAnalyzer

# Create molecule from SMILES
smiles = 'CC(C)Cc1ccc(cc1)C(C)C(=O)O'  # Ibuprofen
mol = Chem.MolFromSmiles(smiles)
mol = Chem.AddHs(mol)

# Generate 3D coordinates
AllChem.EmbedMolecule(mol)
AllChem.MMFFOptimizeMolecule(mol)

# Extract torsion angles
analyzer = MolecularTorsionAnalyzer()
torsion_info = analyzer.extract_torsion_angles(mol)

print(f"Found {len(torsion_info)} torsion angles")
for i, torsion in enumerate(torsion_info, 1):
    print(f"Torsion {i}: {torsion.dihedral_atoms} = {torsion.angle:.2f} rad")
```

Output
```
Found 4 torsion angles
Torsion 1: (0, 1, 3, 4) = -2.99 rad
Torsion 2: (1, 3, 4, 5) = -1.88 rad
Torsion 3: (6, 7, 10, 11) = 1.06 rad
Torsion 4: (7, 10, 12, 13) = 0.56 rad
```

**Efficiency gain**: Ibuprofen has 33 atoms × 3 coordinates = 99 dimensions in full 3D space, but only **4 torsional dimensions** → 24x reduction!

-----

### Component 2: EGNN (E(n)-Equivariant GNN)

### What is E(3) Equivariance and Why Does It Matter?

Before diving into the architecture, we must understand a critical concept: **E(3) equivariance**. The **E(3) group** (Euclidean Group in 3D) represents all possible **rotations, translations, and reflections** in 3D space—the fundamental **symmetries** of molecular structures.

Here's the problem: A molecule's chemical properties (like binding affinity) are **independent** of its orientation in space. Whether you rotate it $45^\circ$, translate it $10 \text{ Ångströms}$, or mirror it (reflection), it remains the same chemical entity. Standard neural networks, which treat coordinates as raw numbers, would learn *different* patterns for the same molecule in different poses, drastically wasting model capacity and requiring massive amounts of augmented data.

**Equivariance** solves this symmetry problem: if you transform the input coordinates, the network's output transforms in **exactly the same way**.

$$\text{If input } \mathbf{x} \text{ is transformed by } \mathcal{R} \text{, then output } f(\mathbf{x}) \text{ is transformed by } \mathcal{R}$$

This principle ensures:

1.  The network learns **physical and chemical laws**, not arbitrary coordinate systems.
2.  **Training efficiency** improves dramatically (one orientation generalizes to all).
3.  Predictions are **perfectly consistent** across any molecular pose.

-----

#### The EGNN Architecture: Invariance + Equivariance

The **E(n) Equivariant Graph Neural Network (EGNN)** achieves this by strictly separating the flow of information:

1.  **Invariant Quantities:** Features, distances, and angles (which **do not change** under E(3) transformations). These are used to compute the message strength.
2.  **Equivariant Quantities:** Coordinates and displacement vectors (which **rotate and translate** with the input). These are used to update positions.

#### The Core Mechanism: Scalar × Vector

The genius of the EGNN lies in how it updates the coordinates:

| Term | Quantity Type | Role |
| :--- | :--- | :--- |
| $\mathbf{h}_i, \mathbf{h}_j$ | Invariant | Node features |
| $||\mathbf{x}_i - \mathbf{x}_j||^2$ | Invariant | Squared distance |
| $\mathbf{x}_i - \mathbf{x}_j$ | Equivariant | Displacement vector |

The layer computes a **scalar weight** (or attention score) $\phi_x$ from the **invariant** information (features and distance, specifically the intermediate edge feature $\mathbf{e}_{ij}$). This scalar is then multiplied by the **equivariant** displacement vector:

$$\Delta \mathbf{x}_{ij} = \phi_x(\mathbf{e}_{ij}) \cdot (\mathbf{x}_i - \mathbf{x}_j)$$

The final coordinate update $\Delta \mathbf{x}_i$ is the aggregation (sum) of these messages:

$$\mathbf{x}_i' = \mathbf{x}_i + \sum_{j \in \mathcal{N}(i)} \Delta \mathbf{x}_{ij}$$

**Why This Works:** Multiplying an **equivariant vector** ($\mathbf{x}_i - \mathbf{x}_j$) by an **invariant scalar** ($\phi_x$) produces a new vector ($\Delta \mathbf{x}_{ij}$) that is *still* **equivariant**. This is the mathematical key to maintaining E(n) symmetry throughout the layer.

-----

#### Implementation Focus (Aligning with Code)

In the provided implementation, the coordinate update is separated from the PyTorch Geometric (`PyG`) `propagate` call, which is used only for **feature message aggregation**.

1.  The `message` function computes and returns the **invariant feature message** ($\mathbf{e}_{ij}$) while **caching** it for later use.
2.  The `forward` function then manually recomputes the displacement vectors ($\mathbf{x}_i - \mathbf{x}_j$) and uses the cached feature messages to calculate the coordinate weights $\phi_x$.
3.  Finally, it calculates the coordinate messages ($\Delta \mathbf{x}_{ij}$) and manually uses `index_add_` (or an equivalent manual approach) to **aggregate the coordinate updates** ($\Delta \mathbf{x}_i$).

The layer also includes a **residual connection** option, ensuring the layer's output feature $\mathbf{h}'$ can be defined as $\mathbf{h}' = \mathbf{h} + \text{NodeUpdate}(\mathbf{h}, \text{AggregatedMessages})$, provided the input and output dimensions match.

```python
# The message function computes and caches the feature message (Invariant)
def message(self, ..., pos_i, pos_j):
    # ... compute edge_message (Invariant)
    self.edge_features = edge_message # Cache this
    return edge_message 

# The forward function then handles coordinate updates (Equivariant)
def forward(self, h, pos, edge_index):
    # Aggregation of features happens here (via propagate)
    agg_h = self.propagate(...) 
    
    # Coordinate update is manual:
    pos_diff = pos_i - pos_j       # Equivariant Vector
    coord_weight = self.coord_mlp(self.edge_features) # Invariant Scalar
    coord_messages = pos_diff * coord_weight # Equivariant Vector
    
    # Manual aggregation of coordinate messages
    agg_pos_delta.index_add_(0, edge_index_i, coord_messages) 
    pos_updated = pos + agg_pos_delta
    
    # ... update features (h_updated)
    return h_updated, pos_updated
```

**Stacking Layers for Deep Geometry Learning:**

The `EGNN` wrapper class stacks these layers:

```python
class EGNN(nn.Module):
    """Multi-layer EGNN that refines both features and coordinates."""
    # ... __init__ includes an option for residual connections
    def forward(self, h, pos, edge_index):
        # ...
        for layer in self.layers:
            h, pos = layer(h, pos, edge_index)
        return h, pos
```

Each layer refines the geometric understanding, propagating information through the molecular graph while maintaining exact **E(n) equivariance**.

-----

### Component 3: Torsion Denoiser Network

The denoiser $\epsilon_\theta(\boldsymbol{\phi}_t, t)$ is the core of our diffusion model. Its job is to predict the noise added to the torsion angles ($\boldsymbol{\phi}_t$) at a given timestep ($t$), while simultaneously being aware of the full 3D molecular geometry.

#### Architecture and Initialization

The `TorsionDenoiser` is a modular network composed of specialized components for each type of input (time, features, geometry).

```python
class TorsionDenoiser(nn.Module):
    """
    Neural network for predicting noise in torsion angles.

    Architecture:
    1. Time embedding: encode timestep t
    2. Node embedding: encode atomic features + time
    3. EGNN: geometric message passing (maintains E(3) symmetry)
    4. Torsion prediction: per-torsion noise from 4-atom features
    """
    def __init__(self, atom_feature_dim, hidden_dim=128, num_egnn_layers=3):
        super().__init__()
        self.hidden_dim = hidden_dim
        # ... (MLPs and EGNN initialization code follows, using atom_feature_dim)
```

The network requires the **atomic feature dimension** (`atom_feature_dim`) at initialization to correctly size the first linear layer, making the model flexible for different input feature sets.

#### Time Embedding: Teaching the Network About Noise Levels

The network needs to know *how much* noise is currently in the input data. We use **sinusoidal positional embeddings**, a technique borrowed from Transformers, to encode the discrete timestep $t$ into a continuous, high-dimensional vector.

This encoding provides the network with two types of information:

  * **Low frequencies** capture the coarse progression (e.g., distinguishing early vs. late diffusion).
  * **High frequencies** help distinguish neighboring timesteps.

<!-- end list -->

```python
def get_time_embedding(self, timesteps):
    # ... (Sinusoidal encoding logic)
    # The output is passed through a simple MLP (time_mlp) for projection.
    return self.time_mlp(emb)
```

-----

#### Forward Pass: From Noisy Torsions to Noise Prediction

The forward pass is a sequence of highly optimized, fully **vectorized** operations that process all atoms and torsions in a single batch efficiently.

##### 1\. Feature Preparation and Time Broadcast (Efficient)

The input atomic features (`node_features`) are first embedded. Critically, the time embedding is then **vectorized and broadcast** to every atom belonging to its respective molecule:

```python
# 1. Embed time and node features
time_embed = self.get_time_embedding(t)    # [batch, hidden_dim]
h = self.node_embedding(node_features)     # [total_atoms, hidden_dim]

# 2. Vectorized Time Embedding Broadcast
# time_embed[batch] gathers the correct time embedding for each atom in O(1) time.
time_embed_per_node = time_embed[batch]
h = h + time_embed_per_node
```

This vectorized approach replaces the slow Python loop with a single tensor operation, ensuring high performance.

##### 2\. Geometry Learning via EGNN

The combined features (`h`) and the 3D coordinates (`pos`) are passed through the EGNN backbone.

```python
# 3. Run EGNN for geometry-aware node features
h, pos_updated = self.egnn(h, pos, edge_index)
# h now contains invariant features incorporating rich geometric information.
```

This step is where the $\text{E}(3)$ equivariance is enforced. The final features $\mathbf{h}$ are **invariant** to molecular rotation/translation but contain information about the relative positions and local chemistry of neighboring atoms.

##### 3\. Torsion Prediction and Circular Encoding

The final stage extracts the refined features necessary to predict the noise for each torsion angle.

1.  **Feature Gathering:** The network uses the `torsion_to_atoms` index map to gather the four corresponding refined features ($\mathbf{h}_{a}, \mathbf{h}_{b}, \mathbf{h}_{c}, \mathbf{h}_{d}$) for every torsion in the batch.
2.  **Circular Encoding:** The current noisy torsion angle ($\boldsymbol{\phi}_t$) is encoded using **$\sin$ and $\cos$** to respect its **periodicity** ($0^\circ = 360^\circ$). This prevents the network from perceiving large numerical differences between chemically identical angles (e.g., $359^\circ$ and $1^\circ$).

<!-- end list -->

```python
# Gather features from all 4 atoms defining each torsion
atom_features = torch.cat([h[torsion_to_atoms[:, i]] for i in range(4)], dim=-1)

# Circular encoding of current (noisy) torsion angles (torsions_t is now flat)
circular_enc = torch.stack([torch.sin(torsions_t), torch.cos(torsions_t)], dim=-1)

# Predict noise
torsion_input = torch.cat([atom_features, circular_enc], dim=-1)
noise_pred_flat = self.torsion_mlp(torsion_input).squeeze(-1)
```

The network returns a **flat tensor** of predicted noise values ($\epsilon_{pred}$), corresponding directly to the input `torsions_t`, simplifying loss calculation and avoiding complex padding/reshaping logic for molecules of different sizes.

---
### Component 4 : Torsional Diffusion Model (Detailed Implementation Guide)

The **Torsional Diffusion Model** serves as the complete engine for conformation generation, integrating the $\text{E}(3)$-Equivariant Graph Neural Network (EGNN) denoiser ($\boldsymbol{\epsilon}_\theta$) with the mathematical framework of **Denoising Diffusion Probabilistic Models (DDPM)**. This implementation follows modern best practices for batched graph data and advanced DDPM sampling.

---

#### Class Initialization and Buffer Registration

`__init__`:
```python
# Noise schedule parameters
self.register_buffer('betas', self.cosine_schedule(num_timesteps))
self.register_buffer('alphas', 1 - self.betas)
self.register_buffer('alpha_bars', torch.cumprod(self.alphas, dim=0))
```

The model pre-computes and registers three critical tensors as **buffers** (not trainable parameters, but part of the model state):

1. **$\boldsymbol{\beta}_t$** (`self.betas`): Noise schedule values for each timestep $t \in [0, T-1]$
2. **$\boldsymbol{\alpha}_t = 1 - \boldsymbol{\beta}_t$** (`self.alphas`): Complementary noise retention coefficients
3. **$\bar{\boldsymbol{\alpha}}_t = \prod_{i=1}^t \boldsymbol{\alpha}_i$** (`self.alpha_bars`): Cumulative product enabling direct sampling at any timestep

Using `register_buffer` ensures these tensors move to the correct device (CPU/GPU) with the model and are saved/loaded with checkpoints, but don't receive gradients during training.

---

#### Noise Schedule: Cosine Schedule Implementation

Why Cosine Schedule?

`cosine_schedule`:
```python
def cosine_schedule(self, timesteps: int, s: float = 0.008) -> torch.Tensor:
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)
```

The **cosine schedule** is superior to the original linear schedule for two key reasons:

1. **Preserves Structure Early**: Compute $\bar{\alpha}_t = \cos^2\left(\frac{t/T + s}{1 + s} \cdot \frac{\pi}{2}\right)$, which decreases slowly at first, adding minimal noise in early timesteps and retaining molecular structure.

2. **Efficient Noise Addition**: The schedule accelerates noise addition in later timesteps, ensuring the data quickly reaches pure noise by $t=T$.

clip $\beta_t$ to $[0.0001, 0.9999]$ for numerical stability, preventing degenerate cases where noise is too small (no learning) or too large (complete information loss).

**Reference**: Nichol & Dhariwal, "Improved Denoising Diffusion Probabilistic Models" (2021)

---

#### Forward Diffusion Process


`add_noise_to_torsions`:

The forward process corrupts clean torsion angles $\boldsymbol{\phi}_0$ with Gaussian noise according to:

$$\boldsymbol{\phi}_t = \sqrt{\bar{\boldsymbol{\alpha}}_t} \boldsymbol{\phi}_0 + \sqrt{1 - \bar{\boldsymbol{\alpha}}_t} \boldsymbol{\epsilon}, \quad \boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$$


```python
def add_noise_to_torsions(
    self,
    torsions_0: torch.Tensor,  # [total_torsions]
    t: torch.Tensor            # [total_torsions]
) -> Tuple[torch.Tensor, torch.Tensor]:
    # Sample random noise
    noise = torch.randn_like(torsions_0)
    
    # Get alpha_bar values for each timestep
    alpha_bar_t = self.alpha_bars[t]
    
    # Compute coefficients
    sqrt_alpha_bar_t = torch.sqrt(alpha_bar_t)
    sqrt_one_minus_alpha_bar_t = torch.sqrt(1 - alpha_bar_t)
    
    # Standard DDPM forward process
    torsions_t = sqrt_alpha_bar_t * torsions_0 + sqrt_one_minus_alpha_bar_t * noise
    
    # Wrap angles to [-π, π]
    torsions_t = torch.atan2(torch.sin(torsions_t), torch.cos(torsions_t))
    
    return torsions_t, noise
```

**Key Implementation Details**

**Flattened Data Structure**:
- Input `torsions_0` has shape `[total_torsions]` (all torsions from all molecules concatenated)
- Input `t` has shape `[total_torsions]` (one timestep per torsion)
- The indexing `self.alpha_bars[t]` performs **advanced indexing**, fetching the correct $\bar{\alpha}_t$ value for each torsion's timestep
- This enables efficient batching of molecules with different numbers of torsions

**Circular Wrapping**:
- Torsion angles are **circular quantities** (e.g., 179° and -179° are nearly identical)
- After adding Gaussian noise (which doesn't respect periodicity), we must wrap angles back to $[-\pi, \pi]$
- `torch.atan2(sin(x), cos(x))` is the numerically stable way to perform this wrapping
- This is a **simplification** compared to proper circular diffusion (von Mises distributions), but works well in practice

---

#### Training Step: The Simple Loss Objective


`training_step`:

The DDPM training objective is remarkably simple—predict the noise that was added:

$$\mathcal{L}_{\text{simple}} = \mathbb{E}_{\boldsymbol{\phi}_0, \boldsymbol{\epsilon}, t} \left[ \left\| \boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta(\boldsymbol{\phi}_t, t) \right\|^2 \right]$$

Where:
- $\boldsymbol{\phi}_0$: Clean torsion angles (ground truth)
- $\boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$: Random noise sampled once
- $t \sim \text{Uniform}(0, T)$: Random timestep
- $\boldsymbol{\epsilon}_\theta$: EGNN denoiser network


```python
def training_step(
    self,
    clean_torsions_flat: torch.Tensor,  # [total_torsions]
    # ... other args ...
    batch: torch.Tensor,                # [total_atoms]
    torsion_batch_idx: torch.Tensor     # [total_torsions]
) -> torch.Tensor:
    # Validation
    mol_batch_size = batch.max().item() + 1
    assert torsion_batch_idx.max().item() + 1 == mol_batch_size
    assert len(torsion_batch_idx) == total_torsions
    
    # Sample random timesteps per MOLECULE
    t_mol = torch.randint(0, self.num_timesteps, (mol_batch_size,), device=device)
    
    # Broadcast timestep for each TORSION
    t = t_mol[torsion_batch_idx]  # [total_torsions]
    
    # Add noise to torsions
    noisy_torsions, true_noise = self.add_noise_to_torsions(clean_torsions_flat, t)
    
    # Predict noise using EGNN denoiser
    predicted_noise = self.denoiser(
        torsions_t=noisy_torsions,
        t=t_mol,  # Pass molecular timesteps [batch_size]
        # ... other args ...
    )
    
    # Compute loss
    loss = F.mse_loss(predicted_noise, true_noise)
    
    return loss
```

#### Robust Batching Strategy

**Why Two Batch Indices?**

The code uses PyTorch Geometric's flat batching convention, which requires tracking two separate mappings:

1. **`batch`**: Maps **atoms** to molecules
   - Shape: `[total_atoms]`
   - Example: `[0, 0, 0, 1, 1, 1, 1]` means atoms 0-2 belong to molecule 0, atoms 3-6 to molecule 1

2. **`torsion_batch_idx`**: Maps **torsions** to molecules
   - Shape: `[total_torsions]`
   - Example: `[0, 0, 1, 1, 1]` means torsions 0-1 belong to molecule 0, torsions 2-4 to molecule 1

**Timestep Broadcasting**:
```python
t_mol = torch.randint(0, self.num_timesteps, (mol_batch_size,), device=device)
t = t_mol[torsion_batch_idx]
```

- Sample **one timestep per molecule** (not per torsion), as all torsions in a molecule share the same diffusion timestep
- Use `torsion_batch_idx` to broadcast: if molecule 0 gets $t=500$, all torsions belonging to molecule 0 get $t=500$
- This ensures the denoiser receives consistent temporal information

**Validation**:
The assertions catch common bugs:
- Mismatched batch sizes between atoms and torsions
- Incorrect tensor shapes that would cause silent failures

---

#### Reverse Process: Generating New Conformations

`generate_torsions`:

The reverse process samples clean torsion angles from pure noise through iterative denoising:

$$\boldsymbol{\phi}_{t-1} = \boldsymbol{\mu}_\theta(\boldsymbol{\phi}_t, t) + \boldsymbol{\sigma}_t \mathbf{z}, \quad \mathbf{z} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$$

Where the **denoised mean** is:

$$\boldsymbol{\mu}_{\theta}(\boldsymbol{\phi}_t, t) = \frac{1}{\sqrt{\alpha_t}} \left( \boldsymbol{\phi}_t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}} \boldsymbol{\epsilon}_\theta(\boldsymbol{\phi}_t, t) \right)$$

And the **posterior variance** is:

$$\boldsymbol{\sigma}_t^2 = \beta_t \cdot \frac{1 - \bar{\alpha}_{t-1}}{1 - \bar{\alpha}_t}$$



```python
@torch.no_grad()
def generate_torsions(
    self,
    # ... args ...
    total_torsions: int
) -> torch.Tensor:
    # Validation
    assert batch.max().item() == 0, "Generation only supports single molecule"
    assert torsion_batch_idx.max().item() == 0, "All torsions must belong to molecule 0"
    assert len(torsion_batch_idx) == total_torsions, "Mismatch in torsion count"
    
    # Initialize from random noise
    torsions = torch.randn(total_torsions, device=device) * math.pi
    
    # Reverse diffusion process
    for t in reversed(range(self.num_timesteps)):
        t_tensor = torch.tensor([t], device=device)  # Line 256
        
        # Predict noise
        predicted_noise = self.denoiser(...)
        
        # Get schedule parameters
        alpha_t = self.alphas[t]
        alpha_bar_t = self.alpha_bars[t]
        beta_t = self.betas[t]
        
        # Compute denoised mean
        torsions = (1 / torch.sqrt(alpha_t)) * (
            torsions - (beta_t / torch.sqrt(1 - alpha_bar_t)) * predicted_noise
        )
        
        # Wrap to [-π, π]
        torsions = torch.atan2(torch.sin(torsions), torch.cos(torsions))
        
        # Add noise for all steps except the last
        if t > 0:
            noise = torch.randn_like(torsions)
            
            # Compute posterior variance
            alpha_bar_t_prev = self.alpha_bars[t - 1]
            posterior_variance = beta_t * (1 - alpha_bar_t_prev) / (1 - alpha_bar_t)
            posterior_variance = torch.clamp(posterior_variance, min=1e-20)
            
            sigma_t = torch.sqrt(posterior_variance)
            torsions = torsions + sigma_t * noise
            
            # Wrap again
            torsions = torch.atan2(torch.sin(torsions), torch.cos(torsions))
    
    return torsions
```

#### Critical Implementation Details

**1. Validation for Single-Molecule Mode**:
```python
assert batch.max().item() == 0, "Generation only supports single molecule"
assert torsion_batch_idx.max().item() == 0, "All torsions must belong to molecule 0"
```
- Generation is designed for **one molecule at a time** (batch size = 1)
- All atom and torsion batch indices must be 0
- For multiple molecules, call this function in a loop

**2. Initialization from Noise**:
```python
torsions = torch.randn(total_torsions, device=device) * math.pi
```
- Start from $\mathcal{N}(0, \pi^2 \mathbf{I})$ rather than $\mathcal{N}(0, \mathbf{I})$
- Scaling by $\pi$ ensures initial noise covers the full torsion angle range $[-\pi, \pi]$

**3. Denoised Mean Calculation**:
The formula directly implements the DDPM posterior mean:
$$\boldsymbol{\mu}_\theta = \frac{1}{\sqrt{\alpha_t}} \left( \boldsymbol{\phi}_t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}} \boldsymbol{\epsilon}_\theta \right)$$

This is mathematically equivalent to predicting $\boldsymbol{\phi}_0$ and interpolating, but predicting $\boldsymbol{\epsilon}$ is empirically more stable.

**4. Posterior Variance**:
```python
alpha_bar_t_prev = self.alpha_bars[t - 1]
posterior_variance = beta_t * (1 - alpha_bar_t_prev) / (1 - alpha_bar_t)
posterior_variance = torch.clamp(posterior_variance, min=1e-20)
```

This is the **theoretically correct** posterior variance from the DDPM paper, derived from Bayes' rule:

$$p({\boldsymbol{\phi}}_{t-1} | \boldsymbol{\phi}_t, \boldsymbol{\phi}_0) = \mathcal{N}(\boldsymbol{\mu}_q, \sigma_q^2 \mathbf{I})$$

Where: $\sigma_q^2 = \frac{1 - \bar{\alpha}_{t-1}}{1 - \bar{\alpha}_t} \cdot \beta_t$

**Common mistake**: Using $\sigma_t = \sqrt{\beta_t}$ (the prior variance) instead of the posterior variance. This leads to:
- Over-smoothed samples (too little noise)
- Reduced diversity in generated conformations
- Slower convergence during sampling

The `torch.clamp(min=1e-20)` prevents division by zero when $\bar{\alpha}_t \approx 1$ in early timesteps.

**5. Dual Wrapping**:
```python
torsions = torch.atan2(torch.sin(torsions), torch.cos(torsions))  # After mean
# ... add noise ...
torsions = torch.atan2(torch.sin(torsions), torch.cos(torsions))  # After noise
```

Angles are wrapped **twice per iteration**:
- After computing the mean: Ensures the denoised estimate stays in $[-\pi, \pi]$
- After adding stochastic noise: Corrects angles that may have exceeded the range

This dual wrapping is essential for maintaining valid angular values throughout the sampling process.

**6. No Noise at Final Step**:
```python
if t > 0:
    # Add noise
```
At $t=0$ (final step), we return the deterministic mean $\boldsymbol{\mu}_\theta$ without adding noise, giving us the final clean sample.

---

#### Summary: Key Design Decisions

| Aspect | Implementation Choice |  Rationale |
|--------|----------------------|-----------------|-----------|
| **Noise Schedule** | Cosine schedule | Better structure preservation than linear |
| **Data Format** | Flat tensors `[total_torsions]` | Handles variable-length molecules naturally |
| **Timestep Sampling** | Per-molecule, then broadcast | All torsions in a molecule share one timestep |
| **Circular Handling** | Post-hoc wrapping with `atan2` | Simpler than von Mises, works well in practice |
| **Training Loss** | Simple MSE on noise | Empirically better than predicting $\boldsymbol{\phi}_0$ |
| **Sampling Variance** | Correct posterior variance | Theoretically sound, prevents over-smoothing |
| **Validation** | Explicit assertions | Catches bugs early, enforces single-molecule generation |

This implementation provides a **mathematically rigorous** and **computationally efficient** foundation for torsional diffusion modeling in drug discovery applications.


-----

### Component 5: Training and Generation Pipeline

#### Data Preparation: `smiles_to_graph_data`

The `smiles_to_graph_data` function now includes type hints and a comprehensive docstring detailing the exact structure of the returned PyTorch Geometric `Data` object. It also explicitly shows the steps for feature and graph construction.

```python
def smiles_to_graph_data(
    smiles: str,
    add_hydrogens: bool = True
) -> Tuple[Data, Chem.Mol, List[TorsionInfo]]:
    """
    Convert SMILES string to PyTorch Geometric Data object with torsion info.

    This function:
    1. Parses SMILES into RDKit molecule
    2. Optionally adds hydrogens (important for complete structure)
    3. Generates 3D coordinates
    4. Extracts torsion angles
    5. Creates graph representation for EGNN

    Args:
        smiles: SMILES string representation of molecule
        add_hydrogens: Whether to add explicit hydrogens (recommended)

    Returns:
        data: PyTorch Geometric Data object with:
            - x: Node features [num_atoms, feature_dim]
            - edge_index: Graph connectivity [2, num_edges]
            - pos: 3D coordinates [num_atoms, 3]
            - torsion_angles: Ground truth torsions [num_torsions]
            - torsion_to_atoms: Atom indices [num_torsions, 4]
        mol: RDKit molecule object
        torsion_info: List of TorsionInfo objects
    """
    # Parse and add hydrogens
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    if add_hydrogens:
        mol = Chem.AddHs(mol)

    # Generate 3D structure: Fixed the exclusion of 'randomSeed' for diversity
    AllChem.EmbedMolecule(mol)
    AllChem.MMFFOptimizeMolecule(mol)  # Energy minimization

    # Explicitly extract features, positions, and edge index
    # Feature extraction (e.g., one-hot atomic number) is now explicitly defined.
    # [Code for feature (x), position (pos), and edge_index extraction]

    # Extract torsion information
    analyzer = MolecularTorsionAnalyzer()
    torsion_info = analyzer.extract_torsion_angles(mol)

    # Convert to tensors
    torsion_angles = torch.tensor([t.angle for t in torsion_info], dtype=torch.float32)
    torsion_to_atoms = torch.tensor([list(t.dihedral_atoms) for t in torsion_info], dtype=torch.long)

    # Create PyG Data object
    return Data(x=x, edge_index=edge_index, pos=pos,
                torsion_angles=torsion_angles,
                torsion_to_atoms=torsion_to_atoms), mol, torsion_info
```

-----

#### Complete Training Loop: `train_torsional_diffusion`

The training process is now encapsulated in a dedicated function, `train_torsional_diffusion`. **Crucially, the batching logic for torsions has been corrected** to match the common implementation of models like Torsional Diffusion, which often expects *flat* torsion tensors and separate `torsion_batch_idx` tensors.

```python
def train_torsional_diffusion(
    dataset: List[Tuple[Data, Chem.Mol, str]],
    atom_feature_dim: int= 128,
    num_epochs: int = 50,  # Reduced for demo
    lr: float = 1e-4,
    hidden_dim: int = 64,
    num_timesteps: int = 100  # Reduced for demo
):
    """Train the torsional diffusion model on a dataset."""
    # ... setup and model creation ...

    for epoch in range(num_epochs):
        total_loss = 0.0
        num_processed = 0 # Track actual number of molecules processed

        for data, mol, smiles in dataset:
            # Skip molecules with no torsions
            if len(data.torsion_angles) == 0:
                continue

            optimizer.zero_grad()

            # FIX 2: Prepare atom batch indices (all zeros for single molecule)
            batch = torch.zeros(data.x.shape[0], dtype=torch.long)

            # FIX 3: Create torsion batch indices (all zeros for single molecule)
            num_torsions = len(data.torsion_angles)
            torsion_batch_idx = torch.zeros(num_torsions, dtype=torch.long)

            # FIX 4: Use flat torsion tensor
            # training_step expects [total_torsions], not [1, num_torsions]
            clean_torsions_flat = data.torsion_angles  # [num_torsions]

            # Training step with corrected arguments
            loss = model.training_step(
                clean_torsions_flat=clean_torsions_flat, # FIX: Renamed parameter
                node_features=data.x,
                pos=data.pos,
                edge_index=data.edge_index,
                torsion_to_atoms=data.torsion_to_atoms,
                batch=batch,
                torsion_batch_idx=torsion_batch_idx  # FIX: Added missing parameter
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            num_processed += 1

        # FIX 5: Use num_processed for accurate average loss calculation
        if num_processed > 0:
            avg_loss = total_loss / num_processed
        else:
            avg_loss = 0.0

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, Molecules: {num_processed}")

    # ... return model ...
```

-----

#### Generating New Conformations: `generate_conformer`

The generation logic is now in a reusable function, `generate_conformer`, which correctly sets the model to evaluation mode (`model.eval()`) and passes the necessary `torsion_batch_idx` and `total_torsions` parameters to the model's generation method.

```python
@torch.no_grad()
def generate_conformer(
    model: TorsionalDiffusionModel,
    data: Data,
    mol: Chem.Mol
) -> Tuple[torch.Tensor, np.ndarray]:
    """
    Generate a new molecular conformer using the trained diffusion model.
    """
    model.eval()

    # Prepare input, including torsion batch index
    batch = torch.zeros(data.x.shape[0], dtype=torch.long)
    num_torsions = len(data.torsion_angles)
    torsion_batch_idx = torch.zeros(num_torsions, dtype=torch.long)

    # Generate torsions with corrected arguments
    generated_torsions = model.generate_torsions(
        node_features=data.x,
        pos=data.pos,
        edge_index=data.edge_index,
        torsion_to_atoms=data.torsion_to_atoms,
        batch=batch,
        torsion_batch_idx=torsion_batch_idx,
        total_torsions=num_torsions
    )

    return generated_torsions.squeeze().cpu(), data.torsion_angles.numpy()
```

-----

#### Main Execution and Summary

The final section is replaced with a complete execution block (`if __name__ == "__main__":`) demonstrating the full pipeline on an example dataset of 8 molecules, including Ibuprofen and Caffeine. This section now includes:

1.  **Dataset Preparation** (calling `prepare_dataset`).
2.  **Torsion Statistics** (printing atom/torsion counts for molecules).
3.  **Model Training** (calling `train_torsional_diffusion`).
4.  **Conformer Generation** (calling `generate_conformer` and printing results **in degrees** for better interpretability).

**Example Output from Main Execution (Simulated):**

```
# ... processing and training logs ...

STEP 3: Generating new conformers
----------------------------------------------------------------------

1. CC(C)Cc1ccc(cc1)C(C)C
   Original torsions (deg): [ 14.5, -92.1,  88.3, 175.0]
   Generated torsions (deg): [ 17.2, -95.0,  89.1, 178.1]
   Difference (deg): [ 2.7, -2.9,  0.8,  3.1]

2. CN1C=NC2=C1C(=O)N(C(=O)N2C)C
   No rotatable bonds - skipping

3. CC(=O)Oc1ccccc1C(=O)O
   Original torsions (deg): [ 45.6, -11.2]
   Generated torsions (deg): [ 47.1, -10.5]
   Difference (deg): [ 1.5,  0.7]
```


#### Efficiency Gains

This section remains, highlighting the key advantage of the torsional diffusion approach.

For Ibuprofen (C₁₃H₁₈O₂):

  - **Full 3D space**: 33 atoms $\times$ 3 coords = **99 dimensions**
  - **Torsional space**: 4 rotatable bonds = **4 dimensions**
  - **Reduction**: 24$\times$ fewer dimensions\!
  - **Sampling speed**: $\sim$100$\times$ faster than GeoDiff (5-20 steps vs. 5000 steps)

This demonstrates why torsional diffusion is the current state-of-the-art for conformer generation: by focusing on the truly flexible degrees of freedom and using SE(3)-equivariant architectures, it achieves both **accuracy** and **efficiency**.

---

## 6. Applications and Practical Considerations

### 6.1 When to Use Diffusion Models

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

## 7. Summary and Next Steps

### What We Covered

In this blog, we explored diffusion models for molecular generation:

1. **Core concepts**: Forward/reverse processes, noise schedules, training objectives
2. **3D generation**: GeoDiff (full coordinates) vs. Torsional Diffusion (efficient)
3. **Implementation**: Complete torsional diffusion from scratch with EGNN backbone
4. **Applications**: State-of-the-art methods like DiffSBDD for structure-based drug design

**Key takeaways:**
- Diffusion models achieve superior sample quality and diversity
- Torsional diffusion reduces dimensionality by 10-1000x (focus on flexible degrees of freedom)
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

[^3]: Satorras, Victor Garcia, Emiel Hoogeboom, and Max Welling. "E(n) equivariant graph neural networks." *International Conference on Machine Learning*. PMLR, 2021.

[^4]: Ho, Jonathan, Ajay Jain, and Pieter Abbeel. "Denoising diffusion probabilistic models." *Advances in Neural Information Processing Systems* 33 (2020): 6840-6851.

[^5]: Schneuing, Arne, et al. "Structure-based drug design with equivariant diffusion models." *arXiv preprint arXiv:2210.13695* (2022).

[^6]: Guan, Jiaqi, et al. "3D equivariant diffusion for target-aware molecule generation and affinity prediction." *arXiv preprint arXiv:2303.03543* (2023).

[^7]: Xu, Minkai, et al. "Geodiff: A geometric diffusion model for molecular conformation generation." arXiv preprint arXiv:2203.02923 (2022).