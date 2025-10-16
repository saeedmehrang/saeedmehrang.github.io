---
title: "Computational Drug Discovery Part 3: AlphaFold and the Protein Structure Prediction Revolution"
date: 2025-10-13
draft: false
summary: "How DeepMind's AlphaFold2 solved the 50-year grand challenge in biology -- the protein folding problem -- using transformers, evolutionary information, and geometric reasoning and what it means for drug discovery."
tags: ["Computational Drug Discovery", "AlphaFold", "Deep Learning", "Protein Structure", "Transformers", "Machine Learning"]
series_order: 3
series: ["Computational Drug Discovery"]
showToc: true
disableAnchoredHeadings: false
cover:
  image: "cover.png"
  image_alt: "alphafold overview"
---

| Characteristic | Detail |
| :--- | :--- |
| **Estimated Reading Time** | 45-60 minutes |
| **Technical Level** | Advanced (requires understanding of deep learning, basic chemistry) |
| **Prerequisites** | [Blogs 1 and 2](../) in this series required |


## 1. Introduction

### The 50-Year Major Problem

In 1972, Christian Anfinsen won the **Nobel Prize** in Chemistry for a surprisingly straightforward idea: **a protein's chain of amino acids holds all the instructions needed to decide its 3D structure**. This rule, called **Anfinsen's dogma**, suggested something important: if we know a protein's chain, we should be able to guess how it folds into its useful 3D form.



This possibility was exciting. The truth, however, was upsetting. For the next **50 years**, despite huge computer and lab efforts, reliably guessing **protein structures** from their chains stayed one of biology's major challenges. Scientists could find structures in the lab using methods like **X-ray crystallography** or **cryo-electron microscopy**, but these techniques were **slow**, **costly**, and often didn't work. Computer guessing methods made small steps forward but were still far from the correct level needed for real-world use.


### AlphaFold 1's First Success

Then, in **2018**, the landscape began to shift. DeepMind entered the **Critical Assessment of protein Structure Prediction (CASP13)** competition with their first system, **AlphaFold 1**. This version used a **deep neural network** to predict the **distances between pairs of amino acids**. This distance information was then used by a classic, physics-based method to find the final 3D shape. AlphaFold 1 performed better than all other groups in the most difficult category, **setting a new standard** for accuracy and showing that machine learning was the key to future progress. However, it still **lacked the precision** that structural biologists needed to truly trust the computer models.


### The Breakthrough

Then, in November 2020, everything became different.

At the 14th CASP competition, **CASP14**, DeepMind introduced the completely redesigned **AlphaFold2**. It achieved an average accuracy of **92.4 GDT** (Global Distance Test) across all tests, basically matching **lab-test accuracy**. The second-best contestant scored only about 75 GDT. The science world was shocked. Computer biologists called it "amazing," "a huge success," and said it solved a problem many thought would take many more years.

Demis Hassabis, John Jumper, and David Baker won the 2024 Nobel Prize in Chemistry for their work on AI for protein structure prediction and design. Hassabis and Jumper, of Google DeepMind, won for developing AlphaFold, while Baker was recognized for his contributions to computational protein design.

### Why This Matters for Drug Discovery

Recall from our first blog in this series: **structure determines function**. A protein's 3D shape defines its binding sites the pockets and surfaces where drug molecules can attach and modulate the protein's activity. Without knowing a protein's structure, designing drugs that specifically target it is like trying to design a key without seeing the lock.

Before AlphaFold, structure determination was a major bottleneck:
- **Experimental methods** (X-ray crystallography, NMR, cryo-EM) take months to years per protein
- **Costs** range from $ 50,000 to 200,000 per structure
- **Success rate** is far from guaranteed many proteins resist crystallization or are too flexible for structure determination
- **Coverage**: Only about 200,000 protein structures had been experimentally solved out of roughly 200 million known protein sequences about 0.1% coverage

AlphaFold's impact was immediate and dramatic. Within months of CASP14, DeepMind and EMBL-EBI released the AlphaFold Database containing predicted structures for over 200 million proteins essentially covering all known proteins in public databases. Structural coverage went from 0.1% to over 50% overnight.

For drug discovery, this means:
- Targets that were once "impossible to drug" now have predicted structures, letting us use **structure-based design**.
- Proteins linked to **rare diseases** that had no lab structures can now be studied.
- We can understand **drug resistance** at the structural level.
- Improving a drug candidate can be guided by **structural insights**.
- The time it takes to get from choosing a target to starting structure-based design **shrunk from years to just days**.


### What We'll Cover Next

In this article, we will look at:
1. **Why guessing protein shape is so hard:** The challenge of having too many possible shapes (Levinthal's paradox) and the search for the lowest energy shape.
2. **How life uses evolution to store shape information:** The key idea that when amino acids change together, it shows they are touching in the 3D structure.
3. **The design of AlphaFold2:** How its parts (like the Evoformer, triangle attention, and Invariant Point Attention) pull out and process this evolutionary data.
4. **The results of this revolution:** What happened at CASP14, the impact of the AlphaFold Database, and how it changes drug discovery.
5. **What problems are still left:** Including protein movement, how things stick to proteins, and the limits of guessing a static (still) structure.

Let's begin by understanding why this problem took 50 years to solve.

---

## 2. The Protein Folding Problem

### 2.1 What Is Protein Folding?

**The Process:**

As we discussed in Blog 1, proteins are polymers linear chains of amino acids connected by peptide bonds (primary structure). When a ribosome synthesizes a protein, it emerges as an unfolded polypeptide chain. What happens next is remarkable: in milliseconds to seconds, the chain spontaneously folds into a specific, complex three-dimensional structure. This structure is:
- **Highly specific**: Each protein sequence folds to essentially the same structure every time
- **Reproducible**: The same sequence produces the same structure across millions of molecules
- **Spontaneous**: No external "instruction manual" is needed; it's a purely physical/chemical process
- **Functional**: The folded structure enables the protein's biological function

**Anfinsen's Dogma:**

Christian Anfinsen demonstrated this principle elegantly in the 1960s with ribonuclease A. He showed that:
1. Denaturing (unfolding) the protein with chemicals destroys its structure and function
2. Removing the denaturant allows the protein to spontaneously refold
3. The refolded protein is structurally and functionally identical to the original

The profound implication: **all information needed for folding is encoded in the amino acid sequence**. The protein doesn't need chaperones, templates, or external information to find its native structure. The folded state is simply the conformation with the **lowest free energy**, the thermodynamically most stable arrangement.

If the information is in the sequence, and folding is just energy minimization, why can't we just predict structures computationally?
"

### 2.2 Why Is Guessing the Shape So Hard?

**Levinthal's Puzzle:**

In 1969, Cyrus Levinthal pointed out a basic, confusing problem. Think about a small protein with 100 amino acids:
- Each part of the chain (residue) can settle into about **three stable positions** (or shapes).
- The total number of possible shapes is huge: $3^{100}$, or about $10^{48}$.
- If the protein tried out these shapes one after another at a very fast rate (1 nanosecond per shape), finding the single correct shape would take $10^{39}$ seconds — **far longer than the universe has existed** ($10^{17}$ seconds).

Yet, proteins fold up correctly in just a **few thousandths of a second (milliseconds)**.



The answer to this puzzle is that proteins **don't search randomly**. Folding happens by following specific steps, helped by local forces: parts of the chain that dislike water (hydrophobic residues) quickly gather in the center, simple shapes like twists and flat sections (**secondary structures**) form early on, and these pieces then come together to make the final complex shape (**tertiary structure**).

But this is the problem for computers trying to guess the shape: **we don't fully know what these folding steps are**. Without knowing the path, we are left trying to find the single best, most stable shape (the **lowest energy spot**) in a possible search space that is still **too big to search** completely.


**Energy Landscape Complexity:**

Protein stability arises from many weak interactions acting cooperatively:
- **Hydrogen bonds**: backbone and sidechain N-H and C=O groups form networks of hydrogen bonds
- **Hydrophobic effect**: nonpolar residues bury themselves in the core to avoid water
- **Van der Waals forces**: weak attractions between atoms in close proximity
- **Electrostatic interactions**: salt bridges between charged residues
- **Disulfide bonds**: covalent S-S bonds between cysteine pairs

These interactions are highly **cooperative**: removing one hydrogen bond can destabilize a whole region, causing cascading structural changes. The energy landscape has countless local minima partially folded states that are locally stable but not the global minimum. Optimization algorithms easily get trapped in these local minima.

Moreover, accurate energy calculation requires quantum-level precision for some interactions (electron distributions, polarization effects), but quantum calculations are computationally prohibitive for proteins with thousands of atoms.

**Long-Range Interactions:**

One of the most challenging aspects is that amino acids far apart in the sequence can be close in 3D space. Consider:
- Residue 50 might be a cysteine that forms a disulfide bond with a cysteine at position 200
- Residue 25 might have a charged sidechain that forms a salt bridge with residue 180
- A hydrophobic core might contain residues from positions 20, 50, 100, 150, 200, and 250

Traditional sequence models like Recurrent Neural Networks (RNNs) struggle with such long-range dependencies. Even LSTMs, designed to handle long-range correlations, have difficulty capturing interactions spanning hundreds of positions.


### 2.3 Old Ways to Guess Shape and Why They Failed

For over 50 years, scientists used different methods to guess protein structures. They had some success, but all hit major walls.

| Method | The Idea | Successes | Limitations |
| :--- | :--- | :--- | :--- |
| **Physics-Based (Molecular Dynamics)** | **Simulate** the protein's atomic movement and folding process using known physics rules (force fields). | Works for **very small** and fast-folding proteins (around 50 parts). | **Too slow**; takes months even on supercomputers to simulate just milliseconds of the actual folding process. **Rarely works** for proteins larger than 100 parts. |
| **Homology Modeling (Template-Based)** | Use the known structure of a **similar protein** as a guide (**template**). | Highly accurate when a protein is very similar to a known one (e.g., >50% sequence match). | **Fails** for proteins that are only distantly related (the "twilight zone," <30% match). **Can't predict new shapes**. |
| **Threading/Fold Recognition** | Try to **fit** the unknown protein sequence onto a library of all *known* protein shapes to find the best match. | Better than Homology Modeling for **distant relatives**. | **Limited** to shapes that are already known. **Can't predict new shapes**. |
| **De Novo Prediction (Ab Initio)** | Guess the shape from the sequence **without templates** using computer search algorithms and energy rules. | Can guess the shape of **small proteins** with moderate accuracy. | **Extremely slow** (days per protein). **Struggles** with large proteins. Needs perfect energy rules, which are **hard to create**. |

**The Assessment: CASP**

The **Critical Assessment of protein Structure Prediction (CASP)** contest, started in 1994, tested these methods blindly. Researchers submitted guesses for structures that were **about to be solved in a lab** but were not yet public.

For 25 years (CASP1 to CASP13), progress was **very slow**. Methods improved little by little, and guessing totally new shapes remained poor.

- **CASP13 (2018)**: **AlphaFold 1** showed the first major promise, beating all traditional methods on the toughest targets.
- **CASP14 (2020)**: **AlphaFold2** changed everything, achieving a median accuracy (92.4 GDT) that was **as good as lab methods**.

The 50-year-old difficulty was finally broken.

---

## 3. Evolution as a Structure Encoder

To understand how AlphaFold works, we first need to grasp a profound insight about evolution and protein structure.

### 3.1 The Key Insight: Co-Evolution

**Why Multiple Sequence Alignments (MSAs) Matter:**

Proteins don't evolve in isolation. Homologous proteins, those descended from a common ancestor, accumulate mutations over millions of years. Yet despite these sequence changes, the 3D structure is typically **conserved**. Why? Because structure is necessary for function, and natural selection eliminates mutations that destroy functional structures.

Here's the key observation: **When two amino acids are close in 3D space, mutations in one position often correlate with mutations in the other.**

**Co-Evolution Example:**

Imagine a protein where:
- Position 25 is on one alpha helix
- Position 100 is on a different helix
- These positions are far in sequence (75 residues apart) but close in 3D space (within 5 Angstroms)
- Position 25 has a negatively charged **glutamate (E)**
- Position 100 has a positively charged **lysine (K)**
- They form a salt bridge an electrostatic interaction stabilizing the structure

Now suppose a mutation occurs at position 25, changing E (negative) to **arginine R** (positive). This disrupts the salt bridge now both positions are positive and repel each other. The structure becomes unstable, and the organism might suffer reduced fitness.

But if a subsequent mutation at position 100 changes K (positive) to **glutamate E** (negative), the salt bridge is restored. The structure stabilizes, and function is recovered.

Over evolutionary time, we observe **correlated mutations**:
- Original: E(25) - K(100)
- Species 1: E(25) - K(100)
- Species 2: R(25) - E(100) [both changed]
- Species 3: R(25) - D(100) [both changed again, maintaining opposite charges]
- Species 4: K(25) - E(100) [both changed, charges swapped]

Positions 25 and 100 **co-evolve**: they mutate together more often than expected by chance.

**How MSAs Capture This:**

To detect co-evolution, we:
1. **Collect homologous sequences**: Search databases for proteins similar to our target (using BLAST, HHblits, or Jackhmmer)
2. **Align sequences**: Arrange them so equivalent positions line up
3. **Analyze covariation**: Look for positions that mutate together

Example MSA (simplified):

```
Position:      25  26  27 ... 100 101 102
Human:         E   L   A  ...  K   V   T
Mouse:         E   L   A  ...  K   V   T
Chicken:       E   L   S  ...  K   I   T
Fish:          D   L   S  ...  R   I   A
Bacteria:      D   L   T  ...  R   I   S
Yeast:         E   M   A  ...  K   V   T
Insect:        D   L   S  ...  R   I   A
Plant:         E   L   A  ...  K   V   S
```

Notice:
- Position 25: E or D (both negative)
- Position 100: K or R (both positive)
- When position 25 is D, position 100 is R (Fish, Bacteria, Insect)
- When position 25 is E, position 100 is K (Human, Mouse, Chicken, Yeast, Plant)

The pattern suggests these positions are **structurally coupled** likely close in 3D space.

### 3.2 Contact Prediction from MSAs

A **Multiple Sequence Alignment (MSA)** lists the amino acid chains of many related proteins, lined up to show which parts match.

**Contact prediction from MSA** is a smart method that uses this list to figure out which amino acids are physically touching in the protein's final 3D shape.

### Intuitive Definition

Imagine you have many versions of a phone book from different years and cities (your **MSA**). You notice that every time a person's last name changes at **Address A**, their spouse's last name changes at **Address B**.

* This **correlated change** (**co-evolution**) tells you that the people at **Address A** and **Address B** are **structurally connected** (they're a family).
* **Contact prediction** uses math to find these co-evolving pairs in a protein's MSA. Since evolution only allows changes at two distant points in the chain if they still work together to keep the structure stable, finding these pairs means those two points **must be touching** in the folded protein.

Essentially, **evolution acts as a statistical codebook**, and **contact prediction** is the key to reading that code to find the **3D neighborhood** of every amino acid. 


**Direct Coupling Analysis (DCA):**

Early statistical methods like DCA used mathematical techniques to:
- Analyze covariation patterns in MSAs
- Distinguish **direct interactions** (residues actually in contact) from **indirect correlations** (A affects B, B affects C, so A and C appear correlated but aren't directly coupled)
- Output a matrix of probabilities: P(contact between residues i and j)

DCA was revolutionary it showed that evolutionary information alone could predict 3D contacts without any physics-based calculations.

**Why This Works:**

Evolution acts as a statistical recorder of 3D structure:
- Residues in contact must maintain complementary properties (charge, size, hydrophobicity)
- Over millions of years and thousands of species, this leaves a statistical signature
- **Deep MSAs** (1000+ sequences) contain enough signal to reconstruct contact maps with surprising accuracy

For well-conserved protein families, DCA could predict 50-70% of contacts correctly enough to reconstruct approximate structures.

**The Leap to Deep Learning:**

In the early 2010s, researchers began applying neural networks to contact prediction from MSAs:
- Neural networks could learn complex, non-linear patterns that statistical methods missed
- They could integrate additional features (sequence profiles, predicted secondary structure)
- By 2018, deep learning contact predictors became state-of-the-art

The realization: **If we can predict contacts accurately, we can reconstruct 3D structures.** Distance geometry algorithms can convert contact maps to 3D coordinates. But there was a better way: predict the full structure end-to-end.

Enter AlphaFold.

---

## 4. AlphaFold2 Architecture

Now we arrive at the heart of the revolution: how AlphaFold2 actually works.

### 4.1 High-Level Overview

**The Pipeline in Order:**

1. **Input**: Protein sequence (e.g., 300 amino acids)
2. **MSA Construction**: Search sequence databases to find homologs, create multiple sequence alignment
3. **Evoformer**: Neural network processes the MSA to extract evolutionary and structural information
4. **Structure Module**: Converts abstract representations into 3D atomic coordinates
5. **Output**: Predicted 3D structure with per-residue confidence scores

**Key Innovations:**

- **End-to-end differentiable**: All components trained together from sequence to 3D coordinates
- **Not just contacts**: Predicts full atomic positions (backbone and sidechains)
- **Iterative refinement**: Uses recycling feeds predictions back as input to correct mistakes
- **Geometric reasoning**: Incorporates 3D geometric constraints directly into the architecture
- **Physical plausibility**: Triangle attention enforces consistency with 3D geometry

This is not an incremental improvement it's a fundamentally new approach.


### 4.2 Input Representations: Feeding the AlphaFold2 Engine

AlphaFold2's strength begins with how it prepares its input data. It transforms raw biological sequences and evolutionary information into carefully structured numerical representations (tensors) that the neural network (Evoformer) can efficiently process.

#### 1\. Primary Sequence Input (Query Protein)

The initial input is the **target protein sequence** we want to predict. This sequence is encoded in two ways:

  * **One-Hot Encoding:** Each amino acid in the sequence (e.g., `M A D L I...`) is converted into a vector where one position is '1' (indicating its type out of 20 possible amino acids) and others are '0'. For a sequence of length $L$, this creates an $L \times 20$ matrix.
  * **Positional Encodings:** To give the model a sense of order and relative position along the linear chain, additional vectors are added. These encodings help the model understand that residue 5 is adjacent to residue 6 but far from residue 100, which is crucial for how the attention mechanisms process the sequence. Positional encoding methods were original introduced in language modeling where Transformers were introduced. For an in depth learning of positional encoding/embedding methods, please see my other series of [blogs for language modeling](../../transformer-architectures/).

#### 2\. The Multiple Sequence Alignment (MSA): The Evolutionary Blueprint

The MSA is the **most crucial input**, acting as an evolutionary record that reveals which amino acids have co-evolved and, thus, are likely in contact in the 3D structure.

  * **Construction:** AlphaFold2 first performs a deep search across massive sequence databases (like UniRef, BFD, MGnify) to find hundreds, thousands, or even tens of thousands of **homologous proteins**—sequences that share a common evolutionary ancestor with the target protein.

  * **The Alignment Process & Gap Insertion:** In their raw form, these homologous sequences often have varying lengths due to **insertions** (extra amino acids) and **deletions** (missing amino acids) that occurred during evolution. To make these sequences comparable, a specialized **Multiple Sequence Alignment (MSA) algorithm** is used. This algorithm introduces **gap characters** (typically represented by a hyphen '-') into the sequences. This ensures that:

      * All **evolutionarily equivalent amino acids** (those descended from the same position in the ancestor protein) are lined up in the **same column**.
      * All sequences in the MSA are made to have the **exact same length**, which is the length of the longest resulting alignment (including all gaps).

  * **Tensor Representation:** Once aligned, the MSA is then structured into a 3D tensor with the shape: **\[$N\_{sequences} \times L\_{aligned} \times features$\]**.

      * $N_{sequences}$: The total number of homologous sequences found.
      * $L_{aligned}$: The uniform length of all aligned sequences (equal to the length of the query protein plus any introduced gaps).
      * $features$: Each aligned position in each sequence is encoded with information, including its amino acid type (one-hot encoded vector of length 20) and whether it's a gap.

  * **Depth and Diversity are Key:** The accuracy of AlphaFold2 correlates strongly with the **depth** (number of sequences) and **diversity** of the MSA. A deeper and more varied MSA provides a richer statistical signal for detecting those critical co-evolutionary patterns.

**Example MSA Tensor Representation (Simplified):**

Imagine we have our query protein and three homologous proteins.

**Original (unaligned) sequences:**

  * Query: `MAKVLIRPGFES` (length 12)
  * Homolog 1: `MAKILIRPGFESL` (length 13)
  * Homolog 2: `MAVLIPGFQSA` (length 11)
  * Homolog 3: `MAKLIRPFESLK` (length 13)

**After Multiple Sequence Alignment (with gaps for uniform length 16):**

```
Position:      1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16
Query Seq:     M  A  K  V  L  I  R  P  G  F  E  S  -  -  -  -
Homolog 1:     M  A  K  I  L  I  R  P  G  F  E  S  L  -  -  -
Homolog 2:     M  A  -  V  L  I  -  P  G  F  Q  S  A  -  -  -
Homolog 3:     M  A  K  -  L  I  R  P  -  F  E  S  -  L  K  -
```

This aligned MSA (conceptually) forms a 2D matrix of characters. For the Evoformer, it's converted into a 3D tensor, where each cell `[sequence_idx, residue_idx]` contains a feature vector describing that amino acid (or gap).

```
MSA Tensor [N_sequences, L_aligned, features]:
Dimensions: [4, 16, 22]  (e.g., 20 for one-hot AA, 1 for gap, 1 for sequence start/end)

[
  [ [M_feat], [A_feat], [K_feat], ..., [S_feat], [G_feat], [G_feat], [G_feat], [G_feat] ],  // Query (G_feat is Gap features)
  [ [M_feat], [A_feat], [K_feat], ..., [S_feat], [L_feat], [G_feat], [G_feat], [G_feat] ],  // Homolog 1
  [ [M_feat], [A_feat], [G_feat], ..., [S_feat], [A_feat], [G_feat], [G_feat], [G_feat] ],  // Homolog 2
  [ [M_feat], [A_feat], [K_feat], ..., [S_feat], [G_feat], [L_feat], [K_feat], [G_feat] ]   // Homolog 3
]
```

*(Where `M_feat`, `A_feat`, `K_feat`, `G_feat` (for gap), etc., are 22-dimensional vectors representing the amino acid or gap along with other characteristics.)*


#### 3. Initial Pair Representation: The Structural Canvas

The **Pair Representation** is a critical matrix initialized to encode relationships between **every pair of residues in the target protein**, forming the structural "canvas" that the Evoformer will iteratively refine.

* **Fixed Dimension $L \times L$ (Target Length Only):** The most important feature is its dimension: $\mathbf{L \times L}$, where $\mathbf{L}$ is the **Target Protein's physical length** (the number of amino acids in the query sequence). It **excludes** the gaps and insertions accounted for in $L_{aligned}$. This is because the matrix's ultimate purpose is to model the 3D structure of the single target protein, which has $L$ residues, not the evolutionary alignment, which has $L_{aligned}$ positions.
* **Purpose:** The $\mathbf{L \times L}$ matrix is designed to hold the final predictions for the geometry of the folded protein. Each entry $(i, j)$ in this matrix will eventually encode the predicted **distance** and **relative orientation** between residue $i$ and residue $j$ in the 3D structure.
* **Initial Features:** Before any evolutionary information is integrated, the matrix is initialized with basic, sequence-derived features for every pair $(i, j)$:
    * **Sequence Separation:** The absolute distance along the primary chain, $|i - j|$.
    * **Amino Acid Identity:** Information about the types of residues at position $i$ and position $j$.

The $\mathbf{L \times L}$ Pair Representation is the **structural prediction space**, while the $\mathbf{L_{aligned}}$ dimension of the MSA is the **evolutionary information source**. The Evoformer is designed to efficiently extract co-evolutionary signals from the large MSA tensor and project them onto the smaller, structure-focused $L \times L$ Pair Representation.

#### 4\. Template Structures (Optional Guidance)

While AlphaFold2 excels at *de novo* prediction (without templates), it can integrate known 3D structures for assistance.

  * **Role:** If the database search identifies experimentally determined structures (from the Protein Data Bank, PDB) of highly similar homologous proteins, these are provided as **structural templates**. They offer a valuable **geometric prior**—a rough initial hint of the target protein's likely overall shape.
  * **Independence:** A defining feature of AlphaFold2 is its ability to achieve high accuracy even when no suitable templates exist, demonstrating its profound understanding of structural biology derived from MSAs. Templates primarily serve as an accelerant or additional constraint rather than a necessity.


### 4.3 Evoformer: The Heart of AlphaFold2

The Evoformer is the core component of AlphaFold2, responsible for integrating the vast evolutionary data from the MSA with the geometric constraints of a protein structure. It learns to predict structure through a sophisticated two-track, iterative architecture.

The entire Evoformer consists of **48 stacked blocks**, with each block performing a cycle of feature refinement and information exchange between the two parallel tracks.

---

### Two Parallel Tracks

The Evoformer maintains two principal data structures that are continually refined:

| Track | Tensor Shape | Variable Clarification | Purpose |
| :--- | :--- | :--- | :--- |
| **1. MSA Representation** | $\mathbf{[N_{sequences} \times L_{aligned} \times d_{msa}]}$ | $L_{aligned}$: Length *with* gaps. | Encodes the evolutionary context. Captures which amino acids appear at which aligned position across all homologous sequences. |
| **2. Pair Representation** | $\mathbf{[L \times L \times d_{pair}]}$ | $L$: **Target protein's physical length** (no gaps). | Encodes the structural relationship. Represents the predicted distance and orientation between every pair of residues $(i, j)$ in the *target protein only*. Target protein is the Query protein in the tensor above. |

---

### Evoformer Block Components

Each of the 48 Evoformer blocks performs the following key steps:

#### A. Refinement of the MSA Representation (The Evolutionary Track)

1.  **MSA Row-wise Self-Attention:**
    * **Action:** Amino acids **within a single sequence (row)** attend to each other.
    * **Goal:** Helps model **long-range dependencies** along the protein chain in *that specific homolog* (e.g., residues 10 and 200 in Homolog 5 might be related).

2.  **MSA Column-wise Gated Self-Attention:**
    * **Action:** Sequences (rows) attend to each other **at a single, fixed sequence position (column)**.
    * **Goal:** Captures **evolutionary variation and conservation** across different species (e.g., is position 50 conserved as Glycine, or does it vary widely?). The use of a **gating mechanism** helps modulate which information is passed on.

3.  **Communication: Pair $\rightarrow$ MSA (Gated Extractor):**
    * **Action:** Information from the Pair Representation is used to enrich (or gate) the MSA representation.
    * **Goal:** Allows the network's current structural hypothesis (from the Pair track) to inform its interpretation of the evolutionary data. For instance, if the network believes residues $i$ and $j$ are close, it can focus the MSA on sequences that strongly support that hypothesis.

#### B. Refinement of the Pair Representation (The Structural Track)

4.  **Communication: MSA $\rightarrow$ Pair (Outer Product Mean):**
    * **Action:** The MSA tensor is processed to calculate the **outer product mean** across all sequences.
    * **Goal:** This is the primary step where the $\mathbf{L \times L}$ Pair Representation is updated using the co-evolutionary signal from the MSA. If positions $i$ and $j$ consistently co-vary (i.e., when $i$ changes, $j$ also changes in a related manner) across many homologous sequences, the Pair Representation is strengthened, indicating a likely physical contact.

5.  **Triangle Multiplicative Updates (The Geometric Constraint):**
    * **Innovation:** These are among the most critical innovations, designed to ensure **geometric consistency** in the predicted residue-residue relationships.
    * **Mechanism:** The updates enforce the **triangle inequality** from 3D space: if we have three residues $i, j, k$, the distance prediction for $d_{i,k}$ must be consistent with the predictions for $d_{i,j}$ and $d_{j,k}$.
    * **Components:** The Evoformer applies this using two mechanisms (often applied in both directions, "starting node" and "ending node"):
        * **Triangle Multiplicative Update:** Updates edge $(i, k)$ based on a multiplication of features from edges $(i, j)$ and $(j, k)$ for all intermediate residues $j$.
        * **Triangle Attention:** Updates the pair features using attention mechanisms constrained by the triangle geometry.
    * **Why it Matters:** This forces the network to learn physically plausible structures, avoiding geometrically impossible distance combinations.

#### C. Final Transition

6.  **Transition (Feedforward Layers):**
    * Standard, independent multi-layer perceptrons (MLPs) are applied to both the MSA and Pair representations, providing additional non-linear capacity to process the newly refined features before they are passed to the next Evoformer block.

This repeated, intertwined exchange of information over 48 blocks allows the Evoformer to transition from raw evolutionary sequence data to a highly refined, geometrically consistent prediction of residue-residue contacts.


**Why 48 Blocks?**

Information needs to propagate across long protein sequences:
- Early blocks capture local patterns (nearby residues, secondary structure)
- Middle blocks propagate information over medium ranges (domain structure)
- Late blocks refine long-range interactions and global consistency
- Deep stacking allows hierarchical feature learning

Each block refines the representations slightly. The deep architecture is necessary for learning complex structural patterns.


### 4.4 Structure Module: The 3D Translator

The Evoformer's job is to create **abstract numerical predictions** (high-dimensional embeddings) about which residues are close and how they are related. The **Structure Module** takes these abstract predictions and translates them into **actual 3D atomic coordinates**.

| Input | Output |
| :--- | :--- |
| Evoformer **Pair Representation** (predicted distances/orientations) | 3D Cartesian Coordinates ($x, y, z$) for every atom in the target protein. |

---

**From Abstract to 3D: Invariant Point Attention (IPA)**

The transition from abstract features to a physical 3D structure is the key challenge here, as the network must obey the laws of physics and geometry.

The Challenge of 3D Symmetry: A protein structure has **Euclidean Symmetry**, known formally as **SE(3) symmetry** (Special Euclidean group in 3D). This means:

* **Rotation:** If you rotate the entire protein in space, it is still the same physical structure.
* **Translation:** If you move the entire protein across the room, it is still the same physical structure.

A standard neural network might learn patterns that are dependent on the protein's arbitrary starting position or orientation. IPA solves this.

Invariant Point Attention (IPA): IPA is an attention mechanism specifically designed to be **geometry-aware**. It ensures the network's processing is **equivariant**:

> **Equivariance:** If you rotate or translate the input features (the abstract representations), the output coordinates will be rotated or translated **by the exact same amount**.

IPA achieves this by simultaneously attending to **both** abstract features and the current 3D positions:

1.  **Local Reference Frame:** Each residue is assigned its own small, independent **3D coordinate system** (an origin and three orthogonal axes). This allows the network to describe a neighboring residue's position **relative to itself**, independent of the global protein position.
2.  **Geometric Attention:** The attention mechanism calculates its Query, Key, and Value vectors by combining:
    * **Abstract Features:** The vector embeddings from the Evoformer.
    * **Geometric Features:** The current 3D position of the residues relative to the local reference frames.
3.  **Simultaneous Update:** The output of IPA simultaneously updates two things:
    * The **abstract features** (refining the representations).
    * The **3D coordinates** (refining the physical positions).

This integrated approach means the network doesn't have to re-learn physics; the IPA architecture **inherently respects** 3D geometry, dramatically speeding up the learning of accurate physical structures.



**Iterative Refinement and Recycling**

The Structure Module's task is too complex to be solved in one step. It relies on iterative processes at two different levels:

1. Iterative Refinement (8 Inner Steps)

The Structure Module runs **8 passes** internally, using the IPA block in each pass.

* It starts with a very rough structural guess (initialized from the Pair Representation).
* In each of the 8 passes, the IPA refines the coordinates.
* The final passes focus on precise details, such as rotating the chemical bonds around the backbone (**torsion angles**) and determining the position of the **sidechains** (the parts that give amino acids their identity).

2. Recycling (Outer Loop)

After the entire network (Evoformer + Structure Module) has run once, the process repeats in an outer loop called **Recycling**. AlphaFold2 typically performs **3 recycles** (meaning 4 total passes through the entire network).

* **Action:** The final predicted structure and its abstract representations (MSA and Pair) from the first pass are **fed back** into the network as new input features for the next run.
* **Why it Works:** This mimics the **iterative refinement** used in traditional scientific modeling (like molecular dynamics). The initial prediction, though imperfect, is far better than a random guess. By feeding this strong "prior knowledge" back in, the network can:
    * **Correct Mistakes:** Use the self-generated structure as a geometric constraint to clean up noisy signals in the MSA.
    * **Deepen Learning:** Allow the network to learn deeper features that depend on an already mostly folded structure.

Recycling acts as a form of **self-correction** and is critical for achieving the high accuracy seen in AlphaFold2's final predictions. 

## 4.5 Loss Functions and Training

### What Does AlphaFold2 Optimize?

AlphaFold2 is trained by minimizing a combined loss function, which is a weighted sum of several individual loss components. This multi-task approach guides the network to produce geometrically accurate structures while simultaneously learning from the underlying evolutionary and physical principles of proteins.

### Primary Loss: The Geometric Target

The main function that guides the refinement of the final 3D structure is the **Frame Aligned Point Error (FAPE)**.

  * **FAPE (Frame Aligned Point Error):** This is the primary loss component for the Structure Module. It quantifies the geometric error by comparing the predicted atomic coordinates to the true coordinates from experimental data.
  * **Key Innovation:** The core idea of FAPE is that it's calculated within local reference frames. For each residue, a coordinate system is defined by its rigid backbone atoms (**$N, C_α, C$**). The error is then measured by comparing the positions of all other atoms relative to these local frames.
  * **Why it's important:** By using local frames, the loss value is independent of the global position and orientation of the protein. This property, known as **SE(3) equivariance**, prevents the model from being penalized for trivial rotations or translations of the entire structure and forces it to focus only on learning the correct internal arrangement of the atoms.
  * **Scope:** FAPE is applied to all atoms, ensuring high-resolution accuracy for both the protein backbone and the sidechains.


### Auxiliary Losses: Guiding the Intermediate Representations

Auxiliary losses are applied to the intermediate outputs of the **Evoformer**. Their purpose is to ensure the MSA and Pair Representations contain high-quality, physically relevant information before they are fed to the Structure Module.

  * **Distogram Loss:**

      * **Target:** Applied to the Pair Representation.
      * **Mechanism:** This loss trains the network to predict a **binned probability distribution** of distances between the $C_α$ atoms of every pair of residues. Instead of predicting a single distance value, it predicts the likelihood that the distance falls into one of several predefined distance ranges.
      * **Goal:** It forces the model to learn the global distance geometry of the protein early in the process.

  * **Masked MSA Prediction:**

      * **Target:** Applied to the MSA Representation.
      * **Mechanism:** The network must predict the identity of amino acids that were deliberately hidden (masked) in the input MSA.
      * **Goal:** Similar to language models like BERT, this task compels the network to learn rich evolutionary relationships and co-evolutionary patterns from the context provided by the surrounding sequences.

  * **Predicted LDDT (pLDDT) Loss:**

      * **Target:** Applied to the final structure.
      * **Mechanism:** This loss trains the model to predict its own accuracy for each residue. The target metric, LDDT, measures how well the local atomic environment of a residue is predicted.
      * **Goal:** This makes the network "self-aware" of its confidence. It is trained to output a high pLDDT score for residues it predicts accurately and a low score for regions it is uncertain about, which is extremely useful for interpreting the final result.

  * **Predicted Aligned Error (PAE) Loss:**

      * **Target:** Also applied to the final structure.
      * **Mechanism:** This loss trains the network to predict the error in the relative position and orientation between pairs of residues. It outputs a 2D map showing the expected positional error (in Ångströms) between residue *i* and residue *j* if the structures are aligned on residue *i*.
      * **Goal:** This provides crucial information about the confidence in the predicted arrangement of domains or sub-structures relative to one another.

  * **Violation Loss:**

      * **Target:** Applied to the final structure.
      * **Mechanism:** This is a physics-based loss that adds a penalty for unrealistic bond lengths, incorrect stereochemistry (e.g., chirality), and steric clashes where atoms are too close together.
      * **Goal:** It ensures that the final predicted structure is not just geometrically close to the target but is also chemically and physically plausible.

### Training Data and Strategy

  * **Training Data:** The models were trained on experimentally determined protein structures sourced from the **Protein Data Bank (PDB)**.
  * **Self-Distillation:** A key innovation was to use an initial, trained model to make high-confidence structure predictions for millions of sequences from large databases (like **UniRef90**) that do not have experimental structures. The final AlphaFold2 models were then trained on a massive dataset combining the original experimental structures with this high-confidence predicted data, vastly expanding the knowledge base.

---

## 5. AlphaFold2 Performance and Impact

### 5.1 CASP14 Results

**The Benchmark:**

CASP14 (2020) featured about 100 protein targets:
- Predictors submit structures blindly before experimental structures are public
- Evaluated using **GDT (Global Distance Test)**: measures the percentage of residues predicted within certain distance thresholds of the true structure
  - GDT > 90: Competitive with experimental methods
  - GDT 50-90: Useful for some applications (homology modeling quality)
  - GDT < 50: Poor prediction

**AlphaFold2's Breakthrough Performance:**

- **Median GDT: 92.4** across all CASP14 targets
- **87 out of 100 targets** with GDT > 90
- **2/3 of predictions** achieved experimental accuracy

**What Changed:**

- **Pre-AlphaFold**: Accurate structure required months of experimental work, $50K-$200K, and often failed
- **Post-AlphaFold**: Structure prediction in hours on a single GPU with near-experimental accuracy
- **Democratization**: Anyone with a sequence and internet access can predict structures for free

### 5.2 AlphaFold Database

**The Moonshot:**

DeepMind partnered with EMBL-EBI to scale structure prediction to the entire known protein universe:

- **July 2021**: Released structures for human proteome (~20,000 proteins) + 20 model organisms
- **2022**: Expanded to **200+ million protein structures** covering UniProt (essentially all sequenced proteins)
- **Freely accessible**: [alphafold.ebi.ac.uk](https://alphafold.ebi.ac.uk)

**Impact on Science:**

- **Structural coverage**: From ~0.1% to ~50% of known proteins overnight
- **Rare diseases**: Proteins implicated in rare genetic diseases now have predicted structures
- **Understudied organisms**: Model organisms, pathogens, and environmental microbes now have structural proteomes
- **Accelerated research**: Enabled structure-based studies that were previously impossible

As of 2025, the AlphaFold Database has been accessed millions of times and is cited in thousands of research papers across all areas of biology.

### 5.3 Applications in Drug Discovery

**Structure-Based Drug Design:**

Recall from Blog 1: Drug discovery often starts with a target protein. With AlphaFold:

1. **Target Identification**: Identify a disease-relevant protein (e.g., a kinase implicated in cancer)
2. **Structure Prediction**: Predict its structure with AlphaFold (hours, not months)
3. **Binding Site Analysis**: Identify potential druggable pockets
4. **Virtual Screening**: Dock millions of compounds computationally (we'll cover this in Blog 6)
5. **Lead Optimization**: Use structural insights to improve drug candidates

**Real-World Examples:**

- **Neglected disease targets**: Structures for proteins from neglected tropical diseases (e.g., malaria, tuberculosis) that lacked experimental structures
- **Membrane proteins**: AlphaFold can predict structures for GPCRs and ion channels (traditional drug targets) that are difficult to crystallize
- **Drug resistance**: Understanding resistance mutations in cancer and infectious diseases by predicting mutant structures

**Antibody Design:**

AlphaFold can predict antibody structures (though accuracy is somewhat lower for highly variable regions):
- Therapeutic antibody engineering
- Vaccine design (predicting epitopes)
- Immunotherapy development

**De Novo Protein Design:**

Emerging field: Combine AlphaFold (prediction) with generative models (design):
- Design proteins with desired functions (enzymes, binders, materials)
- Validate designs with AlphaFold
- Iterative design-predict-optimize cycles

---

## AlphaFold2: Summary, Mechanism, and Drug Discovery Context

AlphaFold2 achieved a revolutionary breakthrough, solving the **50-year protein folding challenge** by predicting structures with near-experimental accuracy. This breakthrough has **democratized** access to 3D structures, making them instantly available for hundreds of millions of proteins and transforming structure-based drug discovery.

***

### How AlphaFold2 Works

AlphaFold2's power comes from deeply integrating evolutionary data with geometric constraints:

* **Core Principle:** It leverages **Evolutionary Insight**, where co-evolved amino acids in a **Multiple Sequence Alignment (MSA)** encode information about **3D contacts**.
* **Evoformer Architecture:** A deep, **two-track neural network** processes the MSA (evolutionary features) and the Pair Representation (structural features) simultaneously across 48 blocks.
* **Geometric Enforcement:** Key innovations enforce 3D geometry:
    * **Triangle Multiplicative Updates** force distance predictions to obey the triangle inequality, ensuring consistency.
    * **Invariant Point Attention (IPA)** is a geometry-aware attention mechanism that refines coordinates while respecting 3D rotation/translation symmetries.
* **Iterative Refinement:** The process is refined over multiple **Recycles** and inner Structure Module passes, converging the abstract prediction to final 3D coordinates.

***

### Impact and Limitations in Drug Discovery

AlphaFold2 is the essential **first step** in modern computational drug discovery, but its limitations define the remaining challenges:

| Role in Drug Discovery | Key Limitation for Drug Discovery |
| :--- | :--- |
| **Enables SBDD:** Provides structures for previously **inaccessible targets** (Structural Proteomics). | **Static Structures:** Predicts **one static conformation**, missing protein **dynamics** and **induced fit**, which are crucial for drug binding. |
| **Speeds Pipeline:** Enables **virtual screening** and rational design, shortening timelines from years to days. | **No Ligands:** Does **not predict where drugs bind** or how small molecules affect the structure; **docking algorithms** are still required. |
| **Complex Targets:** Provides predictions for **Protein-Protein Interactions** (via AlphaFold-Multimer) and **Membrane Proteins**. | **Lower Accuracy:** Accuracy is lower for **PPI complexes** and challenging targets like **Membrane Proteins** (due to missing membrane context). |
| **Identifies Disorder:** Correctly flags flexible regions (Intrinsically Disordered) with **low $\text{pLDDT}$**. | **No Functional Insight:** Cannot model the functional dynamic behavior of disordered regions. |
| **Final Check:** Provides a **pLDDT confidence score**. | **Confidence Calibration:** The model can sometimes be **overconfident** in incorrect predictions, requiring careful user validation. |

***

### The Future

Future research is focused on integrating AlphaFold's power with other tools: combining it with **molecular dynamics** to capture protein motion and integrating it with **docking/binding prediction** to fully automate structure-to-drug pipelines.


---

## Further Reading and References

### Key Papers

1. **Jumper et al. (2021)**: "Highly accurate protein structure prediction with AlphaFold." *Nature* 596, 583589. [The AlphaFold2 paper]
5. **Senior et al. (2020)**: "Improved protein structure prediction using potentials from deep learning." *Nature* 577, 706-710. [AlphaFold1 at CASP13]
6. **Varadi et al. (2022)**: "AlphaFold Protein Structure Database: massively expanding the structural coverage of protein-sequence space with high-accuracy models." *Nucleic Acids Research* 50, D439D444. [AlphaFold Database paper]

### Resources

- **AlphaFold Database**: [alphafold.ebi.ac.uk](https://alphafold.ebi.ac.uk)
- **AlphaFold Code**: [github.com/deepmind/alphafold](https://github.com/deepmind/alphafold)
- **ColabFold**: Easy-to-use AlphaFold notebook: [github.com/sokrypton/ColabFold](https://github.com/sokrypton/ColabFold)
- **CASP**: [predictioncenter.org](https://predictioncenter.org)

---



**What we have coverd so far (Foundation)**

| Blog | Connection to AlphaFold2 |
| :--- | :--- |
| **Blog 1 (Introduction)** | AlphaFold provides the **tertiary structure** required by **Anfinsen's principle** and removes the major **drug discovery bottleneck** of experimental structure determination. |
| **Blog 2 (Representations)** | AlphaFold uses **MSAs** as the primary input and its **Pair Representation** predicts **distance matrices**—both key molecular representations. The final output is **3D coordinates**. |

---

**What we will cover next (Applications)**

| Blog | Connection to AlphaFold2 |
| :--- | :--- |
| **Blog 4 (Graph Neural Networks)** | AlphaFold's **Triangle Attention** is conceptually related to **GNN message passing**, as both iteratively refine features by aggregating information from neighbors. |
| **Blog 5 (Generative Models)** | AlphaFold is a **prediction model** (sequence $\rightarrow$ structure). Future work involves combining it with **Generative Models** to solve the inverse problem: designing a new protein (desired function $\rightarrow$ sequence). |
| **Blog 6 (Molecular Docking)** | AlphaFold is the crucial **first step** in the drug discovery pipeline: **AlphaFold (Structure) $\rightarrow$ Docking (Binding) $\rightarrow$ Screening**. It provides the necessary receptor structure for docking to predict where drugs bind. |
---

In our next blog, we'll explore **Graph Neural Networks** and how they're used to predict molecular properties and drug-likeness. We'll see how the same principles underlying AlphaFold learning from structure and leveraging geometric constraints apply to small molecules in drug discovery.

The revolution in computational drug discovery is just beginning, and AlphaFold has shown us what's possible when we combine deep learning with fundamental scientific insights.
