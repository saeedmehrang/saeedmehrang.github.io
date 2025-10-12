---
title: "Computational Drug Discovery Part 2: Molecular Representations for Machine Learning"
date: 2025-10-11
draft: false
author: Saeed Mehrang
description: "Exploring how we transform molecules and proteins into data structures that machine learning algorithms can process from SMILES strings to molecular graphs to 3D geometries."
summary: "Molecules are converted into data for ML using SMILES, fingerprints, molecular graphs, and 3D geometry. The trade-off is between compactness (SMILES/fingerprints) and richness (graphs/3D geometry). The representation choice fundamentally determines what models can learn about binding and performance in drug discovery."
tags: ["drug-discovery", "machine-learning", "molecular-representations", "graph-neural-networks", "cheminformatics"]
series: ["Computational Drug Discovery"]
series_order: 2
showToc: true
disableAnchoredHeadings: false
cover:
  image: "cover.jpg"
  image_alt: "molecule sketch"
---

## Introduction

In [Blog 1](/blogs/computational-drug-discovery/basic-molecular-biology/), we learned that proteins are chains of amino acids that fold into precise 3D shapes, and drugs are small molecules with specific functional groups that bind to protein pockets. We explored how structure determines function and why finding the right drug molecule is extraordinarily challenging.

But here's the fundamental problem: machine learning models don't understand chemistry, they process numbers. A neural network can't look at a molecular structure diagram and intuitively grasp that a hydroxyl group makes a molecule more water-soluble, or that an aromatic ring provides rigidity. Before we can apply powerful ML algorithms to drug discovery, we need to answer a crucial question: **How do we translate molecular structures into a form that neural networks can learn from?**

This translation from chemistry to data is far from trivial, and the choices we make have dramatic effects on model performance. Unlike images (which naturally map to pixel arrays) or text (which can be tokenized into sequences), molecules have no single "natural" representation for computers. We must actively choose how to encode them, and each choice captures different aspects of chemistry while losing others.

Consider aspirin: we could represent it as a text string describing its atoms and bonds, as a binary vector indicating which structural features it contains, as a graph connecting atoms through bonds, or as a 3D point cloud of atomic coordinates. Each representation preserves different information and each works better or worse depending on what we're trying to predict.

In this post, we'll explore the major approaches to molecular representation, building from simple one-dimensional text strings to sophisticated three-dimensional geometries. We'll see:

- Why representation choice fundamentally affects what ML models can learn
- The trade-offs between different encoding methods for small molecules
- How to represent both 2D connectivity and 3D geometry
- Why proteins require different representations than small molecules
- What information is preserved versus lost in each approach

By understanding these representational choices, we'll be equipped to appreciate the graph neural networks, generative models, and structure prediction methods coming in later blogs.

## The Representation Challenge

### What Makes Molecules Hard to Represent?

Molecules present unique challenges that don't exist in typical machine learning domains:

**1. Permutation Invariance**: The same molecule can be described in countless equivalent ways. When we list atoms, the ordering is arbitrary "carbon-oxygen-carbon" describes the same molecule as "oxygen-carbon-carbon" if they have the same connectivity. Unlike a sentence where word order matters, molecular identity depends only on which atoms connect to which, not on how we label or traverse them. A good representation should either be **invariant** to these orderings (multiple descriptions map to the same representation) or **canonical** (we agree on one standard ordering).

**2. 3D Conformational Flexibility**: Small molecules aren't rigid objects—they're more like flexible chains that can twist and bend. Think of a single bond between two carbon atoms like a hinge or a rotating joint. The molecule can spin around this bond, creating different 3D shapes called conformers, all while maintaining the same chemical formula and connectivity. Consider ethanol (the alcohol in drinks): it has a carbon-carbon single bond that can rotate freely. Imagine holding a molecular model—you could twist one end while keeping the other fixed, creating different spatial arrangements. The molecule might be stretched out, folded up, or somewhere in between. All these shapes are still ethanol—same atoms, same bonds—but different 3D geometries. This flexibility creates a dilemma for machine learning: which shape do we use to represent the molecule? Should we pick the lowest-energy conformer (the most stable shape)? Should we use all possible conformers? Or perhaps just the shape the molecule adopts when it binds to a protein? This becomes critical for drug discovery because binding depends on 3D shape fitting. A drug needs to fit into a protein's binding pocket like a key into a lock. If we represent the molecule in the wrong conformer—say, a stretched-out shape when it actually binds in a folded shape—our predictions will be wrong. It's like trying to predict whether a key will fit a lock while looking at the key bent at the wrong angle.

**3. Size Variability**: Molecules range dramatically in size. Aspirin has just 21 atoms; proteins can have tens of thousands. Most ML architectures prefer fixed-size inputs. How do we handle this variability? Do we pad small molecules? Truncate large ones? Use architectures that naturally handle variable sizes?

**4. Chirality and Stereochemistry**: Some molecules are like your left and right hands—they're mirror images of each other but not identical. Try to superimpose your left hand onto your right hand and you'll see they don't match up perfectly, even though they have all the same parts. In chemistry, we call these mirror-image molecules enantiomers, and they exhibit a property called chirality (from the Greek word for "hand"). Here's the shocking part: these mirror-image molecules can have completely different biological effects, even though they have the exact same atoms connected in the exact same order. The only difference is their 3D spatial arrangement—one is the "left-handed" version and one is the "right-handed" version. The most tragic example is thalidomide, a drug prescribed in the 1950s-60s for morning sickness during pregnancy. One enantiomer (the right-handed form) safely relieved nausea and helped pregnant women feel better. But its mirror-image twin (the left-handed form) caused catastrophic birth defects, leading to thousands of babies born with severely malformed limbs. The two molecules were chemically identical on paper—same atoms, same bonds—but their opposite 3D handedness made one a helpful medicine and the other a devastating poison. This creates a major challenge for molecular representations: we must capture 3D handedness, not just which atoms connect to which. Many simple representations (like basic SMILES strings or simple fingerprints) only encode 2D connectivity—they're "blind" to the difference between left-handed and right-handed versions. It's like describing your hands by saying "five fingers attached to a palm"—technically correct, but missing the crucial fact that they're mirror images. For drug discovery, this blindness can be dangerous: if our representation can't distinguish enantiomers, our machine learning models can't either, and we might predict that a harmful molecule is safe.

### Desirable Properties in Representations

An ideal molecular representation would have several key properties:

**Uniqueness**: One molecule should map to one representation (or a predictable set of equivalent ones). If the same molecule generates random different representations each time, models can't learn consistent patterns.

**Completeness**: The representation should capture all chemically relevant information for the task at hand. Predicting water solubility requires knowing which functional groups are present; predicting binding affinity might require full 3D geometry.

**Compactness**: Efficient storage and computation matter. A representation using 2,048 bits is more practical than one requiring 100,000 dimensions, assuming both capture necessary information.

**Continuity**: Similar molecules should have similar representations. If changing one atom dramatically alters the entire encoding in unpredictable ways, ML models struggle to interpolate and generalize. Smooth representations enable learning structure-property relationships.

**Invertibility**: For generative models that design new molecules, can we decode the representation back into a valid chemical structure? If the encoding is a one-way transformation, we can only analyze existing molecules, not create new ones.

No single representation satisfies all these properties perfectly. The key is choosing the right trade-offs for your specific task.

## Representing Small Molecules

### 3.1 Text-Based: SMILES Strings

#### What is SMILES?

SMILES (Simplified Molecular Input Line Entry System) represents molecules as text strings by linearizing their molecular graphs[^1]. Where a chemist might draw a 2D structure diagram, SMILES encodes the same information as a compact sequence of characters.

The basic rules are intuitive:
- Write atoms as their chemical symbols: C for carbon, O for oxygen, N for nitrogen
- Single bonds are implicit (just write atoms in sequence)
- Double bonds use `=`, triple bonds use `#`
- Branches use parentheses: `CC(C)C` means a carbon chain with a branch
- Rings use numbers: `C1CC1` closes a three-membered ring
- Aromatic rings use lowercase letters: benzene is `c1ccccc1`

Let's work through an example. **Ethanol** (drinking alcohol) has the structure:

```
    H   H
    |   |
H - C - C - O - H
    |   |
    H   H
```

The SMILES representation is simply: `CCO`

Reading left to right: carbon-carbon-oxygen, with hydrogens implicit. The linear sequence captures that we have two carbons bonded to each other, and the second carbon bonds to an oxygen.

**Aspirin** is more complex, containing a benzene ring, an ester group, and a carboxylic acid:

```
SMILES: CC(=O)Oc1ccccc1C(=O)O
```

Breaking this down:
- `CC(=O)O` - acetyl group: methyl carbon, carbonyl carbon, oxygen
- `c1ccccc1` - benzene ring (aromatic)
- `C(=O)O` - carboxylic acid group

One important subtlety: the same molecule can have multiple valid SMILES strings. We could traverse the graph in different orders: `CC(=O)Oc1ccccc1C(=O)O` and `O=C(C)Oc1ccccc1C(=O)O` both represent aspirin. **Canonical SMILES** uses algorithms to generate a unique, standardized string for each molecule, ensuring uniqueness.

#### ML Usage

SMILES strings are sequences of characters, making them natural inputs for architectures designed for language:

- **Tokenize** the string: each character or multi-character token becomes a unit
- **Apply sequence models**: Recurrent Neural Networks (RNNs), LSTMs, or Transformers process the token sequence
- **Treat like machine translation**: Models can learn to map _SMILES to properties_ or _SMILES to modified SMILES_

Pre-trained models like **ChemBERTa** and **MolGPT** are trained on millions of SMILES strings, learning chemical "grammar" and patterns. These can be fine-tuned for downstream tasks like predicting drug properties or generating novel molecules.

#### Pros and Cons

**Advantages**:
- **Compact and relatively human-readable** once you learn the syntax (chemists can decode simple SMILES by hand)
- **Leverages existing NLP architectures** including Transformers and language models
- **Large pre-trained models exist**, capturing chemical knowledge from vast databases
- **Easy to store and share**: just a text string, no complex data structures

**Disadvantages**:
- **Syntactically fragile**: One wrong character creates an invalid molecule. Inserting a stray parenthesis or mismatched ring number produces chemical nonsense
- **Multiple representations for the same molecule**: Even with canonicalization, different algorithms might produce different strings
- **Doesn't explicitly encode 3D information**: SMILES is fundamentally 2D connectivity; stereochemistry requires special notation
- **Hard to enforce validity in generation**: When generating novel SMILES character-by-character, models often produce syntactically invalid strings that can't be converted back to molecules

### 3.2 Fingerprints: Binary Vectors

#### What are Molecular Fingerprints?

Molecular fingerprints transform variable-sized molecules into fixed-length binary vectors typically 1024 or 2048 bits where each bit is either 0 or 1. Each bit position corresponds to a specific structural feature (or pattern), and the bit is set to 1 if that feature is present in the molecule.

Think of it like a checklist: Does this molecule have an aromatic ring? (bit 47 = 1). Does it contain a carbonyl group? (bit 128 = 1). Does it have a nitrogen connected to three carbons? (bit 891 = 1). The complete pattern of 0s and 1s serves as a "fingerprint" that characterizes the molecule.

#### Types of Fingerprints

**MACCS Keys**: A predefined set of 166 structural features designed by medicinal chemists. Each feature is a specific substructure:
- Bit 1: Has any isotope
- Bit 45: Has aromatic ring
- Bit 79: Has carbonyl oxygen
- Bit 123: Has nitrogen attached to aromatic carbon
- And so on...

The advantage is **interpretability** you know exactly what each bit means. The disadvantage is **limited resolution** only 166 features might miss important patterns.

**ECFP/Morgan Fingerprints** (Extended Connectivity Fingerprints): A more sophisticated, hash-based approach[^2]. For each atom, the algorithm:
1. Looks at its immediate neighbors (radius 1)
2. Encodes the local environment (atom types, bond types)
3. Hashes this pattern to a bit position
4. Repeats for larger neighborhoods (radius 2, 3, 4...)

This captures "what's within 1 bond of each atom," "what's within 2 bonds," etc. Unlike MACCS, ECFP can encode millions of possible substructure patterns through hashing.

The hashing process creates a **collision problem**: different substructures might hash to the same bit position, causing information loss. However, with 2048 bits, collisions are relatively rare for typical drug-like molecules.

#### Example: Simple Binary Coding

Consider **caffeine** (the stimulant in coffee). Setting bits based on features:
- Bit indicating "has aromatic ring": 1
- Bit for "has carbonyl": 1 (caffeine has three)
- Bit for "has nitrogen in ring": 1
- Bit for "has methyl group": 1
- Bits for specific atom neighborhood patterns...
- All other bits: 0

The resulting vector `[0,1,0,0,1,1,0,0,...]` (2048 bits total) serves as caffeine's fingerprint.

#### Example 2: How ECFP Hashing Works

Let's walk through how ECFP generates a fingerprint for a simple molecule like ethanol (CCO). What gets hashed? The algorithm encodes each atom's local environment as a numerical identifier (combining atomic numbers, bond types, and neighbor information into an integer), then applies a hash function to map that integer to a bit position (0-2047). Below, I use text descriptions for human readability, but the computer actually works with numerical encodings: 

**Step 1: Radius 0 (just the atom itself)**

- Atom 1 (Carbon): Environment = {atomic number: 6, degree: 4, implicit H: 3} → Encode as integer (e.g., 6403) → Hash(6403) → maps to bit 457
- Atom 2 (Carbon): Environment = {atomic number: 6, degree: 3, implicit H: 2, bonded to O} → Encode as integer (e.g., 6328) → Hash(6328) → maps to bit 1203
- Atom 3 (Oxygen): Environment = {atomic number: 8, degree: 2, implicit H: 1} → Encode as integer (e.g., 8021) → Hash(8021) → maps to bit 89


Set bits 457, 1203, and 89 to 1.

**Step 2: Radius 1 (atom + immediate neighbors)**

- Atom 1: Environment = {C with neighbors: [C(bonded to O), H, H, H]} → Encode as integer (e.g., 6000782) → Hash(6000782) → maps to bit 782
- Atom 2: Environment = {C with neighbors: [C, O, H, H]} → Encode as integer (e.g., 6008654) → Hash(6008654) → maps to bit 1654
- Atom 3: Environment = {O with neighbors: [C(bonded to another C), H]} → Encode as integer (e.g., 8006234) → Hash(8006234) → maps to bit 234


Set bits 782, 1654, and 234 to 1. 


**Step 3: Radius 2 (atom + neighbors within 2 bonds)**

- Atom 1: Environment = {C with extended neighborhood including O two bonds away} → Encode as integer (e.g., 60008782) → Hash(60008782) → maps to bit 1891
- And so on for other atoms...


Set additional bits based on these larger neighborhoods. 

**Final result:** A 2048-bit vector where bits [89, 234, 457, 782, 1203, 1654, 1891, ...] are set to 1, and all others are 0. The beauty of this approach is that **identical molecular neighborhoods produce identical hash values**, allowing the fingerprint to capture structural similarity. Two molecules that both contain "oxygen bonded to carbon bonded to carbon" will have that same numerical encoding, which hashes to the same bit position, indicating they share that substructure. **The collision issue:** If two **different** substructures happen to encode to integers that hash to the same bit position (say, both map to bit 782), we lose the ability to distinguish them. But with 2048 bits available and typical drug molecules having dozens to hundreds of unique substructures, the probability of many collisions is low.

#### ML Usage

Fingerprints work with **classical machine learning**:
- **Input to Random Forests, SVMs, or logistic regression** for property prediction
- **Similarity search**: Compare fingerprints using the Tanimoto coefficient (also called Jaccard similarity):
  ```
  Similarity = (Number of bits set in both) / (Number of bits set in either)
  ```
  Values range from 0 (no shared bits) to 1 (identical). Molecules with Tanimoto > 0.85 are typically very similar.
- **Fast screening**: Compute fingerprints once, then quickly search millions of molecules for similar ones

#### Pros and Cons

**Advantages**:
- **Fixed size**: Every molecule becomes a 2048-bit vector, regardless of its actual size. Easy for classical ML algorithms that require fixed input dimensions
- **Fast to compute**: Fingerprint generation takes milliseconds
- **Excellent for similarity search**: Finding structurally similar molecules is a core task in drug discovery
- **Works well with limited data**: Classical ML on fingerprints can learn from hundreds of examples, not requiring thousands like deep learning

**Disadvantages**:
- **Fixed length limits information capacity**: Only 2048 bits to describe a molecule with potentially thousands of atoms
- **Hash collisions lose information**: Different substructures mapping to the same bit creates ambiguity
- **Not invertible**: Given a fingerprint, you cannot reconstruct the original molecule. The fingerprint is a lossy compression designed for similarity comparison, not for reconstruction. [bit 457=1, bit 782=1, bit 1203=1...] → you know it has certain substructures, but you can't reconstruct the full molecule. It's like knowing "this person has brown hair and blue eyes" - you can't recreate their exact face from that.
- **Doesn't capture 3D geometry**: Purely 2D structural patterns
- **Somewhat outdated**: Modern graph neural networks generally outperform fingerprint-based models for property prediction

### 3.3 Molecular Graphs

#### Why Graphs Are Natural for Molecules

Molecules are literally graphs in the mathematical sense:
- **Nodes** (vertices) represent atoms
- **Edges** represent chemical bonds

This isn't an analogy or approximation it's a direct structural correspondence. Graph theory notation perfectly captures molecular topology.

Graphs naturally handle key molecular properties:
- **Variable size**: Graphs can have any number of nodes; no padding or truncation needed
- **Permutation invariance**: Relabeling nodes doesn't change the graph structure (graph isomorphism -- a permutation-invariant function will produce the same output for two graphs that are identical but have their nodes listed in a different order)
- **Explicit connectivity**: Bond patterns are directly represented as edge structure

This makes graph representations ideal for **Graph Neural Networks** (GNNs) neural architectures that operate directly on graph-structured data. We'll dive deep into GNNs in Blog 4; here we'll focus on how to encode molecular graphs.

#### Node Features: Encoding Atoms

For each atom (node), we store a feature vector capturing its chemical properties. Before listing these features, let's clarify two important concepts:

**Orbitals**: An orbital is a region of space around an atom's nucleus where there is the highest probability of finding an electron. Unlike the old idea of electrons orbiting the nucleus like planets, an orbital describes a three-dimensional shape—like a sphere (s orbital) or a dumbbell (p orbital)—that represents the electron's "home." Each orbital can hold a maximum of two electrons.

**Hybridization**: Hybridization is simply the idea of mixing an atom's existing electron "homes," or atomic orbitals (like s and p), to create new, better ones called hybrid orbitals. Atoms do this just before they form bonds. The reason they mix is that the original orbitals often don't point in the right direction or aren't all the same energy, which wouldn't result in the equal, perfectly-spaced bonds we observe in many molecules, like methane. By mixing, the atom gets a set of new, identical orbitals that are perfectly shaped and oriented to minimize electron repulsion, allowing for the strong, predictable structures and bond angles seen in nature. This mixing explains the observed molecular geometry and the formation of equivalent, strong chemical bonds that wouldn't be possible with the unmixed orbitals. For example, sp³ hybridization creates four equivalent orbitals arranged in a tetrahedral geometry, while sp² creates three orbitals in a planar arrangement.

Common node features include:

1. **Atomic number** (element type): Carbon = 6, Nitrogen = 7, Oxygen = 8, etc. Often one-hot encoded: [0,0,0,0,0,1,0,0,...] for carbon
2. **Formal charge**: -2, -1, 0, +1, +2 (most atoms are neutral; some carry charges)
3. **Hybridization**: sp, sp², sp³ (determines geometry—sp³ is tetrahedral, sp² is planar)
4. **Aromaticity**: Boolean flag indicating if atom is part of an aromatic ring
5. **Number of hydrogen atoms**: Hydrogens are often implicit in molecular graphs
6. **Ring membership**: Is the atom part of any ring? If so, what size?
7. **Chirality tags**: R/S configuration for chiral centers
8. **Degree**: Number of bonds to this atom

A node feature vector might look like: `[element_one_hot (118 dims), charge (5 dims), hybridization (4 dims), aromatic (1 dim), num_H (5 dims), in_ring (1 dim), chirality (3 dims)]`

The exact features depend on the task predicting toxicity might need charge information, while predicting synthetic accessibility cares more about ring systems and complexity.

#### Edge Features: Encoding Bonds

For each bond (edge), we similarly store features, but first let's clarify one important concept:


**conjugation:** In the context of encoding bonds, conjugation simply means that a bond is part of a chain where single and multiple bonds (double or triple) alternate. For instance, a double bond followed by a single bond followed by another double bond (C=C-C=C) is a conjugated system. This alternating pattern allows the electrons in the multiple bonds to be freely delocalized, or spread out, across the entire chain of atoms instead of being stuck between just two atoms. This delocalization is important because it makes the molecule more stable and often influences its color, energy levels, and reactivity. The "Conjugation" feature is a yes/no indicator that tells a computer model whether a bond is involved in this special, stable alternating system.

1. **Bond type**: Single, double, triple, aromatic (often one-hot encoded)
2. **Stereochemistry**: Cis/trans (E/Z) configuration for double bonds
3. **Ring membership**: Is this bond part of a ring?
4. **Conjugation**: Is the bond part of a conjugated system?
5. **Rotatable**: Can the bond rotate freely? (Important for conformational flexibility)



## Example: Caffeine as a Graph (Improved)

Let's encode **caffeine** ($\text{C}_8\text{H}_{10}\text{N}_4\text{O}_2$) with a focus on chemical accuracy for use in a GNN.

### 1. Node List (Selection of Key Atoms)

| Node ID | Atom Type | Hybridization | Ring Membership | Other Features |
| :---: | :---: | :---: | :---: | :---: |
| **C1** | C | $\text{sp}^2$ | 6-membered | Part of $\text{C=O}$ group |
| **C2** | C | $\text{sp}^2$ | 5-membered | Part of $\text{N-C-N}$ system |
| **N3** | N | $\text{sp}^2$ | 6-membered | Attached to $\text{CH}_3$ |
| **O4** | O | $\text{sp}^2$ | No | Double bonded to C1 |
| **H5** | H | $\text{sp}^3$ | No | Bonded to C2 |
| **Me-C**| C | $\text{sp}^3$ | No | Part of methyl group ($\text{CH}_3$) |
| ... | (Continuing for all 24 atoms) | ... | ... | ... |

***

### 2. Edge List (Selection of Key Bonds)

| Edge (i, j) | Bond Type | Conjugation | Aromatic | Rotatable |
| :---: | :---: | :---: | :---: | :---: |
| **(C1, N3)** | Single | Yes | No | No (in ring) |
| **(C1, O4)** | **Double** | No | No | No (Double) |
| **(N3, C-Me)**| **Single** | No | No | **Yes** (Single, non-ring) |
| **(C2, H5)** | **Single** | No | No | No (no rotation around single atom) |
| **(C2, C-ring)** | Double | Yes | No | No (Double) |
| ... | (Continuing for all bonds) | ... | ... | ... |

***

### 3. Adjacency Matrix

A matrix where entry $\mathbf{A}_{i,j} = 1$ if atoms $i$ and $j$ are bonded (regardless of bond type).

$$
\begin{array}{c|ccccccc}
 & \text{C1} & \text{C2} & \text{N3} & \text{O4} & \text{H5} & \dots \\
\hline
\text{C1} & 0 & 0 & 1 & 1 & 0 & \dots \\
\text{C2} & 0 & 0 & 0 & 0 & 1 & \dots \\
\text{N3} & 1 & 0 & 0 & 0 & 0 & \dots \\
\text{O4} & 1 & 0 & 0 & 0 & 0 & \dots \\
\text{H5} & 0 & 1 & 0 & 0 & 0 & \dots \\
\dots & \dots & \dots & \dots & \dots & \dots & 0
\end{array}
$$
***
This graph representation, with accurate bond types (including the non-aromatic double bonds) and conjugation status (applied to the delocalized $\text{N-C-N}$ ring system), provides a more robust input for GNNs learning chemical properties.

#### ML Usage

Graph representations enable **Graph Neural Networks**:
- **Message passing**: Each node aggregates information from its neighbors iteratively. After several layers, each node's representation captures information from its broader neighborhood
- **Permutation invariance**: GNNs automatically handle atom ordering they operate on graph structure, not on arbitrary node labels
- **Variable-size input**: Graphs naturally accommodate molecules of any size
- **Expressive**: GNNs can learn complex structure-property relationships that simpler representations miss

Popular GNN architectures include **Graph Convolutional Networks (GCN)**, **Graph Attention Networks (GAT)**, **Message Passing Neural Networks (MPNN)**[^3], and **Graph Isomorphism Networks (GIN)**. We'll explore these in detail in Blog 4.

#### Pros and Cons

**Advantages**:
- **Explicit structural representation**: Directly encodes the actual molecular graph
- **Flexible node/edge features**: Can incorporate any relevant chemical information
- **Permutation invariant by design**: No arbitrary ordering issues
- **State-of-the-art for property prediction**: GNNs consistently outperform fingerprints and SMILES-based models on benchmark tasks

**Disadvantages**:
- **Still mostly 2D**: Standard molecular graphs encode connectivity but not full 3D conformation
- **More complex**: Requires specialized GNN architectures and libraries (PyTorch Geometric, DGL)
- **Computationally more expensive** than fingerprints (though much faster than physics simulations)
- **Less interpretable**: Deep GNNs are black boxes; understanding why a prediction was made is challenging

### 3.4 3D Conformations

#### Why 3D Matters

Up to this point, we've focused on 2D representations encoding which atoms connect to which. But molecules exist in three-dimensional space, and **3D geometry is crucial for binding**:

- **Shape complementarity**: A drug must fit into the 3D pocket of a protein like a key into a lock
- **Pharmacophores**: The spatial arrangement of functional groups determines binding. Two molecules with identical connectivity but different 3D conformations can bind very differently
- **Chirality**: Mirror-image molecules (enantiomers) have the same 2D connectivity but opposite 3D handedness. Thalidomide's infamous case illustrates the consequences: one enantiomer safely treated morning sickness; the other caused severe birth defects. A representation that can't distinguish 3D stereochemistry is blind to this critical difference

When predicting **binding affinity** or **docking** a molecule into a protein, 3D structure isn't optional it's essential.

#### How to Represent 3D Geometry

Several approaches exist for encoding three-dimensional molecular structure:

**1. Cartesian Coordinates**: Simply list (x, y, z) coordinates for each atom:
```
Atom 1 (C): (0.00, 0.00, 0.00)
Atom 2 (C): (1.52, 0.00, 0.00)
Atom 3 (O): (2.18, 1.21, 0.00)
...
```
This is straightforward but **not rotationally invariant**: rotating or translating the molecule changes all coordinates, even though the molecule is chemically identical.

**2. Distance Matrices**: Matrix where entry (i,j) is the 3D distance between atoms i and j:
```
     Atom1  Atom2  Atom3
Atom1  0.0    1.52   2.45
Atom2  1.52   0.0    1.43
Atom3  2.45   1.43   0.0
```
This representation is **invariant** to rotation and translation (distances don't change), but loses information about absolute orientation.

**3. Coulomb Matrices**: Encode both distance and atomic number. Entry (i,j) is proportional to the nuclear charges divided by distance:
```
M_ij = Z_i * Z_j / |r_i - r_j|  (for i ` j)
M_ii = 0.5 * Z_i^2.4            (diagonal terms)
```
where Z is atomic number and r is position. This captures both geometry and element identity.

**4. Point Clouds**: Treat the molecule as a set of 3D points (atoms) with associated features:
```
[(x1,y1,z1, C, sp3), (x2,y2,z2, N, sp2), (x3,y3,z3, O, sp3), ...]
```
Each point has spatial coordinates plus chemical features. Point cloud neural networks (used in computer vision for 3D object recognition) can process this representation.

#### The Conformer Problem

Small molecules are **flexible** they can rotate around single bonds, adopting multiple 3D shapes called **conformers**. Ethanol has three major rotational conformers around the C-C bond; larger drug molecules might have dozens or hundreds.

**Which conformer do we use?**

- **Lowest energy conformer**: Compute or sample conformers and use the one with minimum energy. But molecules in solution populate multiple conformers, and the bound conformation might not be the lowest-energy free conformation.
- **Multiple conformers**: Generate an ensemble of conformers and train models on all of them. Expensive but more realistic.
- **Learned conformations**: Train models to predict the relevant 3D structure directly, rather than relying on pre-computed geometries.

This ambiguity is a core challenge in 3D molecular ML.

#### ML Approaches for 3D

Modern neural architectures handle 3D geometry while respecting physical symmetries:

**SE(3)-Equivariant Networks**: These models respect rotational and translational symmetries—if you rotate or translate the input molecule, the output transforms predictably. Architectures like **SchNet**[^4], **DimeNet**, **PaiNN**, and **EGNN** explicitly encode 3D distances and angles in ways that are equivariant to the Euclidean group SE(3).

**3D Message Passing**: Extends standard GNN message passing by incorporating 3D distances. Edge features include not just bond types but actual spatial distances, enabling models to learn how 3D geometry affects properties.

**SE(3)-Transformers**: Attention mechanisms adapted for 3D point clouds with rotational equivariance. These models can weigh the importance of different atoms based on both chemical identity and spatial proximity.

**Geometric Deep Learning**: A broader framework for neural networks on symmetric domains (graphs, meshes, point clouds) with appropriate invariances and equivariances built in.

#### When to Use 3D Representations

**Use 3D when**:
- Predicting **binding affinity**: Drug-protein binding is inherently 3D
- Performing **molecular docking**: Fitting molecules into protein binding sites
- Predicting **conformer-dependent properties**: Like NMR spectra or vibrational frequencies
- Understanding **reaction mechanisms**: Transition states and reactivity depend on 3D approach
- When **2D graph isn't enough**: Some properties simply can't be predicted from connectivity alone

**2D is sufficient when**:
- Predicting **simple molecular properties**: Molecular weight, number of rings, etc.
- **Initial screening** where speed matters more than precision
- You have **limited data**: 3D models are more data-hungry
- **No 3D structures available**: Generating accurate conformers requires computation

#### Pros and Cons

**Advantages**:
- **Captures geometry crucial for binding**: The only representation that directly encodes shape
- **Can model conformational flexibility**: With ensembles or learned conformation prediction
- **Essential for structure-based design**: Docking and binding site analysis require 3D
- **Distinguishes stereoisomers**: Properly handles chirality and other 3D features

**Disadvantages**:
- **Computationally expensive**: 3D-aware models are slower to train and run
- **Conformer ambiguity**: Which 3D structure to use isn't always clear
- **Requires 3D coordinates**: Must generate or compute geometries, adding preprocessing cost
- **More data-hungry**: Learning 3D geometry requires more training examples than 2D patterns
- **Breaking symmetries**: Some SE(3)-equivariant architectures are complex to implement

## Representing Proteins

Proteins differ fundamentally from small molecules in ways that demand different representational strategies.

### Why Proteins Are Different

Several key differences shape how we represent proteins:

**1. Size**: Small molecules typically have 10-100 atoms; proteins have **hundreds to thousands**. Hemoglobin has ~5,000 atoms; antibodies have ~20,000. Representing every atom explicitly becomes computationally prohibitive.

**2. Hierarchical Structure**: Remember from [Blog 1](/blogs/computational-drug-discovery/basic-molecular-biology/) that proteins have four structural levels:
- **Primary**: Amino acid sequence
- **Secondary**: Local patterns (helices, sheets)
- **Tertiary**: Full 3D fold
- **Quaternary**: Multi-chain assemblies

Each level captures different information useful for different tasks.

**3. Standardized Building Blocks**: Proteins use 20 standard amino acids (versus unlimited possible atoms and functional groups in small molecules). This regularity enables more compact representations we can encode residue types rather than individual atoms.

**4. Functional Diversity**: The same protein can have multiple functions and binding partners. We often care about specific regions (binding sites, active sites) rather than the whole protein.

### 4.1 Sequence Representations

#### Primary Structure as Text

At the most basic level, a protein is a sequence of amino acids, which we can write as a string using one-letter codes:

```
MVHLTPEEKSAVTALWGKVNVDEVGGEALGRLLVVYPWTQRFFESFGDLST...
```

This sequence for human hemoglobin (beta chain) lists its 147 amino acids in order:
- M (Methionine)
- V (Valine)
- H (Histidine)
- L (Leucine)
- T (Threonine)
...and so on.

Just like SMILES strings for molecules, protein sequences can be treated as **text for NLP methods**.

#### ML Encodings for Sequences

**One-Hot Encoding**: Represent each amino acid as a 20-dimensional binary vector. Alanine = [1,0,0,...,0], Cysteine = [0,1,0,...,0], etc. A 300-residue protein becomes a 300 x 20 matrix.

**Position-Specific Scoring Matrices (PSSMs)**: Instead of just the amino acid at each position, encode evolutionary information. By aligning homologous protein sequences across species, we can compute how conserved each position is and which substitutions are tolerated. Positions critical for function show strong conservation; variable positions are less important.

**Learned Embeddings**: Similar to word embeddings in NLP (Word2Vec, GloVe), we can learn low-dimensional vectors for amino acids. These embeddings capture biochemical similarities hydrophobic amino acids cluster together, charged residues cluster separately.

**Pre-trained Protein Language Models**: Large transformer models trained on millions of protein sequences:
- **ProtBERT**: BERT architecture applied to protein sequences
- **ESM (Evolutionary Scale Modeling)**: Models trained on 250 million sequences, learning deep patterns of protein structure and function[^5]
- **ProGen**: GPT-style generative models for protein sequences

These models learn "protein grammar"—which amino acid sequences are likely, how mutations affect function, and even predict 3D structure from sequence alone (as AlphaFold[^6] does, which we'll explore in Blog 3).

#### Pros and Cons

**Advantages**:
- **Simple and compact**: Just a string of letters
- **Works with Transformers**: Leverage advances in NLP and sequence modeling
- **Rich pre-trained models available**: Transfer learning from vast protein databases
- **Captures evolutionary information**: Through PSSMs and homology

**Disadvantages**:
- **No 3D information**: Can't predict binding without structure
- **No direct binding site information**: Sequence alone doesn't tell you where the pocket is
- **Requires structure prediction** for many downstream tasks

### 4.2 Structure Representations

When 3D structure is available (from X-ray crystallography, cryo-EM, or AlphaFold prediction), we can represent protein geometry explicitly.

#### Representation Options

**All-Atom Coordinates**: List (x,y,z) positions for every atom. Full resolution but massive 5,000+ atoms for a medium-sized protein. Most atoms are hydrogens, which add little information but much computational cost.

**Backbone Atoms Only**: Proteins have a repeating backbone (N-Cα-C-O) with variable sidechains. Representing just the backbone (especially just the Cα atoms -- the central carbon atom in an amino acid) captures the overall fold while reducing size by ~90%. This is sufficient for many tasks since the backbone determines secondary and tertiary structure.

**Contact Maps**: Binary 2D matrix where entry (i,j) = 1 if residues i and j are spatially close (within ~8 Å -- Angstrom distance) in 3D, 0 otherwise:
```
     Res1  Res2  Res3  Res4  Res5
Res1   1     1     0     0     0
Res2   1     1     1     0     0
Res3   0     1     1     1     0
Res4   0     0     1     1     1
Res5   0     0     0     1     1
```
This 2D representation of 3D structure shows which parts of the sequence come close in space (even if far apart in sequence). Contact maps are easier to predict than full 3D coordinates AlphaFold essentially predicts contact maps (distance maps) which are then used to reconstruct 3D structures.

**Distance Matrices**: Instead of binary contacts, store actual distances between all residue pairs (typically using Cα-Cα distances). This preserves more information than binary contacts.

**Surface Representations**: For understanding binding interfaces, represent the protein's solvent-accessible surface as a point cloud or mesh. This captures the shape of binding pockets without interior atoms.

**Voxel Grids**: Discretize 3D space into a regular grid (like pixels in 3D) and encode occupancy and properties (atom type, charge, hydrophobicity) in each voxel. This treats the protein like a 3D image that can be processed with 3D CNNs.

#### ML Methods

**3D Convolutional Neural Networks**: Treat voxel grids like 3D images. CNNs slide filters through 3D space, learning spatial patterns. Used for binding site prediction and protein-ligand scoring.

**Graph Representations**: Residues as nodes, edges connecting spatially proximal residues (even if far in sequence). This captures long-range interactions in the folded structure. GNNs then learn on the residue-residue contact graph.

**Point Cloud Networks**: Treat Cα atoms as a 3D point cloud with residue type features. PointNet and related architectures process these for structure classification and binding site prediction.

**SE(3)-Equivariant Networks**: Same geometric deep learning approaches used for small molecules apply to proteins SchNet, EGNN, and others respect rotational symmetry.

### 4.3 Binding Site Representations

For drug discovery, we often care specifically about the **binding site** the pocket or groove where drugs bind. This is typically 10-20 amino acids, much smaller than the whole protein.

#### Why Focus on Binding Sites?

**Computational Efficiency**: A 500-residue protein might have a 15-residue binding site. Extracting just the pocket reduces data by 97%.

**Task Relevance**: For predicting binding affinity or docking molecules, what matters is the local chemical environment of the pocket, not distant regions of the protein.

**Generalization**: Binding sites with similar shapes and chemical properties bind similar ligands, even if the overall proteins are unrelated. Focusing on pockets enables cross-protein learning.

#### Pocket Features

Common features for encoding binding sites:

- **Shape descriptors**: Pocket volume, depth, geometric moments
- **Electrostatic potential**: Surface charge distribution (positive, negative, neutral)
- **Hydrophobicity maps**: Which regions are oily vs. water-loving
- **Pharmacophore features**: Spatial locations of hydrogen bond donors/acceptors, aromatic groups, charged groups
- **Flexibility**: Which residues are rigid vs. flexible

#### Used In

- **Molecular docking** (covered in later blogs): Placing and scoring drug candidates in the pocket
- **Binding affinity prediction**: Predicting how tightly a molecule binds
- **De novo drug design**: Generating molecules optimized for a specific pocket's shape and chemistry


## Key Takeaways

Let's consolidate the core lessons from our survey of molecular representations:

**No Single "Best" Representation**: The optimal choice depends on your task, available data, and computational resources. Predicting simple properties might work with fingerprints; predicting binding requires 3D structures.

**SMILES and Fingerprints**: Good for quick property prediction and large-scale virtual screening. Compact, fast, and work with classical ML. But they're limited in expressiveness and don't handle 3D.

**Molecular Graphs**: State-of-the-art for most property prediction tasks. Graph neural networks leverage explicit structure and achieve top performance on benchmarks. However, they require more complex architectures and are still primarily 2D.

**3D Representations**: Essential when geometry matters binding affinity, docking, structure-based design. SE(3)-equivariant networks respect physical symmetries. But 3D models are computationally expensive and data-hungry.

**Protein Sequences**: Sufficient for remote homology detection, evolutionary analysis, and initial function prediction. Pre-trained language models capture rich patterns. But you need structure for binding site analysis.

**Protein Structures**: Required for understanding binding interfaces and rational drug design. AlphaFold's breakthrough (Blog 3) means we now have high-quality structures for millions of proteins.

**The Expressiveness-Cost Trade-off**: More expressive representations (3D > graphs > fingerprints) capture more chemistry but require more computation and data. Choose based on your specific needs don't use a 3D SE(3)-Transformer when a random forest on fingerprints suffices.

**Preprocessing Matters**: Real-world data is messy. Proper sanitization, standardization, and featurization are as important as model architecture.

## Connections and Looking Forward

The representations we've covered here form the foundation for everything that follows in this series.

**Looking back to [Blog 1](/blogs/computational-drug-discovery/basic-molecular-biology/)**:
- Remember how **functional groups determine molecular behavior**? In fingerprints, we explicitly check for these functional groups. In graph representations, node features encode them. The chemical principles map directly to our data structures.
- We learned that **protein structure determines function**. Sequence representations capture the primary structure, while 3D representations encode tertiary structure. Different tasks require different structural levels.

In the next post, we'll see how AlphaFold revolutionized protein structure prediction by combining sequence representations with powerful Transformer architectures, solving a 50-year-old challenge and enabling structure-based drug design at unprecedented scale.

## References

[^1]: Weininger, D. (1988). SMILES, a chemical language and information system. *Journal of Chemical Information and Computer Sciences*, 28(1), 31-36.

[^2]: Rogers, D., & Hahn, M. (2010). Extended-connectivity fingerprints. *Journal of Chemical Information and Modeling*, 50(5), 742-754.

[^3]: Gilmer, J., et al. (2017). Neural message passing for quantum chemistry. *Proceedings of the International Conference on Machine Learning*, 1263-1272.

[^4]: Schütt, K. T., et al. (2017). SchNet: A continuous-filter convolutional neural network for modeling quantum interactions. *Advances in Neural Information Processing Systems*, 30.

[^5]: Rives, A., et al. (2021). Biological structure and function emerge from scaling unsupervised learning to 250 million protein sequences. *Proceedings of the National Academy of Sciences*, 118(15), e2016239118.

[^6]: Jumper, J., et al. (2021). Highly accurate protein structure prediction with AlphaFold. *Nature*, 596(7873), 583-589.
