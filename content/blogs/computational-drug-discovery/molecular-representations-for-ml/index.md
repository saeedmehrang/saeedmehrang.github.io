---
title: "Computational Drug Discovery Part 2: Molecular Representations for Machine Learning"
date: 2025-10-11
draft: false
author: Saeed Mehrang
description: "Exploring how we transform molecules and proteins into data structures that machine learning algorithms can process from SMILES strings to molecular graphs to 3D geometries."
summary: "This blog explores how molecules and proteins are transformed into machine learning representations—from SMILES strings and fingerprints to molecular graphs and 3D geometries. We examine key trade-offs: SMILES and fingerprints are compact but limited; molecular graphs enable state-of-the-art GNNs; 3D representations capture geometry essential for binding but are computationally expensive. The key insight: representation choice fundamentally shapes what models can learn and determines performance in drug discovery applications."
tags: ["drug-discovery", "machine-learning", "molecular-representations", "graph-neural-networks", "cheminformatics"]
series: ["Computational Drug Discovery"]
series_order: 2
showToc: true
disableAnchoredHeadings: false
---

## Introduction

In [Blog 1](/blogs/computational-drug-discovery/basic-molecular-biology/), we learned that proteins are chains of amino acids that fold into precise 3D shapes, and drugs are small molecules with specific functional groups that bind to protein pockets. We explored how structure determines function and why finding the right drug molecule is extraordinarily challenging.

But here's the fundamental problem: machine learning models don't understand chemistry they process numbers. A neural network can't look at a molecular structure diagram and intuitively grasp that a hydroxyl group makes a molecule more water-soluble, or that an aromatic ring provides rigidity. Before we can apply powerful ML algorithms to drug discovery, we need to answer a crucial question: **How do we translate molecular structures into a form that neural networks can learn from?**

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

**2. 3D Conformational Flexibility**: Small molecules aren't rigid. They can rotate around single bonds, adopting different 3D shapes called conformers. Ethanol (drinking alcohol) can twist into many different spatial arrangements, all representing the same chemical compound. Do we represent one conformer? All of them? The lowest-energy one? This becomes critical when predicting binding drugs need the right 3D shape to fit into protein pockets.

**3. Size Variability**: Molecules range dramatically in size. Aspirin has just 21 atoms; proteins can have tens of thousands. Most ML architectures prefer fixed-size inputs. How do we handle this variability? Do we pad small molecules? Truncate large ones? Use architectures that naturally handle variable sizes?

**4. Chirality and Stereochemistry**: Mirror-image molecules (enantiomers) can have completely different biological effects. Thalidomide's tragic history illustrates this: one mirror form treated morning sickness, while the other caused severe birth defects. A molecule's representation must capture this 3D handedness, yet many simple encodings miss it entirely.

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

SMILES (Simplified Molecular Input Line Entry System) represents molecules as text strings by linearizing their molecular graphs. Where a chemist might draw a 2D structure diagram, SMILES encodes the same information as a compact sequence of characters.

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
- **Treat like machine translation**: Models can learn to map SMILES � properties or SMILES � modified SMILES

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

**ECFP/Morgan Fingerprints** (Extended Connectivity Fingerprints): A more sophisticated, hash-based approach. For each atom, the algorithm:
1. Looks at its immediate neighbors (radius 1)
2. Encodes the local environment (atom types, bond types)
3. Hashes this pattern to a bit position
4. Repeats for larger neighborhoods (radius 2, 3, 4...)

This captures "what's within 1 bond of each atom," "what's within 2 bonds," etc. Unlike MACCS, ECFP can encode millions of possible substructure patterns through hashing.

The hashing process creates a **collision problem**: different substructures might hash to the same bit position, causing information loss. However, with 2048 bits, collisions are relatively rare for typical drug-like molecules.

#### Example

Consider **caffeine** (the stimulant in coffee). Setting bits based on features:
- Bit indicating "has aromatic ring": 1
- Bit for "has carbonyl": 1 (caffeine has three)
- Bit for "has nitrogen in ring": 1
- Bit for "has methyl group": 1
- Bits for specific atom neighborhood patterns...
- All other bits: 0

The resulting vector `[0,1,0,0,1,1,0,0,...]` (2048 bits total) serves as caffeine's fingerprint.

#### ML Usage

Fingerprints work with **classical machine learning**:
- **Input to Random Forests, SVMs, or logistic regression** for property prediction
- **Similarity search**: Compare fingerprints using the Tanimoto coefficient (also called Jaccard similarity):
  ```
  Similarity = (bits set in both) / (bits set in either)
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
- **Not invertible**: Given a fingerprint, you cannot reconstruct the original molecule
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
- **Permutation invariance**: Relabeling nodes doesn't change the graph structure (graph isomorphism)
- **Explicit connectivity**: Bond patterns are directly represented as edge structure

This makes graph representations ideal for **Graph Neural Networks** (GNNs) neural architectures that operate directly on graph-structured data. We'll dive deep into GNNs in Blog 4; here we'll focus on how to encode molecular graphs.

#### Node Features: Encoding Atoms

For each atom (node), we store a feature vector capturing its chemical properties. Common features include:

1. **Atomic number** (element type): Carbon = 6, Nitrogen = 7, Oxygen = 8, etc. Often one-hot encoded: [0,0,0,0,0,1,0,0,...] for carbon
2. **Formal charge**: -2, -1, 0, +1, +2 (most atoms are neutral; some carry charges)
3. **Hybridization**: sp, sp�, sp� (determines geometry sp� is tetrahedral, sp� is planar)
4. **Aromaticity**: Boolean flag indicating if atom is part of an aromatic ring
5. **Number of hydrogen atoms**: Hydrogens are often implicit in molecular graphs
6. **Ring membership**: Is the atom part of any ring? If so, what size?
7. **Chirality tags**: R/S configuration for chiral centers
8. **Degree**: Number of bonds to this atom

A node feature vector might look like: `[element_one_hot (118 dims), charge (5 dims), hybridization (4 dims), aromatic (1 dim), num_H (5 dims), in_ring (1 dim), chirality (3 dims)]`

The exact features depend on the task predicting toxicity might need charge information, while predicting synthetic accessibility cares more about ring systems and complexity.

#### Edge Features: Encoding Bonds

For each bond (edge), we similarly store features:

1. **Bond type**: Single, double, triple, aromatic (often one-hot encoded)
2. **Stereochemistry**: Cis/trans (E/Z) configuration for double bonds
3. **Ring membership**: Is this bond part of a ring?
4. **Conjugation**: Is the bond part of a conjugated system?
5. **Rotatable**: Can the bond rotate freely? (Important for conformational flexibility)

#### Example: Caffeine as a Graph

Let's encode **caffeine** explicitly:

**Molecular formula**: C�H��N�O�

**Node list** (simplified):
```
Node 0: C (sp2, aromatic, in 5-membered ring)
Node 1: N (sp2, aromatic, in 5-membered ring)
Node 2: C (sp2, aromatic, in 5-membered ring)
Node 3: N (sp2, aromatic, in 5-membered ring)
Node 4: C (sp2, aromatic, in 5-membered ring)
Node 5: C (sp2, aromatic, in 6-membered ring)
Node 6: N (sp2, aromatic, in 6-membered ring)
... (continuing for all atoms)
```

**Edge list**:
```
Edge (0,1): aromatic bond
Edge (1,2): aromatic bond
Edge (2,3): aromatic bond
Edge (3,4): aromatic bond
Edge (4,0): aromatic bond (closes 5-membered ring)
Edge (0,5): aromatic bond (connects to 6-membered ring)
... (continuing for all bonds)
```

**Adjacency matrix**: A matrix where entry (i,j) = 1 if atoms i and j are bonded:
```
     0  1  2  3  4  5  6 ...
0 [  0  1  0  0  1  1  0 ...]
1 [  1  0  1  0  0  0  0 ...]
2 [  0  1  0  1  0  0  0 ...]
...
```

This graph representation can be fed into GNNs, which learn to aggregate information from neighboring atoms through message-passing algorithms.

#### ML Usage

Graph representations enable **Graph Neural Networks**:
- **Message passing**: Each node aggregates information from its neighbors iteratively. After several layers, each node's representation captures information from its broader neighborhood
- **Permutation invariance**: GNNs automatically handle atom ordering they operate on graph structure, not on arbitrary node labels
- **Variable-size input**: Graphs naturally accommodate molecules of any size
- **Expressive**: GNNs can learn complex structure-property relationships that simpler representations miss

Popular GNN architectures include **Graph Convolutional Networks (GCN)**, **Graph Attention Networks (GAT)**, **Message Passing Neural Networks (MPNN)**, and **Graph Isomorphism Networks (GIN)**. We'll explore these in detail in Blog 4.

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

**SE(3)-Equivariant Networks**: These models respect rotational and translational symmetries if you rotate or translate the input molecule, the output transforms predictably. Architectures like **SchNet**, **DimeNet**, **PaiNN**, and **EGNN** explicitly encode 3D distances and angles in ways that are equivariant to the Euclidean group SE(3).

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

**One-Hot Encoding**: Represent each amino acid as a 20-dimensional binary vector. Alanine = [1,0,0,...,0], Cysteine = [0,1,0,...,0], etc. A 300-residue protein becomes a 300 � 20 matrix.

**Position-Specific Scoring Matrices (PSSMs)**: Instead of just the amino acid at each position, encode evolutionary information. By aligning homologous protein sequences across species, we can compute how conserved each position is and which substitutions are tolerated. Positions critical for function show strong conservation; variable positions are less important.

**Learned Embeddings**: Similar to word embeddings in NLP (Word2Vec, GloVe), we can learn low-dimensional vectors for amino acids. These embeddings capture biochemical similarities hydrophobic amino acids cluster together, charged residues cluster separately.

**Pre-trained Protein Language Models**: Large transformer models trained on millions of protein sequences:
- **ProtBERT**: BERT architecture applied to protein sequences
- **ESM (Evolutionary Scale Modeling)**: Models trained on 250 million sequences, learning deep patterns of protein structure and function
- **ProGen**: GPT-style generative models for protein sequences

These models learn "protein grammar" which amino acid sequences are likely, how mutations affect function, and even predict 3D structure from sequence alone (as AlphaFold does, which we'll explore in Blog 3).

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

**Backbone Atoms Only**: Proteins have a repeating backbone (N-C�-C-O) with variable sidechains. Representing just the backbone (especially just the C� atoms) captures the overall fold while reducing size by ~90%. This is sufficient for many tasks since the backbone determines secondary and tertiary structure.

**Contact Maps**: Binary 2D matrix where entry (i,j) = 1 if residues i and j are spatially close (within ~8 �) in 3D, 0 otherwise:
```
     Res1  Res2  Res3  Res4  Res5
Res1   1     1     0     0     0
Res2   1     1     1     0     0
Res3   0     1     1     1     0
Res4   0     0     1     1     1
Res5   0     0     0     1     1
```
This 2D representation of 3D structure shows which parts of the sequence come close in space (even if far apart in sequence). Contact maps are easier to predict than full 3D coordinates AlphaFold essentially predicts contact maps (distance maps) which are then used to reconstruct 3D structures.

**Distance Matrices**: Instead of binary contacts, store actual distances between all residue pairs (typically using C�-C� distances). This preserves more information than binary contacts.

**Surface Representations**: For understanding binding interfaces, represent the protein's solvent-accessible surface as a point cloud or mesh. This captures the shape of binding pockets without interior atoms.

**Voxel Grids**: Discretize 3D space into a regular grid (like pixels in 3D) and encode occupancy and properties (atom type, charge, hydrophobicity) in each voxel. This treats the protein like a 3D image that can be processed with 3D CNNs.

#### ML Methods

**3D Convolutional Neural Networks**: Treat voxel grids like 3D images. CNNs slide filters through 3D space, learning spatial patterns. Used for binding site prediction and protein-ligand scoring.

**Graph Representations**: Residues as nodes, edges connecting spatially proximal residues (even if far in sequence). This captures long-range interactions in the folded structure. GNNs then learn on the residue-residue contact graph.

**Point Cloud Networks**: Treat C� atoms as a 3D point cloud with residue type features. PointNet and related architectures process these for structure classification and binding site prediction.

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

## Practical Considerations

### Tools and Libraries

Several key software ecosystems support molecular representations:

**RDKit**: The industry-standard open-source toolkit for cheminformatics. Handles SMILES parsing, molecular sanitization, fingerprint generation, 2D depiction, and much more. Essential for any small molecule work.

**DeepChem**: High-level ML library specifically for drug discovery. Provides data loaders, featurizers (representation converters), and model implementations. Integrates with TensorFlow and PyTorch.

**PyTorch Geometric (PyG)** and **Deep Graph Library (DGL)**: Leading frameworks for graph neural networks. Convert molecular graphs into tensors and provide GNN layer implementations.

**OpenMM** and **MDTraj**: Molecular dynamics simulation and trajectory analysis. Generate 3D conformers and analyze structural ensembles.

**Biopython**: Python tools for biological computation, including protein sequence/structure parsing, alignment, and feature extraction.

**Open Babel**: Converts between hundreds of chemical file formats. Useful for integrating diverse data sources.

### Data Sources

Large public databases provide training data:

**ChEMBL**: Bioactivity database with millions of compound-protein interactions, binding affinities, and functional assays. Curated from medicinal chemistry literature.

**PubChem**: Over 100 million chemical structures with associated properties. Searchable by structure, name, or properties.

**Protein Data Bank (PDB)**: Repository of 3D protein structures, including many protein-ligand complexes showing drugs bound to targets.

**UniProt**: Comprehensive protein sequence database with functional annotations.

**ZINC**: Database of commercially available compounds for virtual screening. Useful for finding molecules to actually purchase and test.

**GDB-13/GDB-17**: Enumerated databases of all possible small organic molecules up to a certain size. Used for theoretical studies of chemical space.

### Preprocessing Pipelines

Raw data from databases often requires cleaning and standardization:

**For molecules**:
- **Sanitization**: Remove invalid structures, check valence rules
- **Protonation state assignment**: Adjust charges for physiological pH (7.4)
- **Tautomer standardization**: Pick canonical form when multiple tautomers exist
- **Salt stripping**: Remove counterions (Na+, Cl-) that don't affect activity
- **Stereochemistry completion**: Assign undefined stereocenters

**For proteins**:
- **Structure alignment**: Superpose proteins for comparison
- **Missing residue modeling**: Fill gaps in experimental structures
- **Protonation**: Add hydrogens at physiological pH
- **Binding site extraction**: Define and crop pocket regions

Proper preprocessing is critical "garbage in, garbage out" applies strongly to molecular ML.

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

**Looking forward**:
- **Blog 3 (AlphaFold & Protein Structure Prediction)**: AlphaFold uses sequence representations with positional encodings and multiple sequence alignments (PSSMs capturing evolution). Its output is a distance matrix (2D representation of 3D structure) refined into full 3D coordinates. Understanding these representations explains how Transformers can predict geometry from text.

- **Blog 4 (Graph Neural Networks)**: Now that we understand molecular graphs nodes, edges, features, adjacency matrices we'll see how GNNs process them. Message passing aggregates information along edges, enabling models to learn from molecular topology.

- **Blog 5 (Generative Models)**: Generative models can operate in different representational spaces. Some generate SMILES strings autoregressively (character by character). Others generate molecular graphs directly (adding nodes and edges). The representation determines what the model creates and how we ensure chemical validity.

- **Blog 6 (3D Generative Models & Docking)**: Advanced models generate molecules in 3D, directly placing atoms in space while respecting SE(3) symmetries. We'll also see how docking algorithms use 3D protein and ligand representations to predict binding poses and affinities.

Each of these later topics builds on the representational foundations we've established here. Whether you're training a GNN to predict toxicity, using AlphaFold to get a protein structure, or generating novel drug candidates, you're ultimately manipulating one of these representations SMILES strings, fingerprints, graphs, or 3D geometries.

The key insight is that **representation is not just a preprocessing step it fundamentally shapes what models can learn**. A model operating on 2D graphs cannot predict chirality-dependent effects. A SMILES-based model struggles with molecules containing complex ring systems. Choosing the right representation is as important as choosing the right architecture.

In the next post, we'll see how AlphaFold revolutionized protein structure prediction by combining sequence representations with powerful Transformer architectures, solving a 50-year-old challenge and enabling structure-based drug design at unprecedented scale.

## References

1. Weininger, D. (1988). SMILES, a chemical language and information system. *Journal of Chemical Information and Computer Sciences*, 28(1), 31-36.

2. Rogers, D., & Hahn, M. (2010). Extended-connectivity fingerprints. *Journal of Chemical Information and Modeling*, 50(5), 742-754.

3. Duvenaud, D. K., et al. (2015). Convolutional networks on graphs for learning molecular fingerprints. *Advances in Neural Information Processing Systems*, 28.

4. Gilmer, J., et al. (2017). Neural message passing for quantum chemistry. *Proceedings of the International Conference on Machine Learning*, 1263-1272.

5. Sch�tt, K. T., et al. (2017). SchNet: A continuous-filter convolutional neural network for modeling quantum interactions. *Advances in Neural Information Processing Systems*, 30.

6. Unke, O. T., & Meuwly, M. (2019). PhysNet: A neural network for predicting energies, forces, dipole moments, and partial charges. *Journal of Chemical Theory and Computation*, 15(6), 3678-3693.

7. Jumper, J., et al. (2021). Highly accurate protein structure prediction with AlphaFold. *Nature*, 596(7873), 583-589.

8. Rives, A., et al. (2021). Biological structure and function emerge from scaling unsupervised learning to 250 million protein sequences. *Proceedings of the National Academy of Sciences*, 118(15), e2016239118.

9. Landrum, G. (2016). RDKit: Open-source cheminformatics. *https://www.rdkit.org*

10. Wu, Z., et al. (2018). MoleculeNet: a benchmark for molecular machine learning. *Chemical Science*, 9(2), 513-530.
