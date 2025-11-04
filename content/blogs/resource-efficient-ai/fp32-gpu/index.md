---
title: "Prerequisite: How 32-bit Floating-Point Arithmetic Works in GPUs"
date: 2025-11-04
author: Saeed Mehrang
draft: false
description: "A deep dive into the IEEE 754 Single-Precision (FP32) format, exploring its structure, dynamic range, changing resolution, and the low-level binary operations executed by GPUs in machine learning."
summary: "This blog highlights the FP32 (Single-Precision Floating-Point) format, detailing its division into sign, 8-bit exponent, and 23-bit mantissa. GPU kernels adhere to the IEEE 754 standard by performing complex binary arithmetic (like exponent alignment before mantissa addition) on these bit patterns."
tags: ["FP32", "IEEE 754", "GPU", "CUDA"]
showtoc: true
math: true
TocOpen: true
disableAnchoredHeadings: false
---


| Characteristic | Detail |
| :--- | :--- |
| **Estimated Reading Time** | 15-20 minutes |
| **Technical Level** | Intermediate |
| **Prerequisites** | Numbers Theory |


# Unpacking the Bits: How 32-bit Floating-Point Arithmetic Works in GPUs

The 32-bit Single-Precision Floating-Point (FP32) format is the backbone of modern scientific computing and machine learning. It provides a vast dynamic range necessary for training neural networks and complex simulations. This article dissects the core mechanics of FP32, explaining its structure, the trade-offs it imposes on precision, and the binary operations performed at the hardware level inside a GPU.

---

## The FP32 Structure: The IEEE 754 Standard

FP32 uses 32 bits divided into three distinct fields, as defined by the IEEE 754 standard. This partitioning allows the representation of numbers ranging from approximately $1.18 \times 10^{-38}$ to $3.4 \times 10^{38}$.

| Field | Bit Count | Role |
| :--- | :---: | :--- |
| **Sign Bit** (S) | 1 | Determines if the number is positive (0) or negative (1). |
| **Exponent Bits** (E) | 8 | Determines the scale or magnitude of the number (the power of 2). |
| **Mantissa/Fraction Bits** (F) | 23 | Determines the precision or significant digits. |

The value of a **normalized** FP32 number is calculated using the formula:

$$(-1)^{\text{Sign}} \times 2^{\text{Exponent} - \text{Bias}} \times (1 + \text{Fraction})$$

### The Exponent and Bias

The 8-bit exponent field offers 256 possible patterns (0 to 255). To allow the exponent to represent both positive and negative powers, a **Bias of 127** is subtracted from the stored value.

| Stored Exponent Range | True Exponent ($E_{\text{true}}$) Range | Purpose |
| :---: | :---: | :--- |
| **1 to 254** | -126 to 127 | **Normalized Numbers** (Standard range and precision). |
| **0** (All Zeros) | -127 | **Zero** and **Denormalized Numbers** (Subnormal range). |
| **255** (All Ones) | N/A | **Infinity** ($\pm \infty$) and **NaN** (Not a Number). |

The maximum true exponent is **127** (from $254 - 127$), which, when combined with the mantissa, defines the largest representable number. The minimum normalized true exponent is **-126** (from $1 - 127$), defining the smallest normalized positive number ($2^{-126}$).

### The Mantissa and Precision

The 23 mantissa bits, combined with an **implicit leading 1** (for normalized numbers), give an effective **24 bits of precision**. This is the key factor that determines the number's granularity.

---

## Non-Uniform Resolution and Dynamic Range

A critical concept in floating-point arithmetic is that the **absolute resolution** (the difference between two adjacent representable numbers) is **not constant** across the number line; it changes based on the exponent.

### The Mechanism of Changing Resolution

The resolution (the gap between consecutive representable numbers) is determined by the value of the least significant mantissa bit (LSB) scaled by the current exponent:

$$\text{Resolution} = 2^{\text{Exponent} - 23}$$

This formula shows that resolution **changes with the exponent**—larger exponents mean larger gaps between numbers.

---

#### **For Normalized Numbers** (Exponent: -126 to +127)

Normalized numbers have an implicit leading bit of **1**, giving them full 24-bit precision.


| True Exponent Value | Represented Number Range | Resolution (Gap Between Numbers) | Decimal Precision |
| :---: | :--- | :--- | :--- |
| **-126** (Minimum) | $\approx 1.18 \times 10^{-38}$ to $2.35 \times 10^{-38}$ | $2^{-149} \approx 1.4 \times 10^{-45}$ | ~6-9 digits |
| **0** | 1.0 to 2.0 | $2^{-23} \approx 1.19 \times 10^{-7}$ | ~6-9 digits |
| **1** | 2.0 to 4.0 | $2^{-22} \approx 2.38 \times 10^{-7}$ | ~6-9 digits |
| **10** | 1,024 to 2,048 | $2^{-13} \approx 1.22 \times 10^{-4}$ | ~6-9 digits |
| **127** (Maximum) | $\approx 1.7 \times 10^{38}$ to $3.4 \times 10^{38}$ | $2^{104} \approx 2.0 \times 10^{31}$ | ~6-9 digits |


**Key Insight:** As numbers grow larger (exponent increases), the **absolute gap** between consecutive numbers increases dramatically—from $\sim 10^{-45}$ near the smallest normalized number to $\sim 10^{31}$ near the largest. However, the **relative precision** (6-9 significant decimal digits) remains constant across all normalized numbers.

---

#### **For Subnormal Numbers** (Exponent Field: All Zeros)

When the exponent field is all zeros, the implicit leading bit becomes **0** (not 1), creating subnormal numbers that fill the gap between zero and the smallest normalized number.

| Characteristic | Value |
| :--- | :--- |
| **Effective Exponent** | -126 (same as minimum normalized) |
| **Leading Bit** | 0 (not 1) |
| **Number Range** | $2^{-149}$ to $(1-2^{-23}) \times 2^{-126} \approx 1.18 \times 10^{-38}$ |
| **Resolution** | **Fixed at** $2^{-149} \approx 1.4 \times 10^{-45}$ |
| **Smallest Positive Value** | $2^{-149}$ (mantissa = $0.00...01_2$) |
| **Precision** | **Degraded**: 1 to 23 bits (loses precision as numbers approach zero) |

**Key Insight:** Subnormal numbers provide **gradual underflow** near zero. Unlike normalized numbers where resolution scales with magnitude, subnormal numbers have a **fixed, constant resolution** of $2^{-149}$. This prevents abrupt transitions to zero but at the cost of reduced precision—numbers very close to zero may have only a few bits of precision instead of the full 24 bits.

---

**Summary:** FP32 trades absolute precision for dynamic range. Large numbers sacrifice fine-grained resolution, while tiny numbers (subnormals) sacrifice relative precision—but both strategies keep calculations numerically stable across 76+ orders of magnitude.



---

## FP32 in GPU Workloads: From Decimal to Binary

When training neural networks on an NVIDIA GPU using frameworks like PyTorch or TensorFlow, all high-speed computations occur exclusively on the **FP32 binary code**.

| Stage | Location | Action | Data Format |
| :--- | :--- | :--- | :--- |
| **1. Data Conversion** | CPU (PyTorch/TensorFlow) | Input decimal weights/biases are converted to the **IEEE 754 32-bit binary format**. | Decimal $\rightarrow$ FP32 Binary |
| **2. Data Transfer** | System Bus (PCIe) | The FP32 binary data is moved from CPU RAM to GPU VRAM. | FP32 Binary |
| **3. Operation Execution** | GPU (CUDA Kernels) | Kernels (optimized programs) execute math directly on the **FP32 binary bits**. | **FP32 Binary (Exclusive)** |
| **4. Final Output** | CPU and Software | The resulting FP32 binary data is transferred back and converted to a decimal string for display. | FP32 Binary $\rightarrow$ Decimal |

CUDA kernels, which perform the massive parallel matrix multiplications (e.g., using cuBLAS), are low-level programs designed to execute arithmetic that **strictly respects the rules of the IEEE 754 standard**.

---

## Low-Level Binary Arithmetic in GPU Kernels

The core of any floating-point operation, like addition, is not a single instruction but a multi-step process designed to maintain accuracy and follow the standard. The process operates solely on the 32-bit patterns.

### The FP32 Binary Addition Sequence

To calculate $A + B$ within a GPU kernel:

| Step | Operation on Bits | Description |
| :--- | :--- | :--- |
| **1. Exponent Alignment** | **Compare and Shift** | The 8-bit exponents of $A$ and $B$ are compared. The mantissa of the number with the smaller exponent is **right-shifted** by the difference in the exponents ($E_{max} - E_{min}$). This aligns the binary points. |
| **2. Mantissa Arithmetic** | **Binary Addition/Subtraction** | The aligned 24-bit mantissas (including the implicit '1') are added if signs are the same, or subtracted if signs are different. Subtraction is often implemented via **Two's Complement** arithmetic for efficiency. |
| **3. Normalization** | **Shift and Adjust** | The resulting mantissa is shifted left or right until it is in the standard "1.XXXX..." form. The new 8-bit exponent is adjusted to reflect this shift. |
| **4. Rounding** | **Check and Round** | The mantissa is rounded back to 23 stored bits (24 effective bits) based on the discarded bits, following the "round to nearest, ties to even" rule. |
| **5. Assembly** | **Combine Fields** | The final Sign, 8-bit Exponent, and 23-bit Mantissa are combined into the final 32-bit FP32 result. |

### Binary Addition and Subtraction Fundamentals

The elementary operations performed in Step 2 are pure binary arithmetic:

| Operation | Concept | Binary Logic |
| :--- | :--- | :--- |
| **Addition** ($A+B$) | Same-signed mantissas are added column-by-column, generating carries when $1+1$ occurs. | Simple binary addition with carry logic. |
| **Subtraction** ($A-B$) | Performed when signs are different. The smaller mantissa is typically converted to its **Two's Complement**, and then added to the larger mantissa. | Two's Complement conversion followed by standard binary addition. The final carry is discarded. |

This meticulous, step-by-step bit manipulation is what allows the GPU to correctly execute high-level mathematical operations while conforming to the fixed-size constraints and rules of the FP32 standard.


-----

### Numerical Examples for Binary Arithmetic

The GPU kernel executes binary addition and subtraction on the aligned mantissa bits using the following logic:



#### Binary Addition (Same Signs)

Binary addition follows four simple rules ($0+0=0, 0+1=1, 1+0=1, 1+1=0$ with a carry of 1).

##### Example 1: Simple Addition with One Carry ($5_{10} + 3_{10} = 8_{10}$)


| Carry: | **1** | **1** | **1** | |
| :---: | :---: | :---: | :---: | :---: |
| $A (5_{10})$ | $0$ | $1$ | $0$ | $1$ |
| $+ B (3_{10})$ | $0$ | $0$ | $1$ | $1$ |
| $\text{Sum (8}_{10})$ | **1** | **0** | **0** | **0** |


##### Example 2: Multiple Consecutive Carries ($13_{10} + 11_{10} = 24_{10}$)

| Carry: | | **1** | **1** | **1** | |
| :---: | :---: | :---: | :---: | :---: | :---: |
| $A (13_{10})$ | | $1$ | $1$ | $0$ | $1$ |
| $+ B (11_{10})$ | | $1$ | $0$ | $1$ | $1$ |
| $\text{Sum (24}_{10})$ | **1** | **1** | **0** | **0** | **0** |

---
#### Binary Subtraction (Different Signs via Two's Complement)

**Important Context:** This example demonstrates the binary arithmetic concept using **signed integer representation**. In FP32 hardware, the sign is stored separately in the sign bit, and mantissas are treated as **unsigned magnitude values**. However, the underlying binary addition logic shown here (including carry propagation) is the same mechanism used when the GPU performs mantissa alignment and arithmetic during floating-point operations.

Computers typically perform subtraction ($A - B$) by adding the first number ($A$) to the **Two's Complement** of the second number ($-B$). This is a fundamental technique in digital arithmetic.

##### Two's Complement Subtraction ($9_{10} - 5_{10} = 4_{10}$)

This is a simplified **4-bit signed integer example** to illustrate the binary arithmetic concept:

1.  **Numbers (4-bit):** $A = 9_{10} = 1001_2$, $B = 5_{10} = 0101_2$.
2.  **Find Two's Complement of $B$ (i.e., $-5$):**
    * $B$: $0101$
    * One's Complement (Flip bits): $1010$
    * Two's Complement (Add 1): $1010 + 1 = \mathbf{1011}$ (This represents $-5_{10}$ in 4-bit signed representation)
3.  **Add $A + (-B)$:**

| Carry: | **1** | **1** | **1** | |
| :---: | :---: | :---: | :---: | :---: |
| $A (9)$ | $1$ | $0$ | $0$ | $1$ |
| $+ (-B) (-5)$ | $1$ | $0$ | $1$ | $1$ |
| $\text{Sum}$ | $\mathbf{1}$ | **0** | **1** | **0** | **0** |

4.  **Result:** The final carry (the leading '1') is **discarded**. The 4-bit result is $\mathbf{0100}_2$, which equals $4_{10}$.

**Relation to FP32 Mantissa Arithmetic:** When a GPU performs FP32 addition/subtraction with opposite signs, it doesn't use two's complement on the mantissas themselves (since mantissas are unsigned and the sign is handled separately). Instead, the hardware:
- Compares the two aligned mantissas
- Subtracts the smaller from the larger using similar binary arithmetic
- Sets the result's sign based on which operand had the larger magnitude

The key takeaway is that the **binary addition with carry propagation** shown above is the fundamental operation used throughout FP32 arithmetic in GPU hardware.