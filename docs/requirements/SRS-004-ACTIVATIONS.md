# SRS-004: Deterministic Activations & Layer Utilities

| Field | Value |
|-------|-------|
| **ID** | SRS-004 |
| **Component** | Core / Layers |
| **Status** | In Progress |
| **Compliance** | ISO 26262, MISRA-C:2012 |
| **Applicability** | Neural Network Layers (Dense, Conv2D) |

## 1. Purpose

This module defines the non-linear transformations (activations) and vector utilities required to complete a neural network layer's forward pass. These operations must be deterministic, memory-efficient, and suitable for safety-critical certification.

## 2. Requirements

### 2.1 Memory Efficiency

**SRS-004.1: In-Place Processing**

To minimize memory footprint, activation functions shall operate in-place on existing matrix buffers.

**Rationale:**
- Embedded systems have limited RAM
- Creating temporary buffers doubles memory requirements
- In-place operations maintain O(1) space complexity
- Critical for real-time safety-critical systems

**Implementation:** Functions modify input matrix directly, no new allocations.

**Verification:** Memory profiling confirms no additional allocations during activation.

### 2.2 Functional Requirements

**SRS-004.2: ReLU Determinism**

The ReLU function shall implement f(x) = max(0, x) with bit-perfect consistency across all platforms.

**Mathematical Definition:**
```
ReLU(x) = { x  if x > 0
          { 0  if x ≤ 0
```

**Rationale:**
- ReLU is the most common activation in safety-critical AI
- No transcendental functions (unlike sigmoid/tanh)
- Simple comparison operation is deterministic
- Same result on all architectures

**Verification:** Cross-platform testing confirms bit-identical results.

---

**SRS-004.3: Bias Vector Addition**

The system shall provide a utility to add a 1×N bias vector to each row of an M×N result matrix.

**Mathematical Definition:**
```
For matrix Y (M×N) and bias b (1×N):
Y[i][j] = Y[i][j] + b[j] for all i ∈ [0, M), j ∈ [0, N)
```

**Rationale:**
- Required for standard dense layer: y = activation(Wx + b)
- Broadcasting bias across rows is common NN operation
- Must be deterministic and efficient

**Implementation:** Sequential addition with fixed-point arithmetic.

**Verification:** Unit tests confirm correct broadcasting behavior.

---

**SRS-004.4: Bounded Arithmetic**

All additions in this module shall utilize the `fixed_t` arithmetic defined in SRS-002 to ensure consistent overflow behavior.

**Rationale:**
- Integer overflow is undefined behavior in C
- Fixed-point arithmetic provides defined overflow semantics
- Maintains consistency with rest of system

**Verification:** Static analysis confirms only fixed_t operations used.

### 2.3 Performance Requirements

**SRS-004.5: Predictable Execution Time**

Activation functions shall have predictable execution time proportional only to input size, with no data-dependent branching in inner loops.

**Rationale:**
- Required for real-time systems
- Simplifies worst-case execution time (WCET) analysis
- Prevents timing-based side-channel attacks

**Note:** Conditional for sign check (x < 0) is acceptable as it doesn't vary significantly across platforms.

**Verification:** Timing analysis confirms O(n) behavior.

## 3. Activation Functions

### 3.1 ReLU (Rectified Linear Unit)

**Function:** `void fx_relu(fx_matrix_t* mat)`

**Description:** Most common activation in modern neural networks.

**Properties:**
- Simple: max(0, x)
- Efficient: One comparison per element
- Non-saturating: No vanishing gradient problem
- Deterministic: Bit-perfect across platforms

**Use Cases:**
- Hidden layers in MLPs
- Convolutional neural networks
- Most safety-critical AI applications

---

### 3.2 Leaky ReLU

**Function:** `void fx_leaky_relu(fx_matrix_t* mat, fixed_t alpha)`

**Description:** Variant that allows small negative values.

**Mathematical Definition:**
```
LeakyReLU(x) = { x        if x > 0
               { alpha*x  if x ≤ 0
```

**Typical alpha:** 0.01 (allows 1% gradient for negative values)

**Properties:**
- Prevents "dead neurons" (neurons that never activate)
- Still deterministic with fixed-point multiply
- Slightly more complex than ReLU

**Use Cases:**
- When dead neuron problem observed
- Some computer vision applications

---

### 3.3 Future Activations (Planned)

**SRS-004.6:** (Planned) Softmax for Classification

**Challenge:** Requires exp() which is non-deterministic with floats.

**Solution:** Fixed-point exp approximation using Taylor series with limited terms.

**Status:** Not yet implemented, planned for future release.

## 4. Layer Utilities

### 4.1 Bias Addition

**Function:** `void fx_matrix_add_bias(fx_matrix_t* mat, const fx_matrix_t* bias)`

**Description:** Broadcasts 1×N bias vector to each row of M×N matrix.

**Example:**
```
Matrix (2×3):        Bias (1×3):       Result (2×3):
[1  2  3]            [10  20  30]      [11  22  33]
[4  5  6]                              [14  25  36]
```

**Properties:**
- In-place modification
- Row-major iteration for cache efficiency
- Dimension validation before operation

**Use Cases:**
- Required for every dense layer
- Some convolutional layer variants

## 5. Dense Layer Forward Pass

**Complete equation:** y = activation(Wx + b)

**Implementation sequence:**
1. Matrix multiplication: `fx_matrix_mul(&inputs, &weights, &output)`
2. Bias addition: `fx_matrix_add_bias(&output, &bias)`
3. Activation: `fx_relu(&output)`

**Result:** Fully deterministic neural network layer.

## 6. Verification Criteria

**V-004.1: ReLU Correctness**

Test vectors:
- Positive values remain unchanged
- Negative values become zero
- Zero remains zero

**Pass Criteria:** All test cases bit-identical across platforms.

---

**V-004.2: In-Place Operation**

Verify activation functions:
- Modify input matrix directly
- Do not allocate new memory
- Maintain same memory address

**Pass Criteria:** Memory profiling shows no allocations.

---

**V-004.3: Bias Broadcasting**

Test bias addition with various matrix sizes:
- 1×1, 2×2, 10×10, 100×10
- Verify each row gets bias added
- Check dimension mismatch handling

**Pass Criteria:** All tests pass, invalid operations rejected safely.

## 7. Design Rationale

### Why ReLU Over Sigmoid/Tanh?

| Property | ReLU | Sigmoid | Tanh |
|----------|------|---------|------|
| Computation | max(0,x) | 1/(1+e^-x) | (e^x-e^-x)/(e^x+e^-x) |
| Deterministic | ✅ | ❌ (exp varies) | ❌ (exp varies) |
| Efficient | ✅ | ❌ (expensive) | ❌ (expensive) |
| Certifiable | ✅ | ❌ | ❌ |
| Gradient | Not saturating | Saturates | Saturates |

**Decision:** ReLU is the only practical activation for safety-critical certification.

### In-Place vs Copy Operations

**In-Place (Selected):**
- ✅ O(1) space
- ✅ Cache-friendly
- ✅ Minimal memory footprint

**Copy (Rejected):**
- ❌ 2× memory usage
- ❌ Additional allocation overhead
- ❌ Cache thrashing

## 8. Implementation

**Files:**
- `include/activations.h` - API specification
- `src/core/activations.c` - Implementation
- `src/core/matrix.c` - Bias addition utility
- `tests/unit/test_activations.c` - Verification

**Traceability:**
- Code: `@traceability SRS-004-ACTIVATIONS`
- Tests: Link to this document in header

## 9. Usage Example

```c
/* Neural network layer: y = ReLU(Wx + b) */

// Pre-allocated buffers
fixed_t input_buf[10];
fixed_t weight_buf[10 * 5];  // 10×5 weights
fixed_t bias_buf[5];
fixed_t output_buf[1 * 5];   // 1×5 output

// Initialize matrices
fx_matrix_t input, weights, bias, output;
fx_matrix_init(&input, input_buf, 1, 10);
fx_matrix_init(&weights, weight_buf, 10, 5);
fx_matrix_init(&bias, bias_buf, 1, 5);
fx_matrix_init(&output, output_buf, 1, 5);

// Fill input with sensor data
// ... load weights and bias from trained model

// Forward pass (deterministic inference)
fx_matrix_mul(&input, &weights, &output);  // Wx
fx_matrix_add_bias(&output, &bias);        // + b
fx_relu(&output);                          // activation

// Result in output.data[] is ready for next layer
```

## 10. Performance Characteristics

**Time Complexity:**
- ReLU: O(M×N) where M×N = matrix dimensions
- Leaky ReLU: O(M×N)
- Bias addition: O(M×N)

**Space Complexity:**
- All operations: O(1) (in-place)

**Cache Behavior:**
- Sequential memory access
- Row-major iteration
- Excellent cache locality

## 11. Future Extensions

**SRS-004.7:** (Planned) Fixed-Point Sigmoid Approximation

Use piecewise linear approximation or limited Taylor series for applications requiring bounded outputs.

**Trade-off:** Adds complexity but enables classification layers.

---

**SRS-004.8:** (Planned) Batch Normalization

Normalize activations for training stability.

**Challenge:** Requires statistics (mean/variance) which must be pre-computed and stored deterministically.

## 12. References

- **MISRA-C:2012** - Rule 10.3 (Operand type preservation)
- **ISO 26262-6:2018** - Software unit design
- **IEC 62304:2006** - Medical device software
- **Goodfellow et al.** - Deep Learning (Chapter 6, activation functions)
- **He et al. (2015)** - "Delving Deep into Rectifiers" (ReLU paper)

## 13. Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2026-01-15 | William Murray | Initial version |

---

**Document Classification:** Technical Specification  
**Approval Status:** Approved for Implementation  
**Next Review:** 2026-04-15
