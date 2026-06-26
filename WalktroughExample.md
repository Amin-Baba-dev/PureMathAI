# 🧮 Backpropagation: A Numerical Walkthrough

This document walks through the exact mathematics of the `backward()` method using a **toy network**. All numbers are calculated step‑by‑step so you can see exactly how the gradients flow through the chain rule.

**Toy network setup:**
- Batch size $m = 2$
- Input size $= 3$ (instead of 784)
- Hidden size $= 4$ (instead of 128)
- Output size $= 2$ (instead of 10)
- Learning rate $\text{lr} = 0.1$

---

## 1. The Setup (Forward Pass Cached Values)

Assume we have already run `forward(X)`. The following values are stored inside the network.

**Input batch $X$** (2 samples, 3 features):
$$ X = \begin{bmatrix} 0.5 & 0.2 & 0.9 \\ 0.1 & 0.8 & 0.3 \end{bmatrix} \quad \text{shape: } (2, 3) $$

**Weights & Biases** (randomly initialised):

$$ W_1 = \begin{bmatrix} 0.1 & 0.2 & 0.3 & 0.4 \\ 0.5 & 0.6 & 0.7 & 0.8 \\ 0.9 & 1.0 & 1.1 & 1.2 \end{bmatrix} \quad \text{shape: } (3,4) $$

$$ b_1 = \begin{bmatrix} 0.1 & 0.2 & 0.3 & 0.4 \end{bmatrix} \quad \text{shape: } (4,) $$

$$ W_2 = \begin{bmatrix} 0.1 & 0.2 \\ 0.3 & 0.4 \\ 0.5 & 0.6 \\ 0.7 & 0.8 \end{bmatrix} \quad \text{shape: } (4,2) $$

$$ b_2 = \begin{bmatrix} 0.1 & 0.2 \end{bmatrix} \quad \text{shape: } (2,) $$

---

### Forward Pass Results (Cached Internally)

**1. Hidden layer linear transform:**
$$ z_1 = X W_1 + b_1 = \begin{bmatrix} 1.06 & 1.32 & 1.58 & 1.84 \\ 0.78 & 1.00 & 1.22 & 1.44 \end{bmatrix} \quad \text{shape: } (2,4) $$

**2. ReLU activation** (all values positive, so unchanged):
$$ a_1 = \text{ReLU}(z_1) = \begin{bmatrix} 1.06 & 1.32 & 1.58 & 1.84 \\ 0.78 & 1.00 & 1.22 & 1.44 \end{bmatrix} \quad \text{shape: } (2,4) $$

**3. Output layer linear transform:**
$$ z_2 = a_1 W_2 + b_2 = \begin{bmatrix} 2.68 & 3.36 \\ 2.096 & 2.64 \end{bmatrix} \quad \text{shape: } (2,2) $$

**4. Softmax output (predicted probabilities):**
$$ a_2 = \text{softmax}(z_2) = \begin{bmatrix} 0.336 & 0.664 \\ 0.367 & 0.633 \end{bmatrix} \quad \text{shape: } (2,2) $$
*(Each row sums to 1.0)*

---

**True labels (one‑hot)** for these 2 samples:
- Sample 0 is class **0** → $[1, 0]$
- Sample 1 is class **1** → $[0, 1]$

$$ y_{\text{onehot}} = \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix} \quad \text{shape: } (2,2) $$

---

## 2. Backward Pass (Step‑by‑Step)

We now call `backward(X, y_onehot)` with $m = 2$.

---

### Step 1: Output Layer Error
The derivative of Cross‑Entropy + Softmax simplifies to:

$$ dz_2 = a_2 - y_{\text{onehot}} = \begin{bmatrix} 0.336 - 1 & 0.664 - 0 \\ 0.367 - 0 & 0.633 - 1 \end{bmatrix} = \begin{bmatrix} -0.664 & 0.664 \\ 0.367 & -0.367 \end{bmatrix} $$

**Shape:** $(2, 2)$

---

### Step 2: Gradient for $W_2$

$$ dW_2 = \frac{a_1^T \cdot dz_2}{m} $$

First, compute $a_1^T \cdot dz_2$:

$$ \begin{bmatrix} 1.06 & 0.78 \\ 1.32 & 1.00 \\ 1.58 & 1.22 \\ 1.84 & 1.44 \end{bmatrix} \cdot \begin{bmatrix} -0.664 & 0.664 \\ 0.367 & -0.367 \end{bmatrix} = \begin{bmatrix} -0.418 & 0.418 \\ -0.509 & 0.509 \\ -0.601 & 0.601 \\ -0.694 & 0.694 \end{bmatrix} $$

Divide by $m = 2$:

$$ dW_2 = \begin{bmatrix} -0.209 & 0.209 \\ -0.2545 & 0.2545 \\ -0.3005 & 0.3005 \\ -0.347 & 0.347 \end{bmatrix} \quad \text{shape: } (4,2) $$

---

### Step 3: Gradient for $b_2$

We sum $dz_2$ across rows (axis=0) and divide by $m$:

$$ db_2 = \frac{1}{2} \left( \begin{bmatrix} -0.664 \\ 0.367 \end{bmatrix} + \begin{bmatrix} 0.664 \\ -0.367 \end{bmatrix} \right) = \frac{1}{2} \begin{bmatrix} -0.297 \\ 0.297 \end{bmatrix} = \begin{bmatrix} -0.1485 \\ 0.1485 \end{bmatrix} $$

**Shape:** $(2,)$

---

### Step 4: Backpropagate to Hidden Layer

We compute the gradient flowing back to $a_1$:

$$ da_1 = dz_2 \cdot W_2^T $$

Where $W_2^T$ is:

$$ W_2^T = \begin{bmatrix} 0.1 & 0.3 & 0.5 & 0.7 \\ 0.2 & 0.4 & 0.6 & 0.8 \end{bmatrix} $$

Perform the multiplication:

$$ da_1 = \begin{bmatrix} -0.664 & 0.664 \\ 0.367 & -0.367 \end{bmatrix} \cdot \begin{bmatrix} 0.1 & 0.3 & 0.5 & 0.7 \\ 0.2 & 0.4 & 0.6 & 0.8 \end{bmatrix} $$

- Row 0: 
  $$ [-0.664(0.1) + 0.664(0.2),\ \dots] = [0.0664,\ 0.0664,\ 0.0664,\ 0.0664] $$
  
- Row 1:
  $$ [0.367(0.1) - 0.367(0.2),\ \dots] = [-0.0367,\ -0.0367,\ -0.0367,\ -0.0367] $$

Thus:

$$ da_1 = \begin{bmatrix} 0.0664 & 0.0664 & 0.0664 & 0.0664 \\ -0.0367 & -0.0367 & -0.0367 & -0.0367 \end{bmatrix} \quad \text{shape: } (2,4) $$

---

### Step 5: Gradient through ReLU

The ReLU derivative is $1$ for $z_1 > 0$, else $0$. Since all $z_1$ values are positive, the gradient passes unchanged:

$$ dz_1 = da_1 \odot \mathbf{1}_{z_1 > 0} = da_1 \quad \text{(element-wise)} $$

**Shape:** $(2, 4)$

---

### Step 6: Gradient for $W_1$

$$ dW_1 = \frac{X^T \cdot dz_1}{m} $$

Where $X^T$ is:

$$ X^T = \begin{bmatrix} 0.5 & 0.1 \\ 0.2 & 0.8 \\ 0.9 & 0.3 \end{bmatrix} \quad \text{shape: } (3,2) $$

Compute $X^T \cdot dz_1$:

- Row 0: 
  $$ [0.5(0.0664) + 0.1(-0.0367),\ \dots] = [0.02953,\ 0.02953,\ 0.02953,\ 0.02953] $$
  
- Row 1:
  $$ [0.2(0.0664) + 0.8(-0.0367),\ \dots] = [-0.01608,\ -0.01608,\ -0.01608,\ -0.01608] $$
  
- Row 2:
  $$ [0.9(0.0664) + 0.3(-0.0367),\ \dots] = [0.04875,\ 0.04875,\ 0.04875,\ 0.04875] $$

Divide by $m = 2$:

$$ dW_1 = \begin{bmatrix} 0.014765 & 0.014765 & 0.014765 & 0.014765 \\ -0.00804 & -0.00804 & -0.00804 & -0.00804 \\ 0.024375 & 0.024375 & 0.024375 & 0.024375 \end{bmatrix} \quad \text{shape: } (3,4) $$

---

### Step 7: Gradient for $b_1$

We sum $dz_1$ across rows (axis=0) and divide by $m$:

$$ dz_1 = \begin{bmatrix} 0.0664 & 0.0664 & 0.0664 & 0.0664 \\ -0.0367 & -0.0367 & -0.0367 & -0.0367 \end{bmatrix} $$

Summing column‑wise:
$$ \sum_{\text{rows}} dz_1 = \begin{bmatrix} 0.0664 + (-0.0367) & \dots & \dots & \dots \end{bmatrix} = \begin{bmatrix} 0.0297 & 0.0297 & 0.0297 & 0.0297 \end{bmatrix} $$

Divide by $m=2$:
$$ db_1 = \frac{1}{2} \begin{bmatrix} 0.0297 & 0.0297 & 0.0297 & 0.0297 \end{bmatrix} = \begin{bmatrix} 0.01485 & 0.01485 & 0.01485 & 0.01485 \end{bmatrix} $$

**Shape:** $(4,)$

---

## 3. Weight Updates (In‑Place)

We apply gradient descent with $\text{lr} = 0.1$:

$$ \theta \leftarrow \theta - \text{lr} \cdot \nabla_\theta $$

**Update $W_2$:**  
Example first row:
$$ \begin{bmatrix} 0.1 & 0.2 \end{bmatrix} \leftarrow \begin{bmatrix} 0.1 & 0.2 \end{bmatrix} - 0.1 \cdot \begin{bmatrix} -0.209 & 0.209 \end{bmatrix} = \begin{bmatrix} 0.1209 & 0.1791 \end{bmatrix} $$

**Update $b_2$:**  
$$ \begin{bmatrix} 0.1 & 0.2 \end{bmatrix} \leftarrow \begin{bmatrix} 0.1 & 0.2 \end{bmatrix} - 0.1 \cdot \begin{bmatrix} -0.1485 & 0.1485 \end{bmatrix} = \begin{bmatrix} 0.11485 & 0.18515 \end{bmatrix} $$

**Update $W_1$:**  
Example first row:
$$ \begin{bmatrix} 0.1 & 0.2 & 0.3 & 0.4 \end{bmatrix} \leftarrow \begin{bmatrix} 0.1 & 0.2 & 0.3 & 0.4 \end{bmatrix} - 0.1 \cdot \begin{bmatrix} 0.014765 & 0.014765 & 0.014765 & 0.014765 \end{bmatrix} = \begin{bmatrix} 0.0985235 & 0.1985235 & 0.2985235 & 0.3985235 \end{bmatrix} $$

**Update $b_1$:**  
$$ \begin{bmatrix} 0.1 & 0.2 & 0.3 & 0.4 \end{bmatrix} \leftarrow \begin{bmatrix} 0.1 & 0.2 & 0.3 & 0.4 \end{bmatrix} - 0.1 \cdot \begin{bmatrix} 0.01485 & 0.01485 & 0.01485 & 0.01485 \end{bmatrix} = \begin{bmatrix} 0.098515 & 0.198515 & 0.298515 & 0.398515 \end{bmatrix} $$

---

## 4. Summary Table of Gradients

| Variable | Derivation | Result Shape |
| :--- | :--- | :--- |
| $dz_2$ | $a_2 - y_{\text{onehot}}$ | $(2, 2)$ |
| $dW_2$ | $\frac{a_1^T \cdot dz_2}{m}$ | $(4, 2)$ |
| $db_2$ | $\frac{\text{sum}(dz_2, \text{axis}=0)}{m}$ | $(2,)$ |
| $da_1$ | $dz_2 \cdot W_2^T$ | $(2, 4)$ |
| $dz_1$ | $da_1 \odot \mathbf{1}_{z_1 > 0}$ | $(2, 4)$ |
| $dW_1$ | $\frac{X^T \cdot dz_1}{m}$ | $(3, 4)$ |
| $db_1$ | $\frac{\text{sum}(dz_1, \text{axis}=0)}{m}$ | $(4,)$ |

---

## 5. Conclusion

These gradients are used to **nudge the weights opposite to the gradient direction**, slowly minimising the cross‑entropy loss. This exact mathematical procedure repeats for every mini‑batch during training. 

By walking through the raw numbers, you can see that backpropagation is **just repeated matrix multiplication and element‑wise operations** — no black boxes, just linear algebra and calculus!
