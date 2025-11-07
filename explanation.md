# Learning from Failure (LfF): How the Loss Function Reduces Bias

## 🧠 Overview

**Learning from Failure (LfF)** (Nam et al., NeurIPS 2020) is a training strategy designed to reduce bias in deep learning models without needing explicit bias labels.  
It works by training two models simultaneously:

1. **Biased model** \( f_B \): intentionally learns the *spurious correlations* (the bias).
2. **Debiased model** \( f_D \): learns from the *failures* of the biased model to focus on truly informative features.

---

## ⚙️ Step 1: Biased Model — Generalized Cross Entropy (GCE)

The biased model \( f_B \) is trained using **Generalized Cross Entropy (GCE)**:

\[
L_{\text{GCE}}(p(x; \theta_B), y) = \frac{1 - p_y(x; \theta_B)^q}{q}, \quad q \in (0, 1]
\]

where \( p_y(x; \theta_B) \) is the softmax probability assigned to the true label \( y \).

- When \( q \to 0 \), it becomes the standard cross-entropy.
- For \( 0 < q < 1 \), it **amplifies the effect of “easy” samples**, giving them larger gradient contributions.

### 🔹 Why This Causes Bias

In biased datasets, **bias-aligned samples** (those following the spurious correlation) are *easier to learn*.  
Thus, GCE emphasizes these samples, pushing \( f_B \) to become *intentionally biased*.  
This model captures the “prejudice” that the algorithm aims to correct later.

---

## ⚙️ Step 2: Debiased Model — Weighted Cross Entropy

The **debiased model** \( f_D \) is trained using a **weighted cross-entropy** loss:

\[
L_D = \sum_{(x, y)} W(x) \cdot CE(f_D(x), y)
\]

The weight \( W(x) \) is computed as:

\[
W(x) = \frac{CE(f_B(x), y)}{CE(f_B(x), y) + CE(f_D(x), y)}
\]

where \( CE(\cdot) \) is the standard cross-entropy loss.

---

## 🧩 Step 3: How Weighting Reduces Bias

| Case | What Happens | Weight \( W(x) \) | Effect on \( f_D \) |
|------|---------------|--------------------|------------------------|
| **Bias-aligned sample** | Biased model \( f_B \) performs well → low loss | Small | Downweighted |
| **Bias-conflicting sample** | Biased model struggles → high loss | Large | Upweighted |

Therefore, \( f_D \) **focuses on the bias-conflicting samples**, i.e., the ones the biased model fails on.  
These samples require understanding the *true* underlying concept rather than shortcut correlations.

---

## 🧮 Training Algorithm Summary

**Algorithm 1: Learning from Failure**

1. Update biased model:
   \[
   \theta_B \leftarrow \theta_B - \eta \nabla_{\theta_B} \sum GCE(f_B(x), y)
   \]
2. Update debiased model:
   \[
   \theta_D \leftarrow \theta_D - \eta \nabla_{\theta_D} \sum W(x) \cdot CE(f_D(x), y)
   \]

The **relative difficulty weight** \( W(x) \) dynamically encourages the debiased model to prioritize difficult (bias-conflicting) samples.

---

## 🎯 Intuitive Summary

| Model | Loss Type | Learns From | Purpose |
|--------|------------|-------------|----------|
| **Biased model** \( f_B \) | Generalized Cross Entropy | Easy, bias-aligned samples | To *amplify* bias |
| **Debiased model** \( f_D \) | Weighted Cross Entropy | Hard, bias-conflicting samples | To *remove* bias |

Hence, the loss functions in LfF play **complementary roles**:
- \( L_{\text{GCE}} \) amplifies bias,
- \( W(x) \cdot CE \) suppresses it.

By learning from where the biased model fails, LfF teaches the debiased model to rely on **true causal features** instead of spurious ones.

---

**Reference:**  
Nam, Junhyun, et al. *"Learning from Failure: Training Debiased Classifier from Biased Classifier."* NeurIPS 2020.
