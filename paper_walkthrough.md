# Neural Probabilistic Language Model (Bengio et al., 2003)

This repository contains an implementation and technical breakdown of the architecture proposed by Bengio et al. in the seminal 2003 paper. This model marked the transition from discrete N-gram counting to continuous **distributed representations** (embeddings) and neural function approximation.



---

## 1. Problem Statement

Traditional N-gram models suffer from the **curse of dimensionality**: as the context length increases, the number of possible sequences grows exponentially, leading to data sparsity.

Bengio et al. proposed:
1.  **Distributed Representations:** Mapping each word to a real-valued vector in **$\mathbb{R}^m$**.
2.  **Probability Function:** Expressing the joint probability of word sequences in terms of these feature vectors.
3.  **Simultaneous Learning:** Learning the word features and the parameters of the probability function (the neural network) at the same time.

---

## 2. Model Architecture
The model predicts the next word $w_t$ given the context of $n-1$ previous words.

### Parameters
| Parameter | Description | Typical Size |
| :--- | :--- | :--- |
| $V$ | Vocabulary size | 10k+ (Words) / ~70 (Chars) |
| $m$ | Embedding dimension | 30 - 100 |
| $n$ | Context window size | 3 - 10 |
| $h$ | Hidden layer units | 100 - 500 |
| $C$ | Embedding Matrix ($V \times m$) | Shared across all positions |

### Forward Pass Logic
1.  **Embedding:** Retrieve $e_j = C[w_{t-n+j}]$ for each word in the context.
2.  **Concatenation:** $x = (e_1, e_2, \dots, e_{n-1}) \in \mathbb{R}^{(n-1)m}$.
3.  **Hidden Layer:** $h = \tanh(Hx + b_h)$.
4.  **Output Logits:** $y = b + Wx + Uh$.
  - *Note*: ${Wx}$ is a direct linear connection from input to output.
5.  **Softmax:** 
  $$
  P(w_t = i) = \frac{e^{y_i}}{\sum e^{y_j}}
  $$

---

## 3. Training & Gradients
The model is trained by maximizing the log-likelihood of the training corpus, or minimizing the **Cross-Entropy Loss**:
$$\mathcal{L} = -\log P(w_t | \text{context})$$

### Backpropagation Formulas
Let $e = p - \mathbf{1}_{\text{target}}$ be the error at the output layer.

* **Output Weights:** $\frac{\partial \mathcal{L}}{\partial U} = e h^\top$ and $\frac{\partial \mathcal{L}}{\partial W} = e x^\top$.
* **Hidden Layer:** $\delta_h = (1 - h^2) \odot (U^\top e)$.
* **Hidden Weights:** $\frac{\partial \mathcal{L}}{\partial H} = \delta_h x^\top$.
* **Embedding Gradients:** $\frac{\partial \mathcal{L}}{\partial x} = W^\top e + H^\top \delta_h$.
    * *The result is sliced to update specific rows in ${C}$.*

---

## 4. Key Advantages
* **Generalization:** Similar words (in embedding space) lead to similar predictions for the next word, even if that specific sequence was never seen in training.
* **Linear Scaling:** The number of parameters grows linearly with $V$ and $n$, rather than exponentially.
* **Smoothness:** The continuous nature of the model provides a smooth probability manifold.

---

## 5. Character-Level Implementation Tips
When implementing this for characters instead of words:
* **Small $V$:** Since the vocabulary is small (~26-100), the bottleneck of the large softmax $O(V)$ disappears.
* **Longer Context:** Use a window of $n=10$ or more, as character dependencies are more local than word dependencies.
* **Preprocessing:** Map every character to an integer index and use a sliding window to create your dataset.

---

## 6. Performance Metrics
* **Perplexity:** The primary metric, calculated as $\exp(\text{Average NLL})$.
* **Comparison:** A well-tuned Neural N-gram typically outperforms Kneser-Ney smoothed models by a significant margin.