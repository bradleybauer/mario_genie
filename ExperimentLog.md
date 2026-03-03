# Experiment Log

## 2026-03-02: Single Sample Overfit Baseline

**Goal:** Verify the tokenizer can reconstruct a single video sequence near-perfectly.

**Setup:**
- `--overfit-one --epochs 50 --lr 3e-4 --val-interval 500`
- `init_dim=32`, `codebook_size=4096`, `seq_len=16`

**Findings:**
- Found a **codebook bypass bug**: `encode()`/`decode()` skipped quantization entirely.
  Fixed by using `tokenize()` + `decode_from_code_indices()`.
- After fix, recon loss drops to ~0.0006 by epoch 7.
- Codebook usage: ~1700/4096 codes for a single sample.

**Artifacts:** `checkpoints/magvit2/metrics.json`, `checkpoints/magvit2/step_*.png`

---

## 2026-03-02: Codebook Size Sweep (Single Sample)

**Goal:** Understand how codebook size affects recon quality and usage.

**Setup:**
| Run | codebook_size | final recon_loss | codes used |
|-----|--------------|-----------------|------------|
| A   | 256          |                 |            |
| B   | 1024         |                 |            |
| C   | 4096         |                 |            |
| D   | 8192         |                 |            |

**Findings:**
I wonder if the codebook usage related loss term is hurting results since I am purposefully overfitting on a single sample?

**AI**
Batch size affects the practical behavior of the codebook usage (entropy/diversity) loss term in Lookup-Free Quantization (LFQ).

In LFQ, the entropy loss term is calculated based on the distribution of quantized values over the current batch to encourage uniform codebook usage and prevent index collapse.

Larger batch sizes provide a better, more stable statistical estimate of the overall usage distribution, leading to more accurate and effective entropy loss gradients.
Smaller batch sizes can result in noisy or skewed entropy calculations, since a small sample of data may not naturally use the full codebook vocabulary. This can make the regularization less effective or lead to training instability.
While the loss is typically mathematically averaged over the batch, the quality of the underlying distribution estimate strictly depends on the batch size.

During a "single sample" sanity check (overfitting a single input), the codebook usage (entropy/diversity) loss will actively interfere with training.

Here is how it affects the process:

Conflict of Objectives: To perfectly memorize a single sample, the model needs to map its features to a specific, small subset of tokens predictably.
Forced Diversity: The entropy loss heavily penalizes this exact behavior. It tries to force the model to use the entire vocabulary. Since the input provides no natural diversity, the loss pushes the model to randomly assign different codes to the same features to artificially inflate entropy.
This conflict causes severe instability and prevents the model from converging and memorizing the sample perfectly.

Solution: When performing a single-sample sanity check, you should set the weight of the entropy/diversity loss (often called entropy_loss_weight or diversity_loss_weight) to 0.

**HUMAN**
So.... I modified the training script to ignore (set to zero) the entropy loss term while overfit_one is true.
The next test showed very poor results early in training.
So I change the script to simply reduce the entropy loss by 1/10th while overfit_one is true.
That resulted in this reconstruction after 500 steps.

![alt text](pictures/image.png)

This image seems to represent "codebook collapse".
The model improves later in training but not to the quality it achieved without this change to the entropy loss weight.
Given that neither of these modifications worked well, I will return the script to it's initial state.



---

## Template

**Goal:**
**Setup:**
**Findings:**
**Artifacts:**