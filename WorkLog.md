# Work Log

# Table of Contents

- [Single Sample Overfit Baseline](#single-sample-overfit-baseline)
- [Video Tokenizer Codebook Size Sweep](#video-tokenizer-codebook-size-sweep)
- [Single Sample Dataset & Codebook Entropy Loss Interaction](#single-sample-dataset--codebook-entropy-loss-interaction)
- [Which Tokens Correspond To Which World And Stages](#which-tokens-correspond-to-which-world-and-stages)

<br>
<br>

# Single Sample Overfit Baseline

**Context:** Initial tests showed that validation reconstruction was exceptionally bad. So much so that I suspected the training implementation contained a bug.

**Approach:** As a sanity check I thought to verify the tokenizer can reconstruct a single video sequence near-perfectly.

**Result:**
The validation reconstruction was still extremely poor even on the single sample dataset.
I then found a **codebook bypass bug**: `encode()`/`decode()` skipped quantization entirely in the validation path.
This was fixed by using `tokenize()` + `decode_from_code_indices()`.

<br>
<br>

# Single Sample Dataset & Codebook Entropy Loss Interaction

**Context:**
Intuitively the codebook size does not need to be large when overfitting on a single sample.

If the latent space is of size Lx16x16 then with sequences of length L=16 the maximum number of codes is 4096.

**Approach:**
To investigate the compatibility of the codebook usage loss term with single-sample datasets, I modified the training script to disable the entropy loss when `overfit_one` is true. This initial test yielded poor results early in training. 

Subsequently, I adjusted the script to merely scale down the entropy loss by a factor of 10 during the `overfit_one` condition. This resulted in the following reconstruction after 500 steps:

![What looks to me like codebook collapse](pictures/codebook_collapse.png)

This image seems to exhibits signs of "codebook collapse." Although the model's performance improves later in training, it fails to match the reconstruction quality achieved with the unmodified entropy loss weight.

**Result:**
Because both modifications degraded performance, I have reverted the script to its original state.

<details>
<summary><b>AI Explanation: Batch Size and Entropy Loss</b></summary>

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
</details>
<br>
<br>

# Which Tokens Correspond To Which World And Stages

**Context:** 
During inference I want the initial world and stage to be configurable.

**Approach:** 
I will use the trained autoencoder to encode the initial frame for each world and stage. Those encodings will later be used as the initial tokens during inference.

**Result:** 
TODO

<br>
<br>

# Uniform Data Collection

**Context:** 
I want a uniform distribution of training data across all world/stage/x-position bins. However, the emulator can only initialize Mario at the start of a stage, so later portions of each stage are naturally underrepresented — reaching them requires playing through the earlier parts first.

**Approach:** 
During collection, every episode's actions and x-positions are recorded as rollouts. At periodic rebalance intervals the collector scans progression coverage across the existing data, computes per-bin deficits (inverse sampling weights), and builds a replay pool of recorded action sequences that reach underrepresented bins. On each episode reset, the environment samples a target bin proportional to its deficit weight, replays the corresponding recorded actions to fast-forward the emulator to that position, and then resumes live collection from there.

**Result:** 
This is not fully solved — some bins remain under represented — but the distribution is significantly more uniform than naive sequential play.
![Progression distribution across world-stage bins](pictures/data_balance.png)

<br>
<br>

#

**Context:** 

**Approach:** 

**Result:** 

<!-- Template -->

<br>
<br>

# Title

**Context:** 

**Approach:** 

**Result:** 