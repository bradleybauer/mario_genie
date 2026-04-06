# Matrix-Game 3.0: Model and Training Methods (Dataset-Minimized)

This document condenses the Matrix-Game 3.0 report into a model/training-focused technical description. It intentionally de-emphasizes dataset and data-engine details.

## 1. Problem Setting and Design Targets

Matrix-Game 3.0 targets an interactive video world model that must satisfy three constraints at once:

1. Long-horizon consistency: preserve scene identity/layout over minute-long rollouts.
2. Action controllability: respond reliably to keyboard and mouse inputs.
3. Real-time throughput: support streaming generation around 720p at up to ~40 FPS.

The key challenge is that these goals are coupled: faster few-step inference often worsens error accumulation, while stronger long-range memory can add compute overhead and latency.

## 2. High-Level Method Stack

The method is a co-design of four components:

1. Error-aware interactive base model.
2. Camera-aware long-horizon memory mechanism.
3. Training-inference aligned few-step distillation.
4. System-level inference acceleration.

The central principle is alignment: train the model under imperfect contexts resembling its own autoregressive inference conditions.

## 3. Core Base Model Architecture

## 3.1 Latent-space diffusion transformer

- Backbone: bidirectional Diffusion Transformer (DiT) in latent space.
- Input partition:
  - Past/history latent frames: $x_{1:k}$
  - Current target frames to predict: $x_{k+1:N}$ (noised during training)
- Objective is applied only to current latent frames.

The model predicts the flow/noise target for denoising current frames while conditioning on history.

## 3.2 Action conditioning

- Discrete keyboard actions are injected via cross-attention.
- Continuous mouse-control signals are injected via self-attention modulation.

This split is intended to improve control precision without destabilizing visual generation quality.

## 3.3 Unified teacher-student architecture choice

Unlike heterogeneous teacher-student pipelines, Matrix-Game 3.0 keeps both base and distilled models on the same bidirectional architecture family. Claimed benefit:

- Reduced architectural mismatch during distillation.
- Better stability of downstream few-step model training.

## 4. Error-Aware Self-Correction Training

To reduce train-test mismatch, the base model is trained with imperfect conditioning contexts using an error buffer.

## 4.1 Error collection

From predicted clean latent estimate $\hat{x}_i$, define residual:

$$
\delta = \hat{x}_i - x_i
$$

Residuals are stored in error buffer $E$.

## 4.2 Error injection

Sample residuals from the buffer and perturb conditioning history:

$$
\tilde{x}_i = x_i + \gamma \delta, \quad \delta \sim \text{Uniform}(E)
$$

Training objective (flow-matching form) uses perturbed context:

$$
\mathcal{L} = \mathbb{E}\left[\left\| (\epsilon - x_{k+1:N}) - v_\theta(x^t_{k+1:N}, t \mid \tilde{x}_{1:k}, c) \right\|_2^2\right]
$$

where $c$ is action conditioning.

Practical effect: the model learns to recover from accumulated rollout errors instead of relying on clean ground-truth context only.

## 5. Long-Horizon Memory Mechanism

## 5.1 Memory retrieval strategy

Memory is selected with camera-aware retrieval (pose/FOV relevance), rather than purely latent similarity routing. Goal: retrieve frames likely to be geometrically useful for current view reconstruction.

## 5.2 Unified memory + history + prediction attention

Instead of a separate memory branch, retrieved memory latents $m_{1:r}$, past latents $x_{1:k}$, and noised current latents are concatenated and processed in the same DiT self-attention space.

Why this matters:

- Memory and prediction features co-evolve in one denoising hierarchy.
- Reduces branch misalignment issues of cross-attention-only memory injection.

## 5.3 Geometry-aware conditioning

Relative camera geometry (Plucker-style cues) is injected so the model can align retrieved memory with current viewpoint. This reduces view-inconsistent memory copy behavior.

## 5.4 Memory-path error robustness

Error collection/injection is extended to both history and memory:

$$
\tilde{x}_{1:k} = x_{1:k} + \gamma_h\delta, \quad \tilde{m}_{1:r} = m_{1:r} + \gamma_m\delta
$$

with shared residual buffer across history/memory/current latents.

Memory-aware training objective:

$$
\mathcal{L}_{\text{mem}} = \mathbb{E}\left[\left\| (\epsilon - x_{k+1:N}) - v_\theta(x^t_{k+1:N}, t \mid \tilde{x}_{1:k}, \tilde{m}_{1:r}, c, g) \right\|_2^2\right]
$$

where $g$ is geometric conditioning.

## 5.5 Temporal encoding refinements

- Inject original frame indices (global temporal position) into temporal RoPE construction.
- Use head-wise perturbed RoPE base to reduce periodic aliasing across distant frames:

$$
\hat{\theta}_h = \theta_{\text{base}}(1 + \sigma_\theta\epsilon_h)
$$

This is intended to preserve long-range memory access while discouraging accidental phase alignment/copy artifacts.

## 6. Training-Inference Aligned Few-Step Distillation

## 6.1 Motivation

Single-window distillation with ground-truth past frames does not match streaming autoregressive inference, producing exposure bias.

## 6.2 Multi-segment self-generated rollout

Student model is trained by rolling out multiple segments:

1. Each segment begins from random noise.
2. Current segment past frames come from previous segment outputs.
3. Memory is retrieved online from an updated memory pool.
4. First segment runs in I2V-style mode (no memory available yet).

Training samples a random stop segment and applies teacher/critic supervision there.

## 6.3 DMD objective framing

Distillation uses Distribution Matching Distillation (DMD), approximating reverse-KL gradient via difference of data and generator scores at timestep $t$.

Conceptually:

- Encourage student distribution to match target data distribution under real rollout conditions.
- Keep student behavior consistent with deployment-time few-step streaming.

## 6.4 Practical schedule (as reported)

- Teacher/critic/student initialized from memory-augmented base checkpoint.
- Cold start:
  - Single-segment mode using ground-truth past frames.
  - Student LR: $5\times10^{-7}$, Critic LR: $1\times10^{-7}$
  - Student updates per iter: 5
  - Duration: 600 steps
- Multi-segment stage:
  - Number of segments sampled from 1..6
  - Student/Critic LR: $1\times10^{-7}$
  - Student updates per iter: 3
  - Duration: 2400 steps
- Past/memory masking probability: 0.2 (keeps I2V compatibility in training).

## 7. Base and Memory Model Training Configuration

Reported base/memory-stage settings:

- Base initialized from Wan2.2-TI2V-5B-style architecture with action modules in early DiT blocks.
- Base fine-tuning: LR $2\times10^{-5}$ for 50K steps.
- Training mix includes:
  - Mostly memory/history-conditioned clips.
  - Some masked-context I2V-style samples for first-segment robustness.
- Memory-augmented stage uses concatenated memory + past + current tokens in one DiT pass.

Core training principle across stages: intentionally include imperfect contexts and masked-context regimes so inference-time conditions are represented during optimization.

## 8. Real-Time Inference Path (Model/System Side)

After distillation, inference speed is improved with stacked optimizations:

1. DiT INT8 quantization (attention projection layers).
2. Lightweight pruned VAE decoder (MG-LightVAE, e.g., 50%/75% pruning variants).
3. GPU-based camera-aware memory retrieval (sampling approximation of overlap).
4. Asynchronous multi-GPU deployment (reported 8 GPUs for DiT + 1 for VAE decode).

Reported result: up to ~40 FPS at 720p in full optimized setup (5B model).

## 9. Scale-Up Strategy to MoE-28B

The report describes a staged capability split:

- High-noise model specializes in action control.
- Low-noise model focuses on visual refinement/generalization.

Also described:

- Progressive scaling of resolution and clip length during training.
- Viewpoint-specialized high-noise models (first-person vs third-person) with shared low-noise refinement model.

Intended outcome: improved dynamics, generalization, and minute-level coherence at larger scale.

## 10. What Is Novel in the Training Recipe

In compact form, the distinctive training choices are:

1. Shared bidirectional architecture for base and distilled models.
2. Error-buffer self-correction in both history and memory pathways.
3. Camera-aware retrieval plus unified self-attention (no isolated memory branch).
4. Multi-segment self-generated distillation aligned with streaming inference.
5. Coupling model-side and system-side acceleration for practical real-time deployment.

## 11. Minimal Reproduction Checklist (Methods Only)

If reproducing only model/training behavior (ignoring data pipeline details), implement:

1. Bidirectional latent DiT with action injection (keyboard cross-attn, mouse self-attn modulation).
2. History/current latent partition with flow-matching loss on current targets only.
3. Error buffer collection/injection over rollout residuals.
4. Camera-aware memory retrieval and joint memory-history-current self-attention.
5. Relative geometry conditioning + global temporal indexing in RoPE (+ optional head-wise base perturbation).
6. Two-stage distillation: cold-start single-segment, then multi-segment self-generated DMD.
7. Inference acceleration stack: attention-layer quantization, pruned decoder, GPU retrieval, async pipeline.
