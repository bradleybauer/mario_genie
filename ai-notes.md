# Mario World Model

### Note on Cross-Episode Sequences
The data collection script writes out fixed-length sequences (e.g. 16 frames). Because environments auto-reset upon death or level completion, **a single sequence can seamlessly cross episode boundaries.** This means a sequence may start in one world/stage and jump to another midway through.
Downstream models utilizing 3D convolutions (like the MAGVIT-2 video tokenizer) will perceive these boundaries as sudden "scene cuts". While standard models typcially learn to compress these jump cuts properly, if you require strictly continuous patches for training, make sure to evaluate the `dones` flag arrays to mask or split the dataset accordingly.

*(Note: The current training scripts keep windows with zero or one scene cut, and skip windows that contain two or more scene cuts within the training span. Scene cuts are detected from `(world, stage)` transitions when metadata is present, with `dones` used as a fallback for older chunks.)*

## Video Tokenizer Architecture Notes

### Spatial Token Alignment
The video tokenizer is configured to produce a **16x16 grid of discrete tokens** per frame. This is a deliberate design choice grounded in the underlying structure of the NES hardware.

**Why 16x16?**
Super Mario Bros. constructs its world almost entirely from **16x16 pixel metatiles** (also called macroblocks) — 2x2 arrangements of the base 8x8 hardware sprites. The NES engine uses a single 8-bit index (0–255) to reference each unique metatile, meaning there are at most 256 unique building blocks per level theme. Every pipe, brick, question block, and ground tile is one of these metatiles.

By targeting a 16x16 latent grid, we achieve **1 discrete token = 1 NES metatile**, which provides a strong structural inductive bias that aligns the learned codebook with the actual primitives the game is built from.

**How it is achieved:**
- Input images are downscaled to `128x128` (from the native `256x240`).
- The tokenizer applies 3 `compress_space` layers, each halving the spatial dimension ($2^3 = 8\times$ total reduction).
- $128 / 8 = 16$, yielding the 16x16 latent grid.

A native `256x256` input with 4 `compress_space` layers would achieve identical token alignment with sharper pixel fidelity, but was ruled out due to the increased computational cost.

### Lookup-Free Quantization (LFQ) Internals

The discrete latent produced by LFQ is effectively `project_out(quantize(project_in(x)))`. The quantization step itself is just a **sign function** — each dimension of the projected input snaps to ±1 (scaled by `codebook_scale`), so there's no learned codebook at all. The codebook is implicit: every possible combination of ±1 across `d` dimensions defines one of the `2^d` codes.

Concretely:
1. `project_in`: linear projection from feature dim → `codebook_dim × num_codebooks`
2. `quantize`: `torch.where(x > 0, +scale, -scale)` — pure sign thresholding
3. `project_out`: linear projection back to feature dim

Gradients pass through the quantization via the straight-through estimator (`x + (quantized - x).detach()`). An entropy auxiliary loss encourages codebook utilization: per-sample entropy is minimized (confident predictions) while batch-wide entropy is maximized (uniform code usage). The discrete indices are just the binary pattern of positive/negative dimensions packed into integers.


# Testing image-only AE with continuous latent and diffusion
Prompted claude to generate a small model that more closely resembles what genie-2 might look like.

It produced a 1.3 million parameter image-only autoencoder which very quickly memorized a small dataset of 5 images.

1. **Frame VAE** — 2D conv encoder/decoder compresses 224×224 palette-indexed frames to continuous 4×14×14 latents (~1.6M params).
2. **Dynamics model** — Causal transformer (6 layers, 256 dim) processes context latents + action embeddings, then a small U-Net denoiser generates the next frame's latent via DDPM (~5.6M params total).

No discrete tokens — continuous latents with diffusion sampling.


### Streaming Decode Investigation

I investigated whether `scripts/play_autoencoder.py` could be sped up by carrying forward causal decoder state so each new frame would be reconstructed exactly as if the full 16-frame window had been decoded together.

**What was verified:**
- The MAGVIT decoder is temporally causal (`CausalConv3d`, causal time downsampling, time upsampling).
- A cached **decoder-only** streaming implementation can reproduce standard full-window decoding to numerical precision **when the discrete latent tokens are already known**.
- This means decoder-state caching is a good fit for a future **world-model rollout path** where the dynamics transformer already emits discrete tokens.

**What failed in `play_autoencoder.py`:**
- `play_autoencoder.py` does not start from discrete tokens. It first encodes live frames, then applies LFQ, then decodes.
- I built a streaming encoder + decoder path and compared it to standard 16-frame encode/decode.
- The streamed encoder matched the full encoder only up to very small floating-point differences.
- Those tiny differences matter because LFQ is sign-based: small pre-quantization changes can flip bits in the discrete code.
- Once LFQ indices differ, reconstructions are no longer equivalent to the normal 16-frame forward pass.

**Conclusion:**
- For the autoencoder viewer, **decoder-state carry alone is not enough** to preserve exact behavior.
- Exact equivalence would require a bit-identical streaming encoder before LFQ, not just a cached decoder.
- Because of that, `play_autoencoder.py` should remain on the correctness-first **full-window encode + decode** path unless approximation is acceptable.

**Practical implication:**
- Use full-window autoencoder reconstruction for live input visualization.
- Revisit decoder-state caching later in the **autoregressive world-model inference** path, where the model operates on already-discrete latent tokens and exact streaming is much more straightforward.

## Theoretical Minimum Reconstruction Loss

The palette tokenizer uses **cross-entropy** (`F.cross_entropy` with default mean reduction) over all pixels. For a single wrong pixel with optimal logits (probability ~0.5 on the correct class), the minimum recon loss is:

$$\text{recon\_loss} = \frac{\ln 2}{B \times T \times H \times W}$$

Example: batch=6, seq_len=16, image_size=224 → $N = 4{,}816{,}896$ pixels → minimum loss ≈ $1.44 \times 10^{-7}$.


## LTX-Video 2.3 Style DiT as Genie Replacement

### Motivation
Explored whether a 100× downscaled LTX-Video 2.3 architecture (DiT-based latent video diffusion) could replace the current three-module Genie-2 pipeline for joint audio-video world-model generation, targeting 20–100M total parameters.

### Current Genie-2 Architecture (for reference)
The existing pipeline in `scripts/train_genie2.py` has three separate modules:
1. **FrameVAE** (~1.6M params) — 2D conv encoder/decoder, compresses 224×224 palette-indexed frames to continuous 4×14×14 latents. Four 2× downsamples (224→112→56→28→14), VAE with KL penalty.
2. **DynamicsTransformer** — Causal transformer (6 layers, dim=256, 4 heads, MLP ratio 4). Patchifies each frame's latent grid (4×14×14 → 196 spatial tokens), interleaves with action embeddings, processes with causal masking. Outputs a context vector by pooling the last frame's spatial tokens.
3. **LatentDenoiser** — Small U-Net on 14×14 latents (down 14→7, up 7→14, base channels=64). Conditioned on DynamicsTransformer context + sinusoidal timestep embedding via additive broadcast. Predicts noise ε for DDPM (200 steps, cosine beta schedule).

Training is two-phase: (1) train autoencoder on reconstruction, (2) freeze AE, train dynamics+denoiser on next-frame latent prediction.

### Why LTX/DiT Is a Better Fit
- **Unified model**: DiT replaces both DynamicsTransformer and LatentDenoiser with a single transformer that does denoising via self-attention over spatiotemporal latent patches. Architecturally simpler.
- **Flow matching over DDPM**: LTX uses rectified flow matching (linear interpolation $x_t = (1-t)x_0 + t\epsilon$, velocity prediction). Trains faster, samples in 5–20 steps instead of 200 DDPM steps.
- **RoPE** for spatiotemporal position encoding instead of learned embeddings (better generalization).
- **Action conditioning via AdaLN** (adaptive layer norm) replaces LTX's text-conditioning cross-attention path — much cheaper and appropriate for our discrete action space.

### Scaling to 20–100M Parameters
LTX-Video full is ~2B. Scaling knobs for 100× reduction:

| Knob             | LTX Full | Target (20–100M) |
|------------------|----------|-------------------|
| Hidden dim       | 2048+    | 256–512           |
| Layers           | 28+      | 6–12              |
| Heads            | 16+      | 4–8               |
| Latent spatial   | variable | 14×14 or 16×16    |
| Latent temporal  | variable | 4–16 frames       |

Example config: dim=384, 8 layers, 6 heads on 4×14×14 latents (196 spatial patches × ~8 frames) ≈ **30–40M params** unified model.

### Causal Temporal Masking
For autoregressive rollout, mask attention so frame $t$ only attends to frames $\leq t$. LTX already does this for video continuation, so the architecture natively supports it.

### Attention Feasibility
With 196 spatial tokens × 16 frames = 3,136 tokens, full causal attention is feasible at this model size. Windowed attention optional but not strictly necessary.

### Key Modifications to LTX for This Project
1. **Keep existing tokenizer** — reuse the MAGVIT-2 (discrete, 16×16) or FrameVAE (continuous, 14×14) to produce latent patches. No need to train a new 3D VAE.
2. **Action conditioning via AdaLN** — replace text cross-attention with adaptive layer norm conditioned on action embeddings.
3. **Flow matching replaces DDPM** — linear interpolant, velocity prediction, 5–20 sampling steps.
4. **Causal temporal masking** built into the attention for autoregressive rollout.

---

## Joint Audio-Video Generation

### Audio Data Source
We have **AVI files with perfectly synced video+audio**, bypassing the earlier limitation of only having APU register values in RAM dumps. The SMB1 RAM map (`src/mario_world_model/smb1_memory_map.py`) does contain audio registers (`area_music` 0x00FB, `event_music` 0x00FC, `sfx_reg1-3` 0x00FD-0x00FF) but these are high-level triggers, not waveform data. Perfect audio reconstruction from registers alone was not feasible — the AVI files solve this.

### Audio Tokenization Strategy
NES audio is low-complexity (4 synthesis channels: 2 pulse, 1 triangle, 1 noise + optional DPCM). At 60fps, one frame ≈ 16.7ms of audio. At 16kHz sample rate (sufficient for NES frequency range), that's ~267 samples per frame.

**Approach**: Train a small 1D convolutional audio VAE on per-frame audio chunks:
- Extract audio: `ffmpeg -i input.avi -ar 16000 -ac 1 output.wav`
- Align frame indices to audio windows (each frame = 1/60s = ~267 samples at 16kHz)
- A 1D conv encoder compressing by 8–16× yields **~16–32 audio latent tokens per frame** (small compared to 196 video tokens)
- The audio VAE should be very small (sub-1M params) given NES audio simplicity

### Joint Sequence Layout in the DiT
Concatenate audio latent patches with video latent patches in the transformer sequence:
```
[action_t, video_patches_t (196 tokens), audio_patches_t (~16-32 tokens), action_{t+1}, ...]
```

The DiT jointly denoises both modalities using a shared diffusion timestep, with modality-specific input/output projection layers. This is how recent joint audio-video diffusion models work.

### Parameter Budget
- Audio VAE: sub-1M params
- Video tokenizer (existing): ~1.6M params (FrameVAE) or existing MAGVIT
- Unified DiT: 30–50M params (handles both dynamics prediction and denoising)
- **Total: well within 20–100M target**

### What Wouldn't Work Well
- Full bidirectional spatiotemporal attention over very long sequences at 20M params (but causal/windowed attention is fine).
- Trying to generate high-fidelity audio waveforms purely from APU register prediction (bypassed by having actual audio in AVIs).
- Training a full 3D video VAE from scratch at this scale (unnecessary — reuse existing tokenizers).