# Mario World Model

## Video Tokenizer Architecture Notes

### Lookup-Free Quantization (LFQ) Internals

The discrete latent produced by LFQ is effectively `project_out(quantize(project_in(x)))`. The quantization step itself is just a **sign function** — each dimension of the projected input snaps to ±1 (scaled by `codebook_scale`), so there's no learned codebook at all. The codebook is implicit: every possible combination of ±1 across `d` dimensions defines one of the `2^d` codes.

Concretely:
1. `project_in`: linear projection from feature dim → `codebook_dim × num_codebooks`
2. `quantize`: `torch.where(x > 0, +scale, -scale)` — pure sign thresholding
3. `project_out`: linear projection back to feature dim

Gradients pass through the quantization via the straight-through estimator (`x + (quantized - x).detach()`). An entropy auxiliary loss encourages codebook utilization: per-sample entropy is minimized (confident predictions) while batch-wide entropy is maximized (uniform code usage). The discrete indices are just the binary pattern of positive/negative dimensions packed into integers.

### Rotation Trick For Vector Quantization

Standard VQ-VAE training usually handles the non-differentiable nearest-code lookup with a straight-through estimator, so the encoder effectively receives gradients that bypass the quantization step rather than gradients that reflect what the quantizer actually did.

One proposed alternative is the **rotation trick** for vector quantization. Instead of treating quantization as a hard stop in backprop, the encoder output is smoothly mapped onto its selected codebook vector with a linear transformation consisting of a rotation plus rescaling. That transformation is treated as constant during backpropagation.

The key idea is that the backward pass can then preserve some geometric information from the quantization event itself:

- the relative angle between the encoder output and the chosen codebook vector
- the relative magnitude mismatch between them

So instead of the encoder seeing only a crude straight-through surrogate, the gradient carries information about how the encoder output needed to move to better align with the assigned code.

Reported benefits from this idea include:

- lower reconstruction error
- better codebook utilization
- lower quantization error

This seems relevant if we return to explicit VQ or codebook-based tokenizers. It is less directly applicable to the current KL-VAE path, but it could be worth testing in any future MAGVIT / LFQ / VQ-VAE-style experiments where codebook collapse or weak quantizer-aware gradients become a bottleneck.

### Context Frames & Temporal Padding in PaletteVideoTokenizer

The `PaletteVideoTokenizer` disables temporal padding on `conv_in` and replaces it with real context frames from the data.

- `conv_in` is a 7×7×7 `CausalConv3d` (stride=1, dilation=1) → original `time_pad = 6`.
- In `__init__`, temporal padding is zeroed out: `self.conv_in.time_pad = 0`. The original value is saved as `self.conv_in_time_pad = 6`.
- `conv_out` is a 3×3×3 `CausalConv3d` → `time_pad = 2`, uses `replicate` padding (not zeros). This padding is **not** disabled.

With 8 context frames and 16 target frames (24 total input):

1. **conv_in** has no temporal padding, so it consumes 6 frames from the input (producing 18 output frames from 24 input).
2. The encoder/quantizer/decoder process these 18 frames.
3. **conv_out** adds 2 replicate-padded frames on the temporal left, then its kernel-3 conv produces 18 output frames.

At loss time: `skip_logits = context_frames - conv_in_time_pad = 8 - 6 = 2`. The first 2 output frames (which saw replicate padding from `conv_out`) are discarded. The remaining 16 output frames align exactly with the 16 target frames. The first loss frame (output index 2) sees zero padding — all 3 of its temporal receptive field positions are real decoder features.

The accounting: 6 context frames consumed by unpadded `conv_in` + 2 context frames discarded as `conv_out` replicate-padding region = all 8 context frames excluded from loss.

## Theoretical Minimum Reconstruction Loss

The palette tokenizer uses **cross-entropy** (`F.cross_entropy` with default mean reduction) over all pixels. For a single wrong pixel with optimal logits (probability ~0.5 on the correct class), the minimum recon loss is:

$$\text{recon\_loss} = \frac{\ln 2}{B \times T \times H \times W}$$

Example: batch=6, seq_len=16, image_size=224 → $N = 4{,}816{,}896$ pixels → minimum loss ≈ $1.44 \times 10^{-7}$.

## What Signal The GAN Gives The Decoder

In the current palette-video GAN setup, the discriminator does **not** see hard argmax palette indices. It sees the decoder's **softmax probabilities** over palette colors for the full video tensor:

$$x_{fake} = \operatorname{softmax}(z)$$

where $z$ are the decoder logits with shape $(B, K, T, H, W)$. The generator-side adversarial loss is then:

$$\mathcal{L}_{gan} = -D(x_{fake})$$

So compared with cross-entropy, the signal is very different:

- **Cross-entropy / focal CE**: local, per-pixel supervision. It says "increase probability of the ground-truth color here."
- **GAN**: global/contextual supervision. It says "change the output in whatever way makes this whole clip look more real to the discriminator."

This means a pixel can be "correct" under cross-entropy and still receive nonzero GAN gradient. If the surrounding context is wrong — edge shape, local neighborhood, sprite silhouette, temporal consistency, etc. — the discriminator can still push on that pixel because its score depends on the whole video, not independent pixel labels.

Conceptually the gradient to one decoder logit depends on the discriminator's judgment of many nearby outputs:

$$
\frac{\partial \mathcal{L}_{gan}}{\partial z_{p,c}}
=
\sum_{q,k}
\frac{\partial \mathcal{L}_{gan}}{\partial x_{q,k}}
\frac{\partial x_{q,k}}{\partial z_{p,c}}
$$

So the GAN can provide signal "through" a pixel the decoder already got right, because that pixel participates in a larger structure the discriminator thinks looks fake.

What this likely helps with in this project:

- crisper sprite and HUD edges
- more coherent local color arrangements
- cleaner object silhouettes
- fewer implausible palette mixtures
- better temporal consistency / less flicker

Important caveat: the current discriminator is a **global** 3D conv discriminator with adaptive average pooling to one logit per sample, not a PatchGAN. So the signal is richer than pure per-pixel cross-entropy, but it is still fairly coarse and global rather than explicitly local.


## LTX-Video 2.3 Style DiT

### Motivation
Explored whether a 100× downscaled LTX-Video 2.3 architecture (DiT-based latent video diffusion) could replace the current three-module Genie-2 pipeline for joint audio-video world-model generation, targeting 20–100M total parameters.

### LTX-2.3 Architecture Findings
- LTX-2.3 is not a game/world-model system like Matrix-Game 3. It is a general-purpose **joint audio-video diffusion transformer** with open weights.
- The public LTX-2 paper describes an **asymmetric dual-stream transformer**: a larger video stream plus a smaller audio stream, coupled by bidirectional cross-modal attention. The model card lists the current open checkpoint family as `ltx-2.3-22b-*`.
- It uses **separate modality-specific autoencoders**, not one shared audiovisual autoencoder:
	- **Video VAE**: a spatiotemporal latent VAE for video.
	- **Audio VAE**: a separate audio VAE over mel spectrograms.
	- **Vocoder**: HiFi-GAN-like waveform decoder for the audio side.
- The public codebase implements the **video VAE** as a 3D-conv spatiotemporal VAE with explicit temporal and spatial compression. Standard latent shape/compression in the repo:
	- input video: `[B, 3, F, H, W]`
	- latent video: `[B, 128, F', H/32, W/32]`
	- temporal compression: `F' = 1 + (F - 1) / 8`
	- example: `33 x 512 x 512 -> 5 x 16 x 16`
- The video VAE code path uses an initial **spatial patchify step** with `patch_size = 4`, then additional encoder blocks compress time and space further.
- The decoder mirrors this with temporal/spatial upsampling plus an unpatchify-style final expansion back to pixels.
- The audio VAE is much smaller and operates on mel spectrograms, with a reported 4x temporal downsample in the public README.
- Practical implication: the LTX design is a good architectural reference for a **unified latent diffusion backbone**, but its autoencoder stack is much more general-purpose and multimodal than what we need for NES-only experiments.

### Why LTX/DiT Is a Better Fit
- **Unified model**: DiT replaces both DynamicsTransformer and LatentDenoiser with a single transformer that does denoising via self-attention over spatiotemporal latent patches. Architecturally simpler.
- **Flow matching over DDPM**: LTX uses rectified flow matching (linear interpolation $x_t = (1-t)x_0 + t\epsilon$, velocity prediction). Trains faster, samples in 5–20 steps instead of 200 DDPM steps.
- **Action conditioning via AdaLN** (adaptive layer norm) replaces LTX's text-conditioning cross-attention path — much cheaper and appropriate for our discrete action space.

### What "Patchifying" Means
Patchifying means bundling a small local pixel block into one coarser cell before the rest of the model processes it.

For a single image with patch size 4, a `4 x 4` neighborhood is rearranged into channels:

$$[C, H, W] \rightarrow [16C, H/4, W/4]$$

For video, LTX's video VAE does this spatially on each frame before later compression stages:

$$[C, F, H, W] \rightarrow [16C, F, H/4, W/4]$$

This is similar to a **space-to-depth** transform or a non-overlapping patch embedding. It is not quantization and it is not a learned codebook by itself. It is just an early restructuring/compression step.

Why this matters:
- it reduces sequence length / spatial resolution early,
- preserves local neighborhoods as a single coarse unit,
- and makes later attention or convolutions cheaper.

In LTX's case, patchifying is only the first compression stage. The encoder then applies additional temporal/spatial compression to reach the final `8x` time and `32x` space latent grid.

### Patchifying Ideas Worth Testing Here
Patchifying is potentially relevant for this project because it gives a clean way to impose local structure **before** the main dynamics model.

Useful interpretations for Mario:
- **Tokenizer-side patchify**: explicitly patchify frames before a continuous VAE/DiT encoder, similar to LTX. This would be the direct architectural imitation.
- **Metatile-aligned patchify**: use `16 x 16` patch units so one coarse token corresponds to one NES metatile region. This is already close to the existing tokenizer bias and may be the more domain-correct option.
- **Latent patchify instead of pixel patchify**: keep the current tokenizer, but patchify the continuous latent grid before the transformer. This is cheaper to test and avoids retraining the image front-end immediately.

Low-risk experiment order:
2. **Pixel-space patchify** in a small continuous autoencoder baseline, to see whether early patch structure improves optimization or harms fine sprite edges.
3. **Metatile-aligned patchify** with `16 x 16` spatial units on `256 x 256` inputs or the nearest compatible resized representation.

Main tradeoff to watch:
- Patchifying improves efficiency and may encourage local compositional structure.
- But if patches are too coarse, it may hurt tiny sprite details, palette edges, and HUD text.
- For Mario specifically, patch boundaries that align with **metatiles** are much more likely to help than arbitrary patch boundaries.

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

### Attention Feasibility
With 196 spatial tokens × 16 frames = 3,136 tokens, full causal attention is feasible at this model size. Windowed attention optional but not strictly necessary.

### Key Modifications to LTX for This Project
2. **Action conditioning via AdaLN** — replace text cross-attention with adaptive layer norm conditioned on action embeddings.
3. **Flow matching replaces DDPM** — linear interpolant, velocity prediction, 5–20 sampling steps.

### DiT Tricks Still Worth Testing For This Project

Assuming the dynamics transformer will always run at the same context length and the same latent spatial resolution seen during training, the most relevant remaining DiT-style improvements are conditioning and normalization tricks.

Priority order:

1. **Use q/k RMSNorm on every attention path**
	- The current model already benefited from RMSNorm.
	- The next clean step is to make it consistent across all self-attention and cross-attention modules rather than only some of them.

2. **Upgrade to full adaLN-zero style conditioning**
	- Use timestep-conditioned modulation on each transformer sublayer, with zero-initialized modulation so the network starts close to an identity residual stack.
	- This is one of the most standard and proven DiT training tricks.

3. **Add residual gating on attention / MLP branches**
	- Per-branch or per-head gating is a standard stability/control trick in modern transformer variants.
	- This fits naturally with adaLN-style conditioning and can help prevent any one branch from dominating early in training.

4. **Consider RMSNorm for the main residual-stream norms too**
	- Beyond q/k normalization, the remaining LayerNorm sites in the residual stack could be converted to RMSNorm.
	- This is a broader change than q/k RMSNorm, so it should be treated as a later ablation rather than the first follow-up.

---

## Joint Audio-Video Generation

### Audio Data Source
We have **AVI files with perfectly synced video+audio**, bypassing the earlier limitation of only having APU register values in RAM dumps. The SMB1 RAM map (`src/mario_world_model/smb1_memory_map.py`) does contain audio registers (`area_music` 0x00FB, `event_music` 0x00FC, `sfx_reg1-3` 0x00FD-0x00FF) but these are high-level triggers, not waveform data. Perfect audio reconstruction from registers alone was not feasible — the AVI files solve this.

### Audio Tokenization Strategy
NES audio is low-complexity (4 synthesis channels: 2 pulse, 1 triangle, 1 noise + optional DPCM). Current measured mel front-end defaults for SMB1 are: 24 kHz mono, 400-sample window, 100-sample hop, 64 mel bins, 40 Hz `fmin`, 8 kHz `fmax`, 80 dB log floor, Hann window.

**Approach**: Train a small audio VAE over causal mel-spectrogram chunks rather than raw per-frame waveform chunks:
- Extract audio: `ffmpeg -i input.avi -ar 24000 -ac 1 output.wav`
- Compute a continuous mel-spectrogram on a fixed STFT grid
- At 24 kHz with a 100-sample hop, each mel step is ~4.17 ms, so one NES frame (~16.64 ms) spans about 4 mel time steps
- A small audio encoder can compress several mel columns per video frame into **~16–32 audio latent tokens per frame** (small compared to 196 video tokens)
- The audio VAE should be very small (sub-1M params) given NES audio simplicity

**Timing / causality note**:
- The raw normalized audio currently stored per video frame is aligned to frame intervals, not instantaneous frame timestamps
- At 24 kHz and ~60.0988 FPS, one video frame is ~399.34 audio samples, so the saved frame-aligned chunks naturally alternate between 399 and 400 valid samples
- That 399 vs 400 mismatch is not a problem for a vocoder-based design because the vocoder consumes a continuous mel sequence on a fixed 400-window / 100-hop grid, not isolated per-frame waveform chunks
- If strict no-future leakage is required, frame `t` should not condition on audio from interval `[t, t+1)`; instead use audio strictly before frame `t`, or build left-aligned causal mel features whose right edge lands at frame `t`

### Audio VAE Loss Scale
The current audio VAE training loss is dominated by `recon_loss`, not KL. In `scripts/train_ltx_audio_vae.py`, total loss is:

$$
	ext{loss} = \text{recon\_loss} + 10^{-5} \cdot \text{kl\_loss}
$$

The reconstruction target is a log-mel spectrogram normalized from an 80 dB window into `[0, 1]`, so the masked L1 reconstruction loss is directly interpretable as average absolute error on that unit interval.

Useful rule of thumb:

$$
	ext{average dB error} \approx 80 \times \text{recon\_loss}
$$

Interpretation for this compressor:
- `recon_loss > 0.08`: weak, likely underfitting or a data/problem mismatch
- `0.04 - 0.06`: decent first working model
- `0.02 - 0.04`: good
- `< 0.02`: very good for a compact mel VAE

Concrete conversions:
- `0.01` recon loss is about `0.8 dB` average absolute error per mel bin
- `0.025` is about `2 dB`
- `0.05` is about `4 dB`

For model selection, prefer eval `recon_loss` over the combined `loss`, since the KL term is intentionally tiny at the current setting.

### Joint Sequence Layout in the DiT
Concatenate audio latent patches with video latent patches in the transformer sequence:
```
[action_t, video_patches_t (196 tokens), audio_patches_t (~16-32 tokens), action_{t+1}, ...]
```

The DiT jointly denoises both modalities using a shared diffusion timestep, with modality-specific input/output projection layers. This is how recent joint audio-video diffusion models work.

### Half-Resolution Warm Start

- Worth testing a staged training setup where the model is first trained at half spatial resolution, then those weights are used to initialize a full-resolution run. The motivation is to learn coarse layout, motion, and game dynamics cheaply before paying for fine sprite edges, text, and palette detail.

- This is closest to progressive resizing or curriculum training, which is a real and fairly common idea. What is less standard is “train part of the model first, then add the rest of the model.” That is a somewhat different idea, closer to staged pretraining or progressive growing, and it only makes sense if the weight shapes and inductive biases transfer cleanly.

- For this project, the cleanest version is probably to keep the architecture as similar as possible and change only the input/output resolution, then continue training at full resolution. A separate variant would be to pretrain only the tokenizer or a 2D spatial model at half res and then add the temporal or full-resolution parts later, but that should be treated as a different experiment rather than the default form of progressive resizing.

- Main question to test: does half-res pretraining improve sample efficiency and stability enough to outweigh the mismatch when switching to full-resolution details?

### 2D Conv Weight Inflation

- If doing a 2D-first initialization, a standard trick is to inflate pretrained 2D conv weights into 3D convs by copying the 2D kernel across the temporal axis.

- The usual choices are either to repeat the 2D weights across time and divide by the temporal kernel size so the activation scale stays similar, or to place the 2D kernel only in the center time slice and zero the others. Repeating is the more common “inflation” version.

- This is a normal video-model initialization idea, especially when bootstrapping from strong image models. It is not the same thing as half-resolution warm start, but the two ideas can be combined: first learn good spatial filters cheaply in 2D or at lower resolution, then transfer them into the spatiotemporal model.

- Main question to test: for this project, does 2D initialization help enough to justify the mismatch with causal temporal structure, or does training the 3D stack directly work just as well at this scale?

### Inference-Time Use Of The GAN Discriminator

- A reasonable question is whether the GAN discriminator could be reused at inference time to help correct bad samples or guide the world model toward more realistic futures.

- In the current codebase, the discriminator is a training-only component:
	- defined in `src/mario_world_model/gan_discriminator.py`
	- used in the GAN training loop in `scripts/train_ltx_video_vae.py`

- So this is not implemented today, but it is technically feasible.

Practical inference-time uses:

- **Best-of-$N$ reranking**: sample $N$ candidate futures, score each with the discriminator, and keep the highest-scoring one. This is the simplest and lowest-risk first experiment.

- **Rejection sampling**: keep sampling until the discriminator score exceeds a threshold.

- **Guided decoding / guided sampling**: choose samples that maximize a combined objective

$$
J(x) = \log p_{\theta}(x \mid c, a) + \lambda D_{\phi}(x)
$$

where $D_{\phi}(x)$ is the discriminator realism score and $\lambda$ controls how much weight to give the discriminator relative to the world-model likelihood.

- **Latent refinement**: generate a sample once, then take a few gradient steps on the latent variables to improve discriminator score while constraining drift from the original sample or from model likelihood.

Main caveat for world models:

- A standard GAN discriminator mainly learns **realism**, not **dynamics correctness**.

- That means it may prefer futures that look plausible frame-by-frame while still being wrong about game state, object identity, action consequences, or temporal consistency.

- For this project, a pure discriminator score is probably not enough. A better target is a combined score, or eventually a more task-specific critic that checks state/action consistency rather than only visual realism.

Best first experiment:

- Start with **best-of-$N$ reranking** on short rollout chunks.

- Score each candidate with something like

$$
S(x) = \log p_{\theta}(x \mid c, a) + \lambda D_{\phi}(x)
$$

using a small $\lambda$ so the discriminator nudges the choice rather than dominating it.

- Evaluate not just visual quality, but also rollout correctness over longer horizons.

- Compare the improvement against the extra inference cost from generating multiple candidates.

- If this helps, the next step would be discriminator-guided sampling rather than pure reranking.

## Video VAE Latents

**Context:**

The diffusion transformer (DiT) trained on the video VAE latent codes showed notably slow convergence and struggled to learn meaningful temporal dynamics. AI suggested the latent space might be either too entangled (mixing visual and temporal information in ways that confuse the transformer) or too noisy for reliable prediction. The issue likely stems from the continuous KL-regularized latent space not being well-suited for discrete token prediction, or the latent representations compressing information in ways that obscure the underlying game dynamics.

**Approach:**

Several engineering strategies were considered to improve the learnability of the video latent space for autoregressive generation:

1. **Increase KL weight** — Boost the KL regularization term to encourage a tighter, more structured latent distribution that may be easier for the transformer to model.

2. **Cross-latent constraints** — Enforce structural alignment between video and RAM latent embeddings via MSE or contrastive losses. Since RAM contains ground-truth game state while video is high-dimensional observations, clipping or matching them could encourage the video encoder to prioritize semantically meaningful features over visual details.

3. **Temporal consistency regularization** — Add explicit smoothness constraints by penalizing divergence between latents from consecutive frames (should be similar) and random temporal frames (should differ). This pushes the latent space to encode meaningful temporal structure.

4. **Factorized latent spaces** — Separate the latent into semantic (game state) and visual (appearance) components. While the decoder could theoretically ignore the semantic part, it may help the transformer focus on the right variable during prediction.

5. **Data augmentation** — Apply more aggressive augmentation during training to improve generalization and robustness of the learned representation.

6. **Decoder architecture reduction** — Simplify the decoder (fewer channels, fewer residual blocks) to reduce model capacity and force the encoder to preserve more information in the latent rather than deferring complexity to reconstruction.

**Result:**

TODO