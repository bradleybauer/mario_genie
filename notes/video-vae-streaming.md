# VAE Decode Streaming for World Model Inference

## Problem

During autoregressive world model inference, the DiT predicts 1 new latent
per step.  Naively decoding only that latent in isolation produces
boundary artifacts because the VAE decoder's 3D causal convolutions need
temporal context from preceding latents.

Decoding the *entire* accumulated latent history every step is wasteful — the
VAE decoder is the dominant cost (~3 GB peak, 33M params, full 224×224 spatial).

## VAE Decoder Temporal Receptive Field

The current VAE (`video_vae_20260411_232136`) has:
- `temporal_downsample=1` (2 frames per latent)
- `global_bottleneck_attn=false` (bounded receptive field)
- All temporal convolutions are **CausalConv3d** (left-pad only, no future leakage)

### Layer-by-layer analysis

**Pre-temporal-upsample** (latent time domain):

| Layer | Temporal convs | Lookback (latents) |
|-------|:-:|:-:|
| `decoder_in` (Conv3d k=1) | 0 | 0 |
| `decoder_mid` (ResBlock: 2× CausalConv3d k=3) | 2 | 4 |
| `decoder_block6` (ResBlock: 2× CausalConv3d k=3) | 2 | 4 |

Subtotal: **8 latents** of past context needed before upsample.

**Temporal upsample** at `decoder_up1`:
- `ConvTranspose3d(k=3, stride=2)` — doubles temporal resolution
- Adds 1 latent of lookback through stride-2 mapping

**Post-temporal-upsample** (raw pixel time domain):

| Layer | Temporal convs | Lookback (frames) |
|-------|:-:|:-:|
| `decoder_block5` (ResBlock) | 2 | 4 |
| `decoder_up2` + `decoder_block4` | 1 + 2 | 2 + 4 |
| `decoder_up3` + `decoder_block3` | 1 + 2 | 2 + 4 |
| `decoder_up4` + `decoder_block2` | 1 + 2 | 2 + 4 |
| `decoder_up5` + `decoder_block1` | 1 + 2 | 2 + 4 |
| `decoder_out` (CausalConv3d k=3) | 1 | 2 |

Subtotal: **30 frames** of past context needed after upsample.

### Full receptive field (temporal_downsample=1)

Working backward from 1 target latent (= 2 output frames):

1. Post-upsample needs 2 + 30 = 32 frames at decoder_up1 output
2. Through stride-2 transpose → 17 latents at decoder_up1 input
3. Pre-upsample layers → 17 + 8 = **25 latents total input**

**To correctly decode the last latent, feed the decoder 25 latents and
discard all but the final 2 frames (1 latent) of output.**

### Practical overlap window

The theoretical receptive field of 24 preceding latents decays exponentially —
each additional layer multiplies the effect of distant frames by the convolution
weights.  In practice, **8–12 latents of overlap** should be sufficient
for artifact-free boundaries.  This can be tuned by visual inspection.

| Overlap | Latent decode window | Frames decoded | Memory |
|---------|:---:|:---:|:---:|
| 0 (isolated) | 1 | 2 | ~25 MB |
| 4 | 5 | 10 | ~125 MB |
| 8 | 9 | 18 | ~225 MB |
| 12 | 13 | 26 | ~325 MB |
| 24 (full RF) | 25 | 50 | ~625 MB |

All well within GPU memory.  Memory scales linearly with temporal window size
since the spatial dimensions (224×224) are fixed.

## Streaming Decode Algorithm

```
decode_overlap = 8   # tunable, 8-12 recommended

# Ring buffer of recent latents (kept on GPU)
latent_buffer: Tensor  # shape (1, C, context_latents, H, W)

each prediction step:
    new_latents = dit.predict(...)           # (1, C, 1, 7, 7)
    latent_buffer = slide(latent_buffer, new_latents)

    # Decode window: overlap + new latent
    window = latent_buffer[:, :, -(decode_overlap + 1):]  # (1, C, overlap+1, 7, 7)
    frames = vae.decode(denormalize(window))               # (1, 23, 2*(overlap+1), 224, 224)
    display_frames = frames[:, :, -2:]                     # last 2 frames only
    palette_indices = display_frames.argmax(dim=1)         # (1, 2, 224, 224)
```

  Only the final 2 frames (from the 1 new latent) are displayed.
The overlap region provides temporal context for the causal convolutions but
its decoded output is discarded.

### Why this works

CausalConv3d pads on the left by repeating `x[:, :, :1]`.  When we include
real preceding latents instead, those convolutions receive proper
temporal context rather than synthetic padding.  The causal structure means
including *more* past never changes the output for any given timestep — it
only improves it.

### First decode (bootstrap)

After the initial context is built from emulator frames, decode the full context
through the VAE once.  This is expensive but only happens once (and again on
recontextualize).  Subsequent steps use the streaming window.

## Context Length for a Mario World Model

### How long is 30 seconds?

| Metric | Value |
|--------|-------|
| NES frame rate | 60 fps |
| 30 seconds of gameplay | 1,800 frames |
| With temporal_downsample=1 | **900 latents** |

### DiT self-attention cost at 900 latents

Each latent has 7×7 = 49 spatial tokens.

| Context (latent) | Encoder tokens | Self-attn cost (per layer) | 6 layers total |
|:---:|:---:|:---:|:---:|
| 62 (current training) | 3,038 | 9.2M pairs | 55M |
| 254 (planned) | 12,446 | 155M pairs | 930M |
| 900 (30 seconds) | 44,100 | 1.94B pairs | 11.7B |

At d_model=128, 4 heads: 11.7B pairs × 128d × 2 (Q·K + attn·V) ≈ **3 TFLOP**
per encoder pass.  On a Blackwell GPU at ~1 PFLOP/s bf16 throughput, that's
~3ms — still fast!  But memory for the attention matrix (44,100² × 2 bytes =
**3.6 GB** per layer, 21.6 GB for 6 layers) becomes significant.

### What context length actually matters for Mario?

Mario gameplay has these temporal scales:

| Scale | Duration | Latents | What happens |
|-------|:---:|:---:|---|
| Immediate | 0.1–0.5s | 3–15 | Jump arcs, enemy collision, block hits |
| Short-term | 1–3s | 30–90 | Platform sequences, pipe transitions |
| Medium-term | 5–10s | 150–300 | Screen scrolling, level section traversal |
| Long-term | 30s+ | 900+ | Level progression, lives/score accumulation |

The world model needs enough context to:
1. **Continue smooth motion** — 0.5–1s minimum (15–30 latents)
2. **Remember what's just off-screen** — 3–5s (90–150 latents)
3. **Maintain level continuity** — 10–15s (300–450 latents)

30 seconds (900 latents) is likely overkill for visual prediction.  The
DiT would spend most of its attention budget on frames that are no longer
visible and don't affect the current screen state.

### Recommendations

| Config | Context latents | Raw duration | Attention memory | Use case |
|--------|:---:|:---:|:---:|---|
| **Conservative** | 62 | ~2s | 36 MB | Fast iteration, proof of concept |
| **Moderate** | 126 | ~4.2s | 150 MB | Good balance, covers most gameplay |
| **Full screen memory** | 254 | ~8.5s | 600 MB | Remembers off-screen state well |
| **30 seconds** | 900 | 30s | 21.6 GB | Likely diminishing returns |

**Suggestion: 126–254 latents (4–8.5 seconds)** covers the useful
temporal range for Mario.  The NES screen is 256px wide and scrolls at
~2–3 px/frame — it takes about 2–3 seconds to scroll one full screen width.
Having 2–3 screens worth of memory (6–9 seconds) is plenty.

If you want 30 seconds, it's computationally feasible on Blackwell but you'd
want to either:
- Use **flash attention** to avoid materializing the 3.6 GB/layer attention
  matrices (flash attn is O(N) memory)
- Switch to a **sliding window** or **chunked** attention pattern in the
  encoder so distant frames get cheaper attention

### max_frames constraint

The DiT uses learned temporal position embeddings of shape `(1, max_frames,
d_model)`.  Currently `max_frames=64`.  To support longer contexts:

- **max_frames=256** works with current training config (clip=256, ctx=254)
- **max_frames=1024** would cover 30 seconds but wastes embedding parameters
  for positions rarely seen during training
- Alternative: switch to **sinusoidal** or **RoPE** temporal embeddings which
  generalize to unseen lengths without a fixed max_frames table
