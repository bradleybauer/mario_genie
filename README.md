# Table of Contents

- [Single Sample Overfit Baseline](#single-sample-overfit-baseline)
- [How Does The Model Represent Counts](#how-does-the-model-represent-counts)
- [Mesen-Based Data Collection](#mesen-based-data-collection)
- [Initial Video Tokenizer Parameter Sweep](#initial-video-tokenizer-parameter-sweep)
- [Dense Cross-Entropy And NES Color Palette](#dense-cross-entropy-and-nes-color-palette)
- [Disentangling Hidden State From RAM](#disentangling-hidden-state-from-ram)
- [SMB3 x Super Mario Land 2](#smb3-x-super-mario-land-2)
- [More Data Artifacts](#more-data-artifacts)
- [Causal Conv Temporal Padding](#causal-conv-temporal-padding)
- [Discrete Tokenizer Variables To Explore](#discrete-tokenizer-variables-to-explore)
- [Onehot Memory Optimizations](#onehot-memory-optimizations)
- [Video VAE Training Updates](#video-vae-training-updates)
- [Continuous Bottleneck Autoencoders](#continuous-bottleneck-autoencoders)
- [Audio Data Exploration](#audio-data-exploration)
- [Initial Video Only World Model Training](#initial-video-only-world-model-training)
- [Video VAE Latents](#video-vae-latents)
- [World Model Example Generation](#world-model-example-generation)

<br>

<p align="center">
  <img src="pictures/meme.png" alt="meme">
</p>

<br>

# Single Sample Overfit Baseline

**Context:**

Initial tests showed that validation reconstruction was exceptionally bad. So much so that I suspected the training implementation contained a bug.

**Approach:**

As a sanity check I thought to verify the tokenizer can reconstruct a single video sequence near-perfectly.

**Result:**

The validation reconstruction was still extremely poor even on the single sample dataset.
I then found a **codebook bypass bug**: `encode()`/`decode()` skipped quantization entirely in the validation path.
This was fixed by using `tokenize()` + `decode_from_code_indices()`.

<br>
<br>

# How Does The Model Represent Counts

**Context:**

I want to know if the model has an interesting representation for any of the numeric features of the data. For ex, time, score, number of coins, lives, world-stage, mario progression through the level.

**Approach:** 

Copy (but learn from ofc) smart stuff from blogs I've seen recently.

**Result:** 

TODO

<br>
<br>

# Mesen-Based Data Collection

**Context:**

I switched data collection over to Mesen. Unlike my previous data collection scripts, Mesen can save emulator state at frequent intervals during collection, which opens the door to restarting from many points in a run instead of always from the beginning of a stage.

**Approach:** 

Mesen records raw gameplay data to disk, and an offline conversion step turns those recordings into session `.npz` files. The important next step is to use the frequent save states as replay anchors, so collection can resume from underrepresented regions and push the dataset toward a more uniform distribution over game world progression.

**Result:** 

TODO

<br>
<br>

# Initial Video Tokenizer Parameter Sweep

**Context:**

Initial training runs for small models (<2M params) trained for an hour or two were unsuccessful. I want to understand if any of the variables I know how tune will give a significant decrease in reconstruction loss.

**Approach:** 

I trained 24 configurations varying `init_dim` (32, 64), `codebook_size` (16k, 32k, 64k), model size (small, base), and attention (with/without), along with a smarter learning rate schedule.
Each model trained for 4 hours on an RTX 4090, across 8 machines.

**Result:** 

Models more or less performed similarly — at this training scale, none of the hyperparameters made a huge difference.
All models learned throughout training, with a clear, but not very steep, downward trend in loss over time.

Given the trend, I think longer training could improve quality to an acceptable degree.
Chatting with AI I found that using a different loss function would likely help.

<br>
<br>

# Dense Cross-Entropy And NES Color Palette

**Context:**

It turns out Super Mario Bros. on the NES only uses ~30 colors. Across my dataset of a million images (covering nearly the entire game), I observe just 23 distinct colors. AI suggested changing the model to use cross-entropy loss on palette probability distributions instead of MSE loss on raw RGB pixels.

**Approach:** 

It was not realistic to change only the VideoTokenizer's output shape, so I proposed changing both the input and output representations. After all, if it is easier for the decoder to produce palette probabilities, then it may also be easier for the encoder to disentangle information from that same representation. The change was straightforward — `magvit2-pytorch` exposes a "number of channels" argument that we set to the number of palette colors. As a bonus, the palette-indexed representation also provides:

- Smaller CPU->GPU bandwidth requirement
- Smaller dataset size on disk and lower network bandwidth usage

**Result:** 

The new model trained faster on my 3070, completing ~60k steps in 3 hours.
It clearly has a much better grasp of spatial layout within the image.


<br>
<br>

# Disentangling Hidden State From RAM

**Context:**

There is some hidden state surrounding 1UP mushrooms and a few other things. A transformer with a short fixed context length will not be able to learn this in training. The game's "hidden state" — which blocks have been hit, which items collected, which enemies defeated — lives in the NES's 2KB of internal RAM and is not fully recoverable from the image alone.

The NES has 2KB (2048 bytes) of internal RAM mapped to `$0000`–`$07FF`, divided into four regions:

- **Zero Page** (`$0000`–`$00FF`, 256 bytes) — The 6502 CPU's fast-access page. Games store their most frequently-read variables here because zero-page addressing modes are shorter and faster. In SMB1 this includes player position (`$0086` X, `$00CE` Y), velocity (`$0057` X speed, `$001D` Y speed), player state machine (`$000E`), moving direction (`$0045`), and the five enemy type slots (`$0016`–`$001A`). Contains some hidden state: off-screen enemy positions, scroll offsets, and physics state that may be ambiguous from a single frame.

- **Stack** (`$0100`–`$01FF`, 256 bytes) — The 6502 hardware stack, growing downward from `$01FF`. Contains return addresses and saved registers from subroutine calls. Mostly noise from the model's perspective — the values change rapidly and don't carry meaningful game state. Safe to exclude or zero-fill.

- **OAM Buffer** (`$0200`–`$02FF`, 256 bytes) — Sprite Object Attribute Memory staging area. The game writes 64 sprites × 4 bytes here (Y position, tile index, attributes, X position), then DMA transfers the whole page to the PPU each frame. This data is *derivable from the image* — it's literally what draws the sprites on screen. SMB1 rotates sprite priority every frame to work around the NES's 8-sprites-per-scanline limit, causing visible flickering in the raw data. Can be excluded since the image encoder already captures this.

- **Game Data** (`$0300`–`$07FF`, 1280 bytes) — The bulk of meaningful game state and the primary source of hidden state. Includes world/stage (`$075F`/`$075C`), lives (`$075A`), score (`$07DE`–`$07E3` BCD), coins (`$07ED`–`$07EE` BCD), timer (`$07F8`–`$07FA` BCD), gameplay mode (`$0770`), power-up status (`$0756`), level layout data, and enemy state arrays. This is where the non-recoverable information lives — which blocks have been hit, which items have been collected, warp zone flags, and the 1UP re-collection prevention flag.

Some cartridges also include **WRAM** (Work RAM, `$6000`–`$7FFF`, up to 8KB) — extra RAM on the cartridge itself, often battery-backed for save files. Games like *The Legend of Zelda* and *Kirby's Adventure* store persistent world state, map data, and save slots here. SMB1 does not use WRAM, but any general NES world model would need to account for it.

For embedding purposes, **Zero Page + Game Data (1,536 bytes)** captures all meaningful state for SMB1. Stack and OAM (512 bytes) can be dropped — they're either noise or redundant with the image. For WRAM-equipped games, the relevant WRAM region would need to be included as well.

The NES CPU runs at ~1.79 MHz (~29,781 cycles per frame), but the game loop is frame-synchronized: the PPU fires a Vertical Blank NMI (Non-Maskable Interrupt) at 60 Hz, the game logic runs atomically within that window, then idles until the next NMI. The emulator's `env.step()` advances one full NMI-to-NMI cycle and then exposes RAM — this is the only coherent snapshot where all variables agree with each other. One RAM snapshot per frame = zero information loss.

**Approach:** 

Potentially add a temporal embedding of the NES 2KB of RAM to the latent space.
Wonder how that would affect what the encoder learns to encode. Probably a shift toward more visual features?
An idea is you could side-step the image encoder altogether. Just predict pixels based on RAM embeddings.
Talking with AI about it leads me to believe using both images & RAM will perform better than either alone.
The image embeddings are a 16x16 grid of features where that grid is a spatial "bias". It may be easier
to disentangle certain spatial information from the image than it is from RAM.

Add a learned embedding of the NES RAM to the latent space via feature concatenation. A small encoder (2–3 FC layers) maps the 1,536-byte RAM vector to a D-dim embedding matching the latent channel dimension, which is then broadcast spatially to 16×16 and concatenated with the image latent. The RAM encoder trains jointly with the rest of the model.

Using both images and RAM should perform better than either alone. The image embeddings are a 16×16 grid of features with inherent spatial bias — it may be easier to disentangle certain spatial information from the image than from RAM, and vice versa.

For storage, the RAM is delta-encoded (XOR against previous frame) before `savez_compressed` — frame-to-frame deltas are ~90% zeros, and zlib compresses that extremely well.

**Result:** 

Added a NES RAM visualizer to both `play_ram_viz.py` (SMB1) and `play_nes.py --ram` (any ROM).
<p align="center">
  <img src="pictures/smb1ram.png" alt="nes ram in game visualization">
</p>
<p align="center">
  <img src="pictures/smb3ram.png" alt="smb3">
</p>

TODO

<br>
<br>

# SMB3 x Super Mario Land 2

**Context:**

I imagine building a SMB3 world model would be hard.
One idea I had for how to make it easier is to clean the data a bit.
Specifically fixing flickering in the smb3 status bar and removing the max 8 sprite limit.
The nes has certain limits on the max number of sprites that can appear on a scan line.
If the game attempts to draw more than the max the sprites begin to flicker in time. Example,

![Sprite limit example](pictures/spritelimit.png)

I do not want the model to have to spend capacity to reverse engineer exactly how this flicker mechanic works.

**Approach:**

The status bar flickering can be fixed by writing 5 bytes to the ROM file. The three `0xEA` bytes are 6502 NOP instructions, effectively disabling the code that caused the flicker. This will cause a minor floor-shaking glitch during the end credits.

| Address   | Before | After  | Note |
|-----------|--------|--------|------|
| `$3F7B2`  | `0x0D` | `0x16` | —    |
| `$3F8E0`  | `0x68` | `0xEA` | NOP  |
| `$3F8E1`  | `0x8D` | `0xEA` | NOP  |
| `$3F8E2`  | `0x10` | `0xEA` | NOP  |
| `$3F8E3`  | `0x40` | `0xEA` | NOP  |

To fix the sprite limit will require using a different emulator.
A good candidate is Mesen which also emulates sound which would be neat to learn how to model (as if an smb3 world model was't ambitions enough lol)

**Result:**

TODO

<br>
<br>

# More data artifacts

**Context:**

The model is starting to learn the finer details in the dataset and unfortunately it's learning mid-frame spilt artifacts.
![mid-frame split](pictures/split.png)

BTW here is an example of a "scene-cut". This one is "natural" meaning it represents real gameplay. In this case mario reached the flag pole in 4-1 and immediately advanced to 4-2.
![scene-cut](pictures/scene-cut.gif)

**Approach:** 

Check if frame spilts can be prevented with Mesen.

**Result:** 

I have not seen any frame splits with mesen. Mesen's data is super duper clean.

<br>
<br>

# Causal Conv Temporal Padding

**Context:**

Reconstruction of frame 1 (0-indexed) is consistently worse than all other frames in a sample. Consider adding context frames.
Those causal conv filters will have to be "dual purpose" to account for being applied on images with zero padding. I imagine that wouldn't be very efficient.

**Approach:**

Instead of zero-padding the causal temporal convolutions at the start of a sample, feed the first few real frames into the left padding region of the conv. This keeps the sequence length unchanged and only changes what the temporal filters see near the beginning of the clip.

**Result:**

Using real data in place of zero padding made a noticeable improvement in reconstruction quality for the earliest frames.

<br>
<br>

# Discrete Tokenizer Variables To Explore

**Context:**

I'm interested in testing different LFQ params. Specifically the num_codebooks and codebook_size params. Currently num_codebooks=1 and codebook_size=65536.
I would like to test num_codebooks=2 and codebook_size=256 which will result in the overall same number of potential discrete codes and the same bottleneck dimension but would decrease the number of entries in the softmax calculation which could improve optimization performance.

Additionally I want to test a version of the video autoencoder with more context frames, a version without temporal downsampling, and a version trained on a cleaner dataset.

**Approach:**

Train more models. One of these variants prepends actual extra context frames to each sample during both training and inference. The dataset returns `seq_len + N_CTX` frames, and the loss is computed only on the final `seq_len` frames, so the prepended prefix is encoded to latents and reconstructed by the decoder but masked out of the loss.

**Result:**

I trained a tiny model with an expanded bottleneck size and it immediately performed better than all previous models in the early training phase on a per-step basis. The key was using many smaller codebooks instead of using one huge codebook which allows to train a larger effective codebook with a smaller number of FLOPs.

For the explicit context-frame variant, non-context images consistently have better reconstructions than context images. The context images are delightfully glitched. Currently using 8 additional context images per sample.


<br>
<br>

# Onehot Memory Optimizations

**Context:**

Training the palette-based video tokenizer is memory-heavy. The one-hot input tensor `(B, K, T, H, W)` is created via `F.one_hot`, which always returns int64 — 8 bytes per element — even though only 0s and 1s are needed. The model forward pass also runs in float32 by default.

**Approach:**

Two changes: (1) replaced `F.one_hot` with a `scatter_`-based `indices_to_onehot` that writes directly into a tensor of the desired dtype (float32, float16, or bfloat16 via `--onehot-dtype`), avoiding the large int64 intermediate entirely. (2) Added Accelerator mixed precision support via `--mixed-precision bf16` to run the model forward/backward in half precision.

**Result:**

Cuts memory enough to increase batch size or use larger models on the same GPU. No measurable impact on training quality.

<br>
<br>

# Video VAE Training Updates

**Context:**

Baseline cross-entropy loss treats all palette colors equally. But the NES color distribution is extremely skewed — sky blue and black dominate, while rare colors like coin-gold or power-up red appear in a tiny fraction of pixels. The model learns the dominant colors quickly and ignores the rare ones.

Dataset palette distribution (~80.8 billion pixels across 23 colors):

| Rank | Color Index | Probability | Cumulative |
|------|-------------|-------------|------------|
| 1    | 4           | 41.60%      | 41.60%     |
| 2    | 14          | 35.00%      | 76.60%     |
| 3    | 9           | 5.31%       | 81.92%     |
| 4    | 18          | 3.85%       | 85.76%     |
| 5    | 6           | 3.71%       | 89.48%     |
| 6    | 5           | 2.49%       | 91.96%     |
| 7–23 | (17 others) | < 2% each   | 100%       |

The top 2 colors alone cover ~77% of all pixels. The rarest color appears in only 0.005% of pixels — a ~8,600× imbalance vs. the most common.

**Approach:**

Three changes layered on top of each other:

1. **Focal loss** (`--focal-gamma`): Down-weights easy (well-classified) pixels so the model spends more capacity on hard ones. Gamma of 1.0 is a mild version; 2.0 is the typical aggressive setting.

2. **Inverse-frequency class weights** (`--use-class-weights`): Weights each palette color by $w_i = p_i^{-\alpha}$ (normalized to mean 1), with default $\alpha = 0.5$. The most frequent color (41.6%) gets weight 0.05, the rarest (0.005%) gets 4.87 — a ~95× ratio. Full inverse-frequency ($\alpha = 1$) would be ~8,600× (probably to large?). Computed offline from `palette_distribution.json`.

3. **GAN with LeCAM regularization** (`--use-gan`, `--use-lecam`): A compact 3D hinge-loss discriminator that pushes reconstructions toward sharper, more realistic output. LeCAM EMA regularization stabilizes discriminator training by penalizing when its real/fake logit gap drifts too far from the running average.

**Result:**

Results look good. LeCAM regularization made the discriminator training actually work. Adding a discriminator improved training stability of the vae model even.

<br>
<br>

# Continuous Bottleneck Autoencoders

**Context:**

The magvit2 tokenizer (from `magvit2-pytorch`) works but getting good reconstruction quality requires very large codebooks. I'm concerned that as the codebook size grows, learning the dynamics model on top of the discrete codes will become too difficult — the transformer has to predict over an enormous vocabulary. A continuous KL-regularized latent avoids this problem entirely. I also wanted an audio counterpart for the mel spectrograms.

**Approach:**

Wrote two custom VAEs from scratch:

- **Video VAE** (`video_vae.py`, ~19.3M params): A 3D convolutional VAE with spatial patchify/unpatchify (4×4), causal temporal convolutions, and no temporal downsampling. The encoder uses strided 3D convs for 2 levels of spatial downsampling, residual blocks, and outputs a mean/logvar pair. The decoder mirrors the encoder with nearest-neighbor upsampling. KL-regularized continuous latent space instead of LFQ discrete codes.

- **Audio VAE** (`audio_vae.py`, ~6.2M params): A 2D convolutional VAE operating on log-mel spectrograms (time × frequency). Same architectural pattern — causal temporal convolutions with 4× temporal compression via strided convs, residual blocks, KL-regularized latent. Outputs L1-reconstructed mel spectrograms.

Both use causal convolutions (replicate-pad the first frame rather than zero-pad) so they can be applied autoregressively without information leaking from the future.

**Result:**

The video VAE converges noticeably faster than the magvit2 baseline on a per-step basis:

| Step  | Magvit2 (13M) recon | Video VAE (19.3M) recon |
|-------|---------------------|-----------------------------|
| 1k    | 0.2202              | 0.2048                      |
| 5k    | 0.0747              | 0.0248                      |
| 10k   | 0.0510              | 0.0114                      |
| 20k   | 0.0257              | 0.0076                      |
| 25k   | 0.0316              | 0.0066                      |

The video VAE reached recon loss ~0.007 at 25k steps — a level the magvit2 only hit around 60k+ steps. The video VAE also runs at batch size 64 vs magvit2's batch size 4, so it sees far more data per step. KL loss stays low (~2.0) and stable.

The VAE does have periodic KL spikes (e.g. step 2k, 6.6k, 10.8k) where recon loss temporarily jumps, but it recovers quickly each time. Not yet clear what causes them.


<br>
<br>

# Audio Data Exploration

**Context:**
I wanted to explore and do some engineering around the audio data.

**Approach:**

I first ran a dataset-wide spectrum analysis over the raw Mesen AVI recordings to choose a compact mel front-end that still preserves most of the useful signal while staying aligned with the NES frame rate.

From there I extended the normalized dataset format to store per-frame audio chunks together with `audio_lengths` and `audio_sample_rate` so the audio models could train from the same `.npz` files as the video models.

Then I built two initial models:

- **Audio VAE**: a causal 2D convolutional VAE over log-mel spectrograms with two stride-2 encoder downsamples, giving `4x` temporal compression. The model reconstructs mel spectrograms and uses a KL-regularized continuous latent.
- **Audio Vocoder**: a compact BigVGAN-style mel-to-waveform decoder with `SnakeBeta` activations, dilated residual blocks, and transposed-convolution upsampling. The default upsampling schedule `(5, 5, 4)` gives an exact `100x` expansion, matching the mel hop length.

Finally, I ran baseline overfit tests on both models and listened to the reconstructions as a sanity check before doing broader hyperparameter work.

![](pictures/audio.png)

**Result:**

Current mel defaults for SMB1 audio:

- Sample rate: `24000` Hz mono
- Window / `n_fft`: `400` samples
- Hop length: `100` samples
- Mel bins / `n_mels`: `64`
- `fmin`: `40` Hz
- `fmax`: `8000` Hz
- Log floor: `80` dB
- STFT window: Hann

These values came from scanning the full dataset audio distribution. At `24` kHz, about `98.5%` of spectral energy is retained relative to the original audio. Only about `2.5%` of energy falls below `40` Hz, and only about `2.4%` sits above `8` kHz. A `400`-sample window is also convenient because it is almost exactly one NES video frame at `~60.1` FPS, while a `100`-sample hop gives about four mel steps per frame.

The initial audio VAE run produced decent reconstructions, though I have not yet measured how aggressively the bottleneck can be tightened before quality falls off. There is still a fair amount of tuning left to do.

The vocoder successfully overfit a single sample and produced high-quality output audio. The sample happened to contain a power-up sound, which made it especially easy to hear that the model was capturing the characteristic NES sound rather than just broad envelope shape.

<br>
<br>

# Initial Video Only World Model Training

**Context:**

The first video-only world model experiments used pre-encoded `VideoVAE` latents and trained an action-conditioned latent DiT with a continuous flow-matching objective rather than discrete next-token prediction.

The initial problem was straightforward: training in full precision was stable but slow, while `bf16` made training roughly `10x` faster but introduced severe gradient blow-ups. At that point it was hard to tell whether the bottleneck was numerical instability, model size, or the latent representation itself.

**Approach:**

Several engineering changes were made to make the latent DiT cheaper and more stable:

1. **Reduced model size**: cut the DiT configuration down by about an order of magnitude in compute. The working assumption was that Mario latents probably do not need the aggressive hidden-dimension inflation common in large image/video transformers.

2. **Switched to Diffusers + Accelerate**: this simplified mixed-precision training, checkpointing, and general training boilerplate.

3. **Added per-channel latent standardization**: latent channels are normalized using dataset mean/std before training. For flow matching, this makes the interpolation

$$
x_t = (1 - t)x_0 + t\epsilon
$$

better behaved by putting the clean latent target $x_0$ and Gaussian noise $\epsilon$ on a more comparable scale.

4. **Added RMS-normalized attention logits**: `qk_norm="rms_norm"` was added to the most sensitive attention paths to reduce exploding attention activations under mixed precision.

5. **Added verbose gradient norm logging**: this made it easier to see whether failures were coming from the attention stack, the output projection, or the optimizer step itself.

The model was still trained in a causal setup: future latent frames are noised, the model predicts velocity targets, and actions are shifted causally so the action at step $t$ only conditions predictions at or after that step.

**Result:**

`qk` RMS normalization fixed the catastrophic gradient blow-ups and made `bf16` training practical.

However, once the numerical instability was under control, a second issue became clearer: the loss still was not dropping as quickly as hoped on a per-step basis.

The current best hypothesis is that the video VAE latent is still too entangled or too reconstruction-oriented for a small dynamics model to predict efficiently. In other words, fixing the training stack made the optimization healthier, but it also made it more obvious that latent quality is now the main bottleneck rather than raw numerical instability.

That is what motivated the next set of experiments in the **Video VAE Latents** section below.

<br>
<br>

# Video VAE Latents

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

<br>
<br>

# World Model Example Generation

**Context:**

This is the first generation from the world model. The DiT was trained on a Video VAE that was both severely under-trained and under-regularized. I stopped VAE training early — just after it started producing decent reconstructions — and the KL weight was low. The VAE also did not use any of the available data augmentation (e.g. pixel palette perturbation), which likely hurt the encoder's robustness.

The previous VAE iteration did not include temporal smoothness regularization, and the latent space was noticeably chaotic — you could watch features jump erratically between channels over the time dimension. The current VAE does use temporal smoothness (cosine-similarity penalty on consecutive posterior means), which helps a lot, but is still under-regularized overall.

There is also a significant train/inference distribution mismatch: the VAE was trained on sequences of length 16 but is run on sequences of length 60 at inference time. With a 16-frame window, a Game Over screen or main menu would typically fill most or all of the context — the model learned to handle these as near-homogeneous sequences. At inference with 60 frames, large contiguous blocks of the context window are occupied by visually distinct regimes (e.g. 30 frames of the black Game Over screen followed by 30 frames of active gameplay). The causal temporal convolutions never saw that kind of within-sequence heterogeneity during training, and it shows — the blackness of the Game Over screen bleeds into the main menu reconstruction.

**Approach:**

The current DiT uses the encoder–decoder split architecture (`VideoLatentDiTDiffusers`): the encoder runs once on the context history, and the decoder uses cross-attention to read those encoded features at every diffusion timestep. This is much faster than a unified architecture because the encoder forward pass is amortized across all denoising steps, but it means the encoded history features are static — they never see the diffusion timestep conditioning.

Two latent-space improvements were made for this run:

1. **Per-component normalization** — Latent standardization was switched from per-channel (one mean/std per channel, shared across all spatial positions) to per-component (one mean/std per `(C, H, W)` element, shared across time). This gives the flow-matching interpolation $x_t = (1 - t)x_0 + t\epsilon$ a more uniform scale across the full latent tensor.

2. **Temporal smoothness regularization** — A cosine-similarity loss on the VAE posterior mean encourages consecutive latent frames to be similar, penalizing $1 - \text{mean}(\cos(z_t, z_{t+1}))$. This is scale-invariant so it doesn't fight with the KL term. The difference versus the previous VAE without this penalty is stark — the latent dynamics are far less chaotic.

**Result:**

![alt text](hmm.gif)

The generated video shows that the model has learned something — it produces recognizable Mario scenes — but there are clear issues:

- **Weak action conditioning.** Generation is not well conditioned on the player's actions. Actions are currently injected via cross-attention: each NES controller byte is decomposed into 8 bits, passed through a 2-layer MLP (`8 → action_dim=32 → 32`), projected to `d_model`, and then cross-attended to by the latent tokens with a causal mask. The diffusion timestep, by contrast, uses FiLM-style modulation (AdaLayerNorm) which directly scales and shifts every token at every layer. The action signal may be too weak relative to the timestep conditioning — cross-attention lets the model learn to ignore it. Potential improvements include using FiLM for actions as well (or in addition), increasing `action_dim`, adding action tokens to the self-attention sequence rather than a separate cross-attention path, or using additive action embeddings directly on the latent tokens.

- **Train/inference distribution mismatch in the VAE.** The VAE trained on 16-frame sequences but runs on 60-frame sequences at inference. The causal temporal convolutions see input patterns they've never encountered, and it shows — especially around scene transitions like Game Over → main menu.

- **Under-trained and under-regularized VAE.** Early stopping plus no data augmentation means the VAE doesn't reconstruct reliably, which compounds the DiT's prediction difficulty since it's predicting latents that are already imprecise.

One thing I'm interested in trying is training the VAE to convergence with a much higher KL weight, temporal smoothness, and potentially adding per-frame dense (fully-connected) layers to both the encoder and decoder that mix spatial information across the 7×7 grid — the idea being to push the encoder toward encoding more semantic features rather than purely spatial ones. Intuition says semantic latents may be more predictable for the DiT, though it's not certain — spatial structure in the latent grid does carry useful inductive bias.

Another idea I'm considering is **two-phase autoencoder training**. In phase 1, train a balanced encoder and decoder (roughly matched in parameter count) with full latent-space regularization (high KL weight, temporal smoothness, data augmentation) to convergence. A modestly-sized decoder can't compensate for a weak encoder, so the encoder is forced to learn semantically meaningful latents. In phase 2, freeze the encoder and swap in a much larger decoder trained to produce pixel-perfect reconstructions from the already-learned latents. The key insight is that training with the big decoder from the start would let it compensate for a lazy encoder — reconstructing well even from poorly structured latents — removing the pressure on the encoder to learn good representations.

Setting the project aside for a few days due to repetitive strain injury. Using speech-to-text software to write these notes, which is not very effective.

<!-- Template -->

<br>
<br>

# Title

**Context:**

**Approach:**

**Result:**

Random picture of me and my mom (her name is claudette)
![alt text](pictures/claudette.png)
