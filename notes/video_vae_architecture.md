# Video VAE Architecture Notes

## Scope

These notes describe the checkpoint in `checkpoints/video_vae_20260406_065725/` and compare it to the standard LTX video VAE implementation under `LTX-2/`.

Primary sources:

- Checkpoint config: `checkpoints/video_vae_20260406_065725/config.json`
- Trained model definition: `src/models/video_vae.py`
- Training script: `scripts/train/train_video_vae.py`
- Training helpers: `src/training/palette_video_vae_training.py`
- LTX VAE implementation: `LTX-2/packages/ltx-core/src/ltx_core/model/video_vae/video_vae.py`
- LTX VAE configurator defaults: `LTX-2/packages/ltx-core/src/ltx_core/model/video_vae/model_configurator.py`


## Checkpoint Summary

This run is the plain symmetric `VideoVAE` from `src/models/video_vae.py`, not the deep-narrow or RAM variants.

Saved config summary:

- `model_name = video_vae`
- `num_colors = 23`
- `frame_size = 224`
- `clip_frames = 16`
- `context_frames = 8`
- `base_channels = 24`
- `latent_channels = 16`
- `temporal_downsample = 1`
- `num_parameters = 17,685,463`

Important detail: the model was trained on 24-frame windows, not 16-frame windows. The training script builds `total_clip_frames = clip_frames + context_frames`, so each training input was `8` context frames plus `16` target frames.


## What This Architecture Is

At a high level, this checkpoint is a compact causal 3D-convolutional VAE with:

- palette one-hot input, not RGB input
- a symmetric encoder/decoder
- 5 learned spatial downsample stages
- 1 learned temporal downsample stage at the deepest encoder level
- residual blocks at every scale
- no attention
- no transformer blocks
- no UNet skip connections between encoder and decoder
- no patchify / unpatchify stem like LTX uses

The model takes palette-index frames, converts them to one-hot tensors of shape `(B, 23, T, H, W)`, encodes them to a latent tensor, samples from a diagonal Gaussian posterior, and decodes back to per-pixel palette logits.


## Input Representation

The VAE does not consume RGB frames directly.

- Training frames are palette indices with shape `(B, T, H, W)`.
- They are converted to one-hot tensors with shape `(B, num_colors, T, H, W)`.
- For this checkpoint, that means `(B, 23, T, 224, 224)`.

This matters because the model is learning a tokenizer over a discrete NES palette layout, not over continuous RGB values.


## Training Window Semantics

The run config says:

- `clip_frames = 16`
- `context_frames = 8`

The training script combines these into a 24-frame input window.

What that means in practice:

- The model encodes and decodes all 24 frames.
- The first 8 frames are warmup context.
- Reconstruction loss is only applied to the last 16 frames.

So the latent time dimension during training corresponds to the full 24-frame input window, not just the 16 target frames.


## Building Blocks

### 1. CausalConv3d

`CausalConv3d` is the fundamental convolution primitive.

- Temporal padding is causal.
- It duplicates the first frame at the clip start for left padding.
- Spatial padding is replicate padding.

Operationally, each output frame can only depend on the current frame and earlier frames, never future frames.


### 2. ResidualBlock3D

Each residual block is:

- GroupNorm
- SiLU
- causal 3D conv
- GroupNorm
- SiLU
- causal 3D conv
- residual add

If channel count changes, the skip path uses a `1x1x1` projection.


### 3. Downsample3D

Spatial downsample is always done with stride `(1, 2, 2)`, so height and width are halved.

When `downsample_time=True`:

- adjacent frame pairs are packed into the channel dimension
- if the number of frames is odd, the last frame is duplicated
- then the packed tensor is convolved with stride `(1, 2, 2)`

So temporal downsampling is implemented by pair-packing before the conv rather than by a raw stride-2 temporal convolution.


### 4. SpatialUpsample3D

Upsampling is the mirror of the down path:

- nearest-neighbor interpolation
- followed by a causal 3D conv

When `upsample_time=True`, the scale factor is `(2, 2, 2)`.
Otherwise it is `(1, 2, 2)`.


## Exact Width Schedule

With `base_channels = 24`, the hidden widths are:

- base: `24`
- hidden_2: `48`
- hidden_4: `96`
- hidden_8: `192`

So the encoder width progression is:

- `23 -> 24 -> 24 -> 48 -> 96 -> 192 -> latent 16`

and the decoder mirrors that back out.


## Layer-By-Layer Shape Walkthrough

Below is the tensor flow for the actual 24-frame training window.

Tensor order is `(B, C, T, H, W)`.

### Encoder

| Stage | Operation | Output shape |
| --- | --- | --- |
| input | one-hot palette frames | `B x 23 x 24 x 224 x 224` |
| encoder_in | causal 3D conv | `B x 24 x 24 x 224 x 224` |
| encoder_block1 | residual block | `B x 24 x 24 x 224 x 224` |
| encoder_down1 | spatial downsample | `B x 24 x 24 x 112 x 112` |
| encoder_block2 | residual block | `B x 24 x 24 x 112 x 112` |
| encoder_down2 | spatial downsample + width x2 | `B x 48 x 24 x 56 x 56` |
| encoder_block3 | residual block | `B x 48 x 24 x 56 x 56` |
| encoder_down3 | spatial downsample + width x2 | `B x 96 x 24 x 28 x 28` |
| encoder_block4 | residual block | `B x 96 x 24 x 28 x 28` |
| encoder_down4 | spatial downsample + width x2 | `B x 192 x 24 x 14 x 14` |
| encoder_block5 | residual block | `B x 192 x 24 x 14 x 14` |
| encoder_down5 | spatial downsample + temporal pair-pack | `B x 192 x 12 x 7 x 7` |
| encoder_block6 | residual block | `B x 192 x 12 x 7 x 7` |
| encoder_mid | residual block | `B x 192 x 12 x 7 x 7` |
| encoder_out | `1x1x1` conv to mean+logvar | `B x 32 x 12 x 7 x 7` |
| posterior split | mean and logvar | `B x 16 x 12 x 7 x 7` each |

### Bottleneck

Sampled latent shape on the training window:

- `B x 16 x 12 x 7 x 7`

Total latent values per sample:

- `16 * 12 * 7 * 7 = 9,408`

Grid compression relative to the input coordinate grid:

- time: `24 -> 12` which is `2x`
- height: `224 -> 7` which is `32x`
- width: `224 -> 7` which is `32x`

Total coordinate-grid reduction:

- `(24 * 224 * 224) / (12 * 7 * 7) = 2048x`

If you ignore the extra 8 context frames and think only about a 16-frame window, the latent would be:

- `B x 16 x 8 x 7 x 7`


### Decoder

| Stage | Operation | Output shape |
| --- | --- | --- |
| decoder_in | `1x1x1` conv | `B x 192 x 12 x 7 x 7` |
| decoder_mid | residual block | `B x 192 x 12 x 7 x 7` |
| decoder_block6 | residual block | `B x 192 x 12 x 7 x 7` |
| decoder_up1 | spatial + temporal upsample | `B x 192 x 24 x 14 x 14` |
| decoder_block5 | residual block | `B x 192 x 24 x 14 x 14` |
| decoder_up2 | spatial upsample | `B x 96 x 24 x 28 x 28` |
| decoder_block4 | residual block | `B x 96 x 24 x 28 x 28` |
| decoder_up3 | spatial upsample | `B x 48 x 24 x 56 x 56` |
| decoder_block3 | residual block | `B x 48 x 24 x 56 x 56` |
| decoder_up4 | spatial upsample | `B x 24 x 24 x 112 x 112` |
| decoder_block2 | residual block | `B x 24 x 24 x 112 x 112` |
| decoder_up5 | spatial upsample | `B x 24 x 24 x 224 x 224` |
| decoder_block1 | residual block | `B x 24 x 24 x 224 x 224` |
| decoder_out | logits over palette classes | `B x 23 x 24 x 224 x 224` |


## What The Decoder Predicts

The decoder output is not RGB.

It predicts:

- per-pixel palette logits
- shape `(B, 23, T, H, W)`

Final reconstruction is obtained by `argmax` over the 23 palette channels.


## Loss Structure

This is a standard VAE objective over palette logits:

- reconstruction term: focal cross-entropy over palette indices
- KL term: diagonal Gaussian posterior penalty

The total loss is:

`loss = recon_loss + kl_weight * kl_loss`

For this run, `kl_weight = 0.0005`.

Because of context cropping, the reconstruction term only applies to frames `8:24` of the decoded output.


## Exact Parameter Breakdown

Total trainable parameters:

- `17,685,463`

### Encoder: 9,341,912 params (52.8%)

| Block | Params |
| --- | ---: |
| encoder_in | 14,928 |
| encoder_block1 | 31,248 |
| encoder_down1 | 15,576 |
| encoder_block2 | 31,248 |
| encoder_down2 | 31,152 |
| encoder_block3 | 124,704 |
| encoder_down3 | 124,512 |
| encoder_block4 | 498,240 |
| encoder_down4 | 497,856 |
| encoder_block5 | 1,991,808 |
| encoder_down5 | 1,990,848 |
| encoder_block6 | 1,991,808 |
| encoder_mid | 1,991,808 |
| encoder_out | 6,176 |

### Decoder: 8,343,551 params (47.2%)

| Block | Params |
| --- | ---: |
| decoder_in | 3,264 |
| decoder_mid | 1,991,808 |
| decoder_block6 | 1,991,808 |
| decoder_up1 | 995,520 |
| decoder_block5 | 1,991,808 |
| decoder_up2 | 497,760 |
| decoder_block4 | 498,240 |
| decoder_up3 | 124,464 |
| decoder_block3 | 124,704 |
| decoder_up4 | 31,128 |
| decoder_block2 | 31,248 |
| decoder_up5 | 15,576 |
| decoder_block1 | 31,248 |
| decoder_norm | 48 |
| decoder_out | 14,927 |


## Where The Capacity Actually Lives

Most of the model lives in the narrow but deep 192-channel core.

These five pieces alone dominate the parameter budget:

- `encoder_block5`
- `encoder_down5`
- `encoder_block6`
- `encoder_mid`
- `decoder_mid`
- `decoder_block6`
- `decoder_up1`
- `decoder_block5`

If you group the 192-channel trunk together, it accounts for the large majority of parameters.

Interpretation:

- This model is small overall.
- It is not wide.
- It concentrates capacity at the lowest-resolution latent-adjacent part of the network.
- The shallow high-resolution stages are comparatively cheap.


## Architectural Character

This design sits in an interesting middle ground:

- much smaller than a modern large video tokenizer
- still fully learned and symmetric
- enough spatial hierarchy to get to a `7 x 7` latent grid
- only light temporal compression, preserving more time steps than an aggressively compressed video tokenizer

In practice that means:

- better temporal detail retention than a heavily time-compressed latent
- less semantic abstraction than a wider, more aggressively compressed tokenizer
- easier to train and cheaper to run than a large latent video autoencoder


## Comparison To Standard LTX VAE

The standard LTX encoder documentation says:

- `patch_size = 4`
- encoder blocks: `1x compress_space_res`, `1x compress_time_res`, `2x compress_all_res`
- final dimensions: `F' = 1 + (F - 1) / 8`, `H' = H / 32`, `W' = W / 32`

The LTX configurator defaults also indicate:

- latent channels default to `128`
- decoder base channels default to `128`
- decoder timestep conditioning defaults to `True`

### Side-by-side summary

| Aspect | This checkpoint | Standard LTX VAE |
| --- | --- | --- |
| Input channels | 23 palette one-hot channels | 3 RGB channels |
| Input representation | discrete palette indices -> one-hot | continuous image/video tensor |
| Stem | direct causal 3D conv | patchify `4x4` then conv |
| Spatial compression | `32x` | `32x` |
| Temporal compression | `2x` | `8x` |
| Latent channels | `16` | `128` default |
| Bottleneck for 224x224 | `7 x 7` spatial grid | `7 x 7` spatial grid |
| Peak encoder width | `192` | on the documented standard path, widths grow much larger |
| Decoder conditioning | none | timestep conditioning on by default |
| Architecture style | compact symmetric conv VAE | much larger patchified spatiotemporal VAE |


### The most important differences

#### 1. Your model is palette-native, LTX is RGB-native

Your tokenizer is specialized for a small discrete NES palette.

That makes a huge difference:

- the input space is simpler
- reconstruction is classification over palette IDs rather than regression over RGB values
- the model can stay much smaller while still modeling the data well


#### 2. Your model compresses time much less aggressively

Your VAE only halves time once:

- `T -> ceil(T / 2)`

Standard LTX compresses time by `8x` overall.

So your latent keeps more temporal resolution, while LTX pushes more abstraction into channel width.


#### 3. Your latent is dramatically narrower

Your latent width is `16`.

Standard LTX defaults to `128` latent channels.

That is an `8x` difference in latent channel count alone.


#### 4. LTX is decoder-heavy

The LTX decoder is much more elaborate:

- patchified pipeline
- wider internal feature widths
- space-to-depth / depth-to-space style resampling blocks
- timestep-conditioning path enabled by default

That makes the standard LTX VAE much larger, especially in the decoder.


## Depth And Component Comparison With LTX

One useful correction to the vague intuition that LTX is simply a "bigger and deeper" model:

- standard LTX is much bigger
- standard LTX is usually much wider
- standard LTX is much more aggressive in temporal compression
- standard LTX uses more specialized blocks
- but the documented standard LTX path is not actually deeper than this project VAE in the plain stacked-conv sense

### Macro-depth comparison

For this checkpoint, the encoder side is:

- 1 stem conv
- 7 residual blocks: `encoder_block1` through `encoder_block6` plus `encoder_mid`
- 5 downsample blocks
- 1 output conv

The decoder mirrors that with:

- 1 input conv
- 7 residual blocks: `decoder_mid`, `decoder_block6`, `decoder_block5`, `decoder_block4`, `decoder_block3`, `decoder_block2`, `decoder_block1`
- 5 upsample blocks
- 1 output conv

So this VAE is a fairly deep plain conv hierarchy.

If we count learned conv-style modules very literally:

- each residual block contributes 2 convs
- each downsample contributes 1 conv
- each upsample contributes 1 conv
- each stem or head contributes 1 conv

That gives roughly:

- encoder: `1 + (7 * 2) + 5 + 1 = 21` learned conv layers
- decoder: `1 + (7 * 2) + 5 + 1 = 21` learned conv layers
- total: about `42` learned conv layers

For the documented standard LTX encoder path, the source describes:

- `patch_size = 4`
- encoder blocks: `1x compress_space_res`, `1x compress_time_res`, `2x compress_all_res`

That standard path is structurally much shorter:

- patchify
- 1 stem conv
- 4 configured down blocks
- 1 output conv

The decoder mirrors that with:

- 1 input conv
- 4 configured up blocks
- 1 output conv
- optional timestep-conditioning heads

So the default documented LTX route is shallower by stage count and shallower by raw conv count, but much wider and more feature-rich.


## Block-By-Block Comparison

### Your `ResidualBlock3D`

The block in `src/models/video_vae.py` is a canonical residual conv block:

- GroupNorm
- SiLU
- causal 3D conv
- GroupNorm
- SiLU
- causal 3D conv
- identity or `1x1x1` skip if channel count changes

It is "ordinary" in the sense that it is the standard simple two-conv residual pattern without extra conditioning or architectural tricks.

That is not a criticism. It means:

- easy to reason about
- computationally straightforward
- no hidden modulation path
- no attention
- no injected noise
- no special shortcut normalization logic beyond the basic skip


### LTX `ResnetBlock3D`

The LTX residual block has the same broad skeleton, but it is more elaborate.

In addition to the two-conv residual structure, it supports:

- PixelNorm by default instead of GroupNorm
- dropout
- optional injected spatial noise
- optional timestep-conditioned scale/shift modulation
- extra normalization on the shortcut path when channel count changes

So the right description is:

- your block is simpler and more canonical
- LTX's residual block is a more feature-rich conditioned residual block


### Your downsample / upsample blocks

Your downsampling and upsampling are simple and conventional.

`Downsample3D`:

- spatially downsamples with stride `(1, 2, 2)`
- optionally halves time by packing adjacent frame pairs into channels first
- then applies one causal 3D conv

`SpatialUpsample3D`:

- nearest-neighbor interpolation
- then one causal 3D conv

These are easy to understand and relatively cheap.


### LTX resampling blocks

LTX uses more specialized resamplers:

- `SpaceToDepthDownsample`
- `DepthToSpaceUpsample`

These are not simple stride-conv or interpolate-conv layers.

Instead, they:

- rearrange space and/or time into channels
- apply a conv in the rearranged representation
- build an explicit residual path through the rearrangement itself
- then combine the conv path with that residual path

This is one of the biggest architectural differences between the models.

So compared to your model:

- your resampling is more conventional
- LTX resampling is more structured and more specialized


### LTX mid-blocks and optional attention

LTX also has a `UNetMidBlock3D` abstraction that can wrap multiple residual blocks, and the decoder block factory supports `attn_res_x` variants.

That means the LTX codebase is built to support:

- multi-layer mid-blocks
- timestep-conditioned residual stacks
- optional attention-bearing mid-blocks

Your current VAE does not use any equivalent attention or multi-block mid-stack abstraction. It just instantiates fixed individual residual blocks.


## What "Ordinary" Meant

When describing your block as "ordinary," the intended meaning was:

- it is a standard two-conv residual block
- it does not have adaptive modulation
- it does not inject explicit noise
- it does not include attention
- it is not fused with patchify or space-to-depth style resampling
- it is not wrapped in a larger configurable mid-block abstraction

That is often a strength, not a weakness.

For this project, the sophistication is mostly in:

- the overall hierarchy
- the temporal downsample choice
- the palette-specific input/output representation
- the bottleneck size and width schedule

rather than in heavily engineered individual block internals.


## Updated LTX Comparison Summary

The cleanest summary is:

- this checkpoint VAE is deeper and narrower
- standard LTX is shallower and much wider
- this checkpoint uses simple conventional residual and resampling blocks
- standard LTX uses more specialized patchified and rearrangement-based resampling blocks
- this checkpoint is a compact task-specific tokenizer
- standard LTX is a larger general-purpose latent video autoencoder design


## Parameter Count Comparison Caveat

The exact parameter count of a specific LTX checkpoint cannot be pinned down from this workspace alone unless a concrete LTX VAE config or checkpoint is loaded.

What is known directly from source:

- standard LTX uses `latent_channels = 128` by default
- standard LTX decoder uses `decoder_base_channels = 128` by default
- standard LTX encoder compression path is documented
- standard LTX decoder is designed to be much wider than this project VAE

Reasonable conclusion without overclaiming exactness:

- this checkpoint at `17.7M` params is far smaller than a standard-width LTX VAE
- the gap is large, not marginal
- the main drivers are much smaller latent width, much smaller internal widths, less aggressive decoder machinery, and the palette-domain input/output

In other words, this model is closer to a compact task-specific tokenizer, while standard LTX is closer to a general-purpose high-capacity latent video autoencoder.


## Rough Intuition For The Compression Tradeoff

Your checkpoint chooses:

- low latent width
- mild temporal compression
- moderate spatial compression
- compact conv-only architecture

Standard LTX chooses:

- high latent width
- strong temporal compression
- similar spatial compression
- much larger decoder and latent interface

So the tradeoff is not simply "smaller vs larger".

It is more like:

- your model preserves more time positions but stores less information per latent position
- LTX preserves fewer time positions but stores much more information per latent position


## Practical Interpretation

For NES-like palette video, this checkpoint makes sense if the goal is:

- compact latent storage
- manageable training cost
- reasonably faithful short-horizon reconstructions
- enough temporal structure to support downstream latent modeling without crushing the time axis too early

It is not trying to be a broad, high-capacity, photorealistic video tokenizer.


## How To Visualize This Model

### Option 1: Print a shape trace with forward hooks

This is the cleanest way to see the architecture without adding dependencies.

Run from the repo root:

```bash
conda run -n mario python - <<'PY'
import json
from pathlib import Path
import torch
from src.models.video_vae import VideoVAE

cfg = json.loads(Path("checkpoints/video_vae_20260406_065725/config.json").read_text())
m = cfg["model"]
d = cfg["data"]
total_t = cfg["training"]["clip_frames"] + cfg["training"]["context_frames"]

vae = VideoVAE(
    num_colors=d["num_colors"],
    base_channels=m["base_channels"],
    latent_channels=m["latent_channels"],
    temporal_downsample=m["temporal_downsample"],
).eval()

hooks = []

def hook(name):
    def _hook(module, inputs, output):
        if isinstance(output, torch.Tensor):
            print(f"{name:35s} -> {tuple(output.shape)}")
    return _hook

for name, module in vae.named_modules():
    if len(list(module.children())) == 0:
        hooks.append(module.register_forward_hook(hook(name)))

x = torch.zeros(1, d["num_colors"], total_t, d["frame_height"], d["frame_width"])
with torch.no_grad():
    out = vae(x, sample_posterior=False)

print("latents:", tuple(out.latents.shape))
print("logits :", tuple(out.logits.shape))

for h in hooks:
    h.remove()
PY
```


### Option 2: Print parameter counts by submodule

```bash
conda run -n mario python - <<'PY'
import json
from pathlib import Path
from collections import defaultdict
from src.models.video_vae import VideoVAE

cfg = json.loads(Path("checkpoints/video_vae_20260406_065725/config.json").read_text())
m = cfg["model"]
d = cfg["data"]

vae = VideoVAE(
    num_colors=d["num_colors"],
    base_channels=m["base_channels"],
    latent_channels=m["latent_channels"],
    temporal_downsample=m["temporal_downsample"],
)

counts = defaultdict(int)
for name, param in vae.named_parameters():
    block = name.split('.')[0]
    counts[block] += param.numel()

total = sum(counts.values())
for key, value in counts.items():
    print(f"{key:20s} {value:>10,d}  {100*value/total:6.2f}%")
print(f"{'total':20s} {total:>10,d}")
PY
```


### Option 3: Draw an autograd graph

If you want a graph image rather than a textual summary, install `torchviz` and use:

```python
from torchviz import make_dot
make_dot(out.logits.mean(), params=dict(vae.named_parameters())).render("video_vae_graph", format="png")
```

This is useful for seeing operator connectivity, but it is usually noisier than the hook-based summary for understanding the architecture.


## Short Takeaways

- This checkpoint is a compact palette-native causal 3D-conv VAE.
- It has exact width schedule `24 -> 48 -> 96 -> 192 -> latent 16` with a symmetric decoder.
- The bottleneck on the 24-frame training window is `16 x 12 x 7 x 7`.
- Total parameter count is `17.7M`.
- Most of the capacity is concentrated in the 192-channel low-resolution trunk.
- Compared to standard LTX, this model is much narrower, much smaller, less decoder-heavy, and much less aggressive in temporal compression.
- Compared to standard LTX, it is also much more domain-specific because it works over discrete palette indices rather than RGB.