Would be really interested to see a no-image-encoder model that uses attention over a time sequence of nes ram dumps to produce the tokens that the transformer predicts and decoder generates images from.....


Skimmed the magvit2 paper, notes: Increase warmup period, push the temporal downsampling layer deeper, codebook size of 2^18 should be enough which would be two 512 sized codebooks


Instead of all combinations of architecture variables maybe just define an order to test them as additions to some base model

Half res / 2d init


Write down some findings on the bottleneck bitwidth calculations


Image inflation with 2d model trained on sprite sheets? is that too much engineering? yes.


Find nicer abstractions between the training scripts, de-duplicate code/fixes/optimizations/...!


Deeper smaller models?


Adding the gan actually made video vae training more stable.


The latent space is "smoother" than pixels? interesting.


Pretty neat idea for vq-vae/lfq to overcome approximation error of the straight-through estimator x + (f(x) - x).detach(): https://github.com/cfifty/rotation_trick

Benchmark xformers or FlashAttention after revisiting the DiT action masking; if I remove the action mask, re-check whether the attention backend can be simplified and sped up.

For the sake of computational cost I might have the DiT target a tick rate of 15hz. Meaning the DiT would predict chunks of 4 frames per tick since mario runs at 60hz.

VS. super mario bros and super mario bros the lost levels exist

train with heavy kl-weight then freeze encoder and train only decoder?

test reconstruction maxxing: increase decoder capacity, turn off next-frame, temporal smoothness, palette corruption, kl warmup schedule, gan for late stage


latent design
big latent channels but heavily regularization to give the network lots of
"slots" to put information?

patchify on the encoder side? doesn't lose many parameters could boost flops

neat idea is ram encoder with discrete codes


which samples do we perform well on? late training to determine sample weights then start new training run with those?
ppu registers?


i've had this idea alot that the gradient signal decreases over training and eventually the only signal remaining is from the hard to learn and sparse features.
like, in every training run all the moving sprites are learned last. even tho the loss decreases more slowly it's still learning its just that the things its learning do not decrease the loss as much. maybe i need to pump up the class weight loss coefficient. also increasing batch size to lessen chance of getting no-signal batches
build a 'curriculum'?


smooth delta weight, reduce class weight, increase focal loss, use p(correct) map?












This is a solid strategy — it's essentially the approach used by VQ-VAE-2, LTX-Video, and other high-quality latent generative models. The core insight is sound: **the encoder's job (structured latent) and the decoder's job (sharp reconstruction) have conflicting optimization pressures, so decouple them.**

**Phase 1 — Structured latent** works because:
- High KL + weak decoder forces the encoder to actually use the latent (can't bypass it)
- Aux losses (RAM alignment, next-frame prediction, temporal smoothness) impose additional structure that's useful for your downstream DiT
- You're not optimizing for visual quality yet, just information content and structure
- This is exactly the "deliberately weak decoder" principle from the VQ-VAE paper

**Phase 2 — Sharp reconstruction** works because:
- Frozen encoder means the latent space can't degrade — its structure is locked in
- A stronger decoder (more channels, more layers, attention) can now focus purely on reconstruction
- GAN training is notoriously unstable with a moving encoder — freezing it removes that instability
- The discriminator pushes toward perceptually sharp outputs without distorting the latent

**Potential issues to watch for:**

1. **Information bottleneck may be too tight.** If Phase 1's KL is so high that the latent discards fine details, no decoder can recover them. Check the effective latent utilization (active dimensions, KL per dimension) before freezing. You can't decode information that isn't encoded.

2. **Decoder capacity jump.** Going from a weak to strong decoder is a big distribution shift on what the decoder expects from the latent. Consider starting Phase 2 from the Phase 1 decoder weights and *gradually* adding capacity (e.g., unfreezing new layers), rather than initializing a totally new decoder from scratch.

3. **GAN mode collapse on a frozen encoder.** The discriminator can only push the decoder — if the latent doesn't encode enough to distinguish certain frames, the decoder may hallucinate details that look plausible but are wrong. This is fine for generative quality but bad if downstream tasks need fidelity.

4. **Your todo note already has this idea** — "train with heavy kl-weight then freeze encoder and train only decoder" (todo.md). It's a good sign that you've been converging on this independently.

**Suggested concrete plan for your codebase:**

- Phase 1: `--kl-weight 1e-2`, keep current base-channels (weak decoder), enable all aux losses, train until KL stabilizes and latent metrics look good
- Phase 2: Freeze encoder, bump `--base-channels` or add decoder-only attention layers, add `--gan-weight`, drop KL and aux losses (they're on the encoder path which is frozen anyway), train with reconstruction + GAN loss only

The one thing I'd add: **validate the latent before committing to Phase 2.** Sample some latents, check that dimensions are active, interpolations are smooth, and your DiT can model the distribution. If the latent is bad, it's much cheaper to re-run Phase 1 than to discover it after Phase 2.



