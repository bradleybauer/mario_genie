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


I've lost my vae checkpoint. Luckily I wanted to retrain anyway since I've had some ideas for how to improve the latent representation. Also I'm going to reimplement to use accelerator and possibly some implementation library for the vae components.

Benchmark xformers or FlashAttention after revisiting the DiT action masking; if I remove the action mask, re-check whether the attention backend can be simplified and sped up.


For the sake of computational cost I might have the DiT target a tick rate of 15hz. Meaning the DiT would predict chunks of 4 frames per tick since mario runs at 60hz.

VS. super mario bros and super mario bros the lost levels exist


RAM alignment loss.
Use a frozen RAM encoder or RAM VAE, then project video posterior means to the RAM latent space and match aligned timesteps with MSE, cosine, or contrastive loss. This is the most direct way to make the video latent care about game state instead of just appearance.

Inverse or forward dynamics head.
Predict action, next RAM latent, or next video latent from current latent. That forces the latent to encode controllable state and transition structure, not just appearance.

Factorized latent.
Split the latent into semantic and appearance branches. Decode from both, but only feed the semantic branch to the world model. This is more work, but it matches your actual use case better than trying to make one monolithic latent do everything perfectly.

The difference is that a forward dynamics head is a representation-shaping loss, while joint DiT training is an end-task loss.

If you only train the encoder jointly with the DiT world model, the encoder is being asked to produce latents that are useful for a very large downstream model under a noisy denoising objective like the one you describe in matrix-game-3.0-model-training.md:24. That can work, but it does not force the latent itself to be clean, smooth, or semantically factorized. A powerful DiT can learn to compensate for an awkward latent. In other words, the world model may learn around the representation instead of improving it.

A forward dynamics head does something narrower and more aggressive. It says: from the current latent and action, predict the next state directly. That creates immediate pressure for the latent to carry control-relevant, transition-relevant information. It is much better at telling the encoder, “encode what changes game state,” rather than “encode whatever a big transformer can eventually use.” That is why it is not redundant, even if you also plan to train the DiT jointly.

The main practical differences are:

Credit assignment. A forward head gives short, direct gradients to the encoder. Joint DiT training gives delayed, entangled gradients through a large generative objective.
Capacity. A small forward head cannot hide bad latents. A large DiT often can.
Objective shape. A forward head encourages Markovian, predictable state structure. A DiT objective encourages sampleable future latents, which is related but not identical.
Cost. A forward head is cheap enough to use as an auxiliary loss every step. DiT-based shaping is expensive.
So if your goal is “best final simulator,” joint training with the DiT may be enough. If your goal is “a latent that is itself semantically meaningful and smooth,” an auxiliary head is still useful because it constrains the encoder before the DiT has a chance to absorb the mess.

For your setup, I would go one step further: do not make the auxiliary head predict raw next video latent only. That still rewards pixel-detail predictability. Better options are:

predict next RAM latent from current video latent plus action
align video posterior means to RAM embeddings as you already noted in ai-notes.md:344
add mild temporal smoothness on posterior means
optionally split the latent into semantic and appearance parts
That gives you the best of both: reconstruction keeps the decoder honest, RAM/state losses make the latent semantic, and the DiT still learns the full stochastic rollout model.

The clean recipe is usually:

L = recon + beta * KL + lambda_ram * state_alignment + lambda_dyn * forward_loss + lambda_smooth * temporal_smoothness + lambda_dit * world_model_loss

Not all terms need to be on from day one, but if you care about the latent itself, I would not rely on DiT joint training alone.

If you want, I can sketch which auxiliary head is most likely to help here:

video latent -> RAM latent
video latent + action -> next RAM latent
semantic/appearance latent split with DiT only on semantic branch




Going in a different direction here: I feel like you are underrating the pixel denoising augmentation a bit. if the model predicts exact pixels then it could (and i'm observing this ni its constructions right now) reconstruct the noise. Since the noise is not in the target the model would be penalized for shoving those pixels through its latent. Maybe I need to crank up the noise rate.

Searched for augmentation details
I’m checking the exact augmentation path and where the loss is applied, because whether denoising pressure is strong enough depends on the corruption distribution and whether the decoder still gets easy shortcuts through temporal context.

Reviewed train_video_vae.py and checked ai-notes.md
I have the loss path. I’m pulling the actual corruption function because the answer depends on whether the augmentation is sparse enough that the model can just memorize and pass through most pixels.

Searched for `def apply_palette_index_augmentation` in files
I found the exact corruption path. The main thing I’m seeing is that your current denoising pressure is real, but much weaker than it looks from the raw hyperparameters.

Searched for regex patterns and analyzed augmentation effects
Yes, this is a real effect, and I think your read is basically correct.

In the current training loop, the model sees corrupted input frames, but it is scored against the clean target frames. The corruption is applied in palette_video_vae_training.py:40, the noisy frames are fed into the model in train_video_vae.py:629, and reconstruction loss is computed against the original clean frames in train_video_vae.py:641. So if the encoder simply shoves corrupted pixels through the bottleneck and the decoder reproduces them, that does get penalized. That is standard denoising-autoencoder pressure, and it is more useful than I gave it credit for.

The reason I still would not rely on it alone is that your current corruption is fairly weak in aggregate. In your run config, only half the samples are corrupted and then only 20 percent of pixels in those samples are replaced in config.json:32 and config.json:33. That means the expected overall corruption rate is only about 10 percent of pixels, and the replacements are independent pixelwise draws from the palette distribution in palette_video_vae_training.py:76. For a conv VAE, that kind of speckle noise is often easy to clean up locally without forcing a deeply semantic latent.

So yes, increasing the noise rate is a very plausible next move. The cleaner sweep is:

Set sample corruption to 1.0 first, so every training example is denoising.
Then raise per-pixel corruption from 0.2 to 0.3 or 0.4 and watch whether reconstructions stop copying noise.
If that still mostly teaches local cleanup, switch from independent pixel noise to structured corruption like rectangular patch dropout, small region replacement, or occasional frame masking.
That last point matters most. If the corruption is local and unstructured, the model can become a good pixel denoiser without becoming a good state encoder. If the corruption removes larger coherent chunks, the model is forced to infer from object identity, temporal context, and game dynamics, which is much closer to the semantic behavior you want.

So my updated take is: you are not wrong to push denoising harder. It is probably underpowered right now. But stronger denoising is best thought of as a robustness and invariance tool, not a full replacement for state-shaped supervision. If your goal is a genuinely smooth, semantically meaningful latent, denoising augmentation is worth pushing, but I would still pair it with something state-linked later if the world model still struggles.

If you want, I can suggest a concrete corruption schedule for the next three VAE runs, ordered from safest to most aggressive.


Class weight smoothing so that local structures local to rare pixels get a boost?