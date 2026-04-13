# Latent Prediction Error, Compounding Drift, and Possible Remedies

This note summarizes a recurring issue in latent autoregressive prediction: a model can be locally good under teacher-forced training and still degrade quickly when its own predictions are fed back in as context.

The discussion here is specifically about the latent DiT training setup in:

- `scripts/train/train_video_latent_dit.py`
- `scripts/train/train_unified_dit.py`

and the related model definitions:

- `src/models/video_latent_dit_diffusers.py`
- `src/models/video_latent_dit_unified.py`


## 1. Core Problem

The current split-architecture trainer mostly learns from real histories sampled from the dataset.

At training time, the pattern is:

1. Take a real latent history `h`.
2. Take the true next latent or future slice `x`.
3. Corrupt `x` with noise to obtain `x_t`.
4. Train the model to predict the correct flow / velocity target that transports `x_t` back toward the real `x`.

At inference time, the pattern changes:

1. The model predicts the next latent.
2. That prediction is appended to history.
3. The next prediction is conditioned on a history that now includes model outputs rather than only real data.
4. This repeats autoregressively.

This creates a train-inference mismatch.

The model is optimized mostly for contexts drawn from the data distribution, but deployed on contexts drawn from its own rollout distribution.


## 2. What the Flow Loss Is Actually Learning

It is tempting to say the model is only learning:

"Given previous latents and actions, predict the delta to the next latent."

That is not quite right.

The flow objective is more properly learning a conditional generative transport field for the future latent distribution:

`p(x_future | history, actions)`

Under flow matching, the model sees noised targets and learns the velocity field that maps those noisy targets back toward real future samples.

So the model is not ignorant of the latent data distribution.

However, what it learns is mostly the local conditional field under training contexts that are close to real histories.

It does **not** necessarily learn the induced long-horizon rollout distribution that appears when the model repeatedly consumes its own predictions.


## 3. Why Errors Compound

Suppose the true training-time history distribution is:

`p_data(h)`

During rollout, the model instead conditions on:

`q_model(h)`

where `q_model` includes histories containing previous predicted latents.

If early predictions are slightly wrong, then later histories drift away from `p_data(h)`.

Once this happens, several things can go wrong:

- The model may be queried on contexts it rarely saw in training.
- The learned flow field may no longer point toward good future latents in those regions.
- The next prediction becomes slightly worse.
- That worse prediction is fed back into context.
- Drift compounds over time.

This is the main reason rollout quality can collapse even when one-step validation metrics look reasonable.


## 4. There Is Not Something "Wrong" With Flow Loss

The failure mode should not be interpreted as "flow matching is broken".

The issue is mostly one of mismatch:

- Training objective: one-step conditional denoising under mostly teacher-forced contexts.
- Deployment objective: stable multi-step self-conditioned rollout.

These are related, but not identical.

The flow objective can be locally correct while still producing fragile autoregressive behavior.


## 5. Role of the Residual Error Buffer

The existing residual error buffer is a sensible partial fix.

It perturbs the history with residual-like errors collected from prior predictions. That pushes the model to recover from slightly corrupted context rather than always seeing perfect history.

This helps because it broadens the context distribution seen during training.

But it is still only a proxy for true autoregressive training.

Why it is limited:

- It injects synthetic or replayed perturbations, not full self-generated rollout histories.
- It usually reflects local corruption, not multi-step compounding dynamics.
- It does not force the model to deal with the exact distribution induced by its own repeated predictions.


## 6. Split vs Unified Architecture in This Context

### Split architecture

In the split model (`video_latent_dit_diffusers.py`):

- History is encoded once.
- The decoder predicts the future conditioned on encoded history.
- History itself is not directly updated as a function of the current diffusion timestep.

This is efficient and well matched to predicting one future frame from a long context.

### Unified architecture

In the unified model (`video_latent_dit_unified.py`):

- History and noisy future tokens are concatenated.
- All tokens are processed jointly through self-attention.
- History representations become timestep-conditioned and can be updated based on the noisy future tokens.

This is more flexible, but also more expensive.

Important point:

Unified architecture does **not** automatically solve compounding error.

It changes the representational class and may help with dynamic history-future interaction, but rollout drift still comes from training on the wrong context distribution.


## 7. Why a Discriminator or Critic Is Appealing

A one-step flow loss is not designed to say:

"This latent sequence is drifting off the rollout manifold."

A discriminator or critic can help because it can learn to detect when self-generated predictions no longer resemble real latent trajectories.

This is especially attractive when the failure mode is compounding error rather than purely local denoising accuracy.

The intuition is:

- Compounding error moves rollouts away from the data-supported trajectory distribution.
- A learned critic can supply a signal that says "this trajectory no longer looks real / coherent / on-manifold".
- That signal may provide corrective pressure that the one-step flow loss does not.


## 8. But a One-Step Latent GAN Is Probably Weak

A discriminator on isolated predicted latents is probably not the most useful version of this idea.

Reasons:

- A latent can look statistically plausible in isolation but still lead to bad future rollouts.
- The real failure mode is often temporal inconsistency across several autoregressive steps.
- The latent space is an internal representation, not the final perceptual object.

So a one-step latent GAN may mostly enforce marginal latent realism, while the harder problem is rollout consistency.


## 9. Better Adversarial Target: Rollout-Level Critic

If adversarial training is added, the most sensible form is likely one of these:

1. A discriminator on short latent rollouts.
2. A discriminator on decoded short video rollouts.
3. A learned rollout critic that scores short generated trajectories conditioned on history and actions.

This is more aligned with the real deployment objective because the model is used autoregressively.

The critic should ideally judge things like:

- Does scene identity remain coherent over multiple self-fed steps?
- Do action-conditioned transitions still look plausible after several predictions?
- Does motion drift or become unstable?
- Do latent trajectories remain consistent with those seen in real data?


## 10. Why Decoded-Frame GAN Pressure Is More Principled Than Latent GAN Pressure

If compute allows it, applying the discriminator on decoded outputs is conceptually stronger than applying it directly on latents.

Reason:

- The VAE decoder maps latents into the perceptual space we actually care about.
- A decoded-video discriminator judges what matters visually.
- Latent-only discrimination risks overfitting to encoder-specific statistics that may not track perceptual quality.

The downside is obvious: decoding predicted latents during training is much more expensive.


## 11. Can We Add a Loss That Guarantees Recovery Toward the True Distribution?

Not in a strong theorem-like sense.

It is possible to add losses that **encourage** recovery toward the data or rollout manifold, but not ordinary losses that **guarantee** it.

Why not:

- We do not know the true conditional rollout distribution analytically.
- We only have samples.
- A learned objective produces empirical pressure, not a certified projector onto the true manifold.
- Sometimes the conditioning latents are already on-distribution, so there is no meaningful sense in which the output must become "closer" than the inputs.

The right target is not generic closeness to the marginal latent distribution.

The real target is something like:

`p_rollout(next_latents | previous_latents, actions)`

or, even more importantly, the multi-step distribution induced by the model's own repeated use.


## 12. Most Promising Practical Direction: Hybrid Training

The most practical approach is probably not a full replacement of the current objective, but a hybrid.

### Keep the current one-step flow loss

This remains a strong and sample-efficient base objective.

### Add short autoregressive rollout training

For example:

- predict 2 to 4 steps autoregressively
- feed previous predictions back in as new context
- apply losses on later steps or all rollout steps

This directly reduces train-inference mismatch.

### Optionally add a rollout-level critic

If compounding error remains severe after adding short rollout training, add a critic on short generated sequences.

This critic could operate either in latent space or decoded video space.

### Keep the residual error buffer

The error buffer remains useful as a cheap broadening mechanism for training contexts, even if more explicit rollout training is added.


## 13. Suggested Priority Order

If improving rollout robustness is the goal, the most sensible order of experiments is:

1. Keep one-step flow training as the base.
2. Add short autoregressive rollout training.
3. Increase or improve off-manifold history corruption if needed.
4. If rollout drift is still the dominant failure mode, add a rollout-level critic.
5. Only after that, consider a one-step latent GAN if there is a specific reason to do so.


## 14. Practical Takeaway

The central issue is not that the model has failed to learn what valid latents look like.

The real issue is that it is trained mostly under teacher-forced contexts but deployed under self-generated contexts.

That mismatch causes compounding error.

Therefore:

- one-step flow loss is necessary but not sufficient for stable rollout
- residual history perturbation helps but is only a partial fix
- short autoregressive training is the most direct remedy
- a rollout-level critic is the most coherent adversarial extension


## 15. Short Summary

- Flow loss learns a conditional generative field, not merely a simple delta regressor.
- Compounding error arises because rollout uses a different context distribution from training.
- Unified architecture changes representation capacity, but does not by itself solve this mismatch.
- A latent GAN on isolated next-step latents is probably weaker than it first appears.
- If adversarial training is added, rollout-level discrimination is more aligned with the true failure mode.
- The most practical next step is a hybrid of one-step flow loss plus short autoregressive rollout training.