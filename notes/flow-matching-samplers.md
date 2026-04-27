# Flow Matching Samplers, Numerical Integrators, and Rectified Flow

This note summarizes the discussion around the current latent DiT sampling loop, what numerical solver it is actually using, what better alternatives are worth testing, and how rectified flow relates to the current training setup.


## 1. Current Sampling Loop in This Repo

The current inference path for the split latent DiT is in:

- `src/models/latent_utils.py`
- `scripts/eval/play_world_model.py`

The rollout procedure is:

1. Start the future latent segment from Gaussian noise.
2. Hold the history latent segment fixed as conditioning.
3. Evaluate the model's predicted velocity field at a sequence of timesteps.
4. Integrate that velocity field backward from noise toward data.

The update used now is:

$$
x \leftarrow x - \Delta t\, v_\theta(x, t, c)
$$

with a uniform step size:

$$
\Delta t = \frac{1}{N}
$$

and timestep samples taken at:

$$
t_k = \frac{k - 0.5}{N}
$$

for `k = N, N-1, ..., 1`.

The default rollout in `play_world_model.py` uses `N = 8` ODE steps.


## 2. What Solver This Actually Is

The docstring currently calls this "midpoint ODE integration", but that is not quite accurate.

True midpoint Runge-Kutta would first evaluate the field at the current state, take a half step in state space, then evaluate again at the midpoint state. The current code does **not** do that. It only evaluates at a midpoint **time** while still using the current state.

So the present method is best described as:

- a deterministic explicit ODE sampler
- single model evaluation per step
- roughly first-order accuracy
- cheaper than RK2 / Heun, but less accurate at low step counts

This matters because at only 8 denoising steps, solver error can be a meaningful part of total generation error.


## 3. Training Objective Used Here

The DiT training target is defined in `scripts/train/train_video_latent_dit.py` as:

$$
x_t = (1-t)x_0 + t\epsilon
$$

with velocity target:

$$
v^* = \epsilon - x_0
$$

That is a straight-line interpolation between clean latent `x_0` and Gaussian noise `\epsilon`, with direct velocity prediction.

This is **not** DDPM training. It is a linear flow-matching objective.


## 4. What Rectified Flow Is

Rectified flow is a particular straight-path formulation of flow-based generative modeling.

The core idea is:

1. Choose a simple path from noise to data, usually a straight line.
2. Train a model to predict the velocity field along that path.
3. Sample by integrating the learned ODE from noise back to data.

The common basic form looks like:

$$
x_t = (1-t)x_0 + t x_1
$$

where `x_1` is often Gaussian noise.

The learned model predicts the transport direction that should move `x_t` toward the data manifold. One of the main motivations is that straight trajectories are easier to learn and can often be sampled in relatively few steps.

Compared with classical diffusion:

- DDPM is usually framed as iterative denoising in a Markov chain.
- Flow matching / rectified flow is framed as learning a continuous transport ODE.
- In practice, both often use similar neural backbones, but the training target and sampler interpretation differ.


## 5. Is This Repo Using Rectified Flow?

Mostly yes, in the practical sense.

The current DiT trainer uses:

- straight interpolation between clean latent and Gaussian noise
- direct velocity prediction
- deterministic ODE-style sampling

So it is already much closer to **rectified-flow-style linear flow matching** than to a DDPM-style noise-prediction system.

However, there are a few caveats:

- The code and notes mostly call it "flow matching", which is the broader umbrella term.
- It does not appear to implement extra rectified-flow-specific tricks such as reflow, trajectory straightening passes, or special distilled samplers.
- The current inference loop is a simple explicit integrator, not a more advanced rectified-flow solver.

So the clean summary is:

> The current world-model DiT is already using a basic rectified-flow-style / linear flow-matching objective.

What is *not* especially advanced yet is the **numerical solver** used at inference.


## 6. Numerical Integrators Worth Testing

### 6.1 Current method: single-eval explicit solver

Pros:

- cheapest per denoising step
- simple and stable to implement
- easy to combine with action CFG

Cons:

- only first-order accurate
- likely suboptimal at 8 steps
- may need more steps than a better solver for the same quality


### 6.2 Heun's method / trapezoidal RK2

This is probably the best next solver to test.

Use:

$$
k_1 = v_\theta(x_n, t_n, c)
$$

$$
\tilde{x} = x_n - \Delta t\, k_1
$$

$$
k_2 = v_\theta(\tilde{x}, t_n - \Delta t, c)
$$

$$
x_{n-1} = x_n - \frac{\Delta t}{2}(k_1 + k_2)
$$

Pros:

- true second-order method
- usually much better than Euler-like methods at low step counts
- conceptually simple

Cons:

- doubles the number of model evaluations


### 6.3 True midpoint RK2

Also a good candidate:

$$
k_1 = v_\theta(x_n, t_n, c)
$$

$$
x_{mid} = x_n - \frac{\Delta t}{2} k_1
$$

$$
k_2 = v_\theta\left(x_{mid}, t_n - \frac{\Delta t}{2}, c\right)
$$

$$
x_{n-1} = x_n - \Delta t\, k_2
$$

This is what the current code would need to do if it really wanted to deserve the "midpoint" name.


### 6.4 Multistep methods

Methods like Adams-Bashforth can improve quality per model evaluation after the first step or two by reusing previous velocity estimates.

These are attractive when model evaluations are expensive, but they are usually a little more fragile when:

- guidance is strong
- the field is noisy or highly curved
- rollout conditioning changes the field sharply from one step to the next

That may make them less ideal as the first thing to try here.


### 6.5 Fancy diffusion-specific solvers

DPM-Solver, UniPC, and related high-order samplers can be excellent for diffusion and flow-style models, but they are most attractive when:

- the noise parameterization matches their assumptions well
- there is already a standard sigma schedule in place
- the main task is standalone image/video generation rather than autoregressive world-model rollout

They are worth considering later, but probably not as the first upgrade for this repo.


## 7. Schedule Choice Also Matters

Solver order is not the only variable.

The current world-model loop uses a **uniform time grid**. That is simple, but many generative samplers benefit from nonuniform schedules that spend more steps where the field is hardest, often near the lower-noise end.

This means there are really two orthogonal choices:

1. The **integrator**: Euler-like, Heun, midpoint RK2, multistep, etc.
2. The **time schedule**: uniform vs nonuniform.

At low step counts, schedule choice can matter almost as much as integrator choice.


## 8. What Is Most Likely Worth It Here

If the goal is a pragmatic next experiment, the best ranking is probably:

1. Keep the same flow-matching objective.
2. Replace the current sampler with true Heun or true midpoint RK2.
3. Compare at fixed NFE, not only fixed step count.
4. Try a simple nonuniform time schedule.

The reason to compare at fixed NFE is that an 8-step single-eval sampler and a 4-step two-eval Heun sampler both cost 8 model evaluations. That is often the fairest comparison.


## 9. Likely Best Interpretation of the Current Setup

The clearest way to describe the current repo is:

- **Training:** linear flow matching, very close to basic rectified flow.
- **Sampling:** simple deterministic first-order ODE integration on a uniform grid.
- **Not yet using:** a truly second-order solver, a specialized flow sampler, or rectified-flow-specific reflow/distillation tricks.

So if asked "Are we already using rectified flow?", the honest answer is:

> Yes, approximately, in the sense that the training objective is straight-line velocity-based flow matching.

If asked "Are we already using the best rectified-flow sampler?", the answer is:

> No. The current inference loop is a simple baseline integrator and there is room to improve it.


## 10. Recommended Ablation

When revisiting this, test at least:

1. Current sampler, 8 steps.
2. Heun, 4 steps.
3. Heun, 8 steps.
4. Current sampler with a nonuniform schedule.
5. One short multi-step rollout metric, not only one-step denoising quality.

The last point matters because better local denoising does not automatically give better long-horizon world-model rollouts.
