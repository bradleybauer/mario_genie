# Relevant Papers List

## Variational Autoencoders — Foundations

- [ ] Kingma & Welling (2014), *"Auto-Encoding Variational Bayes"* (ICLR) — The original VAE paper.
- [ ] Rezende, Mohamed & Wierstra (2014), *"Stochastic Backpropagation and Approximate Inference in Deep Generative Models"* (ICML) — Independent co-discovery of VAEs / reparameterization trick.
- [ ] Higgins et al. (2017), *"β-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework"* (ICLR) — Upweighting KL for disentanglement.
- [ ] Chen et al. (2018), *"Isolating Sources of Disentanglement in Variational Autoencoders"* (NeurIPS) — β-TCVAE; decomposes KL into Index-Code MI, Total Correlation, and Dimension-wise KL.
- [ ] Hoffman & Johnson (2016), *"ELBO Surgery: Yet Another Way to Carve Up the Variational Evidence Lower Bound"* — ELBO decomposition analysis.
- [ ] Razavi, van den Oord & Vinyals (2019), *"Generating Diverse High-Fidelity Images with VQ-VAE-2"* (NeurIPS) — Hierarchical VQ-VAE for high-quality image generation.

## Identifiability & Gauge Symmetry in VAEs

- [ ] Khemakhem et al. (2020), *"Variational Autoencoders and Nonlinear ICA: A Unifying Framework"* (AISTATS) — Proves standard VAEs are non-identifiable; auxiliary conditioning restores identifiability.
- [ ] Locatello et al. (2019), *"Challenging Common Assumptions in the Unsupervised Learning of Disentangled Representations"* (ICML, Best Paper) — Proves unsupervised disentanglement is impossible without inductive biases.
- [ ] Hyvarinen & Morioka (2016), *"Unsupervised Feature Extraction by Time-Contrastive Learning and Nonlinear ICA"* (NeurIPS) — Temporal non-stationarity makes latent features identifiable.
- [ ] Hyvarinen & Morioka (2017), *"Nonlinear ICA of Temporally Dependent Stationary Sources"* (AISTATS) — Identifiability from temporal autocorrelation structure.
- [ ] Klindt et al. (2021), *"Towards Nonlinear Disentanglement in Natural Data with Temporal Sparse Coding"* (ICLR) — Temporal regularization recovers disentangled representations from video.
- [ ] Wiskott & Sejnowski (2002), *"Slow Feature Analysis: Unsupervised Learning of Invariances"* — The slowness principle: useful features change slowly over time.
- [ ] Lippe et al. (2022), *"CITRIS: Causal Identifiability from Temporal Intervened Sequences"* (ICML) — Temporal structure + interventions (actions) for identifiable causal representations.

## Vector Quantization

- [ ] van den Oord, Vinyals & Kavukcuoglu (2017), *"Neural Discrete Representation Learning"* (NeurIPS) — VQ-VAE; the original vector quantization approach for learned discrete latents.
- [ ] Yu et al. (2024), *"Language Model Beats Diffusion — Tokenizer is Key to Visual Generation"* (ICLR) — MagViT-2; lookup-free quantization (LFQ) for visual tokenization.
- [ ] Yu et al. (2023), *"MAGVIT: Masked Generative Video Transformer"* (CVPR) — MagViT; masked generative modeling over quantized video tokens.
- [ ] Mentzer et al. (2024), *"Finite Scalar Quantization: VQ-VAE Made Simple"* (ICLR) — FSQ; replaces codebook learning with simple scalar quantization.

## Transformers — Foundations

- [ ] Vaswani et al. (2017), *"Attention Is All You Need"* (NeurIPS) — Introduces the Transformer architecture and multi-head self-attention.

## Diffusion & Flow Matching

- [ ] Ho, Jain & Abbeel (2020), *"Denoising Diffusion Probabilistic Models"* (NeurIPS) — DDPM; foundational diffusion model paper.
- [ ] Ho & Salimans (2022), *"Classifier-Free Diffusion Guidance"* — The canonical CFG paper; jointly trains conditional and unconditional models and combines their predictions at sampling time without a separate classifier.
- [ ] Song et al. (2021), *"Score-Based Generative Modeling through Stochastic Differential Equations"* (ICLR) — Unifies score matching and diffusion under SDEs.
- [ ] Lipman et al. (2023), *"Flow Matching for Generative Modeling"* (ICLR) — Flow matching / conditional optimal transport; the framework used in your DiT trainer.
- [ ] Liu, Gong & Liu (2023), *"Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow"* (ICLR) — Rectified flow; straight-line ODE paths for fast sampling.

## Diffusion Transformers

- [ ] Peebles & Xie (2023), *"Scalable Diffusion Models with Transformers"* (ICCV) — DiT; replaces U-Net with transformer for diffusion.
- [ ] Esser et al. (2024), *"Scaling Rectified Flow Transformers for High-Resolution Image Synthesis"* (ICML) — Stable Diffusion 3; scaled DiT with flow matching.
- [ ] Chen et al. (2024), *"PixArt-α: Fast Training of Diffusion Transformer for Photorealistic Text-to-Image Synthesis"* — Efficient DiT training strategies.

## Video Generation & World Models

- [ ] Bruce et al. (2024), *"Genie: Generative Interactive Environments"* (ICML) — Action-conditioned video generation as a world model; spatiotemporal VQ tokenizer + dynamics transformer.
- [ ] Valevski et al. (2024), *"Genie 2: A Large-Scale Foundation World Model"* — Scaled-up Genie with latent diffusion backbone.
- [ ] Blattmann et al. (2023), *"Align Your Latents: High-Resolution Video Synthesis with Latent Diffusion Models"* (CVPR) — Video LDM; temporal layers inserted into image diffusion.
- [ ] Gupta et al. (2024), *"Photorealistic Video Generation with Diffusion Models"* — W.A.L.T; latent video diffusion with window attention.

## LTX & MatrixGame

- [ ] HaCohen et al. (2025), *"LTX-Video: Realtime Video Latent Diffusion"* — LTX-Video; efficient video VAE + DiT for real-time video generation.
- [ ] Meiri et al. (2025), *"MatrixGame: Interactive World Foundation Model for Game Engines"* — Joint audiovisual world model for interactive game generation.

## GAN Components & Regularization

- [ ] Kong, Kim & Bae (2020), *"HiFi-GAN: Generative Adversarial Networks for Efficient and High Fidelity Speech Synthesis"* (NeurIPS) — HiFi-GAN vocoder; multi-period / multi-scale discriminator for audio.
- [ ] Tseng et al. (2021), *"Regularizing Generative Adversarial Networks under Limited Data"* (CVPR) — LeCam regularization for stabilizing GAN training with limited data.
- [ ] Isola et al. (2017), *"Image-to-Image Translation with Conditional Adversarial Networks"* (CVPR) — PatchGAN discriminator used in many VAE+GAN hybrids.

## Training Tricks

- [ ] Nikishin et al. (2024), *"The Rotational Trick for Straight-Through Estimators"* — Gradient-preserving rotation for discrete / quantized latent training.
- [ ] Karras et al. (2020), *"Training Generative Adversarial Networks with Limited Data"* (NeurIPS) — Adaptive discriminator augmentation (ADA).
- [ ] Kingma & Ba (2015), *"Adam: A Method for Stochastic Optimization"* (ICLR) — Adam optimizer.
