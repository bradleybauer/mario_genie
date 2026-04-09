# VAE Latent Space & Disentanglement

## Does a Smaller Latent Force Disentanglement?

No — it's actually the opposite. A smaller latent dimension forces the model to **pack more information into fewer dimensions**, which means each dimension encodes a **mixture** of factors (more entangled, not less).

The KL term is the actual disentanglement mechanism. Here's why:

The KL divergence $\text{KL}(q(z|x) \| p(z))$ where $p(z) = \mathcal{N}(0, I)$ can be decomposed (Hoffman & Johnson, 2016; Chen et al., 2018) into three terms:

$$\text{KL} = \underbrace{I_q(x; z)}_{\text{Index-Code MI}} + \underbrace{\text{KL}\!\left(q(z) \| \prod_j q(z_j)\right)}_{\text{Total Correlation}} + \underbrace{\sum_j \text{KL}(q(z_j) \| p(z_j))}_{\text{Dimension-wise KL}}$$

- **Index-Code MI**: Mutual information between data and latent — how much the latent "knows" about which datapoint it came from.
- **Total Correlation (TC)**: $\text{KL}\!\left(q(z) \| \prod_j q(z_j)\right)$ — measures statistical dependence between latent dimensions. Penalizing this is what drives disentanglement: it pushes each $z_j$ to be independent of the others.
- **Dimension-wise KL**: Pushes each marginal $q(z_j)$ toward $\mathcal{N}(0,1)$.

When you increase $\beta$ (the KL weight), you're penalizing all three terms, but the TC term is the one that forces dimensions to become independent — which is disentanglement.

### Why $\beta$-VAE works

$\beta$-VAE (Higgins et al., 2017) simply uses $\beta > 1$ to upweight the KL term. This:
1. **Increases pressure on TC** → dimensions become more independent
2. **Each dimension encodes one factor** because encoding correlated info across dimensions is penalized
3. **Trade-off**: Too high $\beta$ crushes the latent (posterior collapse) — the model ignores $z$ entirely

### Practical implications for our RAM VAE

- `--kl-weight 3e-3` is a relatively low $\beta$. The model will learn a useful latent but dimensions may be correlated.
- Increasing kl-weight pushes toward disentanglement but sacrifices reconstruction quality.
- **KL annealing** (warmup from $0 \to$ target) helps: lets the model first learn good reconstructions, then gradually imposes structure.
- **Free bits** (per-dimension KL floor) prevents posterior collapse: each dimension is guaranteed to carry at least $\lambda$ nats of information.
- Latent dim 32 with 1365 categorical addresses is a $\sim 43\times$ compression. That's aggressive — each latent dimension must encode $\sim 43$ addresses worth of information.

### The sweet spot

The goal is a latent where:
- Each dimension is roughly independent (low TC)
- Each dimension is meaningfully used (no dead dimensions)
- Reconstruction quality is acceptable

This is achieved by tuning $\beta$ (kl-weight), using annealing to avoid early collapse, and using free bits to keep all dimensions alive.

---

## VAE Fundamentals

### What is a VAE?

A **Variational Autoencoder** (Kingma & Welling, 2014) is a generative model that learns a compressed latent representation of data. Unlike a regular autoencoder, a VAE imposes a probabilistic structure on the latent space.

### Architecture

$$x \xrightarrow{\text{Encoder}} (\mu, \log \sigma^2) \xrightarrow{\text{Sample}} z \xrightarrow{\text{Decoder}} \hat{x}$$

- **Encoder** $q_\phi(z|x)$: Maps input to parameters of a distribution (mean $\mu$ and log-variance $\log \sigma^2$)
- **Reparameterization trick**: $z = \mu + \sigma \cdot \epsilon$ where $\epsilon \sim \mathcal{N}(0, I)$ — makes sampling differentiable
- **Decoder** $p_\theta(x|z)$: Maps latent sample back to data space

### The Loss Function (ELBO)

The VAE maximizes the **Evidence Lower Bound (ELBO)**:

$$\mathcal{L} = \mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)] - \text{KL}(q_\phi(z|x) \| p(z))$$

Or equivalently, we minimize:

$$\text{Loss} = \text{Reconstruction Loss} + \beta \cdot \text{KL Loss}$$

- **Reconstruction loss**: How well can the decoder reproduce the input from $z$? For continuous data this is MSE; for categorical data (like our RAM bytes) it's cross-entropy.
- **KL loss**: How far is the encoder's posterior $q(z|x)$ from the prior $p(z) = \mathcal{N}(0, I)$? This regularizes the latent space.

### Why Not Just Use an Autoencoder?

A regular autoencoder can learn arbitrary latent representations with no structure. The latent space might have "holes" — regions where decoding produces garbage. The KL term forces the latent space to:

1. **Be smooth**: Nearby points in latent space decode to similar outputs
2. **Be complete**: Every point sampled from $\mathcal{N}(0, I)$ decodes to something reasonable
3. **Be structured**: The prior gives the space a known, navigable geometry

### The Reparameterization Trick

The key insight that makes VAEs trainable. Instead of sampling $z \sim q(z|x)$ (which isn't differentiable), we compute:

$$z = \mu + \sigma \odot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

Now gradients flow through $\mu$ and $\sigma$ to the encoder, while the randomness comes from $\epsilon$ (which doesn't depend on parameters).

### KL Divergence (Closed Form)

For Gaussian encoder $q(z|x) = \mathcal{N}(\mu, \sigma^2 I)$ and prior $p(z) = \mathcal{N}(0, I)$:

$$\text{KL} = -\frac{1}{2} \sum_{j=1}^{d} \left(1 + \log \sigma_j^2 - \mu_j^2 - \sigma_j^2\right)$$

This is what `RAMVAE.kl_loss()` computes. Each term penalizes:
- $\mu_j^2$: Means drifting from 0
- $\sigma_j^2$: Variances deviating from 1
- $\log \sigma_j^2$: Entropy bonus — prefers higher uncertainty

### Posterior Collapse

A common failure mode where the encoder learns $q(z|x) \approx \mathcal{N}(0, I)$ for all $x$ — the latent carries no information and the decoder ignores it. This happens when:
- KL weight ($\beta$) is too high
- Decoder is too powerful (can reconstruct from just the input structure)
- Training starts with high KL penalty before reconstruction is useful

**Solutions**:
- **KL annealing**: Start $\beta = 0$, linearly ramp to target over $N$ steps
- **Free bits** (Kingma et al., 2016): $\text{KL}_j = \max(\lambda, \text{KL}_j)$ — guarantee each dimension uses at least $\lambda$ nats
- **Cyclical annealing**: Repeatedly ramp $\beta$ from $0 \to$ target

### $\beta$-VAE and Disentanglement

Standard VAE uses $\beta = 1$. $\beta$-VAE uses $\beta > 1$ to put more pressure on the KL term, which encourages disentangled representations where each latent dimension captures one independent factor of variation.

The trade-off: higher $\beta$ = more disentangled but worse reconstruction.

### Common Reconstruction Losses

| Data Type | Loss | Decoder Output |
|-----------|------|---------------|
| Continuous (images, floats) | MSE / L1 | Raw values |
| Binary | Binary cross-entropy | Sigmoid probabilities |
| Categorical (our RAM) | Cross-entropy per class | Logits per category |
| Mixed | Weighted sum | Mixed heads |

### Our RAM VAE Specifics

Our RAMVAE is a **temporal categorical VAE**:
- **Input**: Sequence of NES RAM states `(B, T, 1365)` — each address is a categorical variable with variable cardinality (2–256 possible values)
- **Encoding**: Per-address learned embeddings → temporal convolutions → latent `(B, T/2, 32)`
- **Decoding**: Latent → temporal convolutions → per-address logits over valid values
- **Reconstruction loss**: Per-address cross-entropy with focal loss ($\gamma = 1.0$) to handle class imbalance
- **Temporal downsampling**: $2\times$ via frame-pair packing in channel dimension, matching the video VAE's temporal structure
- **Compile-friendly**: All operations are vectorized with precomputed gather buffers (no Python loops in hot path)
