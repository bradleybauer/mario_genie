# High level overview

I’m going to map the current LTX/MatrixGame pipeline from the code and docs, then summarize the models and their interactions compared to the Genie/MagViT path. I’ll first gather the relevant architecture files and training scripts.I’m running a targeted codebase scan to identify where LTX, MatrixGame, Genie, and MagViT components are defined so the writeup is grounded in actual implementation details.

**LTX/MatrixGame route, as this repo stands**

If you go LTX/MatrixGame instead of Genie/MagViT, your architecture is basically a latent-world-model stack with separate video/audio compression front ends and separate render back ends. The key caveat is that MatrixGame itself is not implemented yet in this workspace; it appears as design direction in ai-notes.md, while implemented training code is currently tokenizer/autoencoder focused.

**1) Data and shared substrate**
- normalize.py builds the multimodal training substrate: palette-index video frames, reduced actions, reduced RAM, and frame-aligned audio in each sample.
- normalized_dataset.py is the common loader that can expose frames, audio, and actions (and metadata like fps/sample-rate), so all model paths share one data contract.

**2) LTX-side model stack**
- Video compressor: ltx_video_vae.py, trained by train_ltx_video_vae.py.
- Audio compressor: ltx_audio_vae.py, trained by train_ltx_audio_vae.py.
- Audio renderer: ltx_audio_vocoder.py, trained by train_ltx_audio_vocoder.py.

How they interact in the intended LTX/MatrixGame route:
- Frames/audio/actions (and optionally RAM) are loaded from normalized samples.
- LTX video/audio VAEs map raw modalities into continuous latents.
- A MatrixGame-like joint world model (from your notes, a unified DiT-style core) would model temporal dynamics in latent space conditioned on actions, and potentially RAM.
- Predicted video latents decode through the video VAE decoder.
- Predicted audio latents decode to mel then to waveform through the vocoder.

Important current-state detail:
- Today, the three LTX trainers are separate. There is no implemented script yet that trains a joint world model over both latents.
- The vocoder is currently trained from mel features extracted from ground-truth waveform, not from audio-VAE latents directly.

**3) What you are replacing from Genie/MagViT**
- Genie/MagViT video tokenizer path was built around the now-removed palette tokenizer stack.
- That path is discrete-token centric (LFQ/codebook behavior), with optional GAN discriminator support in gan_discriminator.py.
- Your own motivation in README.md is exactly the shift: avoid giant discrete vocab prediction pressure in the downstream dynamics model by using continuous KL latents instead.

**4) Full model inventory you currently use/planned**
- Implemented and used now:
1. LTXVideoVAE
2. LTXAudioVAE
3. LTXAudioVocoder
4. PaletteVideoTokenizer (MagViT-derived)
5. Compact 3D GAN discriminator (optional in video vae training)
- Planned/design-direction:
1. MatrixGame-style joint world model (not yet present as code in this repo)
2. Action-conditioned latent dynamics core replacing DynamicsTransformer + LatentDenoiser concept noted in ai-notes.md

I’m pulling the referenced paper and reconnecting it to your current compressor stack so the writeup maps to this repo rather than staying abstract. Next I’ll extract the world-model structure from the paper and compare it to your LTX-style video/audio VAE plus vocoder setup.

The clean world-model design here is a joint latent audiovisual predictor: compress gameplay clips into video latents and audio latents, then train one action-conditioned model to roll those latents forward in time. That is already the direction sketched in ai-notes.md, and it matches the strongest idea in the LTX-2 paper: keep modality-specific compressors, but let a shared backbone learn the joint dynamics. The important caveat is that LTX-2 itself is not a world model. It is a text-conditioned audiovisual generator. To turn that design into a world model, you replace text-first generation with past-state-plus-action conditioning, add causal rollout, and train it to predict the next latent trajectory rather than a prompt-aligned sample.

In this repo, the compressors already line up well for that plan. The video side is a causal 3D palette VAE in ltx_video_vae.py: it patchifies spatially, downsamples space twice, and preserves the time axis. For a 16-frame, 224x224 clip, that gives you a latent video grid that is roughly 16 time steps by 14 by 14 spatial cells. The audio side is a causal 2D mel VAE in ltx_audio_vae.py, with frame audio first stitched into one continuous waveform by audio_features.py and then converted into a mel clip by audio_features.py. The existing test in test_ltx_vaes.py shows a 61 by 64 mel clip compressing to 16 by 16 audio latents, which is exactly the sort of temporal alignment you want for a 16-frame gameplay chunk. That means one clip naturally becomes about 196 video tokens per frame and 16 audio tokens per frame if you flatten the per-time latent slices. The vocoder in ltx_audio_vocoder.py then stays strictly on the decoder side: world model predicts audio latents, audio VAE decodes to mel, vocoder renders waveform.

Architecturally, I would write the world model as an asymmetric dual-stream transformer. The wide stream models video latents because game visuals carry most of the state complexity. The narrow stream models audio latents because NES audio is structurally simpler. Each block does three things: self-attention inside video, self-attention inside audio, and bidirectional audio-video cross-attention. For cross-modal attention, copy the useful LTX-2 idea and align only on time, not on video space. In practice that means video tokens at frame t can attend to audio tokens at audio time t, while the audio side can attend back to all video tokens from that same frame. Conditioning should be actions through AdaLN, not text cross-attention, exactly as noted in ai-notes.md. Since you said you are not conditioning on RAM, the model’s state is simply the latent history itself: recent video latents, recent audio latents, and the action sequence. If you want longer-horizon rollouts, you rely on the transformer context window rather than an explicit RAM branch.

Training should happen in latent space, not pixel space or waveform space. Pretrain the compressors first with the existing VAE and vocoder training paths in train_ltx_video_vae.py and train_ltx_audio_vae.py, then freeze them for the first world-model stage. The actual world-model objective should look more like the LTX-2 flow-matching setup than like the old Genie-style discrete-token pipeline: take a short clean context prefix, corrupt the future latent segment, condition on actions, and train the model to predict the velocity or denoising direction for the future video and audio latents jointly. Causal masking matters here. The model should only see past latents and past actions when predicting the next segment, so it behaves like a simulator rather than a generic audiovisual generator. At inference time the loop is simple: encode initial context, sample the next latent chunk, decode to frames and mel, run the vocoder for audio, append the new latents to context, and repeat.

So the high-level writeup is: this project’s world model should be a MatrixGame-style latent simulator built on top of separate continuous compressors for video and audio, not a discrete MagViT tokenizer plus separate dynamics and denoiser stack. Video VAE gives a compact spatiotemporal state. Audio VAE gives a compact temporally aligned sound state. A dual-stream action-conditioned transformer learns their joint evolution. The audio vocoder is only the final renderer, not part of environment dynamics. That gives you a cleaner system than the Genie route, keeps capacity where it matters most, and uses the fact that your current audio and video compressors already produce nicely aligned 16-step clip latents.

# Transformers

I’m comparing Matrix-Game 3’s bidirectional DiT and memory design against the LTX-style route in this repo, and I’m pulling the specific notes and model files so the comparison stays tied to the actual code and design notes here.

The clean way to think about it is this: Matrix-Game 3 and LTX both use latent-space diffusion transformers, but they solve different problems with that transformer.

In this repo, the LTX side is real code today: the implemented pieces are the video compressor in ltx_video_vae.py, the audio compressor in ltx_audio_vae.py, and the mel-to-waveform decoder in ltx_audio_vocoder.py. The transformer world-model part is still a design in ai-notes.md, not an implemented module under src yet. The tests in test_ltx_vaes.py and test_ltx_vaes.py confirm the current latent contracts.

**Comparison**
- LTX is primarily a latent generator. The notes describe it as an asymmetric dual-stream transformer: a larger video stream plus a smaller audio stream, coupled across modalities, with flow matching and RoPE-style positional structure in ai-notes.md, ai-notes.md, and ai-notes.md. In your planned version, text conditioning gets replaced by action conditioning through AdaLN in ai-notes.md.
- Matrix-Game 3 is a streaming world model. Its transformer is not just denoising a local clip; it is the state-transition engine for an interactive rollout. It takes recent history, current noisy targets, user actions, and retrieved long-range memory, then predicts the current/future latent frames.
- LTX is modality-centric. Its main concern is how to jointly model video and audio latents well. Your notes reflect that in the planned joint token layout in ai-notes.md.
- Matrix-Game 3 is history-centric. Its main concern is how to keep a world stable over long rollouts while remaining interactive and real-time.
- LTX-style transformers are a natural fit for your repo because the compressor stack is already set up around causal video and audio latents. The video VAE uses causal 3D convolutions and spatial patchify in ltx_video_vae.py and ltx_video_vae.py. The audio VAE uses causal 2D convolutions over whole mel sequences in ltx_audio_vae.py. That gives you a clean latent interface for a future DiT.
- Matrix-Game 3 contributes the stronger world-modeling idea: the transformer should not just see the recent clip. It should have a mechanism to recover older state when the agent revisits or reveals parts of the world again.

**Memory**
Matrix-Game 3’s memory is more specific than “give the transformer a longer context window.”

- It keeps an explicit bank of older latent observations.
- It retrieves a small subset of relevant memory frames rather than attending over all past frames.
- Retrieval is camera-aware in the paper: it tries to find earlier observations with viewpoint overlap to the current target.
- The retrieved memory is injected into the same self-attention space as recent history and the current noisy prediction target. That matters. It means memory is not a separate side channel; the model reasons over memory, recent past, and current prediction inside one shared latent workspace.
- The model predicts only the current segment, while memory and recent history act as conditions.
- Matrix-Game 3 also adds error-aware training: it perturbs history and memory with residual errors from prior predictions so the model learns to use imperfect self-generated context, not only clean ground-truth context. That is a big part of why the memory remains useful during long rollouts instead of collapsing from exposure bias.

That memory design is what makes Matrix-Game 3 feel more like a world model than plain LTX. Plain LTX-style generation can be temporally coherent over a short or medium window. Matrix-style memory is trying to preserve scene identity, geometry, and object state over revisitation and minute-long interaction.

For this project, that translates well, but in a simpler form than the paper. Since you do not want RAM conditioning, the memory key cannot be emulator state bytes. It would need to come from visual information: latent similarity, estimated scroll/camera position from frames, or some learned retrieval key derived from recent latents. For Mario specifically, the memory problem is also simpler than a 3D game:
- You mostly care about persistent layout, off-screen objects, backtracking consistency, and room continuity.
- Camera-aware retrieval can likely become scroll-aware or visual-overlap retrieval.
- Video memory matters much more than audio memory. Audio is already processed as a continuous clip waveform and clip-level mel in audio_features.py and audio_features.py, so audio can remain mostly local while video memory handles long-range consistency.

So the hybrid that makes the most sense here is: LTX-style compressors and latent layout, plus a Matrix-style transformer objective. In one sentence: LTX gives you the latent representation and multimodal tokenization; Matrix-Game 3 gives you the long-horizon streaming control logic.

# RAM

Yes. Given that constraint, I would not treat RAM as an external condition that disappears at inference. I would treat it as a third predicted state modality. That is probably one of the more useful additions you could make here, because the repo already identifies important hidden game state that is not fully recoverable from pixels alone in README.md, and the normalizer already keeps reduced RAM and writes it into each training sample in normalize.py and normalize.py.

The main correction to the earlier framing is this: RAM should not be dropped, and it also should not be teacher-forced forever. It should be learned as self-predicted state, then fed back in exactly the way video and audio history are fed back in.

**Recommended Design**
- Use a small RAM encoder that compresses the reduced RAM vector into a few state tokens per frame, not 1500-plus raw byte tokens. The README already points toward a learned RAM embedding in README.md, and the semantic map in smb1_memory_map.py gives you a sensible basis for grouped state tokens.
- Predict that RAM state jointly with video and audio, but do not force RAM to use the same objective as video and audio. I would keep video and audio on the generative latent objective, and make RAM mostly deterministic: next-state prediction, residual prediction, or latent prediction plus a decode loss back to reduced RAM.
- Let video and audio attend to RAM through cross-attention, because RAM contains exactly the kinds of hidden variables that help long-horizon rollout: off-screen enemy state, block-hit flags, timers, progression flags, power-up state, and even high-level audio control registers. That should improve both visual consistency and event timing.
- Keep the RAM stream narrow. A few RAM tokens per frame is enough. If you make RAM a huge stream or let it dominate every block, the model will over-rely on it and small RAM errors will poison everything else.

**Why This Should Help**
- For Mario, RAM is close to the emulator’s true latent state. In world-model terms, it is a supervised belief state. If the model can maintain that state internally, video and audio prediction become easier.
- The video branch benefits because many future observations depend on hidden state that is only partially visible now. The README makes that explicit in README.md and README.md.
- The audio branch benefits too. You already know from smb1_memory_map.py that some high-level sound and music control lives in RAM, even though the actual waveform still has to come from the audio VAE plus vocoder path.

**Important Caveat**
- The biggest risk is train-inference mismatch. If training always gives the model true past RAM, but inference feeds back predicted RAM, the video and audio branches will learn to depend on unrealistically clean state.
- So if you add RAM, train with self-generated RAM context for at least part of training. Scheduled sampling, multi-segment rollout, or a self-forcing setup is the right idea. The same logic Matrix-Game uses for imperfect visual context applies here even more strongly.
- In practice, I would first infer an initial RAM latent from the observed warm-up frames and actions, then let the model roll that state forward autoregressively.

**What I Would Actually Build First**
- First ablation: add a RAM prediction head only, with no feedback into video or audio. That tells you whether the model can predict hidden state at all.
- Second ablation: feed predicted RAM tokens into the next step, but only as light conditioning.
- Third ablation: add explicit bidirectional cross-attention between RAM and video/audio streams every few blocks.

That sequence matters because it separates two questions: whether RAM supervision is useful at all, and whether cross-modal coupling is what gives the gain.

One practical note: the normalized data already contains RAM targets from normalize.py, but the current dataset loader does not expose RAM yet in normalized_dataset.py and normalized_dataset.py. So architecturally this idea is well aligned with the repo, but there is still a loader gap before you can train it.