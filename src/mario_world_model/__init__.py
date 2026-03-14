"""mario_world_model — toolkit for an action-conditioned NES world model.

Provides the full pipeline from NES emulation to video-tokenizer training:

* **envs** — Gym wrappers around gym-super-mario-bros with shimmy compatibility.
* **actions** — Discrete action space mapping NES button combinations.
* **preprocess / palette_mapper** — Frame padding, resizing, and RGB→palette-index
  conversion with a dynamically growing NES colour palette.
* **rollouts / coverage / storage** — Episode tracking, replay indexing, and
  balanced data-collection utilities for progression and action coverage.
* **palette_tokenizer / model_configs** — MAGVIT-2 video tokenizer variants using
  cross-entropy loss over palette indices.
* **config** — Shared constants (image size, sequence length).
"""
