from __future__ import annotations

# Tokenizer / Model configuration
IMAGE_SIZE = 128
CODEBOOK_SIZE = 256
SEQUENCE_LENGTH = 16
TOKENIZER_LAYERS = (
    'residual', 
    'compress_space',
    'residual',
    'compress_space',
    'residual',
    'compress_space',
)
