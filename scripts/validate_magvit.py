import argparse
import os
import torch
import numpy as np
from torchvision.utils import save_image
from magvit2_pytorch import VideoTokenizer

from mario_world_model.config import IMAGE_SIZE, CODEBOOK_SIZE, TOKENIZER_LAYERS

def validate():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-file", type=str, required=True, help="Path to a single .npz chunk for held-out validation")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to magvit2_latest.pt")
    parser.add_argument("--seq-idx", type=int, default=0, help="Index of the sequence in the chunk to validate")
    parser.add_argument("--output", type=str, default="validation_output.png")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    tokenizer = VideoTokenizer(
        image_size=IMAGE_SIZE,
        init_dim=32,
        codebook_size=CODEBOOK_SIZE,
        layers=TOKENIZER_LAYERS,
        use_gan=False,
        perceptual_loss_weight=0.0
    ).to(device)

    tokenizer.load_state_dict(torch.load(args.checkpoint, map_location=device))
    tokenizer.eval()

    # Load data
    npz = np.load(args.data_file)
    frames = npz['frames'][args.seq_idx] # [T, C, H, W]
    
    # [T, C, H, W] -> [1, C, T, H, W]
    video = torch.from_numpy(frames).float() / 255.0
    video = video.permute(1, 0, 2, 3).unsqueeze(0).to(device) # [Batch(1), C, T, H, W]

    with torch.no_grad():
        quantized = tokenizer.encode(video)
        recon_video = tokenizer.decode(quantized)
        
    original = video[0].permute(1, 0, 2, 3) # [T, C, H, W]
    reconstructed = recon_video[0].permute(1, 0, 2, 3).clamp(0, 1) # [T, C, H, W]
    
    # Side-by-side [Original | Reconstructed]
    comparison = torch.cat([original, reconstructed], dim=3)
    
    save_image(comparison, args.output, nrow=1)
    print(f"Validation saved to {args.output}.")
    print(f"Quantized token shape: {quantized.shape}")

if __name__ == "__main__":
    validate()