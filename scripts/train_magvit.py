import argparse
import glob
import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import sys
from pathlib import Path

# Add src to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from torchvision.utils import save_image
from torch.optim import AdamW
from magvit2_pytorch import VideoTokenizer
from tqdm import tqdm

from mario_world_model.config import IMAGE_SIZE, CODEBOOK_SIZE, TOKENIZER_LAYERS, SEQUENCE_LENGTH

class MarioVideoDataset(Dataset):
    def __init__(self, data_dir, seq_len=4):
        self.chunk_files = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
        self.seq_len = seq_len
        self.samples = []
        self.all_data = [] # Store all data in RAM
        
        print(f"Loading {len(self.chunk_files)} chunks into memory...")
        for chunk_idx, f in enumerate(self.chunk_files):
            try:
                npz = np.load(f)
                frames = npz['frames'] # [num_seqs, T, C, H, W]
                dones = npz['dones'] # [num_seqs, T] boolean array
                self.all_data.append(frames)
                num_seqs = frames.shape[0]
                total_t = frames.shape[1]
                
                # If total_t >= seq_len, we can sample sub-sequences
                if total_t >= seq_len:
                    for i in range(num_seqs):
                        for t in range(0, total_t - seq_len + 1, seq_len):
                            # Check if a boundary break occurs inside this sequence
                            # A 'done=True' means the NEXT frame will be a scene reset.
                            # So it's safe if 'done=True' is exactly on the VERY LAST frame
                            # of our window (index -1), but NOT anywhere before that.
                            window_dones = dones[i, t:t+seq_len]
                            if np.any(window_dones[:-1]):
                                continue # Skip sequences that contain an internal boundary break
                            self.samples.append((chunk_idx, i, t))
            except Exception as e:
                print(f"Skipping {f} due to error: {e}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        chunk_idx, seq_idx, t_start = self.samples[idx]
        
        # Access from pre-loaded memory
        frames = self.all_data[chunk_idx][seq_idx, t_start:t_start+self.seq_len]
        
        # Convert to tensor: [T, C, H, W] -> [C, T, H, W] required by VideoTokenizer
        frames = torch.from_numpy(frames).float() / 255.0
        frames = frames.permute(1, 0, 2, 3) 
        
        return frames

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, required=True, help="Path to chunk files")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--output-dir", type=str, default="checkpoints/magvit2")
    parser.add_argument("--val-interval", type=int, default=50)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    dataset = MarioVideoDataset(args.data_dir, seq_len=SEQUENCE_LENGTH)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    print(f"Found {len(dataset)} sequence segments of length {SEQUENCE_LENGTH} frames.")

    tokenizer = VideoTokenizer(
        image_size=IMAGE_SIZE,
        init_dim=32,
        codebook_size=CODEBOOK_SIZE,
        layers=TOKENIZER_LAYERS,
        use_gan=False,
        perceptual_loss_weight=0.0
    ).to(device)

    optimizer = AdamW(tokenizer.parameters(), lr=args.lr)

    global_step = 0
    for epoch in range(args.epochs):
        tokenizer.train()
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for batch in pbar:
            batch = batch.to(device)
            
            optimizer.zero_grad()
            loss, loss_breakdown = tokenizer(batch, return_loss=True)
            loss.backward()
            optimizer.step()
            
            pbar.set_postfix({'loss': f"{loss.item():.4f}", 'recon': f"{loss_breakdown.recon_loss.item():.4f}"})
            
            if global_step % args.val_interval == 0:
                tokenizer.eval()
                with torch.no_grad():
                    # Generate a reconstruction sample
                    quantized = tokenizer.encode(batch)
                    recon_video = tokenizer.decode(quantized)
                    
                    # Take first sequence from batch, slice along time
                    # batch[0] shape: [C, T, H, W]
                    original_frames = batch[0].permute(1, 0, 2, 3) # [T, C, H, W]
                    recon_frames = recon_video[0].permute(1, 0, 2, 3).clamp(0, 1) # [T, C, H, W]
                    
                    # Cat side by side
                    comparison = torch.cat([original_frames, recon_frames], dim=3) # [T, C, H, W*2]
                    save_image(comparison, os.path.join(args.output_dir, f"step_{global_step:06d}.png"), nrow=1)
                tokenizer.train()
            
            global_step += 1
            
        torch.save(tokenizer.state_dict(), os.path.join(args.output_dir, "magvit2_latest.pt"))

if __name__ == "__main__":
    train()