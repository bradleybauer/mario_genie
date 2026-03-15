Ai suggested that a diffusion decoder could work really well since the dataset is "low entropy".
All of the architectures perform similarly on the training time scales I've tested.
Diffusion decoder in combination with NES ram embedding may be the only way to get a fundamental shift in training performance?
Preliminary training runs suggest training to convergence on this dataset will take at least 2 days on an h100
