Would be really interested to see a no-image-encoder model that uses attention over a time sequence of nes ram dumps to produce the tokens that the transformer predicts and decoder generates images from.....


Skimmed the magvit2 paper, notes: Increase warmup period, push the temporal downsampling layer deeper, codebook size of 2^18 should be enough which would be two 512 sized codebooks


Instead of all combinations of architecture variables maybe just define an order to test them as additions to some base model

Half res / 2d init


Write down some findings on the bottleneck bitwidth calculations


Image inflation with 2d model trained on sprite sheets? is that too much engineering? yes.


Find nicer abstractions between the training scripts, de-duplicate code/fixes/optimizations/...!


Deeper smaller models?


Adding the gan actually made video vae training more stable.


The latent space is "smoother" than pixels? interesting.


Could actually do data aug even with NES palette pixels (set random input pixels to random palette indices). I think I'd want to be careful with perturbing to rare colors since then the network would maybe just learn to ignore them, i.e. the overwhelming majority of appearances of the rarest colors would be cases the optimzier teaches the network to ignore.... So, instead I'll sample from the index distribution!


Pretty neat idea for vq-vae/lfq to overcome approximation error of the straight-through estimator x + (f(x) - x).detach(): https://github.com/cfifty/rotation_trick


I've lost my vae checkpoint. Luckily I wanted to retrain anyway since I've had some ideas for how to improve the latent representation. Also I'm going to reimplement to use accelerator and possibly some implementation library for the vae components.


For the sake of computational cost I might have the DiT target a tick rate of 15hz. Meaning the DiT would predict chunks of 4 frames per tick since mario runs at 60hz.