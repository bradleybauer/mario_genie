Put whole dataset on B200 gpu? Doesn't really work as good as I thought


Super Mario Land 2 from the gameboy 1 would be a better next game than smb3


Would be really interested to see a no-image-encoder model that uses attention over a time sequence of nes ram dumps to produce the tokens that the transformer predits and decoder generates images from.....


Remember to write down some thoughts on the bottleneck bitwidth calculations


Skimmed the magvit2 paper, notes: Increase warmup period, push the temporal downsampling layer deeper, codebook size of 2^18 should be enough which would be two 512 sized codebooks


Instead of all combinations of architecture variables maybe just define an order to test them as additions to some base model

The frames will be reduced to a minimal set of palette indices and the ram/wram will be reduced to a minimal set of ram addresses. The idea is I will need a post step after collection that considers the whole dataset and that reduce redundancies in the data. And similarly for the actions.