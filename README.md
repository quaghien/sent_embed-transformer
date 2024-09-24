# Sentence embeddings into Transformer

Incorporating sentence embeddings into Transformer architecture for Neural Machine Translation built on the idea of ​​[this paper](https://ietresearch.onlinelibrary.wiley.com/doi/full/10.1049/ell2.12840).

## Architecture

<img src="https://github.com/QuagHien/se-transformer-mt5/blob/main/images/architect.png" alt="architect" width="600" height="550" />

The illustration of proposed embed-fusion module. Sentence embedding represents the universal sentence embedding from SimCSE model. e1 ,e2 ,…, eL represent word feature vectors of source input from Transformer encoder. Concatmeans concatenation operation, and Dense means fully-connected network.

<img src="https://github.com/QuagHien/se-transformer-mt5/blob/main/images/attention.png" alt="attention" width="346" height="416" />
