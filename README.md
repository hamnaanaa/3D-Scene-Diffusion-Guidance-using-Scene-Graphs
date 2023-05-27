# Guided 3D Scene Synthesis using Denoising Diffusion Probabilistic Models
This repository implements the paper **Guided 3D Scene Synthesis using Denoising Diffusion Probabilistic Models** (tbd link)

## Setup
The easiest way to set up all dependencies is to use conda:
```
conda env create -f environment.yml
```

### FastText Embeddings by Facebook Research
We use the [FastText](https://fasttext.cc/) embeddings by Facebook Research to embed the scene objects descriptions in a more robust way than a standard Word2Vec encoder. You can download the model binary [here](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.en.300.bin.gz) and place them in the `models` folder.