# Contrastive Explanations for Model Interpretability

### This is the repository for the paper "Contrastive Explanations for Model Interpretability", about explaining neural model decisions *against alternative decisions*.

#### Authors: Alon Jacovi, Swabha Swayamdipta, Shauli Ravfogel, Yanai Elazar, Yoav Goldberg, Yejin Choi

### Getting Started

#### Setup
```bash
conda create -n contrastive python=3.8
conda activate contrastive
pip install allennlp==1.2.0rc1
pip install allennlp-models==1.2.0rc1.dev20201014
pip install jupyterlab
bash scripts/download_data.sh
```

#### Contrastive projection

If you're here just to know how we implemented contrastive projection, here it is:
```python
u = classifier_w[fact_idx] - classifier_w[foil_idx]
contrastive_projection = np.outer(u, u) / np.dot(u, u)
```
Very simple :)

`contrastive_projection` is a projection matrix that projects the model's latent representation of some example `h` into the direction of `h` that separates the logits of the fact and foil.

#### Training MNLI/BIOS models
```bash
bash scripts/train_sequence_classification.sh 
```

#### Highlight ranking
Run the `notebooks/mnli-highlight-featurerank.ipynb` or `notebooks/bios-highlight-featurerank.ipynb` jupyter notebooks.

These notebooks load the respective models, and then run the highlight ranking procedure.

#### Foil ranking (WIP)

