# Advanced Machine Learning and Deep Learning Applications

A collection of six end-to-end machine learning projects covering regression, dimensionality reduction, natural language processing, computer vision, physics-guided modelling, and conditional generation. The repository combines scikit-learn baselines with custom PyTorch models and includes the datasets, saved notebook outputs, and selected visualisations needed to study each workflow.

## Projects

| # | Project | Task and approach | Dataset | Saved result |
|---|---|---|---:|---|
| 01 | [Hospital bed-day prediction](<01_hospital days prediction/linear_regression.ipynb>) | OLS, Ridge, Lasso, and ElasticNet regression with preprocessing and cross-validated model selection | 2,400 records | Lasso selected with CV MAE **1.2019 days**; test MAE **1.2185 days** and R² **0.6762** |
| 02 | [Decision tree and PCA](<02_decision Tree and PCA/decision&PCA.ipynb>) | Decision-tree classification on raw and PCA-transformed features | 2,400 records, 10 features | Best raw-feature accuracy **84.83%** at depth 3 |
| 03 | [Word embeddings and semantic similarity](<03_word Embeddings and Semantic Similarity/word-embeddings-nlp.ipynb>) | GloVe embeddings and cosine-similarity search over the *Pride and Prejudice* vocabulary | 128,577 downloaded tokens | Retrieves three corpus-aware neighbours for five query words |
| 04 | [Orbital decay prediction](<04_orbital Decay Neural Network Regression/physics-informed-orbital-decay-prediction.ipynb>) | Physics-guided feature engineering and a PyTorch multilayer perceptron | 2,000 simulations | Best validation MAE **29.1 days** |
| 05 | [Dice product classification](<05_dice-product-classification/dice-product-classification-cnn.ipynb>) | Residual CNN with an auxiliary visible-face prediction task | 10,000 images | Best validation accuracy **99.19%** at epoch 74 |
| 06 | [Conditional font generation](conditional-font-generation-cvae.ipynb) | Conditional variational autoencoder for character-outline generation and latent interpolation | 8,557 outlines | Generates conditioned samples for `c`, `n`, `s`, `v`, and `z` |

The metrics above are taken from the outputs currently saved in the notebooks. Results may vary when cells are rerun because several models use stochastic training.

## Repository structure

```text
.
├── 01_hospital days prediction/
│   ├── datasets/
│   ├── screenshots/
│   └── linear_regression.ipynb
├── 02_decision Tree and PCA/
│   ├── datasets/
│   ├── screenshots/
│   └── decision&PCA.ipynb
├── 03_word Embeddings and Semantic Similarity/
│   └── word-embeddings-nlp.ipynb
├── 04_orbital Decay Neural Network Regression/
│   ├── datasets/
│   ├── physics-informed-orbital-decay-prediction.ipynb
│   └── weights.pkl
├── 05_dice-product-classification/
│   ├── dataset/
│   ├── screenshots/
│   └── dice-product-classification-cnn.ipynb
├── 06_conditional_font generation/
│   ├── datasets/
│   └── screenshots/
└── conditional-font-generation-cvae.ipynb
```

## Technical highlights

### 1. Hospital bed-day prediction

The notebook builds reproducible scikit-learn pipelines for numerical scaling and one-hot encoding. Four linear models are compared with five-fold cross-validation using mean absolute error, then evaluated on a held-out 20% test set. It also examines coefficients, residuals, feature ablation, and a simulated CRP-data outage.

![Hospital bed-day analysis](<01_hospital days prediction/screenshots/hospital bed day.png>)

### 2. Decision tree classification with PCA

This experiment varies tree depth from 1 to 20, then repeats the search after standardisation and PCA. It compares predictive accuracy with the explained-variance trade-off across different numbers of principal components. In the saved run, PCA does not improve on the best raw-feature tree.

![Accuracy against tree depth](<02_decision Tree and PCA/screenshots/accuracy vs max depth.png>)

### 3. Semantic similarity with GloVe

The NLP notebook downloads the Project Gutenberg text of *Pride and Prejudice* and loads the 100-dimensional `glove-wiki-gigaword-100` model through Gensim. After tokenisation, nearest neighbours are selected by cosine similarity and restricted to words that occur in the book.

This notebook requires internet access on its first run. The GloVe model is also downloaded to the local Gensim cache.

### 4. Physics-guided orbital decay regression

The orbital model transforms physical inputs into log-space features and adds the ballistic coefficient, `mass / (drag coefficient × area)`. A fully connected PyTorch network is trained with Huber loss, AdamW, cosine learning-rate annealing, and early stopping. The best state dictionary is stored in `weights.pkl`.

### 5. Dice product classification

The computer-vision pipeline applies image augmentation, a stratified 85/15 split, class weighting, and a residual CNN. Alongside the product class, an auxiliary head predicts the three visible die-face values to encourage the network to learn task-relevant features.

![Dice product predictions](<05_dice-product-classification/screenshots/dice_product.png>)

### 6. Conditional font generation

The CVAE resamples each closed outline to 160 points and conditions both the encoder and decoder on the character label. Its 32-dimensional latent space is trained with geometric augmentation and beta annealing. The final cells visualise conditional samples and interpolations between learned outlines.

| Training loss | Conditional samples | Latent interpolation |
|---|---|---|
| ![CVAE training loss](<06_conditional_font generation/screenshots/01_training loss.png>) | ![Conditional font samples](<06_conditional_font generation/screenshots/02_conditional samples(CVAE).png>) | ![CVAE latent space](<06_conditional_font generation/screenshots/03_latent space.png>) |

## Getting started

### Prerequisites

- Python 3.10 or later
- JupyterLab or Jupyter Notebook
- Internet access for initial package installation and project 03 downloads
- Optional CUDA or Apple Silicon acceleration for the PyTorch projects

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Install the libraries used across the notebooks:

```bash
python -m pip install \
  jupyterlab numpy pandas matplotlib seaborn \
  scikit-learn scipy gensim pillow torch torchvision
```

For hardware-specific PyTorch installation instructions, use the command recommended for your operating system and accelerator by the official PyTorch installer.

## Running the notebooks

Most notebooks use paths relative to their own project directory. Start Jupyter from that directory, for example:

```bash
cd "01_hospital days prediction"
jupyter lab linear_regression.ipynb
```

Use the same pattern for projects 02, 03, and 04. Run the cells from top to bottom so preprocessing objects, data loaders, and models are created in the required order.

### Dice notebook path correction

The dice notebook currently declares:

```python
DATA_DIR = Path("dice_images")
```

The checked-in directory is named `dataset`. Open Jupyter inside `05_dice-product-classification/` and change that line to:

```python
DATA_DIR = Path("dataset")
```

Then run `dice-product-classification-cnn.ipynb`.

### Font notebook path correction

The font notebook is stored at the repository root and currently declares `DATA_ROOT = "font_outlines"`. Change it to the checked-in dataset path before running:

```python
DATA_ROOT = "06_conditional_font generation/datasets"
```

Then launch it from the repository root:

```bash
jupyter lab conditional-font-generation-cvae.ipynb
```

## Reproducibility notes

- The tabular projects use fixed train/test or train/validation seeds.
- The dice notebook seeds Python, NumPy, and PyTorch, but accelerator kernels can still introduce small differences.
- The font CVAE uses random augmentation and does not set a global seed, so generated samples and loss curves will vary.
- Training the CNN and CVAE from scratch is substantially slower than running the classical machine-learning notebooks.
- Saved notebook outputs are included for inspection without retraining.

## Data and model artefacts

All tabular datasets, dice images, and font outlines used by the notebooks are included in the repository. The orbital-decay directory also contains the best saved PyTorch weights from the recorded training run. Dataset usage and redistribution remain subject to the terms of their original sources.
