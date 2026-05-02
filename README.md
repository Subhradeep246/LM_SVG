# SVG Scaling Laws: Transformer Language Models on SVG Code

## Overview

This project explores scaling laws for decoder-only Transformer language models trained on SVG (Scalable Vector Graphics) code. We train models at 5 different scales (1M to 88M parameters), fit power-law scaling curves, investigate muP (Maximal Update Parameterization) for learning rate transfer, and generate/evaluate SVG samples.

## Project Structure

```
svg-scaling-laws/
├── configs/                    # Model & training configurations
│   ├── model_configs.yaml      # 5 model architectures (Tiny → XL)
│   └── training_config.yaml    # Training hyperparameters
├── data/                       # Data pipeline
│   ├── download_data.py        # Download from HuggingFace
│   ├── preprocess.py           # Clean, normalize, validate SVGs
│   ├── train_tokenizer.py      # Train BPE tokenizer (4K vocab)
│   ├── prepare_dataset.py      # Tokenize, split, chunk to binary
│   └── dataset_stats.py        # Compute statistics & visualizations
├── model/                      # Model architectures
│   ├── transformer.py          # GPT model (standard parameterization)
│   ├── transformer_mup.py      # GPT model (muP parameterization)
│   └── utils.py                # Param counting, memory estimation
├── training/                   # Training scripts
│   ├── train.py                # Main training loop (SP)
│   ├── lr_sweep.py             # Learning rate sweep (SP)
│   ├── train_mup.py            # Training loop (muP)
│   └── lr_sweep_mup.py         # Learning rate sweep (muP)
├── evaluation/                 # Evaluation tools
│   ├── validate_svg.py         # XML/SVG validity checks
│   ├── render_svg.py           # Render SVGs to PNG (CairoSVG)
│   └── evaluate.py             # Perplexity, validity metrics
├── generation/                 # SVG generation
│   └── generate.py             # Unconditional + prefix-conditioned
├── analysis/                   # Analysis & plotting
│   ├── scaling_plots.py        # Power law fitting, scaling curves
│   ├── compare_sp_mup.py       # SP vs muP comparison
│   └── extrapolation.py        # Scaling law extrapolation
├── notebooks/                  # Colab notebooks (main interface)
│   ├── 01_data_preprocessing.ipynb
│   ├── 02_scaling_study.ipynb
│   ├── 03_mup_scaling.ipynb
│   ├── 04_generation.ipynb
│   └── 05_analysis.ipynb
├── scratch/                    # Temporary sandbox for experimental scripts
├── requirements.txt
└── README.md
```
### Data produced - https://drive.google.com/drive/folders/1IEiueGU1vSwlwU1V-W9huypLGfRsd8Bb?usp=sharing 
## Setup & Usage

### Prerequisites
- Google Colab with GPU runtime (T4 is sufficient for Tiny/Small/Medium; A100 highly recommended for Large/XL memory requirements)
- Google Drive with sufficient storage (~5GB for data + checkpoints)
- Weights & Biases account (for experiment tracking)

### Quick Start

1. **Upload this folder** to Google Drive at `/MyDrive/svg-scaling-laws/`

2. **Open notebooks in order** in Google Colab:
   - Set runtime to **GPU (T4)**
   - Run cells sequentially

3. **Notebook workflow**:
   - `01_data_preprocessing.ipynb` — Download, clean, tokenize SVG data
   - `02_scaling_study.ipynb` — LR sweep + train 5 model sizes
   - `03_mup_scaling.ipynb` — muP experiments + extrapolation
   - `04_generation.ipynb` — Generate and evaluate SVG samples
   - `05_analysis.ipynb` — Compile all figures and analysis

### Manual Setup (non-Colab)

```bash
pip install -r requirements.txt

# 1. Download data
python data/download_data.py --output_dir ./data/raw

# 2. Preprocess
python data/preprocess.py --input_dir ./data/raw --output_path ./data/cleaned.jsonl

# 3. Train tokenizer
python data/train_tokenizer.py --data_path ./data/cleaned.jsonl --output_dir ./tokenizer

# 4. Prepare dataset
python data/prepare_dataset.py --cleaned_path ./data/cleaned.jsonl --tokenizer_dir ./tokenizer

# 5. LR sweeps (SP and µP)
python training/lr_sweep.py --model_name tiny
python training/lr_sweep_mup.py --model_name tiny

# 6. Train all models (SP and µP)
python training/train.py --models tiny small medium large xl
python training/train_mup.py --models tiny small medium large xl

# 7. Generate SVGs using the best model
python generation/generate.py --model_path checkpoints/xl_mup.pt --num_unconditional 10 --num_prefix 5

# 8. Evaluate generated outputs (Validity, Render Rate, Perplexity)
python evaluation/evaluate.py --samples_dir outputs/ --test_data ./data/cleaned_test.jsonl

# 9. Plot scaling laws and comparison
python analysis/compare_sp_mup.py --results_dir results/
```

## Model Architectures

| Name | ~Params | d_model | n_layers | n_heads | d_ff |
|------|---------|---------|----------|---------|------|
| Tiny | ~1M | 128 | 4 | 4 | 512 |
| Small | ~3M | 192 | 6 | 6 | 768 |
| Medium | ~10M | 384 | 6 | 6 | 1536 |
| Large | ~30M | 512 | 10 | 8 | 2048 |
| XL | ~88M | 768 | 12 | 12 | 3072 |

## Datasets

- **Primary**: `starvector/svg-icons-simple` (~89K simplified SVG icons)
- **Supplementary**: `starvector/svg-emoji-simple`, `starvector/svg-fonts-simple`
- **Tokenization**: BPE with 4096 vocabulary
- **Final Stats**: 133,891 cleaned SVGs yielding **123.3M training tokens** (filtered `max_token_length=4096`)

## Key Design Decisions

1. **Tokenizer**: BPE with 4K vocab — balances SVG tag vocabulary with numeric diversity
2. **Architecture**: Pre-LayerNorm GPT with GELU, Flash Attention when available
3. **Training**: AdamW, cosine LR schedule, FP16 mixed precision, gradient accumulation
4. **muP**: 1/d attention scaling, MuReadout output, MuAdamW optimizer
5. **Evaluation**: Perplexity + XML validity + SVG render rate + structural validity

## Key Results

1. **Standard Parameterization (SP)**: Achieved a clean power-law scaling curve with an exponent of **$\alpha = 0.2206$**. Finding a stable learning rate across widths required a conservative compromise (`1e-3`).
2. **Maximal Update Parameterization (µP)**: Successfully transferred the optimal learning rate (`1e-2`) zero-shot across all widths without divergence, yielding a steeper scaling exponent of **$\alpha = 0.2360$** (+6.9% efficiency).
3. **Generation**: The 88M parameter XL model achieved a test set perplexity of **2.03** and a **23.5% zero-shot valid render success rate**, demonstrating an emergent understanding of geometric continuity and SVG syntax.

## References

- Kaplan et al. (2020): "Scaling Laws for Neural Language Models" — [arXiv:2001.08361](https://arxiv.org/abs/2001.08361)
- Hoffmann et al. (2022): "Training Compute-Optimal Large Language Models" — [arXiv:2203.15556](https://arxiv.org/abs/2203.15556)
- Yang et al. (2022): "Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer" — [arXiv:2203.09789](https://arxiv.org/abs/2203.09789)
- Rodriguez et al. (2023): "StarVector" — [arXiv:2312.11556](https://arxiv.org/abs/2312.11556)

## Code Attribution

The transformer architecture is based on [nanoGPT](https://github.com/karpathy/nanoGPT) by Andrej Karpathy, with modifications for SVG-specific training, muP integration, and evaluation. muP implementation uses the [mup](https://github.com/microsoft/mup) package by Microsoft Research.
