# DNA Methylation Binary Classifier

This script fine-tunes the Peltarion/dnabert-minilm-small model to classify DNA sequences as methylated or unmethylated, then evaluates performance across CG, CHG, and CHH contexts.

## Overview

The script performs the following steps:

1. **Setup**: Detects and uses the appropriate device (MPS for Apple Silicon, CUDA for NVIDIA, CPU otherwise)
2. **Genome Download**: Automatically downloads the Arabidopsis thaliana TAIR10 Chromosome 1 reference genome
3. **Data Processing**: Loads methylation data, filters for quality, creates binary methylation labels, and extracts DNA sequences
4. **Model Training**: Fine-tunes the DNABERT model for binary classification (methylated vs unmethylated)
5. **Evaluation**: Evaluates performance on the overall test set and separately for each context (CG, CHG, CHH)

## Requirements

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Data Format

The script expects a tab-separated file (`1061_10C_MR.txt`) with the following columns:
- `chr`: Chromosome
- `pos`: Position of cytosine
- `context`: 5-base-pair sequence surrounding the cytosine
- `ratio`: Methylation ratio (0-1)
- `eff_CT_count`: Effective CT count for quality filtering

## Methylation Classification

The script creates binary methylation labels based on the ratio column:
- **Methylated (1)**: ratio > 0.6
- **Unmethylated (0)**: ratio < 0.2
- **Ambiguous**: ratio between 0.2-0.6 (discarded)

## Usage

Simply run the script:

```bash
python dna_methylation_classifier.py
```

## Output

The script will:
- Download the Arabidopsis thaliana genome (if not present)
- Process the methylation data and create binary labels
- Balance the dataset to avoid class bias
- Train the model for 3 epochs
- Print overall evaluation metrics
- Print context-specific evaluation metrics (CG, CHG, CHH)
- Save the trained model to `./final_model`

## Model Details

- **Base Model**: Peltarion/dnabert-minilm-small
- **Sequence Window**: 201 base pairs
- **Quality Filter**: Minimum effective CT count of 5
- **Classification**: Binary (methylated vs unmethylated)
- **Training**: 3 epochs with early stopping based on F1-score
- **Dataset**: Balanced to avoid class bias

## Files Generated

- `Chr1.fa`: Arabidopsis thaliana Chromosome 1 reference genome
- `./results/`: Training checkpoints and logs
- `./final_model/`: Final trained model
- `./logs/`: Training logs 