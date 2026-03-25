# Structure-Aware Test-Time Training

This repository contains the code and experiments for a research project on **Test-Time Training (TTT)** with a focus on the role of **inner loss functions** under out-of-distribution (OOD) settings.

## Project Overview

Test-Time Training adapts a model during inference using self-supervised objectives derived from the test input. While prior work has mainly explored heuristic inner losses in perceptual domains, this project studies **structure-aware and heuristic inner losses** on textual data representing numbers and arithmetic operations.

The goal is to analyze how the choice and placement of the inner loss (joint training vs. test-time adaptation) affects OOD generalization.

## Scope

- Autoregressive models operating on textual representations of arithmetic expressions  
- Controlled OOD setting via extrapolation to longer numeric sequences  
- Comparison of heuristic and structure-aware inner loss functions  
- Evaluation across different Test-Time Training regimes

## Training Scripts

The `scripts/` directory contains two main scripts for training models.

---

### 1. `train_seq2seq_model.py`

This script trains a Seq2Seq model (including TTT variants) and automatically evaluates it on test data.

#### Usage

```bash
python scripts/train_seq2seq_model.py \
  --representation binary \
  --num_digits 3 \
  --epochs 400
```

#### Key Arguments

* `--representation` — data format: `binary` or `decimal`
* `--num_digits` — number of digits in training samples
* `--dataset_size` — size of the training dataset
* `--valid_samples` — size of the validation set
* `--batch_size` — batch size
* `--emb_dim` — embedding dimension
* `--hid_dim` — hidden dimension
* `--ttt_bottleneck_dim` — bottleneck size for TTT
* `--epochs` — number of training epochs
* `--learning_rate` — learning rate
* `--patience` — early stopping patience
* `--print_every` — logging frequency
* `--test_num_digits` — number of digits for evaluation

#### Notes

After training, the script automatically runs evaluation:

* on the base model
* on the model with TTT
* on the model with online TTT

---

### 2. `train_sequence_model.py`

This script trains baseline sequence models such as RNN and GRU.

#### Usage

```bash
python scripts/train_sequence_model.py \
  --dataset_size 20000 \
  --mode binary \
  --num_digits 3 \
  --model GRU
```

#### Key Arguments

* `--dataset_size` — dataset size
* `--mode` — data representation: `binary` or `decimal`
* `--num_digits` — number of digits
* `--model` — model type: `GRU` or `RNN`
* `--commutative_loss` — whether to use commutative loss (`True` / `False`)

#### What the Script Does

1. Generates a dataset
2. Splits it into train/test sets
3. Initializes the selected model (RNN or GRU)
4. Trains the model
