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

### Dataset Generation

The `dataset_generation.py` script generates synthetic addition datasets of configurable size, number of digits, and representation (decimal or binary).

Example:
```bash
python dataset_generation.py --dataset-size 10000 --mode decimal --num-digits 3