# Privacy Attack Experiments

## Research Question

How does increasing Gaussian noise affect the measured ROC-AUC of a loss-based membership inference attack (MIA) on GPT-2?

## 1. DBPedia-14: Baseline MIA

- Dataset: DBPedia-14
- Model: Fine-tuned GPT-2
- Members: 100
- Non-members: 100
- Attack: Loss-based membership inference
- Baseline ROC-AUC using negative loss: 0.5841

## 2. DBPedia-14: DP-SGD Comparison

A separate experiment fine-tuned GPT-2 using Opacus DP-SGD.

- Training examples: 100
- Epochs: 1
- Batch size: 4
- Maximum gradient norm: 1.0
- Target epsilon: 8
- Reported achieved epsilon: approximately 7.9955
- Delta: 1e-5
- Reported MIA ROC-AUC: 0.4491

The score direction must be checked before comparing this AUC with the baseline. An AUC below 0.5 may indicate reversed ranking rather than stronger privacy.

## 3. DBPedia-14: Gaussian Noise Comparison

Each condition contains 100 member and 100 non-member records.

| Condition | Mean member loss | Mean non-member loss | ROC-AUC using negative loss |
|---|---:|---:|---:|
| Baseline, no added noise | 3.4648 | 3.6419 | 0.5841 |
| Noise 0.1 | 3.4677 | 3.6467 | 0.5849 |
| Noise 0.5 | 3.5644 | 3.7416 | 0.5818 |
| Noise 1.0 | 3.8488 | 4.0155 | 0.5735 |
| Noise 2.0 | 4.9810 | 5.1382 | 0.5534 |

The measured ROC-AUC decreased from 0.5841 to 0.5534 as the noise level increased from 0 to 2.0.

### Important limitation

In this experiment, Gaussian noise was added to logits during attack-score calculation. This does not, by itself, demonstrate the privacy effect of a deployed inference-time noise mechanism or establish a formal differential privacy guarantee.

The results are preliminary and should be repeated across random seeds with a clearly specified attack and noise mechanism.

## 4. AGNews: Noisy Synthetic Prompt Experiment

Four synthetic examples were generated per run using noisy token probabilities.

Reported accuracy:
- Baseline: 51% original, 52% calibrated
- Noisy run 1: 44% original, 34% calibrated
- Noisy run 2: 48% original, 45% calibrated

These small, stochastic runs are preliminary and are not sufficient to establish a general accuracy trend.

A basic output-matching audit found zero exact/containment matches among the tested member and non-member examples. This is a limited leakage check, not proof that data extraction or membership inference is impossible.

## 5. Main Conclusions

1. The baseline loss-based MIA on DBPedia-14 achieved ROC-AUC 0.5841 using negative loss.
2. The measured AUC decreased as the experimental noise level increased, reaching 0.5534 at noise 2.0.
3. This is an observation about the current experimental scoring procedure, not proof of inference-time privacy improvement.
4. The DP-SGD experiment is separate from the Gaussian-noise scoring experiment.
5. Further validation should include multiple seeds, confidence intervals, a clearly defined inference mechanism, and consistent attack-score direction.

## Reproducibility

The CSV files and JSON outputs in this repository preserve the current experimental results. Check the corresponding scripts for implementation details and command-line requirements before reproducing any experiment.
