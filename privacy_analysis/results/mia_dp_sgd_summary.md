@'
# GPT-2 Privacy Experiment: Baseline vs DP-SGD

## 1. Objective
Evaluate membership inference risk for a normally fine-tuned GPT-2 model and compare it with a GPT-2 model trained using Differentially Private Stochastic Gradient Descent (DP-SGD).

## 2. Baseline GPT-2 Training
- Dataset: DBPedia-14
- Training examples: 100
- Epochs: 1
- Seed: 42
- Method: Normal fine-tuning
- Model path: outputs/gpt2_mia/model

## 3. Baseline Membership Inference Attack (MIA)
- Attack: Loss-based Membership Inference Attack
- Members: 100
- Non-members: 100
- ROC-AUC: 0.5841
- Random-guess reference: 0.5000

## 4. DP-SGD Implementation
The GPT-2 model was fine-tuned using Opacus with DP-SGD.

- Dataset: DBPedia-14
- Training examples: 100
- Epochs: 1
- Batch size: 4
- Gradient clipping norm: 1.0
- Target epsilon: 8.0
- Achieved epsilon: 7.9955
- Delta: 1e-5
- Noise multiplier: 0.5421
- Model path: outputs/gpt2_mia_dp/model

## 5. DP Model Membership Inference Attack
- Attack: Loss-based Membership Inference Attack
- Members: 100
- Non-members: 100
- Member mean loss: 3.9602
- Non-member mean loss: 3.8469
- ROC-AUC: 0.4491
- Random-guess reference: 0.5000

## 6. Baseline vs DP-SGD Comparison

| Metric | Baseline GPT-2 | DP-SGD GPT-2 |
|---|---:|---:|
| Training examples | 100 | 100 |
| Epochs | 1 | 1 |
| MIA ROC-AUC | 0.5841 | 0.4491 |

Raw ROC-AUC difference: 0.5841 - 0.4491 = 0.1350.

The DP model's AUC is below 0.5, so the attack score's ranking direction must be investigated before interpreting this difference as a privacy improvement.

## 7. Saved Results
- DP training metadata: outputs/gpt2_mia_dp/metadata.json
- DP model MIA scores: privacy_analysis/results/mia_dp_scores.csv
- Combined comparison scores: privacy_analysis/results/mia_comparison.csv

## 8. Conclusion and Limitations
Baseline and DP-SGD models were evaluated using the same MIA comparison script, with the same 100 member and 100 non-member examples and the same loss-based scoring method.

Opacus reported an achieved privacy budget of epsilon approximately 7.9955 and delta 1e-5 for the DP-SGD training run, subject to the implementation and accounting assumptions.

The MIA results are preliminary. An AUC below 0.5 does not by itself prove stronger privacy. Multiple seeds, uncertainty analysis, and verification of attack direction are needed.

Important scope note: DP-SGD is a training-time privacy mechanism. It is distinct from the inference-time Differential Privacy method intended in the original project proposal. This experiment should be treated as a supporting experiment, not as completion of the proposal's intended method.
'@ | Set-Content -Encoding UTF8 privacy_analysis\mia_dp_sgd_summary.md