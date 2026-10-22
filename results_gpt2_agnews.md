# AGNEWS Experiment Results — Local GPT-2

## 1. Experiment Setup
- Dataset: AGNEWS
- Model: Local GPT-2
- Test examples: 100
- Synthesis seed: 0
- Evaluation seed: 0
- Sigma: 0.39
- Private training examples: 20
- Private training splits: 10
- API: Local model inference; no OpenAI API key used for the reported evaluation

## 2. Results

| Metric | Without DP | With DP |
|---|---:|---:|
| Original accuracy | 51% | 51% |
| Calibrated accuracy | 52% | 44% |

## 3. Observations
- Original accuracy was 51% in both conditions.
- Calibration increased accuracy from 51% to 52% without DP.
- Calibration reduced accuracy from 51% to 44% in the DP-enabled condition.
- Some generated synthetic news examples were repetitive or incoherent.

## 4. Limitations
- These are preliminary results from 100 test examples and one seed.
- The results are based on a local GPT-2 adaptation, not an exact reproduction of the original paper's Babbage experiments.
- The accuracy results alone do not establish a formal differential privacy guarantee.
- Multiple seeds, a larger evaluation set, and verification of the privacy mechanism are still required.

## 5. Conclusion
In this preliminary run, uncalibrated accuracy was unchanged between the two conditions, while calibrated accuracy was lower in the DP-enabled condition. Further experiments are needed before drawing a reliable conclusion.
