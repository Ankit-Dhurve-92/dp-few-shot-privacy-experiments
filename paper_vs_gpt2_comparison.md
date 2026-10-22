# Paper vs Local GPT-2: AGNEWS Comparison

## Paper Results
- Model: GPT-3 Babbage
- Zero-shot accuracy: 47.9%
- 4-shot without private data: 68.0%
- DP (epsilon = 1): 64.1%
- Non-private: 69.3%

## Our Results
- Model: Local GPT-2
- Dataset: AGNEWS
- Test examples: 100
- Sigma: 0.39
- Without DP: Original 51%, calibrated 52%
- DP-enabled: Original 51%, calibrated 44%

## Numerical Difference
- DP: 13.1 percentage points
- Non-private: 18.3 percentage points

## Limitations
- Different models and inference implementations.
- Our experiment used 100 test examples; paper reports 5-run averages.
- Sigma = 0.39 is not confirmed equivalent to epsilon = 1.
- Results are preliminary, not an exact reproduction.

## Next Step
Implement our proposed method and compare its results.
