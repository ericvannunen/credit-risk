# credit-risk

Trains and compares two one-year corporate bankruptcy models on the UCI Polish Companies dataset:

- Interpretable logistic regression
- Small PyTorch neural network

Both produce bankruptcy probabilities, A-E risk grades, and top contributing financial ratios.

## Run

```bash
credit-risk --data-dir data
```

If network access is blocked, download `1year.arff` separately and run:

```bash
credit-risk --data-path /absolute/path/to/1year.arff
```