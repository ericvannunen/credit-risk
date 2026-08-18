# credit-risk

Trains and compares two one-year corporate bankruptcy models on the `5year` subset of the UCI Polish Companies dataset. The subset uses financial information from the fifth observed year to predict bankruptcy within the following year:

- Interpretable logistic regression
- Small PyTorch neural network

Both produce bankruptcy probabilities, A-E risk grades, and top contributing financial ratios.

## Run

```bash
credit-risk --data-dir data
```

If the dataset is already downloaded, run directly against the extracted ARFF file:

```bash
credit-risk --data-path /absolute/path/to/5year.arff
```

## Dashboard

Install the project dependencies, then start the interactive dashboard:

```bash
streamlit run credit_risk/app.py
```

Open the local URL shown by Streamlit, usually `http://localhost:8501`.