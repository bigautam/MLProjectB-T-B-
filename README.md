# [Automatic Fraud Detection in Financial Transactions](https://docs.google.com/document/d/1TUafunj8z6kLILLWd9zjG88X3mLwtAVWBDSH_SbVQV4/edit?usp=sharing)
**Team:** B^T B

Leveraging and comparing machine learning techniques to identify fraudulent transactions in obfuscated financial environments.

## Data cleaning
```bash
usage: clean_data.py [-h] --dataset_file DATASET_FILE --out_file OUT_FILE [--sample_n SAMPLE_N] [--fraud_ratio FRAUD_RATIO]

options:
  -h, --help            show this help message and exit
  --dataset_file DATASET_FILE
                        Raw dataset (as CSV)
  --out_file OUT_FILE   File to save dataset (as SVMLight)
  --sample_n SAMPLE_N   Number of examples to sample
  --fraud_ratio FRAUD_RATIO
                        Ratio of fraudulent to non-fraudulent transactions
```

## Evaluation
This implementation is still in progress, features to come include:
- Multiple evaluator comparison
- Visualization access

```bash
usage: evaluate.py [-h] --dataset_file DATASET_FILE

options:
  -h, --help            show this help message and exit
  --dataset_file DATASET_FILE
                        Dataset (as SVMLight)
```