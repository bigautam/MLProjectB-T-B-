import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import dump_svmlight_file, load_svmlight_file

import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_file',
                        type=str,
                        required=True,
                        help='Raw dataset (as CSV)')
    parser.add_argument('--out_file',
                        type=str,
                        required=True,
                        help='File to save dataset (as SVMLight)')
    parser.add_argument('--sample_n',
                        type=int,
                        default=-1,
                        help='Number of examples to sample')
    parser.add_argument('--fraud_ratio',
                        type=float,
                        default=2,
                        help='Ratio of fraudulent to non-fraudulent transactions')
    return parser.parse_args()

def clean(df, sample_n=None, fraud_ratio=2):
    # Recur on 24 hour format
    df["step"] = df["step"] % 24
    df.loc[df["step"] == 0, "step"] = 24

    # Filter for customers only
    df = df[df["nameDest"].str.startswith("C")]

    # Now sample 10,000 rows for modeling
    #sample_df = costomer_df.sample(n=10000, random_state=42)
    # Sample 5000 rows FIRST to reduce memory use, then balance
    if sample_n > 0:
        df = df.sample(n=sample_n, random_state=42)

    # Balance dataset by upsampling minority class
    fraud = df[df["isFraud"] == 1]
    non_fraud = df[df["isFraud"] == 0].sample(n=int(len(fraud) * fraud_ratio), random_state=42)  # Downsample non-fraud to avoid massive duplication

    balanced_df = pd.concat([fraud, non_fraud], axis=0).sample(frac=1, random_state=42).reset_index(drop=True)

    # Predicted variable separation
    X = balanced_df.drop(columns=['isFraud'])
    y = balanced_df['isFraud']

    # Drop unused features
    X = X.drop(columns=['nameOrig', 'nameDest', 'isFlaggedFraud', 'type'])

    # Scale numeric columns
    numeric_cols = ['amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']
    scaler = StandardScaler()
    X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

    return X, y

def save_dataset(X, y, filename):
    dump_svmlight_file(X, y, f=filename)

def get_dataset(filename):
    X, y = load_svmlight_file(f=filename)
    return X, y

def main():
    args = get_args()

    print(f'Reading raw dataset')
    raw_dataset = pd.read_csv(args.dataset_file)

    print(f'Cleaning dataset')
    X, y = clean(raw_dataset, sample_n=args.sample_n, fraud_ratio=args.fraud_ratio)

    print(f'Writing cleaned dataset')
    save_dataset(X, y, args.out_file)

if __name__ == '__main__':
    main()