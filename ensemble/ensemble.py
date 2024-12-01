import argparse
import os
import pandas as pd
from pathlib import Path

def load_csv_files(directory):
    csv_files = []
    for filename in os.listdir(directory):
        if filename.endswith('.csv'):
            filepath = os.path.join(directory, filename)
            csv_files.append(pd.read_csv(filepath))
    return csv_files

def weighted_average_ensemble(csv_files, weights):
    if len(csv_files) != len(weights):
        raise ValueError("The number of weights must match the number of CSV files.")
    weights = [float(w) / sum(weights) for w in weights]
    ensemble_df = csv_files[0] * weights[0]
    for i in range(1, len(csv_files)):
        ensemble_df += csv_files[i] * weights[i]
    return ensemble_df

def main(before_dir='before', after_dir='after', weights=None):
    os.makedirs(after_dir, exist_ok=True)
    csv_files = load_csv_files(before_dir)
    if weights is None:
        weights = [1 / len(csv_files)] * len(csv_files)
    ensemble_df = weighted_average_ensemble(csv_files, weights)
    output_path = os.path.join(after_dir, 'ensemble_result.csv')
    ensemble_df.to_csv(output_path, index=False)
    print(f"Ensembled file saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ensemble multiple CSV files with specified weights.")
    parser.add_argument('--weights', nargs='+', type=float, help="Weights for each CSV file in the ensemble")
    args = parser.parse_args()
    main(weights=args.weights)


# 예시 weights가 0.4, 0.4, 0.2일 때
# python ensemble/ensemble.py --weights 0.4 0.4 0.2
# 