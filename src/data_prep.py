# Description: Prepare data for training and testing.

import argparse
import json
import os

import mlflow
import pandas as pd
from sklearn.model_selection import train_test_split


def parse_args():
    """Parse input arguments"""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_input_dir", type=str, help="input data directory"
    )
    parser.add_argument(
        "--data_output_dir", type=str, help="output data directory"
    )
    parser.add_argument("--config_file", type=str, help="config file path")
    parser.add_argument(
        "--no_logging", action="store_true", help="disable logging to MLflow"
    )

    args = parser.parse_args()

    return args


def main(args):
    """Main function to prepare data for training and testing."""

    # Read config file
    with open(args.config_file, "r") as f:
        config = json.load(f)

    # Set file paths
    data_config = config["data"]
    print(
        "Data config:", ", ".join(f"{k}={v}" for k, v in data_config.items())
    )

    input_data_path = os.path.join(
        args.data_input_dir, data_config["data_file"]
    )
    train_data_path = os.path.join(
        args.data_output_dir, data_config["train_file"]
    )
    test_data_path = os.path.join(
        args.data_output_dir, data_config["test_file"]
    )

    # Read data
    credit_df = pd.read_excel(input_data_path, header=1, index_col=0)

    # Split data into train and test
    test_train_ratio = data_config["test_train_ratio"]
    credit_train_df, credit_test_df = train_test_split(
        credit_df,
        test_size=test_train_ratio,
    )

    # Save train and test data
    credit_train_df.to_csv(train_data_path, index=False)
    credit_test_df.to_csv(test_data_path, index=False)

    # Data stats
    data_stats = {
        "num_samples": credit_df.shape[0],
        "num_features": credit_df.shape[1] - 1,
        "num_train_samples": credit_train_df.shape[0],
        "num_test_samples": credit_test_df.shape[0],
    }
    print("Data stats", ", ".join(f"{k}={v}" for k, v in data_stats.items()))

    # Log data stats to MLflow
    if not args.no_logging:
        mlflow.log_metrics(data_stats)


# Usage: python data_prep.py --data_input_dir ../data/input --data_output_dir ../data/output --config_file ../config/modelling.json --no_logging (optional: disable logging to MLflow) # noqa
if __name__ == "__main__":
    args = parse_args()
    print(
        "Input argument:", ", ".join(f"{k}={v}" for k, v in vars(args).items())
    )

    # Start Logging
    if not args.no_logging:
        mlflow.start_run(run_name="Credit_Default_Data_Prep")

    main(args)

    # Stop Logging
    if mlflow.active_run():
        mlflow.end_run()
