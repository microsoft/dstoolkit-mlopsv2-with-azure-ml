# Description: Train the model.

import argparse
import json
import os

import joblib
import mlflow
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier


def parse_args():
    """Parse input arguments"""

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, help="data directory")
    parser.add_argument("--model_dir", type=str, help="model directory")
    parser.add_argument("--config_file", type=str, help="config file path")

    parser.add_argument(
        "--no_logging", action="store_true", help="disable logging to MLflow"
    )

    args = parser.parse_args()

    return args


def prep_model_dir(model_dir):
    """Prepare model directory"""

    if os.path.exists(model_dir):
        # Remove files within the directory
        for filename in os.listdir(model_dir):
            file_path = os.path.join(model_dir, filename)
            # If file is a file
            if os.path.isfile(file_path):
                os.remove(file_path)
    else:
        # Create the directory
        os.makedirs(model_dir)


def main(args):
    """Main function to train the model"""

    # Read config file
    with open(args.config_file, "r") as f:
        config = json.load(f)

    data_config = config["data"]
    train_config = config["train"]
    model_config = config["model"]

    print(
        "Data config:", ", ".join(f"{k}={v}" for k, v in data_config.items())
    )
    print(
        "Training config:",
        ", ".join(f"{k}={v}" for k, v in train_config.items()),
    )
    print(
        "Model config:", ", ".join(f"{k}={v}" for k, v in model_config.items())
    )

    # Set file paths
    train_data_path = os.path.join(args.data_dir, data_config["train_file"])
    model_path = os.path.join(args.model_dir, model_config["model_file"])

    # Read training data
    train_df = pd.read_csv(train_data_path)

    # Extract the label column
    y_train = train_df.pop("default payment next month")

    # Convert the dataframe values to array
    X_train = train_df.values

    print(f"Training with data of shape {X_train.shape}")

    # Train the model
    clf = GradientBoostingClassifier(
        n_estimators=train_config["n_estimators"],
        learning_rate=train_config["learning_rate"],
    )
    clf.fit(X_train, y_train)

    # Prepare model directory
    prep_model_dir(args.model_dir)

    # Option 1: Save the model locally
    with open(model_path, "wb") as f:
        joblib.dump(clf, f)
        print(f"Model saved at: {model_path}")

    # Log model parameters and model to MLflow
    if not args.no_logging:
        mlflow.log_param(
            "model", "sklearn.ensemble.GradientBoostingClassifier"
        )
        mlflow.log_param("n_estimators", train_config["n_estimators"])
        mlflow.log_param("learning_rate", train_config["learning_rate"])
        mlflow.log_artifact(model_path)

    # # Option 2: Save model via MLflow sklearn wrapper.
    # # Model will be saved locally as well as logged
    # # to AML workspace with built-in sklearn model meta data.
    # mlflow.sklearn.save_model(
    #     sk_model=clf,
    #     path=model_config["model_dir"],
    # )


# Usage: python train.py --data_dir ../data/output --model_dir ../models --config_file ../config/modelling.json --no_logging (optional: disable logging to MLflow) # noqa
if __name__ == "__main__":

    # Parse arguments
    args = parse_args()
    print(
        "Input argument:", ", ".join(f"{k}={v}" for k, v in vars(args).items())
    )

    # Start Logging
    if not args.no_logging:
        mlflow.start_run(run_name="Credit_Default_Model_Training")

    main(args)

    # Stop Logging
    if mlflow.active_run():
        mlflow.end_run()
