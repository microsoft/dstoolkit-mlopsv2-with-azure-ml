# Description: Evaluate the current model and compare to
# the last registered model.

import argparse
import json
import os

import joblib
import mlflow
import mlflow.pyfunc
import pandas as pd
import util
from mlflow.tracking import MlflowClient
from sklearn.metrics import classification_report


def parse_args():
    """Parse input arguments"""

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, help="data directory")
    parser.add_argument("--model_dir", type=str, help="model directory")
    parser.add_argument("--eval_dir", type=str, help="evaluation directory")
    parser.add_argument("--config_file", type=str, help="config file path")

    parser.add_argument(
        "--no_logging", action="store_true", help="disable logging to MLflow"
    )

    args = parser.parse_args()

    return args


def prep_test_data(test_data_path):
    """Prepare test data for evaluation"""

    print(f"Loading test data from: {test_data_path}")

    # Load test data
    test_df = pd.read_csv(test_data_path)

    # Extract the label column
    y_test = test_df.pop("default payment next month")

    # Convert the dataframe values to array
    X_test = test_df.values

    return X_test, y_test


def eval_current_model(model_path, eval_dir, eval_path, X_test, y_test):
    """Evaluate the current model on test data"""

    # Load the model
    model = joblib.load(model_path)
    print(f"Model loaded from: {model_path}")

    # Evaluate the model
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)

    # Save evaluation metrics locally
    with open(eval_path, "w") as f:
        json.dump(report, f, indent=4)
    print(f"Metrics of the current model saved at: {eval_path}")

    # Plot confusion matrix and save locally
    plot_path = os.path.join(eval_dir, "confusion_matrix.png")
    util.plot_confusion_matrix(y_test, y_pred, plot_path)

    return report, plot_path


def eval_last_registered_model(model_name, eval_path, X_test, y_test):
    """Evaluate the last registered model if any"""

    # Load the last registered the model
    client = MlflowClient()
    model_versions = [
        model_run.version
        for model_run in client.search_model_versions(f"name='{model_name}'")
    ]
    if model_versions:
        latest_version = max(model_versions)
        last_model = mlflow.pyfunc.load_model(
            model_uri=f"models:/{model_name}/{latest_version}"
        )
        print(f"Last registered model loaded: {model_name}/{latest_version}")

        # Evaluate the model
        y_pred = last_model.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True)

        # Add model name and version to the report
        report["model_name"] = model_name
        report["model_version"] = latest_version

        # Save evaluation metrics locally
        with open(eval_path, "w") as f:
            json.dump(report, f, indent=4)

        print(f"Metrics of the last registered model saved at: {eval_path}")
    else:
        # No registered model found
        report = None

    return report


def compare_to_the_last(eval_path, perf_current, perf_last):
    """Compare current model performance to the last registered model"""

    # Default to no improvement
    better_than_last = 0

    # Compare f1-score
    f1_current = perf_current["weighted avg"]["f1-score"]
    f1_last = None

    # Compare accuracy
    if perf_last:
        f1_last = perf_last["weighted avg"]["f1-score"]
        print(
            f"weighted avg f1: current={f1_current:.3f}, "
            f"last={f1_last:.3f}"
        )
        if f1_current > f1_last:
            better_than_last = 1
    else:
        better_than_last = 1
        print("No registered model found. Current model is the best so far.")

    # Write the comparison result to a file
    with open(eval_path, "w") as f:
        f.write(str(better_than_last))
    print(f"Record whether this model is better than the last at: {eval_path}")

    return perf_current, perf_last, better_than_last


def main(args):
    """Main function to evaluate the model"""

    # Read config file
    with open(args.config_file, "r") as f:
        config = json.load(f)

    data_config = config["data"]
    model_config = config["model"]
    eval_config = config["eval"]
    print(
        "Data config:", ", ".join(f"{k}={v}" for k, v in data_config.items())
    )

    print(
        "Model config:", ", ".join(f"{k}={v}" for k, v in model_config.items())
    )
    print(
        "Evaluation config:",
        ", ".join(f"{k}={v}" for k, v in eval_config.items()),
    )

    # Set file paths
    test_data_path = os.path.join(args.data_dir, data_config["test_file"])
    model_path = os.path.join(args.model_dir, model_config["model_file"])
    eval_path_current_model = os.path.join(
        args.eval_dir, eval_config["eval_file"]
    )
    eval_path_last_model = os.path.join(
        args.eval_dir, eval_config["eval_file_last_model"]
    )
    eval_path_compare = os.path.join(
        args.eval_dir, eval_config["better_than_last_file"]
    )

    # Prepare test data
    X_test, y_test = prep_test_data(test_data_path)

    # Evaluate the model
    report_current, plot_path = eval_current_model(
        model_path,
        args.eval_dir,
        eval_path_current_model,
        X_test,
        y_test,
    )

    # Evaluate the last registered model
    report_last = eval_last_registered_model(
        model_config["model_name"], eval_path_last_model, X_test, y_test
    )

    # Compare current model performance to the last registered model
    perf_current, perf_last, better_than_last = compare_to_the_last(
        eval_path_compare, report_current, report_last
    )

    # Log evaluation metrics to MLflow
    if not args.no_logging:
        # Log accuracy
        mlflow.log_metric(
            "current model accuracy-", report_current["accuracy"]
        )
        # Log weighted average f1, recall, and precision
        mlflow.log_metrics(report_current["weighted avg"])
        # Log confusion matrix plot
        mlflow.log_artifact(plot_path)
        # Log f1 of the last registered model
        if perf_last:
            mlflow.log_metrics(
                {
                    "current model f1-": perf_current["weighted avg"][
                        "f1-score"
                    ],
                    "last model f1-": perf_last["weighted avg"]["f1-score"],
                }
            )
        # Log whether the current model is better than the last
        mlflow.log_metric("better than last", better_than_last)


# Usage: python evaluate.py --data_dir ../data/output --model_dir ../models --eval_dir ../eval --config_file ../config/modelling.json --no_logging (optional: disable logging to MLflow) # noqa
if __name__ == "__main__":
    args = parse_args()

    print(
        "Input argument:", ", ".join(f"{k}={v}" for k, v in vars(args).items())
    )

    # Start Logging
    if not args.no_logging:
        mlflow.start_run(run_name="Credit_Default_Evaluation")

    main(args)

    # Stop Logging
    if mlflow.active_run():
        mlflow.end_run()
