# Description: Register the current model if the current
# model is better than the last.

import argparse
import json
import os

import joblib
import mlflow
import mlflow.pyfunc


def parse_args():
    """Parse input arguments"""

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, help="model directory")
    parser.add_argument(
        "--registry_dir", type=str, help="model registry directory"
    )
    parser.add_argument(
        "--eval_dir", type=str, help="evaluation result directory"
    )
    parser.add_argument("--config_file", type=str, help="config file path")

    args = parser.parse_args()

    return args


def main(args):
    """Main function to register the model if the current model
    is better than the last"""

    # Read config file
    with open(args.config_file, "r") as f:
        config = json.load(f)
    model_config = config["model"]
    print(
        "Model config:", ", ".join(f"{k}={v}" for k, v in model_config.items())
    )
    eval_config = config["eval"]
    print(
        "Eval config:", ", ".join(f"{k}={v}" for k, v in eval_config.items())
    )
    registry_config = config["registry"]
    print(
        "Registry config:",
        ", ".join(f"{k}={v}" for k, v in registry_config.items()),
    )

    # Read evaluation flag
    eval_path = os.path.join(
        args.eval_dir, eval_config["better_than_last_file"]
    )
    with open(eval_path, "r") as f:
        better_than_last = int(f.read())

    # Register the model if the current model is better than the last
    if better_than_last == 1:
        mlflow.log_metric("better_than_last", better_than_last)

        print("Registering ", registry_config["model_name"])

        # Load the local model
        model_path = os.path.join(args.model_dir, model_config["model_file"])
        model = joblib.load(model_path)

        # Set MLflow model path
        mlflow_model_path = os.path.join(
            args.registry_dir, registry_config["model_name"]
        )

        # Remove the model if it already exists
        if os.path.exists(mlflow_model_path):
            # Remove files within the directory
            for filename in os.listdir(mlflow_model_path):
                file_path = os.path.join(mlflow_model_path, filename)
                os.remove(file_path)

        # Save the MLflow model to local
        mlflow.sklearn.save_model(sk_model=model, path=mlflow_model_path)

        # Load the MLflow model and log the model to MLflow
        mlflow_model = mlflow.sklearn.load_model(mlflow_model_path)
        mlflow.sklearn.log_model(mlflow_model, model_config["model_name"])

        # Register logged model to model registry using mlflow
        run_id = mlflow.active_run().info.run_id
        model_uri = f"runs:/{run_id}/{model_config['model_name']}"
        mlflow_model = mlflow.register_model(
            model_uri, model_config["model_name"]
        )
    else:
        print("Model is not better than the last. Skip registering")


# Usage: python register.py --model_dir ../models --eval_dir ../eval --registry_dir ../registry --config_file ../config/modelling.json # noqa
if __name__ == "__main__":
    args = parse_args()
    print(
        "Input argument:", ", ".join(f"{k}={v}" for k, v in vars(args).items())
    )

    # Start Logging
    mlflow.start_run(run_name="Credit_Default_Model_Registering")

    main(args)

    # Stop Logging
    if mlflow.active_run():
        mlflow.end_run()
