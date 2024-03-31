# Description: Test the train.py script.

import json
import os
import subprocess


def test_train():
    # Data directory
    data_dir = os.path.join("tests", "data")

    # Modelling config file
    config_file = os.path.join("tests", "modelling_test.json")

    # Make a temp directory to store the train and test files
    result_dir = os.path.join("tests", "train_temp")
    os.makedirs(result_dir, exist_ok=True)

    # Run the train.py script as a subprocess
    result = subprocess.run(
        [
            "python",
            "src/train.py",
            "--data_dir",
            data_dir,
            "--model_dir",
            result_dir,
            "--config_file",
            config_file,
        ],
        capture_output=True,
        text=True,
    )

    print(
        f"result.stdout: {result.stdout}",
        f"result.stderr: {result.stderr}",
        f"result.returncode: {result.returncode}",
    )

    # Check the return code to ensure the script executed successfully
    assert result.returncode == 0

    # Read modelling_test.json
    with open(config_file) as f:
        config = json.load(f)

    # Check model was created
    model_file = os.path.join(result_dir, config["model"]["model_file"])
    assert os.path.isfile(model_file)

    # Clear and remove the temp directory
    for filename in os.listdir(result_dir):
        file_path = os.path.join(result_dir, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)
    os.rmdir(result_dir)


if __name__ == "__main__":
    test_train()
