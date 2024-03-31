# Description: Test the evaluate.py script

import json
import os
import subprocess


def test_evaluate():
    # Data directory
    data_dir = os.path.join("tests", "data")

    # Model directory
    model_dir = os.path.join("tests", "model")

    # Modelling config file
    config_file = os.path.join("tests", "modelling_test.json")

    # Make a temp directory to store the evaluation files
    result_dir = os.path.join("tests", "evaluation_temp")
    os.makedirs(result_dir, exist_ok=True)

    # Run the evaluate.py script as a subprocess
    result = subprocess.run(
        [
            "python",
            "src/evaluate.py",
            "--data_dir",
            data_dir,
            "--model_dir",
            model_dir,
            "--eval_dir",
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

    # Check evaluation files were created
    eval_file = os.path.join(result_dir, config["eval"]["eval_file"])
    assert os.path.isfile(eval_file)

    better_than_last_file = os.path.join(
        result_dir, config["eval"]["better_than_last_file"]
    )
    assert os.path.isfile(better_than_last_file)

    # Clear and remove the temp directory
    for filename in os.listdir(result_dir):
        file_path = os.path.join(result_dir, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)
    os.rmdir(result_dir)


if __name__ == "__main__":
    test_evaluate()
