# Description: Utility functions for the project.

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


def plot_confusion_matrix(y_true, y_pred, file_path):
    """Create a confusion matrix plot."""
    # Generate confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Create heatmap plot
    sns.heatmap(cm, annot=True, cmap="Blues", fmt="g")

    # Add labels, title and ticks
    plt.xlabel("Predicted labels")
    plt.ylabel("True labels")
    plt.title("Confusion Matrix")
    plt.xticks(ticks=[0, 1], labels=["Class 0", "Class 1"])
    plt.yticks(ticks=[0, 1], labels=["Class 0", "Class 1"])

    # Save plot as png file
    plt.savefig(file_path)
