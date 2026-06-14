import subprocess
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report
)

def get_git_commit():
    commit_hash = subprocess.check_output(
        ["git", "rev-parse", "HEAD"]
    ).decode().strip()
    return commit_hash

def create_confusion_matrix(
    y_true,
    y_pred,
    filepath="models/confusion_matrix.png"
):
    cm = confusion_matrix(
        y_true,
        y_pred
    )
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm
    )
    fig, ax = plt.subplots(
        figsize=(6,6)
    )
    disp.plot(ax=ax)
    plt.savefig(
        filepath,
        bbox_inches="tight"
    )
    plt.close()
    return filepath

def save_classification_report(
    y_true,
    y_pred,
    filepath="models/classification_report.txt"
):
    report = classification_report(
        y_true,
        y_pred
    )
    with open(
        filepath,
        "w"
    ) as f:
        f.write(report)
    return filepath