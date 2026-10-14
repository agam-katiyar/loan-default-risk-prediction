import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score,
    classification_report,
    roc_curve,
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_recall_curve,
    average_precision_score,
)


def evaluate(model, X_test, y_test, model_name: str = "model"):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, y_prob)
    print(f"\n{'='*40}")
    print(f"Model: {model_name}")
    print(f"AUC-ROC: {auc:.4f}")
    print(f"{'='*40}")
    print(classification_report(y_test, y_pred))

    return y_pred, y_prob


def plot_roc(y_test, y_prob, model_name: str = "model", save: bool = True):
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)

    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, lw=2, label=f"{model_name} (AUC = {auc:.2f})")
    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.tight_layout()

    if save:
        plt.savefig(f"reports/figures/roc_{model_name}.png", dpi=150)

    plt.show()


def plot_precision_recall(y_test, y_prob, model_name: str = "model", save: bool = True):
    # PR curve is more informative than ROC when classes are imbalanced
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    ap = average_precision_score(y_test, y_prob)

    plt.figure(figsize=(7, 5))
    plt.plot(recall, precision, lw=2, label=f"{model_name} (AP = {ap:.2f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.tight_layout()

    if save:
        plt.savefig(f"reports/figures/pr_{model_name}.png", dpi=150)

    plt.show()


def plot_confusion_matrix(y_test, y_pred, model_name: str = "model", save: bool = True):
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="Blues")
    plt.title(f"Confusion Matrix — {model_name}")
    plt.tight_layout()

    if save:
        plt.savefig(f"reports/figures/cm_{model_name}.png", dpi=150)

    plt.show()
