import os
import re
import pandas as pd

def extract_metrics(path):
    """Extract accuracy, f1, and AUC from a classification report."""
    if not os.path.exists(path):
        print(f"File not found: {path}")
        return None

    with open(path) as f:
        text = f.read().lower()

    # Extract metrics
    acc_match = re.search(r"accuracy\s+([\d.]+)", text)
    f1_match = re.search(r"weighted avg\s+[\d.]+\s+[\d.]+\s+([\d.]+)", text)
    auc_match = re.search(r"auc[:\s]+([\d.]+)", text)

    accuracy = float(acc_match.group(1)) * (100 if float(acc_match.group(1)) <= 1 else 1) if acc_match else None
    f1 = float(f1_match.group(1)) if f1_match else None
    auc = float(auc_match.group(1)) if auc_match else None

    return {"accuracy": accuracy, "f1": f1, "auc": auc}



A_path = "results/classifier_A/report_A.txt"
B_path = "results/classifier_B/report_B.txt"

A = extract_metrics(A_path)
B = extract_metrics(B_path)


print("\n Model Comparison (A vs B)\n")
if A and B:
    df = pd.DataFrame({
        "Metric": ["Accuracy (%)", "F1-score", "AUC"],
        "Model A (Original HR)": [A["accuracy"], A["f1"], A["auc"]],
        "Model B (SRGAN)": [B["accuracy"], B["f1"], B["auc"]],
    })
    print(df.to_string(index=False, float_format="%.3f"))
    print("\nAccuracy = %.3f%%" % (B["accuracy"] - A["accuracy"]))
else:
    print("One or both report files missing or unparseable.")
