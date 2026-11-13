import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


def print_table(title, frame):
    """DataFrame'i ek bağımlılık gerektirmeden çerçeveli tablo olarak yazdır."""
    formatted = frame.to_string(index=False, float_format=lambda value: f"{value:.4f}",
                                na_rep="N/A")
    lines = formatted.splitlines()
    width = max(len(line) for line in lines)
    border = "+-" + "-" * width + "-+"
    print(f"\n{title}")
    print(border)
    for index, line in enumerate(lines):
        print(f"| {line.ljust(width)} |")
        if index == 0:
            print(border)
    print(border)


def evaluate_model(frame, probabilities):
    """Genel ve kural bazında yerel doğrulama metriklerini hesapla."""
    target = frame["rule_violation"]
    predictions = (probabilities >= 0.5).astype(int)
    scored = frame[["rule", "rule_violation"]].copy()
    scored["probability"] = probabilities
    rule_scores = []
    for rule, group in scored.groupby("rule"):
        auc = (roc_auc_score(group["rule_violation"], group["probability"])
               if group["rule_violation"].nunique() == 2 else float("nan"))
        rule_scores.append({"rule": rule, "roc_auc": auc, "count": len(group)})
    per_rule = pd.DataFrame(rule_scores)
    metrics = {
        "roc_auc": roc_auc_score(target, probabilities),
        "mean_rule_auc": per_rule["roc_auc"].mean(),
        "accuracy": accuracy_score(target, predictions),
        "f1": f1_score(target, predictions, zero_division=0),
    }
    return metrics, per_rule
