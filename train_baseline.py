"""Çalıştır: python train_baseline.py"""
import argparse
from pathlib import Path

import pandas as pd

from src.data import load_data, make_text, split_data
from src.models import build_models
from src.utils import evaluate_model, print_table


def main():
    parser = argparse.ArgumentParser(description="Jigsaw başlangıç modellerini karşılaştır.")
    parser.add_argument("--seeds", nargs="+", type=int, default=[7, 21, 42])
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    output = root / "outputs"
    output.mkdir(exist_ok=True)
    train, test = load_data(root / "data")
    results, rule_results = [], []

    for seed in dict.fromkeys(args.seeds):
        training, validation = split_data(train, seed)
        train_text, valid_text = make_text(training), make_text(validation)
        for name, model in build_models().items():
            model.fit(train_text, training["rule_violation"])
            probabilities = model.predict_proba(valid_text)[:, 1]
            metrics, per_rule = evaluate_model(validation, probabilities)
            results.append({"model": name, "seed": seed,
                            "validation_rows": len(validation), **metrics})
            rule_results.append(per_rule.assign(model=name, seed=seed))

    results = pd.DataFrame(results)
    results.to_csv(output / "results.csv", index=False)
    pd.concat(rule_results, ignore_index=True).to_csv(output / "per_rule.csv", index=False)
    summary = results.groupby("model")[["roc_auc", "mean_rule_auc", "accuracy", "f1"]].mean()
    print_table("Ortalama doğrulama sonuçları", summary.reset_index())

    # Dummy yalnızca referans; metin modellerini aynı bölmelerde karşılaştır.
    best_name = summary.drop(index="dummy")["mean_rule_auc"].idxmax()
    final_model = build_models()[best_name]
    final_model.fit(make_text(train), train["rule_violation"])
    submission = pd.DataFrame({
        "row_id": test["row_id"],
        "rule_violation": final_model.predict_proba(make_text(test))[:, 1],
    })
    sample = pd.read_csv(root / "data" / "sample_submission.csv")
    if not submission["row_id"].equals(sample["row_id"]):
        raise ValueError("Test ve sample_submission satır sıraları eşleşmiyor.")
    submission.to_csv(output / "submission.csv", index=False)
    print(f"Eğitim: {len(train)} satır, test: {len(test)} satır")
    print(f"Seçilen model: {best_name}. Sonuçlar: {output}")


if __name__ == "__main__":
    main()
