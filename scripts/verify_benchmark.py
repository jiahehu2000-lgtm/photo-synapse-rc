"""Check a fresh default-protocol evaluation against retained public predictions."""
import argparse
import csv
import json
from pathlib import Path


def verify(output, method):
    root = Path(__file__).resolve().parents[1]
    output = Path(output)
    names = {"softmax": "LINEAR_SOFTMAX", "svm": "LINEAR_SVM", "ridge": "RIDGE"}
    summary = json.loads((output/"summary.json").read_text(encoding="utf-8"))
    protocol = json.loads((output/"protocol.json").read_text(encoding="utf-8"))
    if not summary["completed"] or summary["cached_results_read"] or summary["method"] != method:
        raise ValueError("Expected a complete freshly trained evaluation for the requested method")
    if protocol["adc_bits"] != 2 or protocol["reads"] != 40 or protocol["reset"] or protocol["encoder_only"]:
        raise ValueError("Retained references apply only to the default 2-bit, 40-read protocol")
    with (root/"results/public_reference_predictions.csv").open(encoding="utf-8-sig", newline="") as h:
        reference = {
            (row["condition"], int(row["record_id"])):
            (int(row["subject"]), int(row["true_class"]), int(row["predicted_class"]))
            for row in csv.DictReader(h) if row["classifier"] == names[method]
        }
    with (output/"predictions.csv").open(encoding="utf-8-sig", newline="") as h:
        rows = list(csv.DictReader(h))
    actual = {}
    for row in rows:
        key = (row["condition"], int(row["record_id"]))
        if key in actual:
            raise ValueError(f"Duplicate outer prediction: {key}")
        actual[key] = (int(row["subject"]), int(row["true"]), int(row["predicted"]))
    if len(reference) != 638 or actual != reference:
        differing = sum(actual.get(key) != value for key, value in reference.items())
        raise ValueError(f"Reference mismatch: {differing} predictions differ or record membership changed")
    return dict(method=method, checked_outer_predictions=638, reference_predictions_match=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="核对从头训练的默认方案与638条公共留出参考预测")
    p.add_argument("--output", required=True)
    p.add_argument("--method", choices=["softmax", "svm", "ridge"], default="softmax")
    args = p.parse_args()
    print(json.dumps(verify(args.output, args.method), indent=2))
