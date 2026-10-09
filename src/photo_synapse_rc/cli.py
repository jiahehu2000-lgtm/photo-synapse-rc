"""Portable command line; no Origin or historical local folder dependencies."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from .config import device_models
from .core import bits_for, encode_xor, current, features


def demo(output):
    """Synthetic smoke demonstration; deliberately reports no research accuracy."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    path = output/"demo_states.npz"
    if path.exists():
        raise FileExistsError(f"Demo output already exists: {path}")
    rng = np.random.default_rng(20261009)
    t = np.linspace(0, 2*np.pi, 40, endpoint=False)
    x = np.stack([np.column_stack([np.sin(t*(1+j%3)+j*.7+r*.2)
                                  for j in range(6)]) for r in range(20)])
    x += rng.normal(0, .03, x.shape)
    train = np.arange(15)
    bits, scaler = bits_for(x, train)
    encoded = encode_xor(bits)
    arrays = {"synthetic_inputs": x, "events": encoded}
    report = dict(synthetic=True, research_accuracy=None, records=len(x), input_shape=list(x.shape),
                  event_shape=list(encoded.shape), input_scaler=scaler, conditions={})
    for condition, params in device_models().items():
        i = current(encoded, params)
        f = features(encoded, params)
        arrays[f"{condition}_current_nA"] = i
        arrays[f"{condition}_features"] = f
        report["conditions"][condition] = dict(features=f.shape[1],
            min_current_nA=float(i.min()), max_current_nA=float(i.max()),
            adc_levels_nA=np.unique(f).tolist())
    np.savez_compressed(path, **arrays)
    (output/"demo_summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    return report


def main(argv=None):
    # Windows redirected consoles may otherwise encode Chinese output as GBK.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="光暗器件模型驱动RC：demo演示 / evaluate重训评价 / predict推断")
    commands = parser.add_subparsers(dest="command", required=True)
    d = commands.add_parser("demo", help="合成输入演示，不能作为论文数据")
    d.add_argument("--output", default="outputs/demo")
    e = commands.add_parser("evaluate", help="真实UTD前十类319条；嵌套LOSO完整重训")
    e.add_argument("--inertial-dir", required=True)
    e.add_argument("--index", default=None, help="可选CSV；保留参考记录顺序")
    e.add_argument("--output", default="outputs/benchmark", help="新的空目录；保留全部预测与失败")
    e.add_argument("--method", choices=["softmax", "svm", "ridge"], default="softmax")
    e.add_argument("--adc-bits", type=int, default=2, help="0为理想浮点，正式方案2")
    e.add_argument("--reset-memory", action="store_true")
    e.add_argument("--encoder-only", action="store_true")
    e.add_argument("--reads", type=int, default=40)
    p = commands.add_parser("predict", help="使用冻结部署模型推断一条完整MAT记录")
    p.add_argument("--model", required=True)
    p.add_argument("--mat", required=True)
    args = parser.parse_args(argv)
    try:
        with threadpool_limits(limits=1):
            if args.command == "demo":
                result = demo(args.output)
            elif args.command == "evaluate":
                from .data import load_public_records
                from .evaluation import evaluate
                x, y, subjects, metadata = load_public_records(args.inertial_dir, args.index)
                result = evaluate(x, y, subjects, device_models(), args.output, args.method,
                                  args.adc_bits, args.reset_memory, args.encoder_only, args.reads)
                Path(args.output, "input_manifest.json").write_text(
                    json.dumps(metadata, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
                result = {key: result[key] for key in ("completed", "method", "scope", "records",
                          "outer_results", "predictions", "seconds", "cached_results_read")} | {
                    "output_directory": str(args.output),
                    "conditions": {name: {key: values[key] for key in (
                        "subject_mean_accuracy_pct", "subject_sd_accuracy_pct", "pooled_correct",
                        "records", "pooled_accuracy_pct", "pooled_macro_f1_pct")}
                        for name, values in result["conditions"].items()},
                }
            else:
                from .inference import predict_mat
                result = predict_mat(args.mat, args.model)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except (ValueError, FileNotFoundError, FileExistsError) as exc:
        parser.exit(2, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
