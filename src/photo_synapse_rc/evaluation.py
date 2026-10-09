"""Nested subject-grouped evaluation of the frozen model RC pipeline.

This is a public-database development re-evaluation, not a new blind test.
Every split refits its input and feature standardization using training only.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import time
from collections.abc import Mapping
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from . import core, readout


_METHODS = {"softmax": "LINEAR_SOFTMAX", "svm": "LINEAR_SVM", "ridge": "RIDGE"}
_SCOPE = (
    "Public-database development re-evaluation with nested subject-grouped "
    "LOSO; previously used database and encoder, not new independent blind "
    "data. Device currents are simulated from frozen measured-response fits."
)


def _json(path, value):
    # Exclusive creation also prevents an accidental concurrent overwrite.
    with Path(path).open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def _csv(path, rows, fields):
    with Path(path).open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _input(x, y, subjects, params, method, adc_bits, reset, encoder_only, reads):
    if method not in _METHODS:
        raise ValueError("method must be softmax, svm, or ridge")
    if np.iscomplexobj(x):
        raise ValueError("x must contain real prepared six-axis inputs")
    x = np.asarray(x, dtype=float)
    if x.ndim != 3 or x.shape[1:] != (40, 6) or len(x) < 2 or not np.isfinite(x).all():
        raise ValueError("x must be a finite prepared array of shape (records, 40, 6)")
    clean = []
    for name, values in [("y", y), ("subjects", subjects)]:
        values = np.asarray(values)
        if (
            values.ndim != 1 or len(values) != len(x) or values.dtype.kind not in "iuf"
            or not np.isfinite(values).all() or np.any(values != np.floor(values))
        ):
            raise ValueError(f"{name} must contain one integer ID per input record")
        clean.append(values.astype(int, copy=False))
    y, subjects = clean
    if not np.array_equal(np.unique(y), readout.CLASSES):
        raise ValueError("y must contain the ten class IDs 1 through 10")
    if not np.array_equal(np.unique(subjects), np.arange(1, 9)):
        raise ValueError("subjects must contain exactly the eight subject IDs 1 through 8")
    if isinstance(adc_bits, (bool, np.bool_)) or not isinstance(adc_bits, (int, np.integer)):
        raise ValueError("adc_bits must be an integer from 0 through 52")
    if not 0 <= int(adc_bits) <= 52:
        raise ValueError("adc_bits must be an integer from 0 through 52")
    if isinstance(reads, (bool, np.bool_)) or not isinstance(reads, (int, np.integer)):
        raise ValueError("reads must be an integer from 1 through 40")
    if not 1 <= int(reads) <= 40:
        raise ValueError("reads must be an integer from 1 through 40")
    if not isinstance(reset, (bool, np.bool_)) or not isinstance(encoder_only, (bool, np.bool_)):
        raise ValueError("reset and encoder_only must be boolean")
    if encoder_only:
        parameters = {"ENCODER": None}
    else:
        if not isinstance(params, Mapping) or not params:
            raise ValueError("params must map condition names to frozen device parameters")
        parameters = {}
        for name, value in params.items():
            if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", name):
                raise ValueError("Condition names must contain only letters, digits, underscore, or hyphen")
            if not isinstance(value, Mapping) or "tau_s" not in value or "gain_nA" not in value:
                raise ValueError(f"{name} parameters must contain tau_s and gain_nA")
            if np.iscomplexobj(value["tau_s"]) or np.iscomplexobj(value["gain_nA"]):
                raise ValueError(f"{name} parameters must be real")
            tau, gain = np.asarray(value["tau_s"], float), np.asarray(value["gain_nA"], float)
            if (
                tau.shape != (3,) or gain.shape != (3,) or not np.isfinite(tau).all()
                or not np.isfinite(gain).all() or np.any(tau <= 0) or np.any(gain < 0)
            ):
                raise ValueError(f"{name} requires three positive time constants and nonnegative gains")
            parameters[name] = {"tau_s": tau.tolist(), "gain_nA": gain.tolist()}
    return x, y, subjects, parameters


def _output(output, save_models):
    output = Path(output)
    if output.exists():
        if not output.is_dir() or any(output.iterdir()):
            raise FileExistsError("Evaluation output must be a new or empty directory; existing results are preserved")
    else:
        output.mkdir(parents=True, exist_ok=False)
    (output / "folds").mkdir(exist_ok=False)
    if save_models:
        (output / "outermodels").mkdir(exist_ok=False)
    return output


def _feature_sets(x, train, parameters, adc_bits, reset, encoder_only, reads):
    bits, scaler = core.bits_for(x, train)
    encoded = core.encode_xor(bits)
    features = {
        name: core.features(
            encoded, par, adc_bits=adc_bits, reset=reset, reads=reads,
            encoder_only=encoder_only,
        )
        for name, par in parameters.items()
    }
    return features, scaler


def _scaler_info(features, train):
    mean = features[train].mean(0)
    sd = features[train].std(0, ddof=1)
    sd[sd < 1e-12] = 1.0
    return mean, sd


def _training_span(features, train):
    # Reset-dark at a coarse ADC can be identically zero. Its valid classifier
    # is bias-only; a zero dummy coordinate lets the same solvers fit that bias.
    if np.all(features[train] == features[train][0]):
        mu, sd = _scaler_info(features, train)
        return (
            np.zeros((len(features), 1)), np.zeros((features.shape[1], 1)), mu, sd,
            dict(rank=0, cutoff=1e-10, gram_reconstruction_error=0.0,
                 basis_orthogonality_error=None, bias_only_dummy_coordinate=True),
        )
    return readout.training_span(features, train)


def _array_hash(array):
    array = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(array.shape).encode("ascii"))
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def evaluate(
    x, y, subjects, params, output, method="softmax", adc_bits=2,
    reset=False, encoder_only=False, reads=40,
):
    """Refit eight outer LOSO folds, with three grouped inner splits per fold.

    ``x`` contains already resampled and per-record prepared 40-by-6 signals.
    ``params`` maps condition names (normally LIGHT/DARK) to three-component
    ``tau_s`` and ``gain_nA`` arrays. Extra named conditions are supported.
    Record IDs in exported artifacts are one-based indices of the supplied x.

    A nonempty output directory is rejected. No cache or old predictions are
    consumed. On failure, completed folds remain and partial.json records the
    failure; summary.json is created only after the whole evaluation succeeds.
    """
    x, y, subjects, parameters = _input(
        x, y, subjects, params, method, adc_bits, reset, encoder_only, reads,
    )
    classifier = _METHODS[method]
    candidates = list(readout.LAMBDAS if method == "ridge" else readout.C_VALUES)
    parameter_name = "lambda" if method == "ridge" else "C"
    output = _output(output, save_models=method != "ridge")
    started = time.perf_counter()
    results, predictions = [], []
    held = None
    try:
        _json(output / "protocol.json", dict(
            method=method, classifier=classifier, scope=_SCOPE,
            records=len(x), subject_ids=list(range(1, 9)), classes=readout.CLASSES.tolist(),
            parameters=parameters, outer="8 leave-one-subject-out folds",
            inner="3 grouped folds; sorted remaining subjects assigned round-robin",
            selection="Equal subject mean accuracy; each of 7 validation subjects weighted 1/7; first candidate within 1e-10 percentage points of maximum",
            parameter_name=parameter_name, candidates=candidates,
            input="Prepared complete-record 40-by-6 signals; channel centering and acceleration/gyro group RMS precede this function",
            standardization="Input mean/sample SD, feature mean/sample SD, and training span refit only from each split's training records",
            encoding="Fixed 150 threshold streams; adjacent-time causal XOR; first stimulus zero",
            adc_bits=int(adc_bits), adc_fullscale_nA=3.0, reset=bool(reset),
            encoder_only=bool(encoder_only), reads=int(reads), features=150 * int(reads),
            slot_s=core.SLOT_S, delay_s=core.READ_DELAY_S, added_noise=False,
            index_base=1, cached_results_read=False,
            input_sha256=dict(x=_array_hash(x), y=_array_hash(y), subjects=_array_hash(subjects)),
        ))
        with threadpool_limits(limits=1):
            for held in range(1, 9):
                remaining = sorted(set(subjects.tolist()) - {held})
                groups = [remaining[j::3] for j in range(3)]
                validation_totals = {name: np.zeros(len(candidates)) for name in parameters}
                inner_rows = {name: [] for name in parameters}
                inner_splits = []
                for inner, group in enumerate(groups, 1):
                    train = np.flatnonzero((subjects != held) & ~np.isin(subjects, group))
                    validation = np.flatnonzero(np.isin(subjects, group))
                    if (
                        held in subjects[train] or held in subjects[validation]
                        or set(subjects[train]) & set(subjects[validation])
                    ):
                        raise ArithmeticError("Inner subject groups overlap")
                    features, scaler = _feature_sets(
                        x, train, parameters, adc_bits, reset, encoder_only, reads,
                    )
                    split = dict(
                        inner=inner, train_ids=(train + 1).tolist(),
                        validation_ids=(validation + 1).tolist(),
                        train_subjects=sorted(set(subjects[train].tolist())),
                        validation_subjects=group, input_scaler=scaler,
                        readout_standardization_fit_ids=(train + 1).tolist(), span_audits={},
                    )
                    for name, f in features.items():
                        if method == "ridge":
                            scores = readout.ridge_scores(f, y, train, validation, candidates)
                            diagnostics = [None] * len(candidates)
                            split["span_audits"][name] = dict(
                                solver="dual ridge on all training-standardized feature columns",
                                fit_records=len(train),
                            )
                        else:
                            phi, _, _, _, span = _training_span(f, train)
                            fits = readout.fit_candidates(phi[train], y[train], classifier, candidates)
                            scores = np.stack([
                                phi[validation] @ fit["weights"] + fit["bias"] for fit in fits
                            ])
                            diagnostics = fits
                            split["span_audits"][name] = span
                        predicted = scores.argmax(2) + 1
                        for ci, parameter in enumerate(candidates):
                            for subject in group:
                                keep = subjects[validation] == subject
                                correct = int(np.sum(predicted[ci, keep] == y[validation][keep]))
                                accuracy = 100.0 * correct / int(keep.sum())
                                validation_totals[name][ci] += accuracy / 7.0
                                row = dict(
                                    inner=inner, validation_subject=subject,
                                    parameter=float(parameter), parameter_name=parameter_name,
                                    correct=correct, total=int(keep.sum()), accuracy_pct=accuracy,
                                    training_records=len(train),
                                )
                                if diagnostics[ci] is not None:
                                    fit = diagnostics[ci]
                                    row.update(
                                        iterations=fit["iterations"],
                                        softmax_gradient_max=fit["softmax_gradient_max"],
                                        initial_convergence_warnings=fit["initial_convergence_warnings"],
                                    )
                                inner_rows[name].append(row)
                    inner_splits.append(split)

                train = np.flatnonzero(subjects != held)
                test = np.flatnonzero(subjects == held)
                features, scaler = _feature_sets(
                    x, train, parameters, adc_bits, reset, encoder_only, reads,
                )
                folds = []
                for name, f in features.items():
                    cv = validation_totals[name]
                    ci = int(np.flatnonzero(cv >= cv.max() - 1e-10)[0])
                    selected = float(candidates[ci])
                    if method == "ridge":
                        scores = readout.ridge_scores(f, y, train, test, [selected])[0]
                        mu, sd = _scaler_info(f, train)
                        span = dict(solver="dual ridge on all training-standardized feature columns", fit_records=len(train))
                        fit_info, model_file = {}, None
                    else:
                        phi, basis, mu, sd, span = _training_span(f, train)
                        # Fresh estimator with the chosen parameter: no inner-fit warm start.
                        fit = readout.fit_candidates(phi[train], y[train], classifier, [selected])[0]
                        weights = basis @ fit["weights"]
                        scores = phi[test] @ fit["weights"] + fit["bias"]
                        direct = ((f[test] - mu) / sd) @ weights + fit["bias"]
                        score_error = float(np.max(np.abs(direct - scores)))
                        if score_error >= 1e-7:
                            raise ArithmeticError("Exported full-feature scores differ from training-span scores")
                        mask, thresholds = core.encoder()
                        par = parameters[name]
                        model_file = f"outermodels/{classifier}_{name}_subject{held}.npz"
                        np.savez_compressed(
                            output / model_file, weights=weights, bias=fit["bias"], mu=mu, sd=sd,
                            feature_mean=mu, feature_sd=sd, basis_weights=fit["weights"],
                            input_mean=np.array(scaler["mean"]), input_sd=np.array(scaler["std"]),
                            mask=mask, thresholds=thresholds,
                            tau_s=np.array(par["tau_s"] if par is not None else [], float),
                            gain_nA=np.array(par["gain_nA"] if par is not None else [], float),
                            slot_s=core.SLOT_S, delay_s=core.READ_DELAY_S,
                            ADC_bits=int(adc_bits), ADC_fullscale_nA=3.0,
                            classes=readout.CLASSES, train_ids=train + 1, test_ids=test + 1,
                            C=selected, reads=int(reads), reset=bool(reset), encoder_only=bool(encoder_only),
                        )
                        fit_info = dict(
                            iterations=fit["iterations"],
                            softmax_gradient_max=fit["softmax_gradient_max"],
                            initial_convergence_warnings=fit["initial_convergence_warnings"],
                            linear_score_error=score_error,
                        )
                    predicted = scores.argmax(1) + 1
                    par = parameters[name]
                    fold = dict(
                        condition=name, classifier=classifier, held_subject=held,
                        features=f.shape[1], train_count=len(train), test_count=len(test),
                        train_ids=(train + 1).tolist(), test_ids=(test + 1).tolist(),
                        record_ids=(test + 1).tolist(), true=y[test].tolist(),
                        predicted=predicted.tolist(), scores=scores.tolist(),
                        tau_s=par["tau_s"] if par is not None else [],
                        gain_nA=par["gain_nA"] if par is not None else [],
                        input_scaler=scaler, span_audit=span,
                        feature_scaler=dict(mean=mu.tolist(), sample_sd=sd.tolist(), fit_records=len(train)),
                        selected_parameter=selected, parameter_name=parameter_name,
                        selected_at_grid_boundary=ci in [0, len(candidates) - 1],
                        inner_subject_mean_accuracy_pct=cv.tolist(),
                        inner_splits=inner_splits, inner_validation=inner_rows[name],
                        metric=readout.metric(y[test], predicted), model_file=model_file,
                        zero_fraction=float(np.mean(f[test] == 0)),
                        constant_train_columns=int(np.sum(f[train].std(0) < 1e-12)),
                        **fit_info,
                    )
                    folds.append(fold)
                    for record_id, truth, pred, score in zip(test + 1, y[test], predicted, scores):
                        row = dict(
                            classifier=classifier, condition=name, subject=held,
                            record_id=int(record_id), true=int(truth), predicted=int(pred),
                            correct=int(truth == pred),
                        )
                        row.update({f"score_{k}": float(score[k - 1]) for k in range(1, 11)})
                        predictions.append(row)
                _json(output / "folds" / f"subject{held}.json", folds)
                results.extend(folds)
                print(f"LOSO subject {held}/8 completed ({method}, {len(folds)} conditions)", flush=True)

            conditions = {}
            for name in parameters:
                folds = [fold for fold in results if fold["condition"] == name]
                record_ids = np.concatenate([fold["test_ids"] for fold in folds])
                if not np.array_equal(np.sort(record_ids), np.arange(1, len(x) + 1)):
                    raise ArithmeticError("Every record must receive exactly one outer prediction per condition")
                truth = np.concatenate([fold["true"] for fold in folds])
                predicted = np.concatenate([fold["predicted"] for fold in folds])
                pooled = readout.metric(truth, predicted)
                accuracy = np.array([fold["metric"]["accuracy_pct"] for fold in folds])
                f1 = np.array([fold["metric"]["macro_f1_pct"] for fold in folds])
                conditions[name] = dict(
                    subject_mean_accuracy_pct=float(accuracy.mean()),
                    subject_sd_accuracy_pct=float(accuracy.std(ddof=1)),
                    subject_mean_macro_f1_pct=float(f1.mean()),
                    subject_sd_macro_f1_pct=float(f1.std(ddof=1)),
                    pooled_correct=pooled["correct"], records=pooled["n"],
                    pooled_accuracy_pct=pooled["accuracy_pct"],
                    pooled_macro_f1_pct=pooled["macro_f1_pct"],
                    subject_metrics=[dict(subject=fold["held_subject"], **fold["metric"]) for fold in folds],
                    confusion=pooled["confusion"],
                )
                counts = np.array(pooled["confusion"], int)
                totals = counts.sum(1, keepdims=True)
                proportions = np.divide(counts, totals, out=np.zeros_like(counts, float), where=totals > 0)
                for kind, matrix in [("counts", counts), ("proportion", proportions)]:
                    rows = [dict(
                        true_class=i + 1,
                        **{f"predicted_{j + 1}": matrix[i, j].item() for j in range(10)},
                    ) for i in range(10)]
                    _csv(output / f"{name}_confusion_{kind}.csv", rows,
                         ["true_class"] + [f"predicted_{j}" for j in range(1, 11)])
            _csv(output / "predictions.csv", predictions,
                 ["classifier", "condition", "subject", "record_id", "true", "predicted", "correct"]
                 + [f"score_{j}" for j in range(1, 11)])
            summary = dict(
                completed=True, method=method, classifier=classifier, scope=_SCOPE,
                records=len(x), subjects=8, conditions=conditions,
                outer_results=len(results), predictions=len(predictions),
                adc_bits=int(adc_bits), adc_fullscale_nA=3.0, reads=int(reads),
                reset=bool(reset), encoder_only=bool(encoder_only),
                features=150 * int(reads), seconds=time.perf_counter() - started,
                cached_results_read=False,
            )
            _json(output / "summary.json", summary)
            return summary
    except Exception as exc:
        # The directory and successful folds remain useful audit evidence.
        partial = dict(
            completed=False, status="PARTIAL_FAILED", held_subject=held,
            completed_outer_results=len(results), error_type=type(exc).__name__,
            error=str(exc), scope=_SCOPE, seconds=time.perf_counter() - started,
        )
        try:
            _json(output / "partial.json", partial)
        except OSError:
            pass
        raise
