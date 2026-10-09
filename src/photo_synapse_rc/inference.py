"""Inference using the frozen public-only Softmax deployments."""
from pathlib import Path
import hashlib

import numpy as np
from scipy.io import loadmat
from scipy.special import softmax

from .config import ACTION_LABELS
from .core import prepare, resample, encode_xor, features


def predict_raw(raw, model_path):
    """Predict one complete six-channel sequence; no rotation/label selection."""
    x = prepare(resample(raw))[None, :, :]
    with np.load(model_path, allow_pickle=False) as model:
        required = {"weights", "bias", "mu", "sd", "input_mean", "input_sd", "mask",
                    "thresholds", "tau_s", "gain_nA", "slot_s", "delay_s", "ADC_bits",
                    "ADC_fullscale_nA", "classes"}
        if not required.issubset(model.files):
            raise ValueError(f"Missing deployment arrays: {sorted(required-set(model.files))}")
        a = {key: model[key] for key in required}
        options = {key: model[key] for key in ("reads", "reset", "encoder_only") if key in model.files}
    for key, value in options.items():
        if value.shape != () or not np.isfinite(value):
            raise ValueError(f"Invalid scalar deployment setting: {key}")
    reads = int(options.get("reads", 40))
    if reads != float(options.get("reads", 40)) or not 1 <= reads <= 40:
        raise ValueError("Deployment reads must be an integer in 1..40")
    if any(float(options.get(key, False)) not in (0., 1.) for key in ("reset", "encoder_only")):
        raise ValueError("Deployment reset/encoder_only flags must be boolean")
    reset, encoder_only = bool(options.get("reset", False)), bool(options.get("encoder_only", False))
    feature_count = 150*reads
    shape = {"weights": (feature_count, 10), "bias": (10,), "mu": (feature_count,), "sd": (feature_count,),
             "input_mean": (6,), "input_sd": (6,), "mask": (150, 6),
             "thresholds": (150,), "tau_s": ((0,) if encoder_only else (3,)),
             "gain_nA": ((0,) if encoder_only else (3,)), "classes": (10,)}
    shape.update({key: () for key in ("slot_s", "delay_s", "ADC_bits", "ADC_fullscale_nA")})
    for key, value in a.items():
        if not np.isfinite(value).all() or (key in shape and value.shape != shape[key]):
            raise ValueError(f"Invalid deployment array: {key}")
    if not np.array_equal(a["classes"], np.arange(1, 11)):
        raise ValueError("Deployment must use ordered classes 1..10")
    if np.any(a["sd"] <= 0) or np.any(a["input_sd"] <= 0):
        raise ValueError("Deployment standard deviations must be positive")
    if float(a["ADC_fullscale_nA"]) != 3.:
        raise ValueError("This deployment schema uses a fixed 3 nA full-scale")
    adc_bits = int(a["ADC_bits"])
    if adc_bits != float(a["ADC_bits"]) or not 0 <= adc_bits <= 52:
        raise ValueError("Deployment ADC bits must be an integer in 0..52")
    z = np.clip((x-a["input_mean"])/a["input_sd"], -3, 3)/3
    encoded = encode_xor(z @ a["mask"].T > a["thresholds"])
    params = None if encoder_only else {"tau_s": a["tau_s"], "gain_nA": a["gain_nA"]}
    f = features(encoded, params, adc_bits, float(a["slot_s"]), float(a["delay_s"]),
                 reset=reset, reads=reads, encoder_only=encoder_only)
    scores = ((f-a["mu"])/a["sd"]) @ a["weights"]+a["bias"]
    prob = softmax(scores, axis=1)[0]
    predicted = int(scores.argmax(1)[0])+1
    return dict(predicted_class=predicted, predicted_action=ACTION_LABELS[predicted-1],
                scores=scores[0].tolist(), softmax_probabilities=prob.tolist(),
                model_sha256=hashlib.sha256(Path(model_path).read_bytes()).hexdigest(),
                feature_count=f.shape[1], records=1,
                scope="Full-record model inference; probabilities are not calibrated confidence.")


def predict_mat(mat_path, model_path):
    data = loadmat(mat_path)
    if "d_iner" not in data:
        raise ValueError("MAT must contain d_iner: finite N-by-6 acceleration/angular-velocity data")
    return predict_raw(data["d_iner"], model_path)
