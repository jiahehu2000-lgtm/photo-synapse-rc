"""Frozen device-memory RC preprocessing and features / 冻结器件记忆 RC 核心。

The three exponential states are linear and independent. Digital thresholding,
temporal XOR and ADC quantization supply the pipeline's nonlinear operations.
三指数状态为独立线性动态；非线性来自数字阈值、时间 XOR 和 ADC 量化。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]
DeviceParams = Mapping[str, Sequence[float] | FloatArray]
SLOT_S = 0.13906
READ_DELAY_S = 0.07416


def _array(value: Any, name: str, ndim: int | None = None) -> FloatArray:
    """Require a finite real array / 检查有限实数数组。"""
    try:
        if np.iscomplexobj(value):
            raise ValueError(f"{name} must contain real values / 必须为实数")
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a numeric real array / 必须为数值实数数组") from exc
    if ndim is not None and result.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions / 数组维数不符")
    if result.size == 0 or not np.isfinite(result).all():
        raise ValueError(f"{name} must be nonempty and finite / 数组不能为空且必须有限")
    return result


def _integer(value: Any, name: str, minimum: int = 1) -> int:
    """Require an integer in range / 检查整数取值范围。"""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer / 必须为整数")
    value = int(value)
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum} / 超出合法范围")
    return value


def _scalar(value: Any, name: str, *, zero_allowed: bool = False) -> float:
    """Require finite positive timing or scale / 检查有限时间或量程。"""
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a real scalar / 必须为实数标量")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a real scalar / 必须为实数标量") from exc
    if not np.isfinite(result) or result < 0 or (result == 0 and not zero_allowed):
        rule = "nonnegative" if zero_allowed else "positive"
        raise ValueError(f"{name} must be finite and {rule} / 必须有限且在合法范围内")
    return result


def _binary(value: Any, name: str) -> BoolArray:
    """Require binary record/time/branch data / 检查记录×时间×支路二值数据。"""
    result = _array(value, name, 3)
    if not np.all((result == 0) | (result == 1)):
        raise ValueError(f"{name} must contain only 0 and 1 / 只能包含 0 和 1")
    return result.astype(np.bool_, copy=False)


def _parameters(params: DeviceParams) -> tuple[FloatArray, FloatArray]:
    """Require the three frozen exponential components / 检查三指数参数。"""
    if not isinstance(params, Mapping) or "tau_s" not in params or "gain_nA" not in params:
        raise ValueError("params must contain tau_s and gain_nA / 参数缺少时间常数或增益")
    tau = _array(params["tau_s"], "tau_s", 1)
    gain = _array(params["gain_nA"], "gain_nA", 1)
    if tau.shape != (3,) or gain.shape != (3,):
        raise ValueError("tau_s and gain_nA must each have three components / 参数须各有三个分量")
    if np.any(tau <= 0) or np.any(gain < 0):
        raise ValueError("tau_s must be positive and gain_nA nonnegative / 时间常数须为正且增益非负")
    return tau, gain


def resample(raw: Any, steps: int = 40) -> FloatArray:
    """Cumulative trapezoid phase-bin averages / 累计梯形积分后按动作相位分段平均。"""
    raw = _array(raw, "raw", 2)
    steps = _integer(steps, "steps")
    if raw.shape[1] != 6 or raw.shape[0] < 2:
        raise ValueError("raw must have six channels and at least two samples / 原始数据须为六路且至少两点")
    cumulative = np.vstack(
        [np.zeros((1, 6)), np.cumsum((raw[:-1] + raw[1:]) / 2, axis=0)]
    )
    edges = np.linspace(0, len(raw) - 1, steps + 1)
    sampled = np.stack(
        [np.interp(edges, np.arange(len(raw)), cumulative[:, i]) for i in range(6)],
        axis=1,
    )
    result = np.diff(sampled, axis=0) * steps / (len(raw) - 1)
    if not np.isfinite(result).all():
        raise ValueError("resample overflowed / 重采样数值溢出")
    return result


def prepare(x: Any) -> FloatArray:
    """Center channels and apply acceleration/gyro group RMS / 各通道去均值后分别归一化加速度与角速度组 RMS。"""
    x = _array(x, "x", 2)
    if x.shape[1] != 6:
        raise ValueError("x must have six channels / 输入须为六路")
    result = x - x.mean(axis=0)
    for group in [slice(0, 3), slice(3, 6)]:
        rms = np.sqrt(np.mean(result[:, group] ** 2))
        result[:, group] /= rms if rms >= 1e-12 else 1.0
    if not np.isfinite(result).all():
        raise ValueError("prepare overflowed / 输入归一化数值溢出")
    return result


def encoder() -> tuple[FloatArray, FloatArray]:
    """Return the fixed 54 axis and 96 sign-projection streams / 返回固定 54 路单轴及 96 路符号投影编码。"""
    mask = np.repeat(np.eye(6), 9, axis=0)
    thresholds = np.tile(np.linspace(-0.8, 0.8, 9), 6)
    signs = np.ones((32, 6))
    for bit in range(1, 6):
        signs[:, bit] = 2 * ((np.arange(32) >> (5 - bit)) & 1) - 1
    return (
        np.vstack([mask, np.repeat(signs / np.sqrt(6), 3, axis=0)]),
        np.r_[thresholds, np.tile([-0.3, 0.0, 0.3], 32)],
    )


def bits_for(x: Any, train: Sequence[int] | NDArray[np.integer]) -> tuple[BoolArray, dict[str, Any]]:
    """Fit channel mean/sample SD on training records, then threshold / 仅用训练记录拟合输入均值和样本标准差并编码。"""
    x = _array(x, "x", 3)
    if x.shape[2] != 6:
        raise ValueError("x must have shape (records, steps, 6) / 输入须为记录×时间×六路")
    indices = np.asarray(train)
    if indices.ndim != 1 or indices.size == 0 or indices.dtype.kind not in "iu":
        raise ValueError("train must be a nonempty 1D integer index array / 训练索引须为非空一维整数")
    if np.any(indices < 0) or np.any(indices >= len(x)) or len(np.unique(indices)) != len(indices):
        raise ValueError("train indices must be unique and within record bounds / 训练索引须唯一且不越界")
    selected = x[indices].reshape(-1, 6)
    if len(selected) < 2:
        raise ValueError("training requires at least two time samples / 训练至少需要两个时间采样点")
    mean = selected.mean(axis=0)
    sd = selected.std(axis=0, ddof=1)
    sd[sd < 1e-12] = 1.0
    z = np.clip((x - mean) / sd, -3, 3) / 3
    mask, thresholds = encoder()
    bits = z @ mask.T > thresholds
    return bits, {"mean": mean.tolist(), "std": sd.tolist(), "fit_records": len(indices)}


def encode_xor(bits: Any) -> BoolArray:
    """Causal adjacent-time XOR with the first stimulus zero / 相邻时间因果 XOR，首步刺激固定为零。"""
    bits = _binary(bits, "bits")
    result = np.zeros_like(bits)
    result[:, 1:] = bits[:, 1:] != bits[:, :-1]
    return result


def current(
    encoded: Any,
    params: DeviceParams,
    slot: float = SLOT_S,
    delay: float = READ_DELAY_S,
    reset: bool = False,
) -> FloatArray:
    """Simulate independent linear three-exponential states / 模拟独立线性三指数状态，各动作从零开始。"""
    encoded = _binary(encoded, "encoded")
    tau, gain = _parameters(params)
    slot = _scalar(slot, "slot")
    delay = _scalar(delay, "delay", zero_allowed=True)
    if not isinstance(reset, (bool, np.bool_)):
        raise ValueError("reset must be boolean / reset 须为布尔值")
    state = np.zeros((len(encoded), encoded.shape[2], 3))
    result = np.zeros(encoded.shape, dtype=np.float64)
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        decay = np.exp(-slot / tau)
        read = np.exp(-delay / tau)
        for step in range(encoded.shape[1]):
            state = (0 if reset else state * decay) + encoded[:, step, :, None] * gain
            result[:, step] = np.sum(state * read, axis=2)
    if not np.isfinite(result).all():
        raise ValueError("current overflowed / 器件电流模拟数值溢出")
    return result


def quantize(a: Any, bits: int = 2, full_scale_nA: float = 3.0) -> FloatArray:
    """Nearest uniform ADC levels with clipping; bits=0 preserves floats / 最近均匀 ADC 等级并裁剪，0 位保留浮点参考。"""
    a = _array(a, "current")
    bits = _integer(bits, "bits", minimum=0)
    full_scale_nA = _scalar(full_scale_nA, "full_scale_nA")
    if bits > 52:
        raise ValueError("bits must be <= 52 for float64 levels / 位宽超出 float64 可表达等级范围")
    if bits == 0:
        return a.copy()
    levels = 2**bits - 1
    return np.floor(np.clip(a / full_scale_nA, 0, 1) * levels + 0.5) * full_scale_nA / levels


def features(
    encoded: Any,
    params: DeviceParams | None,
    adc_bits: int = 2,
    slot: float = SLOT_S,
    delay: float = READ_DELAY_S,
    reset: bool = False,
    reads: int = 40,
    *,
    encoder_only: bool = False,
) -> FloatArray:
    """Flatten branch-major, time-fast observations / 按支路优先、时间最快展平特征；150×40 得到 6000 维。"""
    encoded = _binary(encoded, "encoded")
    reads = _integer(reads, "reads")
    if reads > encoded.shape[1]:
        raise ValueError("reads cannot exceed the number of time steps / 读取次数不能多于时间步数")
    if not isinstance(encoder_only, (bool, np.bool_)):
        raise ValueError("encoder_only must be boolean / encoder_only 须为布尔值")
    if encoder_only:
        observed = encoded.astype(np.float64)
    else:
        if params is None:
            raise ValueError("device features require params / 器件特征需要模型参数")
        observed = quantize(current(encoded, params, slot, delay, reset), adc_bits)
    indices = (
        np.arange(encoded.shape[1])
        if reads == encoded.shape[1]
        else np.rint(np.linspace(0, encoded.shape[1] - 1, reads)).astype(int)
    )
    return observed[:, indices, :].transpose(0, 2, 1).reshape(len(encoded), -1)
