"""Training-only standardized linear readouts for ten-class RC experiments."""
from __future__ import annotations

import time
import warnings

import numpy as np
from scipy.linalg import eigh
from scipy.special import logsumexp
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from threadpoolctl import threadpool_limits


SEED = 2026100307
C_VALUES = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.]
LAMBDAS = [.001, .01, .1, 1., 10.]
CLASSES = np.arange(1, 11)


def _features(features):
    array = np.asarray(features, dtype=float)
    if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] == 0 or not np.isfinite(array).all():
        raise ValueError("Features must be a nonempty finite two-dimensional array")
    return array


def _indices(indices, count, *, minimum=1):
    array = np.asarray(indices)
    if array.ndim != 1 or len(array) < minimum or array.dtype.kind not in "iu":
        raise ValueError(f"Indices must be a one-dimensional integer array of length >= {minimum}")
    if np.any(array < 0) or np.any(array >= count) or len(np.unique(array)) != len(array):
        raise ValueError("Indices contain duplicate or out-of-range records")
    return array.astype(int, copy=False)


def _labels(labels, count, *, require_all=False):
    array = np.asarray(labels)
    if array.ndim != 1 or len(array) != count or array.dtype.kind not in "iuf" or not np.isfinite(array).all():
        raise ValueError("Labels must be a finite one-dimensional numeric array matching the records")
    if np.any(array != np.floor(array)) or np.any(array < 1) or np.any(array > 10):
        raise ValueError("Labels must be integer class IDs from 1 to 10")
    array = array.astype(int, copy=False)
    if require_all and not np.array_equal(np.unique(array), CLASSES):
        raise ValueError("Training requires all ten action classes; partial-class training is unsupported")
    return array


def training_span(features, train):
    """Retain every nonzero standardized training direction, using only train.

    Return ``phi, basis, mean, sample_sd, diagnostic``. This is the full training
    span, with the same numerical rank cutoff as the retained implementation.
    It does not tune or truncate a PCA dimension using validation records.
    """
    features = _features(features)
    train = _indices(train, len(features), minimum=2)
    mu = features[train].mean(0)
    sd = features[train].std(0, ddof=1)
    sd[sd < 1e-12] = 1.
    z = (features-mu)/sd
    gram = z[train] @ z[train].T
    values, vectors = eigh(gram, check_finite=False, driver="evr")
    cutoff = max(float(values[-1])*1e-12, 1e-10)
    keep = values > cutoff
    if not keep.any():
        raise ValueError("Training features have no nonzero standardized directions")
    basis = (z[train].T @ vectors[:, keep])/np.sqrt(values[keep])[None, :]
    phi = np.ascontiguousarray(z @ basis)
    reconstruction_error = float(np.max(np.abs(phi[train] @ phi[train].T-gram)))
    orthogonality_error = float(np.max(np.abs(basis.T @ basis-np.eye(keep.sum()))))
    if reconstruction_error >= 1e-6 or orthogonality_error >= 1e-7:
        raise ArithmeticError("Training-span reconstruction or orthogonality tolerance was exceeded")
    return phi, basis, mu, sd, dict(
        rank=int(keep.sum()), cutoff=cutoff,
        gram_reconstruction_error=reconstruction_error,
        basis_orthogonality_error=orthogonality_error,
    )


def softmax_gradient(phi, labels, weights, bias, c):
    """Gradient of mean cross-entropy plus the source L2 penalty."""
    scores = phi @ weights+bias
    prob = np.exp(scores-logsumexp(scores, axis=1)[:, None])
    target = np.eye(10)[labels-1]
    residual = (prob-target)/len(labels)
    return np.r_[(phi.T @ residual+weights/(len(labels)*c)).ravel(), residual.sum(0)]


def fit_candidates(phi, labels, method, cs):
    """Fit ordered C candidates with the original solver and warm-start rules.

    ``weights`` has shape (input columns, 10); ``bias`` has shape (10,).
    Multinomial Softmax keeps one estimator across candidates. SVM refits each
    candidate independently. All ten class IDs must be present in training.
    """
    phi = _features(phi)
    labels = _labels(labels, len(phi), require_all=True)
    if method not in {"LINEAR_SVM", "LINEAR_SOFTMAX"}:
        raise ValueError(f"Unsupported linear readout: {method}")
    cs = list(cs)
    if not cs or any(not np.isfinite(c) or c <= 0 for c in cs):
        raise ValueError("C candidates must be finite positive numbers")
    fits, estimator = [], None
    with threadpool_limits(limits=1):
        for c in cs:
            start = time.perf_counter()
            if method == "LINEAR_SVM":
                estimator = LinearSVC(
                    C=c, penalty="l2", loss="squared_hinge", dual=True,
                    multi_class="ovr", fit_intercept=True, intercept_scaling=1.,
                    tol=1e-7, max_iter=200000, random_state=SEED,
                )
            elif estimator is None:
                estimator = LogisticRegression(
                    C=c, penalty="l2", solver="lbfgs", fit_intercept=True,
                    tol=1e-9, max_iter=10000, warm_start=True,
                )
            else:
                estimator.set_params(C=c)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                estimator.fit(phi, labels)
            convergence = [str(w.message) for w in caught if issubclass(w.category, ConvergenceWarning)]
            if convergence:
                estimator.set_params(max_iter=500000)
                with warnings.catch_warnings(record=True) as retry:
                    warnings.simplefilter("always")
                    estimator.fit(phi, labels)
                if any(issubclass(w.category, ConvergenceWarning) for w in retry):
                    raise RuntimeError(f"{method} failed to converge for C={c}")
            if not np.array_equal(estimator.classes_, CLASSES):
                raise ValueError("The fitted readout does not contain all ten ordered classes")
            weights = estimator.coef_.T.copy()
            bias = estimator.intercept_.copy()
            gradient_max = (
                float(np.max(np.abs(softmax_gradient(phi, labels, weights, bias, c))))
                if method == "LINEAR_SOFTMAX" else None
            )
            if gradient_max is not None and gradient_max >= 2e-5:
                raise ArithmeticError(f"Softmax gradient tolerance exceeded for C={c}: {gradient_max}")
            scores = phi @ weights+bias
            predicted = scores.argmax(1)+1
            if not np.array_equal(predicted, estimator.predict(phi)):
                raise ArithmeticError("Exported linear scores disagree with the fitted estimator")
            fits.append(dict(
                C=c, weights=weights, bias=bias,
                iterations=int(np.max(estimator.n_iter_)),
                seconds=time.perf_counter()-start,
                softmax_gradient_max=gradient_max,
                initial_convergence_warnings=convergence,
                scores=scores, predicted=predicted, classes=estimator.classes_.copy(),
            ))
    return fits


def ridge_scores(f, y, train, evaluation, lambdas=LAMBDAS):
    """Source dual ridge with train-only scaling and an unpenalized bias."""
    f = _features(f)
    y = _labels(y, len(f))
    train = _indices(train, len(f), minimum=2)
    evaluation = _indices(evaluation, len(f))
    _labels(y[train], len(train), require_all=True)
    lambdas = list(lambdas)
    if not lambdas or any(not np.isfinite(lam) or lam <= 0 for lam in lambdas):
        raise ValueError("Ridge lambdas must be finite positive numbers")
    a = f[train]
    mu, sd = a.mean(0), a.std(0, ddof=1)
    sd[sd < 1e-12] = 1.
    z, q = (a-mu)/sd, (f[evaluation]-mu)/sd
    target = np.eye(10)[y[train]-1]
    bias = target.mean(0)
    values, vectors = eigh(z @ z.T, check_finite=False, driver="evr")
    values = np.maximum(values, 0)
    rhs, cross = vectors.T @ (target-bias), (q @ z.T) @ vectors
    return np.stack([cross @ (rhs/(values[:, None]+len(train)*lam))+bias for lam in lambdas])


def metric(y, predicted):
    """Ten-class confusion, percent accuracy, and zero-safe macro F1."""
    y = np.asarray(y)
    y = _labels(y, len(y))
    predicted = _labels(predicted, len(y))
    if not len(y):
        raise ValueError("Metrics require at least one record")
    confusion = np.zeros((10, 10), dtype=int)
    np.add.at(confusion, (y-1, predicted-1), 1)
    tp, true, called = confusion.diagonal(), confusion.sum(1), confusion.sum(0)
    precision = np.divide(tp, called, out=np.zeros(10, float), where=called > 0)
    recall = np.divide(tp, true, out=np.zeros(10, float), where=true > 0)
    f1 = np.divide(2*precision*recall, precision+recall, out=np.zeros(10, float), where=precision+recall > 0)
    return dict(
        n=int(len(y)), correct=int((y == predicted).sum()),
        accuracy_pct=float(100*np.mean(y == predicted)),
        macro_f1_pct=float(100*f1.mean()), confusion=confusion.tolist(),
    )
