"""Read the official first-ten-action UTD-MHAD inertial recordings."""
from __future__ import annotations

import csv
import hashlib
import re
from pathlib import Path

import numpy as np
from scipy.io import loadmat

from .core import prepare, resample


_NAME = re.compile(r"a([1-9]\d*)_s([1-9]\d*)_t([1-9]\d*)_inertial\.mat")
EXPECTED_RECORDS = 319
MISSING_OFFICIAL_RECORD = "a8_s1_t4_inertial.mat"


def _identity(filename: str) -> tuple[int, int, int]:
    match = _NAME.fullmatch(filename)
    if match is None:
        raise ValueError(f"Invalid UTD-MHAD inertial filename: {filename!r}")
    action, subject, trial = map(int, match.groups())
    if not 1 <= subject <= 8 or not 1 <= trial <= 4:
        raise ValueError(f"Subject/trial outside the official dataset: {filename}")
    return action, subject, trial


def load_public_records(inertial_dir, index_path=None):
    """Return ``prepared, labels, subjects, metadata`` for all 319 real records.

    A supplied CSV controls record order through its ``file`` column. Original
    machine-specific ``source_path`` columns are ignored. Without a CSV, records
    are ordered numerically by action, subject, and trial. An archive's nested
    ``Inertial`` directory is accepted. No missing trial is fabricated.
    """
    root = Path(inertial_dir).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Inertial data directory does not exist: {root}")

    available = {}
    for path in root.rglob("*.mat"):
        if _NAME.fullmatch(path.name) is None:
            continue
        action, _, _ = _identity(path.name)
        if not 1 <= action <= 10:
            continue
        resolved = path.resolve()
        if not resolved.is_relative_to(root):
            raise ValueError(f"Inertial file resolves outside the data directory: {path.name}")
        if path.name in available:
            raise ValueError(f"Duplicate inertial filename: {path.name}")
        available[path.name] = resolved

    if index_path is None:
        records = [{"file": name} for name in sorted(available, key=_identity)]
    else:
        with Path(index_path).open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames or "file" not in reader.fieldnames:
                raise ValueError("Public index must have a file column")
            records = []
            seen = set()
            for row in reader:
                filename = row["file"]
                identity = _identity(filename)
                if filename in seen:
                    raise ValueError(f"Duplicate public-index filename: {filename}")
                seen.add(filename)
                for field, value in zip(("action_id", "subject_id", "trial_id"), identity):
                    if row.get(field, "") != "" and int(row[field]) != value:
                        raise ValueError(f"Index {field} disagrees with filename: {filename}")
                if 1 <= identity[0] <= 10:
                    records.append(row)

    missing = [row["file"] for row in records if row["file"] not in available]
    if missing:
        raise FileNotFoundError(f"Missing indexed inertial file(s): {', '.join(missing)}")
    if len(records) != EXPECTED_RECORDS:
        raise ValueError(f"Expected {EXPECTED_RECORDS} real first-ten-action recordings, found {len(records)}")
    selected = {row["file"] for row in records}
    if selected != set(available):
        raise ValueError("The index does not cover every available first-ten-action recording")
    expected = {
        f"a{a}_s{s}_t{t}_inertial.mat"
        for a in range(1, 11) for s in range(1, 9) for t in range(1, 5)
    } - {MISSING_OFFICIAL_RECORD}
    if selected != expected:
        raise ValueError("Record membership differs from the official 319-record subset")

    prepared, labels, subjects, metadata = [], [], [], []
    hashes = set()
    for record_id, row in enumerate(records, 1):
        filename = row["file"]
        path = available[filename]
        data = loadmat(path)
        if "d_iner" not in data:
            raise ValueError(f"Missing d_iner variable: {filename}")
        original = data["d_iner"]
        if np.iscomplexobj(original):
            raise ValueError(f"Complex-valued d_iner is unsupported: {filename}")
        raw = np.asarray(original, dtype=float)
        if raw.ndim != 2 or raw.shape[0] < 2 or raw.shape[1] != 6 or not np.isfinite(raw).all():
            raise ValueError(f"d_iner must be a finite N-by-6 matrix with N >= 2: {filename}")
        for field, value in (("sample_count", len(raw)), ("channel_count", 6)):
            if row.get(field, "") != "" and int(row[field]) != value:
                raise ValueError(f"Index {field} disagrees with d_iner: {filename}")
        action, subject, trial = _identity(filename)
        features = np.asarray(prepare(resample(raw)), dtype=float)
        if features.shape != (40, 6) or not np.isfinite(features).all():
            raise ValueError(f"Preprocessing produced invalid 40-by-6 data: {filename}")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        expected_digest = row.get("sha256", "")
        if expected_digest:
            if re.fullmatch(r"[0-9a-f]{64}", expected_digest) is None:
                raise ValueError(f"Index sha256 must be 64 lowercase hexadecimal characters: {filename}")
            if digest != expected_digest:
                raise ValueError(f"Index sha256 disagrees with the inertial file: {filename}")
        if digest in hashes:
            raise ValueError(f"Duplicate inertial file content: {filename}")
        hashes.add(digest)
        prepared.append(features)
        labels.append(action)
        subjects.append(subject)
        metadata.append(dict(
            record_id=record_id, file=filename, action_id=action,
            subject_id=subject, trial_id=trial, sample_count=len(raw),
            channel_count=6, nominal_elapsed_s=(len(raw)-1)/50.,
            sha256=digest,
        ))
    return np.stack(prepared), np.asarray(labels, dtype=int), np.asarray(subjects, dtype=int), metadata
