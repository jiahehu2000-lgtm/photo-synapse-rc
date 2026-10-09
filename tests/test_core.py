"""Numerical implementation checks / 数值实现检查，非科学结论证明。"""

import unittest

import numpy as np

from photo_synapse_rc.core import (
    bits_for,
    current,
    encode_xor,
    encoder,
    features,
    prepare,
    quantize,
    resample,
)


PARAMS = {"tau_s": [0.08, 0.8, 6.3], "gain_nA": [1.2, 0.2, 0.03]}


class CoreTests(unittest.TestCase):
    def test_recurrence_matches_independent_convolution(self):
        """Replay an event kernel sum without recursive state updates."""
        events = np.random.default_rng(7).integers(0, 2, size=(3, 40, 5))
        tau = np.array(PARAMS["tau_s"])
        gain = np.array(PARAMS["gain_nA"])
        expected = np.zeros(events.shape, float)
        slot, delay = 0.13906, 0.07416
        for time in range(40):
            for past in range(time + 1):
                kernel = np.sum(gain * np.exp(-((time - past) * slot + delay) / tau))
                expected[:, time] += events[:, past] * kernel
        np.testing.assert_allclose(current(events, PARAMS), expected, atol=1e-14, rtol=1e-14)
        no_memory = events * np.sum(gain * np.exp(-delay / tau))
        np.testing.assert_allclose(current(events, PARAMS, reset=True), no_memory, atol=1e-14)

    def test_adc_half_levels_and_clipping(self):
        values = [-1.0, 0.0, 0.5, 1.5, 2.5, 3.0, 4.0]
        np.testing.assert_array_equal(quantize(values), [0, 0, 1, 2, 3, 3, 3])
        np.testing.assert_array_equal(quantize(values, bits=0), values)
        np.testing.assert_array_equal(quantize([0.0, 1.0, 2.0], 1, 2.0), [0, 2, 2])

    def test_fixed_mask_and_threshold_bank(self):
        mask, thresholds = encoder()
        self.assertEqual(mask.shape, (150, 6))
        self.assertEqual(thresholds.shape, (150,))
        np.testing.assert_array_equal(mask[:54], np.repeat(np.eye(6), 9, axis=0))
        np.testing.assert_array_equal(thresholds[:54], np.tile(np.linspace(-0.8, 0.8, 9), 6))
        projections = mask[54::3]
        self.assertEqual(len(np.unique(projections, axis=0)), 32)
        np.testing.assert_allclose(projections[:, 0], np.ones(32) / np.sqrt(6))
        expected = np.array(
            [[1] + [1 if number & (1 << bit) else -1 for bit in range(4, -1, -1)] for number in range(32)]
        ) / np.sqrt(6)
        np.testing.assert_array_equal(projections, expected)
        np.testing.assert_array_equal(mask[54:], np.repeat(expected, 3, axis=0))
        np.testing.assert_array_equal(thresholds[54:], np.tile([-0.3, 0, 0.3], 32))

    def test_xor_is_causal_and_resets_each_record(self):
        bits = np.array([[[1, 0], [1, 1], [0, 1], [1, 0]], [[0, 1], [1, 1], [1, 0], [0, 0]]])
        expected = np.array([[[0, 0], [0, 1], [1, 0], [1, 1]], [[0, 0], [1, 0], [0, 1], [1, 0]]], bool)
        np.testing.assert_array_equal(encode_xor(bits), expected)
        changed = bits.copy()
        changed[:, 2:] = 1 - changed[:, 2:]
        np.testing.assert_array_equal(encode_xor(changed)[:, :2], expected[:, :2])
        np.testing.assert_array_equal(encode_xor(bits[:, :1]), np.zeros((2, 1, 2), bool))

    def test_resample_constant_and_linear_signal(self):
        levels = np.arange(1.0, 7.0)
        np.testing.assert_allclose(resample(np.tile(levels, (81, 1))), np.tile(levels, (40, 1)))
        # Integer bin edges at 0,2,...,80 integrate a linear signal exactly.
        raw = np.tile(np.arange(81.0)[:, None], (1, 6))
        expected = np.tile(np.arange(1.0, 80.0, 2.0)[:, None], (1, 6))
        np.testing.assert_allclose(resample(raw), expected, atol=1e-12)
        # A manual trapezoid accumulation also covers fractional phase edges.
        raw = np.random.default_rng(19).normal(size=(57, 6))
        integrated = np.zeros_like(raw)
        for row in range(1, len(raw)):
            integrated[row] = integrated[row - 1] + (raw[row - 1] + raw[row]) / 2
        edge_values = []
        for edge in np.linspace(0, 56, 41):
            left = int(np.floor(edge))
            right = min(left + 1, 56)
            edge_values.append(integrated[left] + (edge - left) * (integrated[right] - integrated[left]))
        expected = np.diff(edge_values, axis=0) * 40 / 56
        np.testing.assert_allclose(resample(raw), expected, atol=1e-12)

    def test_prepare_group_rms_and_input_preservation(self):
        raw = np.random.default_rng(3).normal(size=(40, 6))
        before = raw.copy()
        result = prepare(raw)
        np.testing.assert_array_equal(raw, before)
        np.testing.assert_allclose(result.mean(axis=0), 0, atol=1e-14)
        for group in [slice(0, 3), slice(3, 6)]:
            self.assertAlmostEqual(float(np.mean(result[:, group] ** 2)), 1.0)
        np.testing.assert_array_equal(prepare(np.ones((40, 6))), np.zeros((40, 6)))

    def test_input_scaler_uses_training_records_only(self):
        x = np.random.default_rng(23).normal(size=(4, 40, 6))
        bits, scaler = bits_for(x, [0, 2])
        selected = x[[0, 2]].reshape(-1, 6)
        np.testing.assert_array_equal(scaler["mean"], selected.mean(0))
        np.testing.assert_array_equal(scaler["std"], selected.std(0, ddof=1))
        self.assertEqual(scaler["fit_records"], 2)
        changed = x.copy()
        changed[[1, 3]] = 1e6
        new_bits, new_scaler = bits_for(changed, [0, 2])
        self.assertEqual(scaler, new_scaler)
        np.testing.assert_array_equal(bits[[0, 2]], new_bits[[0, 2]])

    def test_feature_flatten_order_and_read_subsets(self):
        events = np.zeros((1, 40, 150), bool)
        events[0, 0, 0] = True
        events[0, 3, 1] = True
        observed = current(events, PARAMS)
        full = features(events, PARAMS, adc_bits=0)
        self.assertEqual(full.shape, (1, 6000))
        for branch in range(150):
            np.testing.assert_array_equal(full[0, branch * 40 : (branch + 1) * 40], observed[0, :, branch])
        selected = np.rint(np.linspace(0, 39, 10)).astype(int)
        reduced = features(events, PARAMS, adc_bits=0, reads=10)
        np.testing.assert_array_equal(reduced, observed[:, selected].transpose(0, 2, 1).reshape(1, 1500))
        bypass = features(events, None, encoder_only=True)
        np.testing.assert_array_equal(bypass, events.transpose(0, 2, 1).reshape(1, 6000))

    def test_invalid_data_indices_and_parameters(self):
        x = np.ones((3, 40, 6))
        invalid = [[], [0, 0], [-1], [3], [0.0], [[0]], [True]]
        for indices in invalid:
            with self.subTest(indices=indices), self.assertRaises(ValueError):
                bits_for(x, indices)
        with self.assertRaises(ValueError):
            resample(np.ones((1, 6)))
        with self.assertRaises(ValueError):
            prepare(np.full((40, 6), np.nan))
        with self.assertRaises(ValueError):
            encode_xor(np.full((1, 40, 150), 0.5))
        with self.assertRaises(ValueError):
            current(np.zeros((1, 40, 150)), {"tau_s": [0, 1, 2], "gain_nA": [1, 1, 1]})
        with self.assertRaises(ValueError):
            current(np.zeros((1, 40, 150)), {"tau_s": [1, 2], "gain_nA": [1, 1]})
        with self.assertRaises(ValueError):
            features(np.zeros((1, 40, 150)), PARAMS, reads=41)
        with self.assertRaises(ValueError):
            quantize([1.0], bits=-1)
        with self.assertRaises(ValueError):
            quantize([1.0], full_scale_nA=0)


if __name__ == "__main__":
    unittest.main()
