import unittest
from pathlib import Path
import tempfile

import numpy as np

from photo_synapse_rc.inference import predict_raw
from photo_synapse_rc.core import prepare, resample, encode_xor, features


class InferenceTests(unittest.TestCase):
    def test_frozen_models_are_self_contained_and_finite(self):
        root = Path(__file__).resolve().parents[1]
        t = np.linspace(0, 2*np.pi, 100)
        raw = np.column_stack([np.sin(t*(j+1))+0.1*np.cos(t*.7) for j in range(6)])
        for condition in ("LIGHT", "DARK"):
            with self.subTest(condition=condition):
                result = predict_raw(raw, root/"models"/f"{condition}_softmax_public319.npz")
                self.assertEqual(result["feature_count"], 6000)
                self.assertIn(result["predicted_class"], range(1, 11))
                self.assertEqual(len(result["scores"]), 10)
                p = np.array(result["softmax_probabilities"])
                self.assertTrue(np.isfinite(p).all())
                self.assertTrue(np.all(p >= 0))
                self.assertAlmostEqual(p.sum(), 1., places=12)

    def test_nonfinite_input_rejected(self):
        root = Path(__file__).resolve().parents[1]
        with self.assertRaises(ValueError):
            predict_raw(np.full((40, 6), np.nan), root/"models/LIGHT_softmax_public319.npz")

    def test_exported_variant_settings_are_honored(self):
        root = Path(__file__).resolve().parents[1]
        with np.load(root/"models/LIGHT_softmax_public319.npz", allow_pickle=False) as original:
            arrays = {key: original[key] for key in original.files}
        t = np.linspace(0, 3*np.pi, 180)
        raw = np.column_stack([np.sin((j+1)*t) for j in range(6)])
        x = prepare(resample(raw))[None, :, :]
        z = np.clip((x-arrays["input_mean"])/arrays["input_sd"], -3, 3)/3
        events = encode_xor(z @ arrays["mask"].T > arrays["thresholds"])
        rng = np.random.default_rng(91)
        for reads, reset, encoder_only in [(40, True, False), (40, False, True), (4, False, False)]:
            with self.subTest(reads=reads, reset=reset, encoder_only=encoder_only):
                a = dict(arrays)
                a.update(reads=np.array(reads), reset=np.array(reset), encoder_only=np.array(encoder_only))
                a.update(weights=rng.normal(0, .1, (150*reads, 10)),
                         mu=np.zeros(150*reads), sd=np.ones(150*reads))
                if encoder_only:
                    a["tau_s"], a["gain_nA"] = np.array([]), np.array([])
                par = None if encoder_only else {"tau_s": a["tau_s"], "gain_nA": a["gain_nA"]}
                expected = features(events, par, int(a["ADC_bits"]), float(a["slot_s"]),
                                    float(a["delay_s"]), reset=reset, reads=reads,
                                    encoder_only=encoder_only) @ a["weights"]+a["bias"]
                with tempfile.TemporaryDirectory() as temp:
                    model = Path(temp)/"variant.npz"
                    np.savez_compressed(model, **a)
                    result = predict_raw(raw, model)
                np.testing.assert_allclose(result["scores"], expected[0], rtol=0, atol=1e-12)
                self.assertEqual(result["feature_count"], 150*reads)


if __name__ == "__main__":
    unittest.main()
