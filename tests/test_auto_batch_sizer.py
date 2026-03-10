import io
import unittest
from contextlib import redirect_stdout
from types import SimpleNamespace
from unittest.mock import patch

import torch

from mario_world_model.auto_batch_sizer import (
    _ACTIVATION_RATIO,
    _estimate_memory_ceiling,
    find_max_batch_size,
)


class _DummyModel:
    def __init__(self) -> None:
        self.training = False

    def train(self, mode: bool = True) -> None:
        self.training = mode


class _DummyModelWithParams(_DummyModel):
    """Minimal model stub that exposes parameters() for estimation tests."""

    def __init__(self, param_bytes: int = 20 * 1024**2) -> None:
        super().__init__()
        # Create a flat parameter with the desired byte size (float32 → /4 elements)
        self._param = torch.nn.Parameter(torch.zeros(param_bytes // 4))

    def parameters(self):
        return iter([self._param])


# Shared patch that disables the memory estimation (returns None) so that
# existing binary-search tests are deterministic regardless of GPU presence.
_no_estimate = patch(
    "mario_world_model.auto_batch_sizer._estimate_memory_ceiling",
    return_value=(None, {}),
)


class AutoBatchSizerTests(unittest.TestCase):
    def test_uses_confirmed_fit_without_retesting_same_batch(self) -> None:
        model = _DummyModel()
        probed_batches: list[int] = []
        outcomes = {
            16: False,
            8: True,
            12: True,
            14: True,
            15: True,
        }

        def fake_try_batch(*args, **kwargs):
            batch_size = args[1]
            probed_batches.append(batch_size)
            if batch_size not in outcomes:
                raise AssertionError(f"Unexpected probe for batch size {batch_size}")
            return outcomes[batch_size]

        with (
            _no_estimate,
            patch("mario_world_model.auto_batch_sizer._try_batch", side_effect=fake_try_batch),
            patch("mario_world_model.auto_batch_sizer._clear_gpu"),
            patch(
                "torch.cuda.get_device_properties",
                return_value=SimpleNamespace(total_memory=24 * 1024**3),
            ),
            patch("torch.cuda.get_device_name", return_value="Fake GPU"),
        ):
            batch_size = find_max_batch_size(
                model,
                image_size=128,
                seq_len=16,
                device="cuda",
                floor=1,
                ceiling=16,
            )

        self.assertEqual(batch_size, 8)
        self.assertEqual(probed_batches, [16, 8, 12, 14, 15])
        self.assertFalse(model.training)

    def test_warns_only_after_explicit_floor_probe(self) -> None:
        model = _DummyModel()
        probed_batches: list[int] = []
        outcomes = {
            5: False,
            4: False,
        }

        def fake_try_batch(*args, **kwargs):
            batch_size = args[1]
            probed_batches.append(batch_size)
            if batch_size not in outcomes:
                raise AssertionError(f"Unexpected probe for batch size {batch_size}")
            return outcomes[batch_size]

        stdout = io.StringIO()
        with (
            _no_estimate,
            patch("mario_world_model.auto_batch_sizer._try_batch", side_effect=fake_try_batch),
            patch("mario_world_model.auto_batch_sizer._clear_gpu"),
            patch(
                "torch.cuda.get_device_properties",
                return_value=SimpleNamespace(total_memory=24 * 1024**3),
            ),
            patch("torch.cuda.get_device_name", return_value="Fake GPU"),
            redirect_stdout(stdout),
        ):
            batch_size = find_max_batch_size(
                model,
                image_size=128,
                seq_len=16,
                device="cuda",
                floor=4,
                ceiling=5,
            )

        self.assertEqual(batch_size, 4)
        self.assertEqual(probed_batches, [5, 4])
        self.assertIn("even batch_size=4 OOMs", stdout.getvalue())
        self.assertFalse(model.training)


class MemoryEstimationTests(unittest.TestCase):
    """Tests for the analytical memory-ceiling estimator."""

    def test_clamps_ceiling_for_small_gpu(self) -> None:
        """On an 8 GB GPU the ceiling should be far below 6976."""
        model = _DummyModelWithParams(param_bytes=20 * 1024**2)  # 20 MB
        device = torch.device("cuda")

        free = 6 * 1024**3  # 6 GB free
        total = 8 * 1024**3

        with (
            patch("torch.cuda.mem_get_info", return_value=(free, total)),
            patch("torch.cuda.get_device_name", return_value="FakeGPU-8GB"),
        ):
            est, info = _estimate_memory_ceiling(
                model, image_size=128, seq_len=16, device=device,
            )

        self.assertIsNotNone(est)
        # Should be a small number (roughly 10-30), certainly < 100
        self.assertGreater(est, 0)
        self.assertLess(est, 100)
        self.assertIn("gpu_name", info)
        self.assertAlmostEqual(info["total_gb"], 8.0, places=1)

    def test_larger_gpu_allows_higher_ceiling(self) -> None:
        """A 48 GB GPU should produce a much higher ceiling than 8 GB."""
        model = _DummyModelWithParams(param_bytes=20 * 1024**2)
        device = torch.device("cuda")

        small_free = 6 * 1024**3
        large_free = 42 * 1024**3

        with patch("torch.cuda.get_device_name", return_value="FakeGPU"):
            with patch("torch.cuda.mem_get_info", return_value=(small_free, 8 * 1024**3)):
                est_small, _ = _estimate_memory_ceiling(
                    model, image_size=128, seq_len=16, device=device,
                )
            with patch("torch.cuda.mem_get_info", return_value=(large_free, 48 * 1024**3)):
                est_large, _ = _estimate_memory_ceiling(
                    model, image_size=128, seq_len=16, device=device,
                )

        self.assertIsNotNone(est_small)
        self.assertIsNotNone(est_large)
        self.assertGreater(est_large, est_small * 3)

    def test_estimation_clamps_search_ceiling(self) -> None:
        """find_max_batch_size should use the estimate to reduce the ceiling."""
        model = _DummyModelWithParams(param_bytes=20 * 1024**2)
        probed_batches: list[int] = []

        def fake_try_batch(*args, **kwargs):
            bs = args[1]
            probed_batches.append(bs)
            return bs <= 8  # fits up to 8

        free = 6 * 1024**3
        total = 8 * 1024**3

        with (
            patch("mario_world_model.auto_batch_sizer._try_batch", side_effect=fake_try_batch),
            patch("mario_world_model.auto_batch_sizer._clear_gpu"),
            patch("torch.cuda.mem_get_info", return_value=(free, total)),
            patch("torch.cuda.get_device_name", return_value="FakeGPU-8GB"),
            patch(
                "torch.cuda.get_device_properties",
                return_value=SimpleNamespace(total_memory=total),
            ),
        ):
            bs = find_max_batch_size(
                model,
                image_size=128,
                seq_len=16,
                device="cuda",
                floor=1,
                ceiling=6976,  # absurdly high — should be clamped
            )

        # No probe should ever have been attempted at the absurd original ceiling
        self.assertTrue(
            all(b < 6976 for b in probed_batches),
            f"Probed at the unclamped ceiling: {probed_batches}",
        )
        # The highest probed batch must be well below the original 6976
        self.assertLess(max(probed_batches), 200)
        self.assertGreaterEqual(bs, 1)

    def test_returns_none_without_cuda(self) -> None:
        """Non-CUDA environments should get (None, {}) gracefully."""
        model = _DummyModel()
        device = torch.device("cpu")

        with patch("torch.cuda.mem_get_info", side_effect=RuntimeError("no CUDA")):
            est, info = _estimate_memory_ceiling(
                model, image_size=128, seq_len=16, device=device,
            )

        self.assertIsNone(est)


if __name__ == "__main__":
    unittest.main()