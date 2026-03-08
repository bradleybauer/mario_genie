import io
import unittest
from contextlib import redirect_stdout
from types import SimpleNamespace
from unittest.mock import patch

from mario_world_model.auto_batch_sizer import find_max_batch_size


class _DummyModel:
    def __init__(self) -> None:
        self.training = False

    def train(self, mode: bool = True) -> None:
        self.training = mode


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


if __name__ == "__main__":
    unittest.main()