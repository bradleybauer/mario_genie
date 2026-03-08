import importlib.util
import sys
import unittest
from pathlib import Path
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def load_script_module(script_name: str, module_name: str):
    script_path = PROJECT_ROOT / "scripts" / script_name
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


capacity_sweep_genie = load_script_module(
    "sweep_dataset_capacity_genie.py",
    "test_sweep_dataset_capacity_genie_binary_search",
)


class CapacitySweepGenieBinarySearchTests(unittest.TestCase):
    def test_global_best_pass_continues_searching_upward(self) -> None:
        model = capacity_sweep_genie.MODEL_CONFIGS[0]
        probed_sizes: list[int] = []

        def fake_run_trial(model_config, dataset_size, *, dry_run, resume, **train_kwargs):
            probed_sizes.append(dataset_size)
            return capacity_sweep_genie.TrialResult(
                dataset_size=dataset_size,
                best_recon=0.0 if dataset_size <= 70 else 1.0,
                passed=dataset_size <= 70,
                run_name=f"run_{dataset_size}",
            )

        with patch.object(capacity_sweep_genie, "run_trial", side_effect=fake_run_trial):
            result = capacity_sweep_genie.binary_search_max_size(
                model,
                100,
                threshold=0.5,
                global_best_max=40,
                dry_run=False,
                resume=False,
                data_dir="data/human_play",
                output_dir="checkpoints/test_capacity_sweep_genie",
                max_minutes=1,
                lr=1e-4,
                patience=60,
                val_interval=10,
                max_batch_size=16,
                seed=0,
                num_workers=0,
            )

        self.assertGreaterEqual(len(probed_sizes), 2)
        self.assertEqual(probed_sizes[:2], [40, 70])
        self.assertEqual(result.max_dataset_size, 70)


if __name__ == "__main__":
    unittest.main()