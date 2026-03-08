import importlib.util
import sys
import unittest
from pathlib import Path


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


capacity_sweep = load_script_module(
    "sweep_dataset_capacity.py",
    "test_sweep_dataset_capacity",
)
capacity_sweep_genie = load_script_module(
    "sweep_dataset_capacity_genie.py",
    "test_sweep_dataset_capacity_genie",
)


class CapacitySweepInitialGlobalBestTests(unittest.TestCase):
    def test_override_is_used_without_completed_results(self) -> None:
        for module in (capacity_sweep, capacity_sweep_genie):
            with self.subTest(module=module.__name__):
                resolved = module.resolve_global_best_max(
                    completed_results={},
                    initial_global_best=37,
                    total_samples=100,
                )
                self.assertEqual(resolved, 37)

    def test_completed_results_take_precedence_when_larger(self) -> None:
        for module in (capacity_sweep, capacity_sweep_genie):
            with self.subTest(module=module.__name__):
                completed_results = {
                    "best_model": module.ModelResult(name="best_model", max_dataset_size=52),
                }
                resolved = module.resolve_global_best_max(
                    completed_results=completed_results,
                    initial_global_best=37,
                    total_samples=100,
                )
                self.assertEqual(resolved, 52)

    def test_override_is_clamped_to_total_samples(self) -> None:
        for module in (capacity_sweep, capacity_sweep_genie):
            with self.subTest(module=module.__name__):
                resolved = module.resolve_global_best_max(
                    completed_results={},
                    initial_global_best=150,
                    total_samples=100,
                )
                self.assertEqual(resolved, 100)

    def test_negative_override_is_rejected(self) -> None:
        for module in (capacity_sweep, capacity_sweep_genie):
            with self.subTest(module=module.__name__):
                with self.assertRaises(ValueError):
                    module.resolve_global_best_max(
                        completed_results={},
                        initial_global_best=-1,
                        total_samples=100,
                    )


if __name__ == "__main__":
    unittest.main()