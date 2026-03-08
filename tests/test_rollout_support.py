import json
import tempfile
import unittest
from pathlib import Path

from mario_world_model.rollouts import Rollout, RolloutIndex


class RolloutSupportTests(unittest.TestCase):
    def test_discontinuous_positions_do_not_create_phantom_bins(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            data_dir = Path(tmp_dir)
            rollout = Rollout(
                world=8,
                stage=4,
                actions=[1, 2, 3],
                x_positions=[10, 2560, 3900],
                max_x=3900,
                outcome="transition",
                num_steps=3,
            )
            with (data_dir / "rollouts.jsonl").open("w", encoding="utf-8") as fh:
                fh.write(json.dumps(rollout.__dict__) + "\n")

            index = RolloutIndex(data_dir)
            reachable = index.reachable_bins(bin_size=64)

            self.assertIn((8, 4, 0), reachable)
            self.assertIn((8, 4, 40), reachable)
            self.assertNotIn((8, 4, 1), reachable)
            self.assertNotIn((8, 4, 20), reachable)

    def test_find_all_replay_actions_requires_actual_bin_visit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            data_dir = Path(tmp_dir)
            rollout = Rollout(
                world=8,
                stage=4,
                actions=[1, 2, 3],
                x_positions=[10, 2560, 3900],
                max_x=3900,
                outcome="transition",
                num_steps=3,
            )
            with (data_dir / "rollouts.jsonl").open("w", encoding="utf-8") as fh:
                fh.write(json.dumps(rollout.__dict__) + "\n")

            index = RolloutIndex(data_dir)
            self.assertEqual(index.find_all_replay_actions(8, 4, 20, bin_size=64), [])
            self.assertEqual(index.find_all_replay_actions(8, 4, 40, bin_size=64), [([1, 2, 3], 2)])


if __name__ == "__main__":
    unittest.main()