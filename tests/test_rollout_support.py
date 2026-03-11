import json
import tempfile
import unittest
from pathlib import Path

from mario_world_model.actions import get_action_meanings
from mario_world_model.rollouts import Rollout, RolloutIndex, remove_rollout_from_jsonl
from scripts.collect_vector import HumanActionPolicy, VectorActionPolicy

import numpy as np


class RolloutSupportTests(unittest.TestCase):
    def test_vector_action_policy_uses_running_jump_during_recovery(self) -> None:
        action_meanings = get_action_meanings()
        policy = VectorActionPolicy(mode="heuristic", num_envs=1, action_meanings=action_meanings, seed=0)
        policy._jump_until[0] = 5
        policy._sprint_until[0] = 5

        action = int(policy.sample()[0])

        self.assertEqual(
            frozenset(token.lower() for token in action_meanings[action]),
            frozenset({"right", "a", "b"}),
        )

    def test_vector_action_policy_triggers_recovery_when_stuck(self) -> None:
        action_meanings = get_action_meanings()
        policy = VectorActionPolicy(mode="heuristic", num_envs=1, action_meanings=action_meanings, seed=1)
        info = {
            "world": np.asarray([1]),
            "stage": np.asarray([1]),
            "x_pos": np.asarray([96]),
            "time": np.asarray([300]),
        }

        for _ in range(14):
            policy.sample(info)

        self.assertGreater(policy._jump_until[0], policy._t[0])

    def test_human_action_policy_maps_up_to_standalone_up_action(self) -> None:
        action_meanings = get_action_meanings()
        policy = HumanActionPolicy.__new__(HumanActionPolicy)
        policy._action_to_index = {
            frozenset(token.lower() for token in action): idx
            for idx, action in enumerate(action_meanings)
        }
        policy._noop_index = policy._action_to_index[frozenset({"noop"})]

        action = policy._compose_action(
            right=False,
            left=False,
            up=True,
            down=False,
            jump=False,
            sprint=False,
        )

        self.assertEqual(action_meanings[action], ["up"])

    def test_human_action_policy_prioritizes_up_over_unsupported_combos(self) -> None:
        action_meanings = get_action_meanings()
        policy = HumanActionPolicy.__new__(HumanActionPolicy)
        policy._action_to_index = {
            frozenset(token.lower() for token in action): idx
            for idx, action in enumerate(action_meanings)
        }
        policy._noop_index = policy._action_to_index[frozenset({"noop"})]

        action = policy._compose_action(
            right=True,
            left=False,
            up=True,
            down=False,
            jump=False,
            sprint=False,
        )

        self.assertEqual(action_meanings[action], ["up"])

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

    def test_death_rollout_does_not_support_terminal_bin(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            data_dir = Path(tmp_dir)
            rollout = Rollout(
                world=1,
                stage=1,
                actions=[1, 2, 3],
                x_positions=[10, 80, 150],
                max_x=150,
                outcome="death",
                num_steps=3,
            )
            with (data_dir / "rollouts.jsonl").open("w", encoding="utf-8") as fh:
                fh.write(json.dumps(rollout.__dict__) + "\n")

            index = RolloutIndex(data_dir)

            reachable = index.reachable_bins(bin_size=64)
            self.assertIn((1, 1, 0), reachable)
            self.assertIn((1, 1, 1), reachable)
            self.assertNotIn((1, 1, 2), reachable)

            self.assertEqual(index.find_all_replay_actions(1, 1, 1, bin_size=64), [([1, 2, 3], 2)])
            self.assertEqual(index.find_all_replay_actions(1, 1, 2, bin_size=64), [])

    def test_remove_rollout_from_jsonl_deletes_matching_entry(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            jsonl = Path(tmp_dir) / "rollouts.jsonl"
            r1 = Rollout(world=1, stage=1, actions=[1, 2, 3], x_positions=[10, 80, 150], max_x=150, outcome="death", num_steps=3)
            r2 = Rollout(world=1, stage=2, actions=[4, 5], x_positions=[20, 100], max_x=100, outcome="flag", num_steps=2)
            r3 = Rollout(world=1, stage=1, actions=[6, 7, 8], x_positions=[5, 60, 200], max_x=200, outcome="death", num_steps=3)
            with jsonl.open("w", encoding="utf-8") as fh:
                for r in [r1, r2, r3]:
                    fh.write(r.to_json_line() + "\n")

            removed = remove_rollout_from_jsonl(jsonl, 1, 1, [1, 2, 3])
            self.assertTrue(removed)

            remaining = [Rollout.from_json_line(l) for l in jsonl.read_text().splitlines() if l.strip()]
            self.assertEqual(len(remaining), 2)
            self.assertEqual(remaining[0].actions, [4, 5])
            self.assertEqual(remaining[1].actions, [6, 7, 8])

    def test_remove_rollout_from_jsonl_no_match(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            jsonl = Path(tmp_dir) / "rollouts.jsonl"
            r1 = Rollout(world=1, stage=1, actions=[1, 2], x_positions=[10, 80], max_x=80, outcome="death", num_steps=2)
            with jsonl.open("w", encoding="utf-8") as fh:
                fh.write(r1.to_json_line() + "\n")

            removed = remove_rollout_from_jsonl(jsonl, 1, 1, [9, 9, 9])
            self.assertFalse(removed)

            remaining = [Rollout.from_json_line(l) for l in jsonl.read_text().splitlines() if l.strip()]
            self.assertEqual(len(remaining), 1)

    def test_remove_rollout_from_jsonl_only_removes_first(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            jsonl = Path(tmp_dir) / "rollouts.jsonl"
            r = Rollout(world=2, stage=3, actions=[1, 1], x_positions=[5, 10], max_x=10, outcome="death", num_steps=2)
            with jsonl.open("w", encoding="utf-8") as fh:
                fh.write(r.to_json_line() + "\n")
                fh.write(r.to_json_line() + "\n")

            removed = remove_rollout_from_jsonl(jsonl, 2, 3, [1, 1])
            self.assertTrue(removed)

            remaining = [Rollout.from_json_line(l) for l in jsonl.read_text().splitlines() if l.strip()]
            self.assertEqual(len(remaining), 1)


if __name__ == "__main__":
    unittest.main()