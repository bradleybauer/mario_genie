import unittest

from mario_world_model.rollouts import EpisodeTracker


class RolloutTransitionTests(unittest.TestCase):
    def test_stage_transition_emits_prior_stage_rollout(self) -> None:
        tracker = EpisodeTracker(1)
        tracker.set_initial_info(0, world=1, stage=1, life=2)

        transition_rollout = tracker.record_step(
            0,
            action=3,
            x_pos=40,
            world=1,
            stage=1,
            life=2,
        )
        self.assertIsNone(transition_rollout)

        transition_rollout = tracker.record_step(
            0,
            action=4,
            x_pos=0,
            world=1,
            stage=2,
            life=2,
        )
        self.assertIsNotNone(transition_rollout)
        assert transition_rollout is not None
        self.assertEqual((transition_rollout.world, transition_rollout.stage), (1, 1))
        self.assertEqual(transition_rollout.actions, [3])
        self.assertEqual(transition_rollout.x_positions, [40])
        self.assertEqual(transition_rollout.outcome, "transition")

    def test_tracking_is_suspended_after_transition_until_reset(self) -> None:
        tracker = EpisodeTracker(1)
        tracker.set_initial_info(0, world=1, stage=1, life=2)

        tracker.record_step(0, action=3, x_pos=40, world=1, stage=1, life=2)
        tracker.record_step(0, action=4, x_pos=0, world=1, stage=2, life=2)
        tracker.record_step(0, action=5, x_pos=12, world=1, stage=2, life=2)

        self.assertIsNone(
            tracker.finish_episode(
                0,
                cur_life=2,
                flag_get=0,
                terminated=True,
                truncated=False,
            )
        )

        tracker.set_initial_info(0, world=2, stage=1, life=2)
        tracker.record_step(0, action=7, x_pos=8, world=2, stage=1, life=2)
        rollout = tracker.finish_episode(
            0,
            cur_life=2,
            flag_get=1,
            terminated=True,
            truncated=False,
        )
        self.assertIsNotNone(rollout)
        assert rollout is not None
        self.assertEqual((rollout.world, rollout.stage), (2, 1))
        self.assertEqual(rollout.actions, [7])
        self.assertEqual(rollout.outcome, "flag")


if __name__ == "__main__":
    unittest.main()