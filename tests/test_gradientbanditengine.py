# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring
# pylint: disable=line-too-long

import unittest
from collections import Counter
from context import in3120


class TestGradientBanditEngine(unittest.TestCase):

    def _create_engine(self, actions, options=None, updates=None) -> in3120.GradientBanditEngine:
        options = options if options else in3120.GradientBanditEngine.Options()
        updates = updates if updates else []
        engine = in3120.GradientBanditEngine(actions, options)
        for action, reward in updates:
            engine.update(action, reward)
        return engine

    def test_initial_state(self):
        actions = ["a", "b", "c", "d"]
        engine = self._create_engine(actions)
        state = engine.state()
        self.assertEqual(state.baseline, 0)
        self.assertDictEqual(state.preferences, {a: 0.0 for a in actions})
        self.assertDictEqual(state.policy, {a: 0.25 for a in actions})

    def test_invalid_action_specifications(self):
        for actions in (None, [], ["a", "b", "a"]):
            with self.assertRaises(AssertionError):
                _ = in3120.GradientBanditEngine(actions, in3120.GradientBanditEngine.Options())

    def test_invalid_options(self):
        actions = ["a", "b", "c"]
        invalid = [
            None,
            in3120.GradientBanditEngine.Options(learning_rate=-0.1),
            in3120.GradientBanditEngine.Options(learning_rate=0.0),
            in3120.GradientBanditEngine.Options(discount_factor=-0.1),
            in3120.GradientBanditEngine.Options(discount_factor=0.0),
            in3120.GradientBanditEngine.Options(discount_factor=1.1),
            in3120.GradientBanditEngine.Options(temperature=-0.1),
            in3120.GradientBanditEngine.Options(temperature=0.0),
        ]
        for options in invalid:
            with self.assertRaises(AssertionError):
                _ = in3120.GradientBanditEngine(actions, options)

    def test_dynamic_reward_baseline(self):        
        actions = ["x", "y", "z"]
        updates = [("x", 1.0), ("y", 0.0), ("z", 0.0), ("x", 1.0), ("y", 0.0), ("z", 0.0)]
        options = [
            in3120.GradientBanditEngine.Options(dynamic_baseline=False),
            in3120.GradientBanditEngine.Options(dynamic_baseline=True, discount_factor=0.1),
            in3120.GradientBanditEngine.Options(dynamic_baseline=True, discount_factor=0.7),
        ]
        engines = [self._create_engine(actions, o, updates) for o in options]
        baselines = [engine.state().baseline for engine in engines]
        self.assertEqual(baselines[0], 0.0)
        self.assertGreater(baselines[1], 0.0)
        self.assertGreater(baselines[2], 0.0)
        self.assertGreater(baselines[1], baselines[2])

    def test_histogram(self):
        actions = ["b", "aa", "ccc"]
        updates = [("aa", 1.0), ("ccc", 9.0), ("b", 2.0), ("aa", -2.5), ("ccc", 0.7), ("b", 10.0)]
        engine = self._create_engine(actions, None, updates)
        histogram = engine.histogram()
        lines = histogram.split("\n")
        self.assertListEqual(sorted(actions), [line.split(" ")[0] for line in lines])
        self.assertTrue(all(line[3:5] == " (" for line in lines))
        self.assertTrue(all(line[10:12] == "):" for line in lines))
        self.assertTrue(all(line[i] == "*" for line in lines for i in range(13, len(line))))
        self.assertGreater(len(lines[1]), len(lines[0]))
        self.assertGreater(len(lines[1]), len(lines[2]))
        self.assertGreater(len(lines[2]), len(lines[0]))

    def test_update_with_unknown_action(self):
        engine = self._create_engine(["a", "b", "c"])
        with self.assertRaises(AssertionError):
            engine.update("x", 1.0)

    def test_basic_update(self):
        actions = ["a", "b", "c"]
        engine = self._create_engine(actions)
        state = engine.state()
        before = {a: (state.preferences[a], state.policy[a]) for a in actions}
        engine.update("b", 3.0)
        state = engine.state()
        after = {a: (state.preferences[a], state.policy[a]) for a in actions}
        self.assertTrue(before["a"] > after["a"])
        self.assertTrue(before["b"] < after["b"])
        self.assertTrue(before["c"] > after["c"])
        self.assertTrue(after["a"] == after["c"])
        self.assertEqual(1.0, sum(before[a][1] for a in actions))
        self.assertEqual(1.0, sum(after[a][1] for a in actions))

    def test_greedy_action_selection(self):
        actions = ["x", "y", "z"]
        updates = [("x", 1.0), ("y", 2.0), ("z", 0.0), ("x", 1.0), ("y", 1.0), ("z", -2.0)]
        engine = self._create_engine(actions, None, updates)
        for action, reward in updates:
            engine.update(action, reward)
        self._test_greedy_action_selection_with_invalid_epsilon(engine)
        self._test_greedy_action_selection_with_zero_epsilon(engine)
        self._test_greedy_action_selection_with_nonzero_epsilon(engine)
        self._test_greedy_action_selection_with_invalid_subset(engine)

    def _test_greedy_action_selection_with_invalid_epsilon(self, engine: in3120.GradientBanditEngine):
        for epsilon in (-0.1, 1.1):
            with self.assertRaises(AssertionError):
                _ = engine.greedy(epsilon=epsilon)

    def _test_greedy_action_selection_with_nonzero_epsilon(self, engine: in3120.GradientBanditEngine):
        distribution = Counter(engine.greedy(0.2) for _ in range(1000))
        self.assertEqual(3, len(distribution))
        self.assertTrue(distribution["x"] < distribution["y"])
        self.assertTrue(distribution["z"] < distribution["y"])
        distribution = Counter(engine.greedy(0.2, ["x", "y"]) for _ in range(1000))
        self.assertEqual(2, len(distribution))
        self.assertTrue(distribution["x"] < distribution["y"])

    def _test_greedy_action_selection_with_invalid_subset(self, engine: in3120.GradientBanditEngine):
        for subset in ([], ["w", "x", "y"]):
            with self.assertRaises(AssertionError):
                _ = engine.greedy(subset=subset)

    def _test_greedy_action_selection_with_zero_epsilon(self, engine: in3120.GradientBanditEngine):
        self.assertEqual(engine.greedy(), "y")
        self.assertEqual(engine.greedy(subset=["x", "y", "z"]), "y")
        self.assertEqual(engine.greedy(subset=["y", "z"]), "y")
        self.assertEqual(engine.greedy(subset=["x", "z"]), "x")

    def test_sampling_random_actions(self):
        actions = ["x", "y", "z"]
        updates = [("x", 0), ("y", 2), ("z", 7)]
        engine = self._create_engine(actions, None, updates)
        self._test_sampling_random_actions_with_no_subset(engine)
        self._test_sampling_random_actions_with_subset(engine)
        self._test_sampling_random_actions_with_invalid_subset(engine)
        self._test_sampling_random_actions_with_invalid_sample_size(engine)

    def _test_sampling_random_actions_with_no_subset(self, engine: in3120.GradientBanditEngine):
        distribution = Counter(engine.sample(1000))
        self.assertEqual(3, len(distribution))
        self.assertTrue(distribution["x"] < distribution["y"])
        self.assertTrue(distribution["y"] < distribution["z"])

    def _test_sampling_random_actions_with_subset(self, engine: in3120.GradientBanditEngine):
        distribution = Counter(engine.sample(1000, ["x", "z"]))
        self.assertEqual(2, len(distribution))
        self.assertTrue(distribution["x"] < distribution["z"])

    def _test_sampling_random_actions_with_invalid_sample_size(self, engine: in3120.GradientBanditEngine):
        for k in (-1, 0):
            with self.assertRaises(AssertionError):
                engine.sample(k)

    def _test_sampling_random_actions_with_invalid_subset(self, engine: in3120.GradientBanditEngine):
        for subset in ([], ["x", "w"]):
            with self.assertRaises(AssertionError):
                engine.sample(1000, subset)


if __name__ == '__main__':
    unittest.main(verbosity=2)
