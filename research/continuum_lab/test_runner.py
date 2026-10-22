"""No external dependencies. Run: python -m unittest discover -s research/continuum_lab -v"""
import unittest
from runner import INPUTS, choose_probe, rules, run, survivors
import random


class ContinuumTests(unittest.TestCase):
    def test_unique_probes(self):
        for policy in ("random", "active"):
            obs = []
            for _ in INPUTS:
                x = choose_probe(obs, policy, random.Random(len(obs)))
                self.assertNotIn(x, [sample for sample, _ in obs])
                obs.append((x, 0))
            self.assertEqual(len(obs), len(INPUTS))

    def test_oracle_rule_never_passed_to_policy(self):
        observations = [((0, 0, 0), 0)]
        self.assertGreater(len(survivors(observations)), 1)
        self.assertEqual(len(rules()), 12)

    def test_reproducible_and_same_budget(self):
        a = run(trials=14, budget=3, seed=72)
        self.assertEqual(a, run(trials=14, budget=3, seed=72))
        for strategy in ("random", "active"):
            self.assertEqual(a["strategies"][strategy]["tested"], 14 * (8 - 3))
            self.assertEqual(len(a["strategies"][strategy]["episodes"]), 14)

    def test_invalid_budget(self):
        with self.assertRaises(ValueError):
            run(budget=8)


if __name__ == "__main__":
    unittest.main()
