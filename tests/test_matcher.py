"""Tests for matcher.py: name or Unknown.

Owner: Joenish
"""
import unittest

import numpy as np

from tests.helpers import unit
from urec.matcher import Matcher


class MatcherTest(unittest.TestCase):
    def setUp(self):
        self.m = Matcher({"Nishanth": np.stack([unit(1, 0, 0), unit(0.9, 0.1, 0)]),
                          "Sathish": np.stack([unit(0, 1, 0)])}, threshold=0.5)

    def test_known_face(self):
        name, score = self.m.match(unit(1, 0.05, 0))
        self.assertEqual(name, "Nishanth")
        self.assertGreater(score, 0.9)

    def test_stranger_is_unknown(self):
        name, _ = self.m.match(unit(0, 0, 1))
        self.assertIsNone(name)

    def test_empty_gallery(self):
        self.assertEqual(Matcher({}, 0.5).match(unit(1, 0)), (None, 0.0))


if __name__ == "__main__":
    unittest.main()
