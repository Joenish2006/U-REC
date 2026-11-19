"""Tests for register.py: head pose and the guided scan steps.

Owner: Nishanth
"""
import unittest

from urec.register import PoseGuide, head_pose


class PoseTest(unittest.TestCase):
    # landmarks: eye, eye, nose, mouth, mouth
    STRAIGHT = [[40, 40], [60, 40], [50, 52], [43, 65], [57, 65]]
    TURNED = [[40, 40], [60, 40], [56, 52], [43, 65], [57, 65]]
    TURNED_OTHER = [[40, 40], [60, 40], [44, 52], [43, 65], [57, 65]]

    def test_head_pose(self):
        yaw, pitch = head_pose(self.STRAIGHT)
        self.assertAlmostEqual(yaw, 0.0)
        self.assertAlmostEqual(pitch, 12 / 25)
        self.assertGreater(head_pose(self.TURNED)[0], 0.18)

    def test_guide_needs_both_directions(self):
        g = PoseGuide()
        self.assertTrue(g.ok("straight", *head_pose(self.STRAIGHT)))
        self.assertFalse(g.ok("yaw_a", *head_pose(self.STRAIGHT)))
        self.assertTrue(g.ok("yaw_a", *head_pose(self.TURNED)))
        self.assertFalse(g.ok("yaw_b", *head_pose(self.TURNED)))
        self.assertTrue(g.ok("yaw_b", *head_pose(self.TURNED_OTHER)))


if __name__ == "__main__":
    unittest.main()
