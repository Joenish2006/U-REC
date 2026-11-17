"""Unit tests that need no camera and no models. Run: python -m unittest

Owner: Thanigai
"""
import tempfile
import unittest
from pathlib import Path

import numpy as np

from urec.database import Database, clean_name
from urec.embedder import Face
from urec.identify import assign_faces
from urec.matcher import Matcher
from urec.register import PoseGuide, head_pose
from urec.tracker import Tracker, iou


def unit(*v):
    v = np.asarray(v, dtype=np.float32)
    return v / np.linalg.norm(v)


def face(box):
    return Face(box=box, score=0.9, landmarks=np.zeros((5, 2), np.float32), embedding=unit(1, 0))


class MatcherTest(unittest.TestCase):
    def setUp(self):
        self.m = Matcher({"Omkar": np.stack([unit(1, 0, 0), unit(0.9, 0.1, 0)]),
                          "Sathish": np.stack([unit(0, 1, 0)])}, threshold=0.5)

    def test_known_face(self):
        name, score = self.m.match(unit(1, 0.05, 0))
        self.assertEqual(name, "Omkar")
        self.assertGreater(score, 0.9)

    def test_stranger_is_unknown(self):
        name, _ = self.m.match(unit(0, 0, 1))
        self.assertIsNone(name)

    def test_empty_gallery(self):
        self.assertEqual(Matcher({}, 0.5).match(unit(1, 0)), (None, 0.0))


class DatabaseTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = Database(Path(self.tmp.name) / "t.db")

    def tearDown(self):
        self.db.close()
        self.tmp.cleanup()

    def test_save_list_delete_reset(self):
        self.db.save_person("Joenish", "opencv", np.ones((3, 4), np.float32))
        self.db.save_person("Omkar", "opencv", np.ones((2, 4), np.float32))
        self.assertTrue(self.db.has("joenish"))  # names are case-insensitive
        self.assertEqual([p[0] for p in self.db.people()], ["Joenish", "Omkar"])
        self.assertEqual(self.db.gallery("opencv")["Joenish"].shape, (3, 4))
        self.assertTrue(self.db.delete("Omkar"))
        self.assertFalse(self.db.delete("Omkar"))
        self.assertEqual(self.db.reset(), 1)
        self.assertEqual(self.db.gallery("opencv"), {})

    def test_rescan_replaces_only_that_backend(self):
        self.db.save_person("Joenish", "opencv", np.ones((3, 4), np.float32))
        self.db.save_person("Joenish", "insightface", np.ones((5, 8), np.float32))
        self.db.save_person("Joenish", "opencv", np.zeros((2, 4), np.float32))
        self.assertEqual(self.db.gallery("opencv")["Joenish"].shape, (2, 4))
        self.assertEqual(self.db.gallery("insightface")["Joenish"].shape, (5, 8))

    def test_clean_name(self):
        self.assertEqual(clean_name("  Nishanth   K "), "Nishanth K")
        with self.assertRaises(ValueError):
            clean_name("   ")


class TrackerTest(unittest.TestCase):
    def test_iou(self):
        self.assertAlmostEqual(iou([0, 0, 10, 10], [0, 0, 10, 10]), 1.0)
        self.assertEqual(iou([0, 0, 10, 10], [20, 20, 30, 30]), 0.0)

    def test_same_person_keeps_track_and_votes(self):
        t = Tracker(window=5)
        (a,) = t.update([[0, 0, 100, 200]])
        a.vote("Omkar", 0.7)
        (b,) = t.update([[5, 3, 104, 205]])
        self.assertIs(a, b)
        b.vote("Omkar", 0.8)
        self.assertEqual(b.label(min_votes=2), ("Omkar", 0.8))

    def test_one_vote_is_not_enough(self):
        (a,) = Tracker().update([[0, 0, 100, 200]])
        a.vote("Omkar", 0.7)
        self.assertEqual(a.label(min_votes=2), (None, 0.0))

    def test_unknown_votes_win_ties_against_names(self):
        (a,) = Tracker().update([[0, 0, 100, 200]])
        for name in ("Omkar", "Omkar", None, None, None):
            a.vote(name, 0.5)
        self.assertEqual(a.label(min_votes=2), (None, 0.0))

    def test_lost_track_is_dropped(self):
        t = Tracker(max_misses=2)
        t.update([[0, 0, 10, 10]])
        for _ in range(3):
            t.update([])
        self.assertEqual(t.tracks, [])


class AssignFacesTest(unittest.TestCase):
    def test_face_goes_to_body_around_it(self):
        boxes = [[0, 0, 100, 300], [200, 0, 300, 300]]
        self.assertEqual(assign_faces([face((230, 20, 270, 70))], boxes, (480, 640)), [1])

    def test_close_up_face_in_lower_half_of_body(self):
        # real numbers from a laptop webcam: person close to the camera
        boxes = [[382, 222, 1004, 712], [0, 308, 160, 714]]
        self.assertEqual(assign_faces([face((551, 359, 866, 733))], boxes, (720, 1280)), [0])
        self.assertEqual(len(boxes), 2)  # no extra estimated box

    def test_prefers_body_with_face_in_upper_part(self):
        # face of a person in front (box 1) overlapping the torso of someone behind (box 0)
        boxes = [[100, 0, 500, 600], [200, 250, 400, 700]]
        self.assertEqual(assign_faces([face((270, 270, 330, 340))], boxes, (720, 1280)), [1])

    def test_face_without_body_gets_estimated_box(self):
        boxes = []
        owners = assign_faces([face((300, 100, 360, 170))], boxes, (480, 640))
        self.assertEqual(owners, [0])
        x1, y1, x2, y2 = boxes[0]
        self.assertLess(x1, 300)
        self.assertGreater(y2, 170)


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
