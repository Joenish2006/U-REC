"""Tests for tracker.py and identify.py: stable labels and face-to-body linking.

Owner: Sathish
"""
import unittest

from tests.helpers import face
from urec.identify import assign_faces
from urec.tracker import Tracker, iou


class TrackerTest(unittest.TestCase):
    def test_iou(self):
        self.assertAlmostEqual(iou([0, 0, 10, 10], [0, 0, 10, 10]), 1.0)
        self.assertEqual(iou([0, 0, 10, 10], [20, 20, 30, 30]), 0.0)

    def test_same_person_keeps_track_and_votes(self):
        t = Tracker(window=5)
        (a,) = t.update([[0, 0, 100, 200]])
        a.vote("Nishanth", 0.7)
        (b,) = t.update([[5, 3, 104, 205]])
        self.assertIs(a, b)
        b.vote("Nishanth", 0.8)
        self.assertEqual(b.label(min_votes=2), ("Nishanth", 0.8))

    def test_one_vote_is_not_enough(self):
        (a,) = Tracker().update([[0, 0, 100, 200]])
        a.vote("Nishanth", 0.7)
        self.assertEqual(a.label(min_votes=2), (None, 0.0))

    def test_unknown_votes_win_ties_against_names(self):
        (a,) = Tracker().update([[0, 0, 100, 200]])
        for name in ("Nishanth", "Nishanth", None, None, None):
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


if __name__ == "__main__":
    unittest.main()
