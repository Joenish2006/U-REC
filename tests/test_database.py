"""Tests for database.py: storing, listing and removing people.

Owner: Thanigai
"""
import tempfile
import unittest
from pathlib import Path

import numpy as np

from urec.database import Database, clean_name


class DatabaseTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = Database(Path(self.tmp.name) / "t.db")

    def tearDown(self):
        self.db.close()
        self.tmp.cleanup()

    def test_save_list_delete_reset(self):
        self.db.save_person("Joenish", "opencv", np.ones((3, 4), np.float32))
        self.db.save_person("Nishanth", "opencv", np.ones((2, 4), np.float32))
        self.assertTrue(self.db.has("joenish"))  # names are case-insensitive
        self.assertEqual([p[0] for p in self.db.people()], ["Joenish", "Nishanth"])
        self.assertEqual(self.db.gallery("opencv")["Joenish"].shape, (3, 4))
        self.assertTrue(self.db.delete("Nishanth"))
        self.assertFalse(self.db.delete("Nishanth"))
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


if __name__ == "__main__":
    unittest.main()
