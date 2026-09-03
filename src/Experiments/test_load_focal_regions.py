"""Tests for SetFocalRegions.load_focal_regions."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from Classes.focal_regions import SetFocalRegions


def _write_regions(path: Path, regions: list) -> None:
    path.write_text(json.dumps(regions))


class TestLoadFocalRegions(unittest.TestCase):
    def _sfr(self, max_regions=3, add_empirical=True):
        return SetFocalRegions(
            num_agents=3,
            threshold=2 / 3,
            len_history=2,
            max_regions=max_regions,
            from_file=True,
            add_empirical_focal_regions=add_empirical,
            seed=0,
        )

    def _temp_files(self, theoretical, empirical=None):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        theo_path = Path(tmp.name) / "theoretical.json"
        emp_path = Path(tmp.name) / "empirical.json"
        _write_regions(theo_path, theoretical)
        if empirical is not None:
            _write_regions(emp_path, empirical)
        return theo_path, emp_path

    def test_missing_theoretical_file_raises(self):
        sfr = self._sfr(add_empirical=False)
        sfr.file = Path(tempfile.gettempdir()) / "does_not_exist_focal.json"
        with self.assertRaises(FileNotFoundError):
            sfr.load_focal_regions()

    def test_without_empirical_keeps_first_of_each_category(self):
        theoretical = [
            {"category": "fair", "region": [[1, 0, 1], [1, 1, 0], [0, 1, 1]]},
            {"category": "fair", "region": [[0, 0, 0], [0, 0, 0], [0, 0, 0]]},
            {"category": "segmented", "region": [[1, 1, 0], [1, 0, 1], [0, 1, 1]]},
            {"category": "mixed", "region": [[1, 1, 1], [0, 0, 0], [1, 0, 1]]},
        ]
        theo_path, _ = self._temp_files(theoretical)
        sfr = self._sfr(max_regions=3, add_empirical=False)
        sfr.file = theo_path
        regions = sfr.load_focal_regions()
        self.assertEqual(len(regions), 3)
        self.assertEqual([r.category for r in regions], ["fair", "segmented", "mixed"])
        np.testing.assert_array_equal(
            regions[0].focal_region, np.array(theoretical[0]["region"])
        )

    def test_empirical_region_is_prepended_in_its_category(self):
        theoretical = [
            {"category": "fair", "region": [[1, 0, 1], [1, 1, 0], [0, 1, 1]]},
            {"category": "segmented", "region": [[0, 0, 0], [0, 0, 0], [0, 0, 0]]},
            {"category": "mixed", "region": [[1, 1, 1], [0, 0, 0], [1, 0, 1]]},
        ]
        empirical_region = [[1, 1, 0], [1, 1, 0], [0, 0, 1]]
        empirical = [{"category": "segmented", "region": empirical_region}]
        theo_path, emp_path = self._temp_files(theoretical, empirical)
        sfr = self._sfr(max_regions=3, add_empirical=True)
        sfr.file = theo_path
        sfr.file_empirical = emp_path
        regions = sfr.load_focal_regions()
        self.assertEqual(len(regions), 3)
        segmented = [r for r in regions if r.category == "segmented"]
        self.assertEqual(len(segmented), 1)
        np.testing.assert_array_equal(
            segmented[0].focal_region, np.array(empirical_region)
        )

    def test_new_empirical_category_is_included(self):
        theoretical = [
            {"category": "fair", "region": [[1, 0, 1], [1, 1, 0], [0, 1, 1]]},
            {"category": "segmented", "region": [[1, 1, 0], [1, 0, 1], [0, 1, 1]]},
        ]
        empirical_region = [[0, 1, 0], [1, 0, 1], [1, 1, 0]]
        empirical = [{"category": "mixed", "region": empirical_region}]
        theo_path, emp_path = self._temp_files(theoretical, empirical)
        sfr = self._sfr(max_regions=3, add_empirical=True)
        sfr.file = theo_path
        sfr.file_empirical = emp_path
        regions = sfr.load_focal_regions()
        cats = [r.category for r in regions]
        self.assertIn("mixed", cats)
        mixed = next(r for r in regions if r.category == "mixed")
        np.testing.assert_array_equal(mixed.focal_region, np.array(empirical_region))

    def test_missing_empirical_file_raises_when_requested(self):
        theoretical = [
            {"category": "fair", "region": [[1, 0, 1], [1, 1, 0], [0, 1, 1]]},
        ]
        theo_path, emp_path = self._temp_files(theoretical)
        sfr = self._sfr(add_empirical=True)
        sfr.file = theo_path
        sfr.file_empirical = emp_path
        with self.assertRaises(FileNotFoundError):
            sfr.load_focal_regions()

    def test_exact_duplicates_within_category_are_dropped(self):
        a = [[1, 1, 0], [1, 0, 1], [0, 1, 1]]
        b = [[0, 1, 0], [1, 0, 1], [1, 1, 0]]
        theoretical = [
            {"category": "segmented", "region": a},
            {"category": "segmented", "region": a},
            {"category": "segmented", "region": b},
        ]
        theo_path, _ = self._temp_files(theoretical)
        sfr = self._sfr(max_regions=2, add_empirical=False)
        sfr.file = theo_path
        regions = sfr.load_focal_regions()
        self.assertEqual(len(regions), 2)
        np.testing.assert_array_equal(regions[0].focal_region, np.array(a))
        np.testing.assert_array_equal(regions[1].focal_region, np.array(b))

    def test_cyclic_duplicates_within_category_are_dropped(self):
        a = [[1, 1, 0], [1, 0, 1], [0, 1, 1]]
        a_shifted = [[1, 0, 1], [0, 1, 1], [1, 1, 0]]
        b = [[0, 1, 0], [1, 0, 1], [1, 1, 0]]
        theoretical = [
            {"category": "segmented", "region": a},
            {"category": "segmented", "region": a_shifted},
            {"category": "segmented", "region": b},
        ]
        theo_path, _ = self._temp_files(theoretical)
        sfr = self._sfr(max_regions=2, add_empirical=False)
        sfr.file = theo_path
        regions = sfr.load_focal_regions()
        self.assertEqual(len(regions), 2)
        np.testing.assert_array_equal(regions[0].focal_region, np.array(a))
        np.testing.assert_array_equal(regions[1].focal_region, np.array(b))

    def test_short_category_is_not_padded_with_repeats(self):
        theoretical = [
            {"category": "fair", "region": [[1, 0, 1], [1, 1, 0], [0, 1, 1]]},
            {"category": "segmented", "region": [[1, 1, 0], [1, 0, 1], [0, 1, 1]]},
            {"category": "segmented", "region": [[0, 1, 0], [1, 0, 1], [1, 1, 0]]},
        ]
        theo_path, _ = self._temp_files(theoretical)
        sfr = self._sfr(max_regions=4, add_empirical=False)
        sfr.file = theo_path
        regions = sfr.load_focal_regions()
        self.assertEqual(len(regions), 3)
        self.assertEqual(len({r.focal_region.tobytes() for r in regions}), 3)

    def test_duplicates_across_categories_are_dropped(self):
        shared = [[1, 1, 0], [1, 0, 1], [0, 1, 1]]
        theoretical = [
            {"category": "fair", "region": shared},
            {"category": "segmented", "region": shared},
            {"category": "mixed", "region": [[1, 1, 1], [0, 0, 0], [1, 0, 1]]},
        ]
        theo_path, _ = self._temp_files(theoretical)
        sfr = self._sfr(max_regions=6, add_empirical=False)
        sfr.file = theo_path
        regions = sfr.load_focal_regions()
        self.assertEqual(len(regions), 2)
        self.assertEqual([r.category for r in regions], ["fair", "mixed"])

    def test_never_returns_more_than_max_regions(self):
        theoretical = [
            {"category": "fair", "region": [[1, 0, 1], [1, 1, 0], [0, 1, 1]]},
            {"category": "fair", "region": [[0, 0, 1], [1, 1, 0], [0, 1, 1]]},
            {"category": "segmented", "region": [[1, 1, 0], [1, 0, 1], [0, 1, 1]]},
            {"category": "mixed", "region": [[1, 1, 1], [0, 0, 0], [1, 0, 1]]},
        ]
        theo_path, _ = self._temp_files(theoretical)
        sfr = self._sfr(max_regions=2, add_empirical=False)
        sfr.file = theo_path
        regions = sfr.load_focal_regions()
        self.assertEqual(len(regions), 2)
        self.assertEqual(len({r.focal_region.tobytes() for r in regions}), 2)

    def test_empirical_duplicate_of_theoretical_keeps_empirical_first(self):
        theoretical_region = [[1, 1, 0], [1, 0, 1], [0, 1, 1]]
        theoretical = [
            {"category": "fair", "region": [[1, 0, 1], [1, 1, 0], [0, 1, 1]]},
            {"category": "segmented", "region": theoretical_region},
            {"category": "mixed", "region": [[1, 1, 1], [0, 0, 0], [1, 0, 1]]},
        ]
        empirical = [{"category": "segmented", "region": theoretical_region}]
        theo_path, emp_path = self._temp_files(theoretical, empirical)
        sfr = self._sfr(max_regions=3, add_empirical=True)
        sfr.file = theo_path
        sfr.file_empirical = emp_path
        regions = sfr.load_focal_regions()
        segmented = [r for r in regions if r.category == "segmented"]
        self.assertEqual(len(segmented), 1)
        np.testing.assert_array_equal(
            segmented[0].focal_region, np.array(theoretical_region)
        )


if __name__ == "__main__":
    unittest.main()
