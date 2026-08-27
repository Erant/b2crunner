"""`timestamped_run_name`'s collision guard.

The name is second-resolution (`%Y%m%d-%H%M%S`) and, before this, carried
no other disambiguator. Once more than one run can start on this machine at
once — the whole point of `pipeline.gpu_scheduler` — two runs landing in
the same wall-clock second is routine, not a corner case, and a collision
here means two runs silently sharing one output directory and one log file.
"""

from __future__ import annotations

import unittest

from pipeline.logging_setup import timestamped_run_name


class TestTimestampedRunName(unittest.TestCase):
    def test_rapid_repeated_calls_are_all_distinct(self):
        names = [timestamped_run_name("fast_helical") for _ in range(200)]
        self.assertEqual(len(names), len(set(names)))

    def test_it_still_starts_with_the_prefix_and_timestamp(self):
        name = timestamped_run_name("fast_helical")
        self.assertRegex(name, r"^fast_helical-\d{8}-\d{6}-[0-9a-f]{6}$")

    def test_the_default_prefix_is_run(self):
        self.assertTrue(timestamped_run_name().startswith("run-"))


if __name__ == "__main__":
    unittest.main()
