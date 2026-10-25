import unittest

import numpy as np

from stingray.pulse.search import _folding_search


class TestFoldingSearchLastPoint(unittest.TestCase):
    def test_folding_search_uses_last_data_point(self):
        times = np.arange(11.0)
        n_used = []

        def stat_func(t, f, fd=0, **kwargs):
            n_used.append(len(t))
            return 0

        _folding_search(stat_func, times, [1.0], use_times=True)

        self.assertEqual(sum(n_used), len(times))
