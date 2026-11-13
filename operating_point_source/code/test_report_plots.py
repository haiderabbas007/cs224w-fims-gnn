"""Checks for display transforms and event-tail denominator semantics."""
import unittest
import warnings
import numpy as np
from report_plots import sigmoid_for_display, histogram_data, event_loss_survival


class PlotSemantics(unittest.TestCase):
    def test_sigmoid_extreme_scores_without_overflow(self):
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            values = sigmoid_for_display(np.array([-1e6, -1., 0., 1., 1e6]))
        self.assertTrue(np.isfinite(values).all())
        self.assertTrue(np.all(values[1:] >= values[:-1]))
        np.testing.assert_array_equal(values[[0, 2, 4]], [0., .5, 1.])

    def test_histograms_count_ignored_primary_charge(self):
        arrays = dict(logits=np.array([-1000., -1., 0., 1., 1000.]),
                      target=np.array([-1, 0, 1, -1, 1]),
                      primary_electron_weight=np.array([5, 7, 0, 11, 0], dtype=np.int64))
        hist = histogram_data(arrays, bins=10)
        for space in ['logit', 'score']:
            self.assertEqual(hist[space+'_microcurler_sites'].sum(), 2)
            self.assertEqual(hist[space+'_other_sites'].sum(), 1)
            self.assertEqual(hist[space+'_primary_electrons'].sum(), 23)
            self.assertEqual(hist[space+'_ignored_primary_electrons'].sum(), 16)

    def test_empty_and_constant_histogram_inputs(self):
        for z in [np.array([], dtype=float), np.ones(4)]:
            hist = histogram_data(dict(logits=z, target=np.zeros(len(z), dtype=np.int8),
                primary_electron_weight=np.zeros(len(z), dtype=np.int64)))
            self.assertTrue(np.all(np.diff(hist['logit_edges']) > 0))
            self.assertEqual(hist['logit_other_sites'].sum(), len(z))

    def test_event_survival_is_strict_and_primary_bearing_only(self):
        rows = [dict(primary_electrons=100, primary_loss_fraction=x) for x in [0., .05, .1, 1.]]
        rows.append(dict(primary_electrons=0, primary_loss_fraction=None))
        x, y = event_loss_survival(rows)
        np.testing.assert_array_equal(x, [0., .05, .1, 1.])
        np.testing.assert_array_equal(y, [.75, .5, .25, 0.])

    def test_no_primary_is_undefined(self):
        x, y = event_loss_survival([dict(primary_electrons=0, primary_loss_fraction=None)])
        self.assertTrue(np.isnan(y).all())


if __name__ == '__main__':
    unittest.main()
