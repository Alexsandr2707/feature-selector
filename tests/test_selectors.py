"""Selection contracts for named and positional feature matrices."""

import unittest

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from feature_selector.selectors import ElasticNetSelector, PLSSelector, StaticSelector


class SelectorTests(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(7)
        self.X = pd.DataFrame(rng.normal(size=(40, 4)), columns=list('abcd'))
        self.y = pd.Series(3 * self.X.a - 2 * self.X.c)

    def test_static_modes_and_output_order(self) -> None:
        for X in (self.X, self.X.to_numpy()):
            selector = StaticSelector(indices=[3, 0]).fit(X)
            result = selector.transform(X)
            self.assertIsInstance(result, type(X))
            np.testing.assert_array_equal(np.asarray(result), self.X.iloc[:, [3, 0]])
            np.testing.assert_array_equal(selector.get_selected_indices(), [3, 0])
            np.testing.assert_array_equal(selector.get_selection_mask(), [True, False, False, True])
            positions = selector.get_selected_indices()
            positions[0] = 1
            np.testing.assert_array_equal(selector.get_selected_indices(), [3, 0])
        selector = StaticSelector(columns=['d', 'a']).fit(self.X)
        assert_frame_equal(selector.transform(self.X), self.X[['d', 'a']])
        self.assertEqual(clone(selector).columns, ['d', 'a'])

    def test_names_refit_and_nonstring_labels(self) -> None:
        selector = StaticSelector(indices=[2, 0]).fit(self.X)
        np.testing.assert_array_equal(selector.get_feature_names_out(), ['c', 'a'])
        selector.fit(self.X.to_numpy())
        self.assertFalse(hasattr(selector, 'feature_names_in_'))
        np.testing.assert_array_equal(selector.get_feature_names_out(), ['x2', 'x0'])
        np.testing.assert_array_equal(selector.get_feature_names_out(list('abcd')), ['c', 'a'])
        frame = self.X.set_axis([10, 20, 30, 40], axis=1)
        named = StaticSelector(columns=[30, 10]).fit(frame)
        assert_frame_equal(named.transform(frame), frame.iloc[:, [2, 0]])
        self.assertFalse(hasattr(named, 'feature_names_in_'))

    def test_learned_selectors_match_containers_and_keep_missing_rows(self) -> None:
        X = self.X.copy()
        y = self.y.copy()
        X.iloc[2, 1] = np.nan
        y.iloc[4] = np.nan
        for prototype in (ElasticNetSelector(top_k=2), PLSSelector(depth=1)):
            with self.subTest(selector=type(prototype).__name__):
                frame_selector = clone(prototype).fit(X, y)
                array_selector = clone(prototype).fit(X.to_numpy(), y.to_numpy())
                np.testing.assert_array_equal(
                    frame_selector.get_selected_indices(), array_selector.get_selected_indices()
                )
                np.testing.assert_allclose(
                    frame_selector.transform(X), array_selector.transform(X.to_numpy()),
                    equal_nan=True,
                )
                self.assertEqual(frame_selector.n_training_pairs_, len(X) - 2)
                self.assertEqual(len(frame_selector.transform(X)), len(X))
                np.testing.assert_allclose(
                    frame_selector.transform(X.to_numpy()), array_selector.transform(X.to_numpy()),
                    equal_nan=True,
                )

    def test_output_configuration_and_pipeline(self) -> None:
        array = self.X.to_numpy()
        selector = StaticSelector(indices=[3, 0]).set_output(transform='pandas')
        result = selector.fit_transform(array)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(list(result.columns), ['x3', 'x0'])
        pipeline = Pipeline([
            ('scale', StandardScaler()), ('select', StaticSelector(indices=[2, 0]))
        ])
        self.assertIsInstance(pipeline.fit_transform(array), np.ndarray)
        pipeline.set_output(transform='pandas')
        result = pipeline.fit_transform(self.X)
        self.assertEqual(list(result.columns), ['c', 'a'])
        self.assertTrue(result.index.equals(self.X.index))

    def test_invalid_selection_schema_and_unfitted_access(self) -> None:
        for kwargs in ({}, {'columns': ['a'], 'indices': [0]}, {'indices': []},
                       {'indices': [-1]}, {'indices': [True]}, {'indices': [1.2]},
                       {'indices': [0, 0]}, {'indices': np.array(0)},
                       {'indices': np.array([[0]])}, {'indices': [4]}, {'columns': ['missing']}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                StaticSelector(**kwargs).fit(self.X)
        with self.assertRaises(ValueError):
            StaticSelector(columns=['a']).fit(self.X.to_numpy())
        selector = StaticSelector(indices=[0]).fit(self.X)
        for bad in (self.X[['b', 'a', 'c', 'd']], self.X.to_numpy()[:, :2],
                    np.ones(4), np.ones((2, 2, 2))):
            with self.assertRaises(ValueError):
                selector.transform(bad)
        for method in ('get_selected_indices', 'get_selection_mask', 'get_feature_names_out'):
            with self.assertRaises(NotFittedError):
                getattr(StaticSelector(indices=[0]), method)()
        for cls in (ElasticNetSelector, PLSSelector):
            with self.assertRaisesRegex(ValueError, 'indexes'):
                cls().fit(self.X, self.y.set_axis(range(1, len(self.y) + 1)))
