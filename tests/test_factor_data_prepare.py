#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: Dly
@version: 1.0.0
@license: Apache Licence
@file: test_factor_data_prepare.py
@time: 2022/2/14 12:29
"""
import unittest

import numpy as np
import pandas as pd

from work_file.data_prepare_model.factor_data_preprocess import DataPreprocess

factor_data_ = pd.DataFrame([[0.784, 0.283, 0.96, 0.443, 0.805, 0.942, 0.942, 0.464, 0.443, 0.358],
                             [0.42, 0.133, 0.987, 0.52, 0.062, 0.205, 0.701, 0.459, 0.501, 0.068],
                             [0.502, 0.835, 0.121, 0.916, 0.26, 0.563, 0.025, 0.597, 0.373, 0.575],
                             [0.678, 0.442, 0.106, 0.282, 0.571, 0.475, 0.208, 0.015, 0.208, 0.292],
                             [0.179, 0.269, 0.2, 0.617, 0.433, 0.224, 0.796, 0.63, 0.838, 0.921]],
                            columns=['00001', '00002', '00003', '00004', '00005', '00006', '00007', '00008',
                                     '00009', '00010'],
                            index=['20000101', '20000102', '20000103', '20000104', '20000105'])
factor_prepare = DataPreprocess()


class TestDataPrepare(unittest.TestCase):

    def test_data_winsorize_mad(self):
        self.assertNotIn(0.96, factor_prepare.data_winsorize(factor_data_, winsorize_method='MAD').values)
        self.assertIn(0.575, factor_prepare.data_winsorize(factor_data_, winsorize_method='MAD').values)

    def test_data_winsorize_3_sigma(self):
        self.assertNotIn(0.08, factor_prepare.data_winsorize(factor_data_, winsorize_method='3-sigma').values)
        self.assertIn(0.563, factor_prepare.data_winsorize(factor_data_, winsorize_method='3-sigma').values)
        pd.testing.assert_frame_equal(factor_data_,
                                      factor_prepare.data_winsorize(factor_data_, winsorize_method='3-sigma'))

    def test_data_winsorize_3_percentile(self):
        self.assertNotIn(0.08, factor_prepare.data_winsorize(factor_data_, winsorize_method='percentile').values)
        self.assertIn(0.0594, factor_prepare.data_winsorize(factor_data_, winsorize_method='percentile').values)

    def test_data_standardize_z_score(self):
        self.assertIn(0.40965975, factor_prepare.data_standardize(factor_data_, standard_method='z_score').values)
        self.assertIn(-0.958706, factor_prepare.data_standardize(factor_data_, standard_method='z_score').values)

    def test_data_standardize_min_max(self):
        self.assertIn(0.4601626, factor_prepare.data_standardize(factor_data_, standard_method='min_max_-1_1').values)
        self.assertNotIn(-0.958706, factor_prepare.data_standardize(factor_data_, standard_method='min_max_-1_1').values)

    def test_data_standardize_min_max_0_1(self):
        self.assertNotIn(0.409660, factor_prepare.data_standardize(factor_data_, standard_method='min_max_0_1').values)
        self.assertIn(0.21367521, factor_prepare.data_standardize(factor_data_, standard_method='min_max_0_1').values)


if __name__ == '__main__':
    unittest.main(verbosity=0)
