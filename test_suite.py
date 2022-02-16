#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: Dly
@version: 1.0.0
@license: Apache Licence
@file: test_suite.py
@time: 2022/2/14 13:03
"""
import unittest

if __name__ == '__main__':
    suite = unittest.TestSuite()
    all_cases = unittest.defaultTestLoader.discover('tests', 'test_*.py')
    for case in all_cases:
        suite.addTests(case)

    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)
