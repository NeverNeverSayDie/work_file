#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: Dly
@version: 1.0.0
@license: Apache Licence
@file: factor_data_process.py
@time: 2022/2/10 16:31
"""


class DataProcess(object):
    """
    量化交易--数据预处理
    """

    def __init__(self, factor_data, winsorize_method, ploidy, percentile_list, standard_method, neutralization_method,
                 factor_value_list, dummies_variable, add_constant):
        """

        Args:
            factor_data: 因子值 pd.DataFrame / pd.Series
            winsorize_method: 去极值方法 "MAD", "3-sigma", "percentile"
            ploidy:  倍数
            percentile_list: 百分位数列表，默认为None, 当 winsorize_method=percentile 时，percentile_list不能为None
            standard_method:
            neutralization_method:
            factor_value_list: 连续性因子值列表, [因子值1, 因子值2, ....]， 因子值类型: DataFrame/Series
            dummies_variable:
            add_constant:
        """
        self.factor_data = factor_data
        self.factor_value_list = factor_value_list
        self.dummies_variable = dummies_variable

        self._winsorize_method = winsorize_method
        self._ploidy = ploidy
        self._percentile_list = percentile_list
        self._standard_method = standard_method
        self._neutralization_method = neutralization_method
        self._add_constant = add_constant

    def data_winsorize(self, factor_data=None, winsorize_method=None, ploidy=None, percentile_list=None):
        """
        去极值
        Args:
            factor_data: 输入的因子值
            winsorize_method: 去极值方法
            ploidy: 倍数
            percentile_list: 百分位数列表

        Returns:

        """
        if factor_data is None:
            factor_data = self.factor_data
        if winsorize_method is None:
            winsorize_method = self._winsorize_method
        if ploidy is None:
            ploidy = self._ploidy
        if percentile_list is None:
            percentile_list = self._percentile_list

        if winsorize_method == 'MAD':
            median = factor_data.median()
            new_median = abs(factor_data.sub(median, axis=1)).median(axis=0)
            upper = median + ploidy * 1.4826 * new_median
            lower = median - ploidy * 1.4826 * new_median

        elif winsorize_method == "3-sigma":
            mean = factor_data.mean()
            std = factor_data.std()
            upper = mean + ploidy * std
            lower = mean - ploidy * std
        elif winsorize_method == "percentile":
            if percentile_list is not None:
                quantile = factor_data.quantile([min(percentile_list), max(percentile_list)])
                upper = quantile.iloc[0]
                lower = quantile.iloc[1]
            raise ValueError("使用百分位法去极值, percentile_list不能为空")
        else:
            raise ValueError("输入的去极值方法必须在['MAD', '3-sigma', 'percentile']之内")
        return factor_data.clip(lower, upper, axis=1)

    def data_standardize(self, factor_data=None, standard_method=None):
        """
        标准化
        Args:
            factor_data: 传入的因子值
            standard_method: 标准化方法

        Returns:

        """
        if factor_data is None:
            factor_data = self.factor_data
        if standard_method is None:
            standard_method = self._standard_method

        if standard_method == 'z_score':
            standard_factor_data = (factor_data - factor_data.mean()) / factor_data.std()
        elif standard_method == 'min_max_-1_1':
            standard_factor_data = 2 * (factor_data - factor_data.min()) / (factor_data.max() - factor_data.min()) - 1
        elif standard_method == "min_max_0_1":
            standard_factor_data = (factor_data - factor_data.min()) / (factor_data.max() - factor_data.min())
        else:
            raise ValueError("输入的标准化方法必须在['z_score', 'min_max_-1_1', 'min_max_0_1']之内")
        return standard_factor_data

    def data_neutralization(self, factor_data=None, neutralization_method=None, factor_value_list=None,
                            dummies_variable=None, add_constant=None):
        """
        中性化
        Args:
            factor_data:
            neutralization_method:
            factor_value_list:
            dummies_variable:
            add_constant:

        Returns:

        """
        if factor_data is None:
            factor_data = self.factor_data
        if neutralization_method is None:
            neutralization_method = self._neutralization_method
        if factor_value_list is None:
            factor_value_list = self.factor_value_list
        if dummies_variable is None:
            dummies_variable = self.dummies_variable
        if add_constant is None:
            add_constant = self._add_constant
