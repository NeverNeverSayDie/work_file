#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: Dly
@version: 1.0.0
@license: Apache Licence
@file: factor_data_prepare.py
@time: 2022/2/10 16:31
"""
import pandas as pd
import statsmodels.api as sm


class DataPrepare(object):
    """
    量化交易--数据预处理
    """

    def __init__(self, factor_data, winsorize_method=None, ploidy=2, percentile_list=None, standard_method=None,
                 neutralization_method=None, factor_value_list=None, dummies_variable=None, regress_method=None,
                 add_constant=None, weights=None):
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
        self.weights = weights

        self._winsorize_method = winsorize_method
        self._ploidy = ploidy
        self._percentile_list = percentile_list
        self._standard_method = standard_method
        self._neutralization_method = neutralization_method
        self._regress_method = regress_method
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
            if percentile_list is None:
                raise ValueError("使用百分位法去极值, percentile_list不能为空")
            quantile = factor_data.quantile([min(percentile_list), max(percentile_list)])
            upper = quantile.iloc[0]
            lower = quantile.iloc[1]
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

    def data_neutralization(self, factor_data=None, factor_value_list=None, dummies_variable=None, regress_method=None,
                            add_constant=None, weights=None):
        """
        中性化
        Args:
            factor_data:
            factor_value_list:
            dummies_variable:
            regress_method:
            add_constant:
            weights:

        Returns:

        """
        if factor_data is None:
            factor_data = self.factor_data
        if factor_value_list is None:
            factor_value_list = self.factor_value_list
        if dummies_variable is None:
            dummies_variable = self.dummies_variable
        if regress_method is None:
            regress_method = self._regress_method
        if add_constant is None:
            add_constant = self._add_constant
        if weights is None:
            weights = self.weights

        # 市值中性化
        factor_value_df = pd.DataFrame(factor_data.index)
        if factor_value_list is not None and dummies_variable is None:
            for factor_value in factor_value_list:
                factor_value_standard = self.data_standardize(factor_value, standard_method='z-score')
                factor_value_df = pd.merge(factor_value_df, factor_value_standard, how='inner', right_index=True,
                                           left_index=True)

        # 行业中性化
        if factor_value_list is None and dummies_variable is not None:
            for dummies in dummies_variable:
                dummies_ = pd.get_dummies(dummies.set_index(['date', 'code'])).unstack()
                factor_value_df = pd.merge(factor_value_df, dummies_, how='inner', right_index=True, left_index=True)

        # 市值和行业中性化
        if factor_value_list is not None and dummies_variable is not None:
            _factor_value = pd.DataFrame(factor_data.index)
            _dummies_variable = pd.DataFrame(factor_data.index)
            for factor_value in factor_value_list:
                factor_value_standard = self.data_standardize(factor_value, standard_method='z-score')
                _factor_value = pd.merge(_factor_value, factor_value_standard, how='inner', right_index=True,
                                         left_index=True)
            for dummies in dummies_variable:
                dummies_ = pd.get_dummies(dummies.set_index(['date', 'code'])).unstack()
                _dummies_variable = pd.merge(_dummies_variable, dummies_, how='inner', right_index=True, left_index=True)

            factor_value_df = pd.merge(
                pd.merge(_factor_value, _dummies_variable, how='index', left_index=True, right_index=True), how='index',
                left_index=True, right_index=True)

        # 回归求残差
        trade_date_list = factor_data.index.unique().tolist()
        for trade_date in trade_date_list:
            x = factor_value_df.loc[trade_date]
            y = factor_data.loc[trade_date]
            if add_constant:
                x = sm.add_constant(x)
            # 最小二乘法
            if regress_method == 'OLS':
                model = sm.OLS(y, x)
            # 加权最小二乘法
            elif regress_method == 'WLS':
                if weights is not None:
                    weights = self.data_standardize(weights, standard_method='z_score')
                model = sm.WLS(y, x, weights=weights)

            # 稳健回归
            elif regress_method == 'RLM':
                model = sm.RLM(y, x, M=sm.robust.norms.HuberT())
            else:
                raise ValueError()

            res = model.fit()


if __name__ == '__main__':
    import numpy as np

    factor_data_ = pd.DataFrame(np.random.rand(5, 10),
                                columns=['00001', '00002', '00003', '00004', '00005', '00006', '00007', '00008',
                                         '00009', '00010'],
                                index=['20000101', '20000102', '20000103', '20000104', '20000105'])
    market_factor = pd.DataFrame(np.random.rand(5, 10),
                                 columns=['00001', '00002', '00003', '00004', '00005', '00006', '00007', '00008',
                                          '00009', '00010'],
                                 index=['20000101', '20000102', '20000103', '20000104', '20000105'])

    data_prepare = DataPrepare(factor_data_, winsorize_method='MAD')
    print(data_prepare.data_standardize(standard_method='min_max_0_1'))
