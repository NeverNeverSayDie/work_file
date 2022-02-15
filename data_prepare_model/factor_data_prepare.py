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
from loguru import logger


class DataPreprocess(object):
    """
    量化交易--数据预处理
    """

    @classmethod
    def data_winsorize(cls, factor_data, winsorize_method='MAD', n=3, percentile_list=(0.025, 0.975)):
        """
        去极值
        Args:
            factor_data: 输入的因子值
            winsorize_method: 去极值方法
            n: 倍数
            percentile_list: 百分位数列表

        Returns:

        """

        if winsorize_method == 'MAD':
            median = factor_data.median()
            new_median = abs(factor_data.sub(median, axis=1)).median(axis=0)
            upper = median + n * 1.4826 * new_median
            lower = median - n * 1.4826 * new_median

        elif winsorize_method == "3-sigma":
            mean = factor_data.mean()
            std = factor_data.std()
            upper = mean + n * std
            lower = mean - n * std

        elif winsorize_method == "percentile":
            if percentile_list is None:
                raise ValueError("使用百分位法去极值, percentile_list不能为空")
            quantile = factor_data.quantile([min(percentile_list), max(percentile_list)])
            upper = quantile.values[0]
            lower = quantile.values[1]
        else:
            raise ValueError("输入的去极值方法必须在['MAD', '3-sigma', 'percentile']之内")
        return factor_data.clip(lower, upper, axis=1)

    @classmethod
    def data_standardize(cls, factor_data, standard_method='z_score'):
        """
        标准化
        Args:
            factor_data: 传入的因子值
            standard_method: 标准化方法

        Returns:

        """
        if standard_method == 'z_score':
            standard_factor_data = (factor_data - factor_data.mean()) / factor_data.std()
        elif standard_method == 'min_max_-1_1':
            standard_factor_data = 2 * (factor_data - factor_data.min()) / (factor_data.max() - factor_data.min()) - 1
        elif standard_method == "min_max_0_1":
            standard_factor_data = (factor_data - factor_data.min()) / (factor_data.max() - factor_data.min())
        else:
            raise ValueError("输入的标准化方法必须在['z_score', 'min_max_-1_1', 'min_max_0_1']之内")
        return standard_factor_data

    @classmethod
    def data_neutralization(cls, factor_data, factor_value_list=(), dummies_variable=(), weights=None,
                            regress_method='OLS', add_constant=False):
        """
        中性化
        Args:
            factor_data: 传入的因子值
            factor_value_list: 连续性因子值
            dummies_variable: 虚拟类型因子
            weights: 权重
            regress_method: 中性化方法
            add_constant: 是否添加截距列

        Returns:

        """

        # # 市值中性化
        factor_value_df = pd.DataFrame(index=factor_data.stack().index)
        # 市值和行业中性化
        _factor_value = pd.DataFrame(index=factor_data.stack().index)
        _dummies_variable = pd.DataFrame(index=factor_data.stack().index).rename_axis(['date', 'code'])
        if len(factor_value_list) > 0:
            for factor_value in factor_value_list:
                factor_value_standard = cls.data_standardize(factor_value, standard_method='z_score')
                _factor_value = pd.merge(_factor_value, factor_value_standard.stack().rename('factor_value'),
                                         how='inner', right_index=True, left_index=True)
        if len(dummies_variable) > 0:
            for dummies in dummies_variable:
                dummies_ = pd.get_dummies(dummies.set_index(['date', 'code']))
                _dummies_variable = pd.merge(_dummies_variable.rename_axis(['date', 'code']), dummies_, how='inner',
                                             right_index=True, left_index=True)

        factor_value_df = pd.merge(factor_value_df.rename_axis(['date', 'code']),
                                   pd.merge(_factor_value.rename_axis(['date', 'code']), _dummies_variable,
                                            how='inner', left_index=True, right_index=True), how='inner',
                                   left_index=True, right_index=True)
        merge_factor_data = pd.merge(factor_data.stack().rename_axis(['date', 'code']).rename('factor_data'),
                                     factor_value_df.rename_axis(['date', 'code']), how='inner', right_index=True,
                                     left_index=True)
        # 回归求残差
        res_df = pd.DataFrame()
        columns_list = merge_factor_data.columns.unique().to_list()
        columns_list.remove('factor_value')
        for dt, grouped in merge_factor_data.groupby(['date']):
            y = grouped['factor_value'].values
            x = grouped[columns_list]
            if add_constant:
                x = sm.add_constant(x)
            # 最小二乘法
            if regress_method == 'OLS':
                model = sm.OLS(y, x)
            # 加权最小二乘法
            elif regress_method == 'WLS':
                if weights is None:
                    raise ValueError("使用WLS方法, 权重weights不能为空")
                model = sm.WLS(y, x, weights=weights.loc[dt])

            # 稳健回归
            elif regress_method == 'RLM':
                model = sm.RLM(y, x, M=sm.robust.norms.HuberT())
            else:
                raise ValueError("输入的标准化方法必须在['OLS', 'WLS', 'RLM']之内")

            res = model.fit()
            res_df = res_df.append(pd.DataFrame(res.resid))

        return res_df.unstack()


if __name__ == '__main__':
    import numpy as np

    factor_data_ = pd.DataFrame([[0.784, 0.283, 0.96, 0.443, 0.805, 0.942, 0.942, 0.464, 0.443, 0.358],
                                 [0.42, 0.133, 0.987, 0.52, 0.062, 0.205, 0.701, 0.459, 0.501, 0.068],
                                 [0.502, 0.835, 0.121, 0.916, 0.26, 0.563, 0.025, 0.597, 0.373, 0.575],
                                 [0.678, 0.442, 0.106, 0.282, 0.571, 0.475, 0.208, 0.015, 0.208, 0.292],
                                 [0.179, 0.269, 0.2, 0.617, 0.433, 0.224, 0.796, 0.63, 0.838, 0.921]],
                                columns=['00001', '00002', '00003', '00004', '00005', '00006', '00007', '00008',
                                         '00009', '00010'],
                                index=['20000101', '20000102', '20000103', '20000104', '20000105'])
    market_factor = pd.DataFrame(np.random.rand(5, 10),
                                 columns=['00001', '00002', '00003', '00004', '00005', '00006', '00007', '00008',
                                          '00009', '00010'],
                                 index=['20000101', '20000102', '20000103', '20000104', '20000105'])
    ind_factor = pd.DataFrame(np.array([['A', 'A', 'B', 'A', 'C', 'B', 'D', 'A', 'C', 'A'],
                                        ['B', 'B', 'B', 'C', 'A', 'A', 'D', 'A', 'C', 'A'],
                                        ['C', 'B', 'A', 'D', 'C', 'C', 'A', 'B', 'C', 'C'],
                                        ['D', 'B', 'D', 'C', 'A', 'B', 'D', 'A', 'B', 'C'],
                                        ['D', 'A', 'B', 'C', 'C', 'D', 'A', 'A', 'B', 'B']]),
                              columns=['00001', '00002', '00003', '00004', '00005', '00006', '00007', '00008',
                                       '00009', '00010'],
                              index=['20000101', '20000102', '20000103', '20000104', '20000105'])
    ind_factor = ind_factor.unstack().reset_index().rename(
        columns={'level_0': 'code', 'level_1': 'date', 0: 'industry'})
    weight = pd.DataFrame(np.random.random_sample((5, 10)).round(2),
                          columns=['00001', '00002', '00003', '00004', '00005', '00006', '00007', '00008',
                                   '00009', '00010'],
                          index=['20000101', '20000102', '20000103', '20000104', '20000105'])
    data_prepare = DataPreprocess()
    factor_data_ = pd.Series([0.784, 0.283, 0.96, 0.443, 0.805, 0.942, 0.942, 0.464, 0.443, 0.358],
                             index=['00001', '00002', '00003', '00004', '00005', '00006', '00007', '00008',
                                    '00009', '00010'])
    print(data_prepare.data_standardize(factor_data=factor_data_, standard_method='z_score'))
    # data_prepare.data_neutralization(factor_value_list=[market_factor], dummies_variable=[ind_factor],
    #                                  regress_method='OLS')
    # res = data_prepare.data_neutralization(factor_data=factor_data_, factor_value_list=[market_factor],
    #                                        regress_method='OLS', weights=weight, add_constant=True)
    # print(res)
