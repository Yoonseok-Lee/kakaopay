# -*- coding: utf-8 -*-
"""
Created on Sat Jan  6 17:28:20 2018

@author: lab07


1. Title: Boston Housing Data

2. Sources:
    (a) Origin: This dataset was taken from the StatLib library which is
        maintained at Carnegie Mellon University.
    (b) Creator: Harrison, D. and Rubinfeld, D.L. 'Hedonic prices and the
        demand for clean air', J. Environ. Economics & Management,
        vol.5, 81-102, 1978.
    (c) Date: July 7, 1993

3. Past Usage:
    - Used in Belsley, Kuh & Welsch, 'Regression diagnostics ...', Wiley,
      1980. N.B. Various transformations are used in the table on
      pages 244-261.
    - Quinlan,R. (1993). Combining Instance-Based and Model-Based Learning.
      In Proceedings on the Tenth International Conference of Machine
      Learning, 236-243, University of Massachusetts, Amherst. Morgan Kaufmann.

4. Relevant Information:
    Concerns housing values in suburbs of Boston.

5. Number of Instances: 506

6. Number of Attributes: 13 continuous attributes (including "class"
   attribute "MEDV"), 1 binary-valued attribute.

7. Attribute Information:

    1. CRIM     per capita crime rate by town
    2. ZN       proportion of residential land zoned for lots over
                25,000 sq.ft.
    3. INDUS    proportion of non-retail business acres per town
    4. CHAS     Charles River dummy variable (= 1 if tract bounds
                river; 0 otherwise)
    5. NOX      nitric oxides concentration (parts per 10 million)
    6. RM       average number of rooms per dwelling
    7. AGE      proportion of owner-occupied units built prior to 1940
    8. DIS      weighted distances to five Boston employment centres
    9. RAD      index of accessibility to radial highways
   10. TAX      full-value property-tax rate per $10,000
   11. PTRATIO  pupil-teacher ratio by town
   12. B        1000(Bk - 0.63)^2 where Bk is the proportion of blacks
                by town
   13. LSTAT    % lower status of the population
   14. MEDV     Median value of owner-occupied homes in $1000's

8. Missing Attribute Values: None.


"""
#import sys, os
#sys.path.append(os.getcwd())  

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

import analysis as an

from xgboost.sklearn import XGBRegressor
from sklearn.tree import DecisionTreeRegressor
from functools import partial


###############################################################################
# Read Data
###############################################################################
housing_header = ['CRIM','ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE',
                  'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
housing_data = pd.read_fwf('housing.data', names=housing_header)

###############################################################################
# EDA
###############################################################################

##### 각 변수(figures)별 기초 통계 확인
print ('Housing_data summary')
print ('row counts=' + str(housing_data.shape[0]) + ' col counts=' + str(housing_data.shape[1]))
print (housing_data.describe())

##### 변수들간의 상관관계 확인

## 전체 상관관계 값 확인
print ('Housing_data Correlations')
corr = housing_data.corr()
print (corr)
print ()

# MEDV 타켓 데이터에 대한 상관관계 값 확인
print ('Target Correlations')
target_corr = corr.iloc[-1][:-1].sort_values(ascending=False)
print(target_corr)
print()

# 상관관계 값을 히트맵으로 표현
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
sns.heatmap(corr,
            cmap=sns.diverging_palette(220, 10, as_cmap=True),
            annot=True,
            mask=mask)

# 입력 변수로 사용할 인자들 간의 주요 상관관계 확인 
print ('Important Correlations between Figures')
attrs = corr.iloc[:-1,:-1]

# 절대값으로 0.5 이상 강한 관계성이 보이는 것만
threshold = 0.5
important_corrs = (attrs[abs(attrs) > threshold][attrs != 1.0]) \
    .unstack().dropna().to_dict()

unique_important_corrs = pd.DataFrame(
    list(set([(tuple(sorted(key)), important_corrs[key]) \
    for key in important_corrs])), columns=['attribute pair', 'correlation'])

unique_important_corrs = unique_important_corrs.iloc[
	abs(unique_important_corrs['correlation']).argsort()[::-1]]
print(unique_important_corrs.sort_values(by='correlation', ascending=False))
print()

# 데이터 분포 확인을 위한 박스차트 확인
housing_data.plot.box()

# scatter plot matrix 차트를 통한 각 변수간의 관계 파악, 상관관계 및 선형 비선형 확인 
pd.plotting.scatter_matrix(housing_data)

# 타켓 데이터와 상관성이 큰 데이터 별도로 차트로 보기
sns.distplot(housing_data['MEDV'])
sns.jointplot(housing_data['MEDV'], housing_data['LSTAT'], kind='scatter', joint_kws={'alpha':0.5})
sns.jointplot(housing_data['MEDV'], housing_data['RM'], kind='scatter', joint_kws={'alpha':0.5})
sns.jointplot(housing_data['MEDV'], housing_data['PTRATIO'], kind='scatter', joint_kws={'alpha':0.5})


###############################################################################
# Modeling Test
###############################################################################

#### 트레이닝 셋과 테스트셋 데이터 분리 
target = housing_data['MEDV']
print ('MEDV (target) statistics:\n%s' % target.describe())

housing_data = housing_data.drop('MEDV', axis=1)
data_train, data_test, target_train, target_test = train_test_split(housing_data, 
                                                                    target, 
                                                                    test_size=.2)

#### 각 모델링 알고리즘 별 테스트 진행
GBRegressor = partial(XGBRegressor, seed=8329)

# 인자간의 상관성이 존재 하지만 먼저 일반적인 회귀 모형 확인
an.LinearModelLearning(data_train, target_train)

# 특성 인자별 상관성이 존재 함으로 PLS + Regression 을 통해 데이터 압축 하여 회귀 모형 테스트 
an.PLSModelLearning(data_train, target_train, 5)

# 주요 인자간의 상관성이 크고 선형적이지 않은 데이터도 존재 하기에
# 결정트리가 좋은 성능을 보일 수 있어 일반적인 결정 트리 모델 테스트 진행
an.TRModelLearning(data_train, target_train, targetModel = DecisionTreeRegressor, title='DecisionTreeRegressor')
an.TRModelComplexity(data_train, target_train, targetModel = DecisionTreeRegressor, title='DecisionTreeRegressor')

# 그라디언트 부스팅 적용 모델 테스트 진행
an.TRModelLearning(data_train, target_train, targetModel = GBRegressor, title='GBRegressor')
an.TRModelComplexity(data_train, target_train, targetModel = GBRegressor, title='GBRegressor')


#### 그라디언트 부스팅 (트리) 가 가장 나은 성능을 모여주므로 최적 모델을 GBRegressor 에서 진행
optimal_model = an.get_optimal_GBModel(data_train, target_train)

## 최적 모델 실행
# 생성된 최적의 모델 정보 확인
print ("Parameter 'max_depth' is {} for the optimal model.".format(optimal_model.get_params()['max_depth']))
print (optimal_model)

#### 테스트 테이터로 해당 모델 실행 후 성능 확인 진행
an.get_performance_model(optimal_model, data_test, target_test)

#### 모델 생성에 영향 준 주요 인자 확인
an.check_model_factors(optimal_model)

#optimal_model.predict(data_test)

