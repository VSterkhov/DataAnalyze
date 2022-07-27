#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 25 14:06:42 2022

@author: vladislav
"""

import pandas
import math
import matplotlib.pyplot as plt

data = pandas.read_csv('./train.csv', delimiter=(','))
data = data[pandas.isnull(data['Age']) == 0 ]
data['Age'][data['Age']<=16]=1
data['Age'][data['Age']>16]=0

young = data[data['Age']==1]
math = data[data['Age']==0]

surv_young_coef = young[young.Survived == 1].shape[0] / (young.shape[0] / 100)
surv_math_coef = math[math.Survived == 1].shape[0] / (math.shape[0] / 100)

plt.bar(['Young (<=16)', 'Mathure (>16)'], [surv_young_coef, surv_math_coef])
plt.title("Percent of survive")
plt.ylabel("Percent")
plt.axis([None, None, 0, 100])
plt.show()
