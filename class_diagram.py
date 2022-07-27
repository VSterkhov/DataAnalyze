#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 25 14:06:42 2022

@author: vladislav
"""

import pandas
import matplotlib.pyplot as plt

data = pandas.read_csv('./train.csv', delimiter=(','))
data = data[pandas.isnull(data['Pclass']) == 0 ]

unsurvived = data[data.Survived == 0]
unsurvived = unsurvived.reset_index()

hashmap = dict()
for index, row in unsurvived.iterrows():
    if hashmap.get(row['Pclass']) == None:
        hashmap[row['Pclass']]=1
    else:
        hashmap[row['Pclass']]=hashmap.get(row['Pclass'])+1

plt.bar(hashmap.keys(), hashmap.values())
plt.title("Unsurvived of Pclass")
plt.xlabel("Pclass")
plt.ylabel("Count")
plt.show()

unsurvived = data[data.Survived == 1]
unsurvived = unsurvived.reset_index()

hashmap = dict()
for index, row in unsurvived.iterrows():
    if hashmap.get(row['Pclass']) == None:
        hashmap[row['Pclass']]=1
    else:
        hashmap[row['Pclass']]=hashmap.get(row['Pclass'])+1

plt.bar(hashmap.keys(), hashmap.values())
plt.title("Survived of Pclass")
plt.xlabel("Pclass")
plt.ylabel("Count")
plt.show()