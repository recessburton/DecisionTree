#! /usr/bin/env python
# -*- coding: utf-8 -*-
'''
Copyright (C),2014-2015, YTC, www.bjfulinux.cn
Created on  2015-04-16 12:54

@author: ytc recessburton@gmail.com
@version: 1.0

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>
'''

from math import log

def calcShannonEnt(dataSet):
    '香农信息熵的计算函数'
    numEntries = len(dataSet)
    labelCounts = {}
    for featVec in dataSet:                         #为所有可能的分类创建字典
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCounts:                         #计算概率，计算信息熵
        prob = float(labelCounts[key])/numEntries
        shannonEnt -= prob * log(prob, 2)
    return shannonEnt

def createDataSet():
    '香农信息熵计算测试数据集生成函数'
    dataSet = [[1, 1, 'yes'], 
               [1, 1, 'yes'], 
               [1, 0, 'no' ], 
               [0, 1, 'no' ], 
               [0, 1, 'no' ]]
    labels = ['no surfacing', 'flippers']
    return dataSet, labels


def splitDataSet(dataSet, axis, value):
    '按照给定的特征划分数据集'
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:              #如果与要找的特征匹配，将其余的特征加入要返回的向量中
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis + 1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet

def shannonSplit(dataSet):
    '利用信息熵来划分数据集'
    numFeatures = len(dataSet[0]) -1
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0
    bestFeature = -1
    for i in range(numFeatures):                        #对每个特征（列）进行循环
        featList = [example[i] for example in dataSet]
        uniqueVals = set(featList)                      #找出该特征所有的可能取值
        newEntropy = 0.0
        for value in uniqueVals:                        #对每个特征的取值考察利用该特征划分得到的信息增益
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet)/float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)
        infoGain = baseEntropy - newEntropy
        if (infoGain > bestInfoGain):                   # 找出带来最大信息增益的特征
            bestInfoGain =infoGain
            bestFeature = i
    return bestFeature


 



if __name__ == '__main__':
    myDat, labels = createDataSet()
    print myDat
    print shannonSplit(myDat)
