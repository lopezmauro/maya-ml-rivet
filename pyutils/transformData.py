# -*- coding: utf-8 -*-
"""This module is mean to be used to get the main training data for train the model to be used on ml_rivets.mll node
This code is to be used on maya with numpy library

MIT License

Copyright (c) 2020 Mauro Lopez

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
from maya import cmds
import numpy as np
import random
import itertools


def setRandomAttributes(control, attribute, limits=[0, 1]):
    cmds.setAttr('{}.{}'.format(control, attribute), random.uniform(limits[0], limits[1]))


def setAtributes(control, attribute, value=0):
    cmds.setAttr('{}.{}'.format(control, attribute), value)


def getControlLocalMatrix(controlName):
    return np.array(cmds.xform(controlName, q=1, os=1, m=1), dtype=np.double)


def getAttrDict(controlList, attrList=['tx', 'ty', 'tz']):
    controlsDict = dict()
    for control, attribute, in itertools.product(controlList, attrList):
        if not cmds.getAttr('{}.{}'.format(control, attribute), k=1):
            continue
        flags = {'q': 1}
        flags[attribute] = 1
        limits = cmds.transformLimits(control, **flags)
        currDict = controlsDict.setdefault(control, {})
        currDict[attribute] = limits
    return controlsDict


def getInverseMatrix(matrix):
    return np.linalg.inv(matrix)

