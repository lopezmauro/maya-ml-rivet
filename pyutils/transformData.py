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

