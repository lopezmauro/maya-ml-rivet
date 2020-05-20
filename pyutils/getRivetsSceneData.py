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
import os
import json
import time
import logging
import meshData
import transformData
import constants
import numpy as np
from maya import cmds

_logger = logging.getLogger(__name__)


def closestTriangleToTransform(transform, meshName):
    """get the closest mesh triangles ids from a transomr position

    Args:
        transform (str): name of the transfomr to get the position
        meshName (str): name of the mesh to get the triangle

    Returns:
        list: 3 mesh vertices ids
    """
    faceVertices, points = meshData.getMeshData(meshName)
    vertexFaces = meshData.getMeshVertexFaces(faceVertices)
    point = np.array(cmds.xform(transform, q=1, ws=1, t=1), dtype=np.double)
    return meshData.getClosestTriangle(point, points, vertexFaces, faceVertices)


def filterUnnecesaryTransforms(meshName, transforms, vertices, tol=0.1):
    """remove all controls that wont affect the driven position

    Args:
        meshName (str): name of the mesh to get the deformation
        transforms (list): transforms names to evaluate
        vertices (list): list of vertices to chekc the deformation
        tol (float, optional): how much vertex deformation is accepted. Defaults to 0.1.

    Returns:
        list: list of drivers that affects the vertex position
    """
    points = meshData.getMeshPoints(meshName)
    contrlDict = transformData.getAttrDict(transforms)
    results = set()
    for control, attrDict in contrlDict.iteritems():
        if testControl(control, attrDict, meshName, vertices, points, tol):
            results.add(control)
    return results


def testControl(control, attrDict, meshName, vertices, points, tol=0.1):
    """test if a control affect any vertices positions
    Args:
        control (sre): name of the control to evaluate
        meshName (str): name of the mesh to get the deformation
        attrDict (str): control attributes to edit to check the vertice deformation
        points (np.array): cached mesh positions
        tol (float, optional): how much vertex deformation is accepted. Defaults to 0.1.

    Returns:
        bool: True if the control modify any of the vertices
    """            
    for attr, limits in attrDict.iteritems():
        for lim in limits:
            transformData.setAtributes(control, attr, value=lim)
            deltaPoints = meshData.getVerticesDeltas(meshName, vertices, points)
            if np.linalg.norm(deltaPoints) > tol:
                transformData.setAtributes(control, attr, value=0)
                return True
        transformData.setAtributes(control, attr, value=0)
    return False


def saveJsonFile(filePath, myData):
    """save data to a json file

    Args:
        filePath (str): filepath where to save the data
        myData (vairable): data to serialize
    """
    with open(filePath, 'wb') as myfile:
        json.dump(myData, myfile, indent=4)


def readJsonFile(filePath):
    """read data from json file

    Args:
        filePath (str): location of the json file

    Returns:
        variable: data read form the json file
    """
    result = None
    with open(filePath, 'r') as myfile:
        result = json.load(myfile)
    return result


def resetControls(controlsDict):
    """set all attributes form the control dict to 0

    Args:
        controlsDict (dict): {controlName;{attrName:limits}}
    """
    for control, attrDict in controlsDict.iteritems():
        for attr in attrDict.keys():
            transformData.setAtributes(control, attr, value=0)


def getData(mesh, driverList, drivenList, folderData, filePrefix='', samples=1000):
    """set random values to the attributes of the driverList to het the data necessary to
    train the model for predict rivets positions

    Args:
        mesh (str): name of the mesh to get the deformation
        driverList (list): name of transforms to set attributes
        drivenList (list): list of transforms that will recieve the rivets
        folderData (str): folder path where to save the data to train
        filePrefix (str, optional): prefix name to the saved files
        samples (int, opitonal): how many random samples to create
            (more smaples, slower but accurated results), default 300
    """
    

    start = time.time()
    vertices = list()
    if not os.path.exists(folderData):
        os.makedirs(folderData)
    for driven in drivenList:
        _logger.info('Getting closest veritces for {}'.format(driven))
        vertices.extend(closestTriangleToTransform(driven, mesh))
    _logger.info('> filtering driver list')
    controlsDict = transformData.getAttrDict(driverList)
    resetControls(controlsDict)
    filterdeDrivers = list(filterUnnecesaryTransforms(mesh, driverList, vertices, tol=0.1))
    _logger.info("Driver filtered from {} to {}".format(len(driverList), len(filterdeDrivers)))
    points = meshData.getMeshPoints(mesh)
    csvIn = os.path.join(folderData, '{}{}.{}'.format(filePrefix,
                                                      constants.INFILESUFIX,
                                                      constants.NUMPYIOFORMAT))
    csvOut = os.path.join(folderData, '{}{}.{}'.format(filePrefix,
                                                       constants.OUTFILESUFIX,
                                                       constants.NUMPYIOFORMAT))
    transformsPath = os.path.join(folderData, '{}{}.{}'.format(filePrefix,
                                                               constants.TRFFILESUFIX,
                                                               constants.DATAIOFORMAT))
    # Iterate meshes over time to sample displacements
    localMatrices = list()
    dsplcs = list()
    for i in range(samples):
        _logger.info('> Building sample ' + str(i))
        localMtx = np.array([], dtype=np.double)
        for control in filterdeDrivers:
            for attr, limits in controlsDict[control].iteritems():
                transformData.setRandomAttributes(control, attr, limits)
            localMtx = np.append(localMtx, transformData.getControlLocalMatrix(control))
        dsplc = np.array(meshData.getVerticesDeltas(mesh, vertices, points), dtype=np.double).flatten()
        localMatrices.append(localMtx)
        dsplcs.append(dsplc)

    localMatrices = np.stack(localMatrices)
    dsplcs = np.stack(dsplcs)
    np.savetxt(csvIn, localMatrices)
    _logger.info("{} saved".format(csvIn))
    np.savetxt(csvOut, dsplcs)
    _logger.info("{} saved".format(csvOut))
    saveJsonFile(transformsPath, {'drivers': filterdeDrivers, 'drivens': drivenList})
    _logger.info("{} saved".format(transformsPath))
    resetControls(controlsDict)
    end = time.time()
    _logger.info('Procces ended in {} sec'.format(end - start))
    return csvIn, csvOut, transformsPath