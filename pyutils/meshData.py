import numpy as np
from maya.api import OpenMaya as om


def getMeshFn(meshName):
    sel = om.MSelectionList()
    sel.add(meshName)
    dag = sel.getDagPath(0)
    return om.MFnMesh(dag)


def getGeomIt(meshName):
    sel = om.MSelectionList()
    sel.add(meshName)
    dag = sel.getDagPath(0)
    return om.MItMeshPolygon(dag)


def getMeshData(meshName):
    faceVertices = list()
    geoIt = getGeomIt(meshName)
    while not geoIt.isDone():
        faceVertices.append(geoIt.getVertices())
        geoIt.next(None)  # geoIt.next expect an input for some reason
    points = getMeshPoints(meshName)
    return faceVertices, points


def getMeshVertexFaces(faceVertices):
    vtxDict = dict()
    result = list()
    for face, vertices in enumerate(faceVertices):
        for vtx in vertices:
            vtxDict.setdefault(vtx, []).append(face)
    # forcing sorted range instead of iterate unsorted items
    for i in range(len(vtxDict)):
        result.append(vtxDict[i])
    return result


def getClosestTriangle(point, points, vertexFaces, faceVertices):
    closesVtx = getClosestPoints(point, points)
    nearVertices = getNearVertices(closesVtx, vertexFaces, faceVertices)
    nearVertices.remove(closesVtx)
    secondVtx = nearVertices[getClosestPoints(point, points[list(nearVertices)])]
    commonFaces = set(vertexFaces[closesVtx]).intersection(set(vertexFaces[secondVtx]))
    closestVertices = set()
    for face in commonFaces:
        closestVertices.update(faceVertices[face])
    closestVertices.remove(closesVtx)
    closestVertices.remove(secondVtx)
    closestVertices_l = list(closestVertices)
    thirdVtx = closestVertices_l[getClosestPoints(point, points[closestVertices_l])]
    return list((closesVtx, secondVtx, thirdVtx))


def normalizeArray(x):
    return x / np.linalg.norm(x)


def getMatrixFromTriangle(trianlgePositions):
    ab = trianlgePositions[1] - trianlgePositions[0]
    ac = trianlgePositions[2] - trianlgePositions[1]
    normal_vector = normalizeArray(np.cross(ac, ab, axis=0))
    tangent_vector = normalizeArray(np.cross(ab, normal_vector, axis=0))
    cross_vector = normalizeArray(np.cross(tangent_vector, normal_vector, axis=0))
    position = np.mean(trianlgePositions, axis=0)
    matrix = np.array([[tangent_vector[0], tangent_vector[1], tangent_vector[2], 0],
                       [normal_vector[0],  normal_vector[1],  normal_vector[2],  0],
                       [cross_vector[0],   cross_vector[1],   cross_vector[2],   0],
                       [position[0],       position[1],       position[2],       1]])
    return matrix


def getNearVertices(vertex, vertexFaces, faceVertices):
    nearFaces = vertexFaces[vertex]
    result = set()
    for face in nearFaces:
        result.update(set(faceVertices[face]))
    return list(result)


def getMeshPoints(meshName):
    mfn = getMeshFn(meshName)
    points4 = np.array(mfn.getPoints(), dtype=np.double)
    points3 = np.delete(points4, -1, axis=1)
    return points3


def getClosestPoints(point, pointList):
    pnt = np.asarray(point, dtype=np.double)
    dist_2 = np.sum((pointList - pnt)**2, axis=1)
    return np.argmin(dist_2)


def getVerticesDeltas(meshName, vtx, cachedPoints):
    newPoints = getMeshPoints(meshName)
    return newPoints[vtx]-cachedPoints[vtx]

