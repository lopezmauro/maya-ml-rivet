# -*- coding: utf-8 -*-
"""This module is mean to be used to train the data gotten form maya scene, and generatr the model
to be used on ml_rivets.mll node
This code needs to be used on python3 with the correct installed depencencies
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
# standard library imprts
import os
import time
import logging
from zipfile import ZipFile
import tempfile
import shutil
# external library imprts
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
# import modules
import constants
_logger = logging.getLogger(__name__)


DEVICE = torch.device("cpu")

if torch.cuda.is_available():
    DEVICE = torch.device("cuda:0")
    torch.cuda.empty_cache()
    _logger.info("Running on the GPU")
else:
    _logger.info("Running on the CPU")


class LRModel(nn.Module):
    """simple procedural linear regresion network with constant hidden layers size
    Heritage:
        torch.nn.Module
    """
    def __init__(self, inputsSize, outputsSize, numLayers=1, layerSize=512):
        """create the base Linear regresion network
        Args:
            inputsSize (int): amount of neurons at input level
            outputsSize (int): amount of neurons at outpur level
            numLayers (int, optional): amount of hidden layers. Defaults to 1.
            layerSize (int, optional): amount of neurons of the hidden layers. Defaults to 512.
        """
        super(LRModel, self).__init__()
        layers = []
        layers.append(nn.Linear(inputsSize, layerSize))
        layers.append(nn.Tanh())
        for x in range(numLayers):
            layers.append(nn.Linear(layerSize, layerSize))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(layerSize, outputsSize))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


def featNorm(features):
    """Normalize features by mean and standard deviation.
    in order to be able to dernomalize them afterwards
    Args:
        features (np.array): un normlized np.array

    Returns:
        tuple: (normalizedFeatures, mean, standardDeviation)
    """

    mean = np.mean(features, axis=0)
    std = np.std(features - mean, axis=0)
    featuresNorm = (features - mean) / (std + np.finfo(np.double).eps)
    return (featuresNorm, mean, std)


def featDenorm(featuresNorm, mean, std):
    """Denormalize features by mean and standard deviation

    Args:
        features_norm (np.array): normlized np.array
        mean (float):  average of the array elements
        std (np.array): standard deviation, a measure of the spread of the array elements

    Returns:
        np.array: un normlized np.array
    """
    features = (featuresNorm * std) + mean
    return features


def getDataFiles(dataFolder, filePrefix=''):
    """Get the data files generated on maya

    Args:
        dataFolder (str): folder path where the files were saved
        filePrefix (str, optional): prefix name to the saved files

    Returns:
        tuple: (inputs (np.array), outputs(np.array))
    """
    csvIn = os.path.join(dataFolder, f'{filePrefix}{constants.INFILESUFIX}.{constants.NUMPYIOFORMAT}')
    csvOut = os.path.join(dataFolder, f'{filePrefix}{constants.OUTFILESUFIX}.{constants.NUMPYIOFORMAT}')
    inputs = np.loadtxt(csvIn)
    outputs = np.loadtxt(csvOut)
    return inputs, outputs


def shufleLists(inLists):
    """random re order several lists with the same shufle fot all the lists
    Args:
        inLists (list): list of lists
    Returns:
        list: list of re ordered list
    """
    indices = np.arange(len(inLists[0]))
    np.random.shuffle(indices)
    return [a[indices] for a in inLists]


def fwdPass(model, X_data, y_data, loss_func, optimizer, train=False):
    """basic forward pass calculation process, set values of the output layers
    from the inputs data by traversing through all neurons from first to last layer
    Args:
        model (toch.nn.Model): network model
        X_data (torch.FloatTensor): all inputs values
        y_data (torch.FloatTensor): all expected output values
        loss_func (torch.nn.modules.loss): function used to calculate the loss between output and y_data
        optimizer (torch.optim): algorithm ussed to reduce the loss
        train (bool, optional): define if should update the model neurons with the new data.
            used to diferenciate the actual traint from the test. Defaults to False.

    Returns:
        tuple: (accuracy, loss) values to mesure the training progress
    """
    X = torch.FloatTensor(X_data).to(DEVICE)
    y = torch.FloatTensor(y_data).to(DEVICE)
    n_items = len(y)
    if train:
        model.zero_grad()
    pred = model(X)
    loss = loss_func(pred, y)
    n_correct = torch.sum((torch.abs(pred - y) < torch.abs(.1 * y)))
    acc = n_correct.item()/(n_items)
    if train:
        loss.backward()
        optimizer.step()
    return acc, loss


def testPass(model, X_data, y_data, loss_func, optimizer, size=32):
    """same process than the foward pass, but withouth changing the values of the
    model (it won't train) and with data never seen for the model (test batch)
    Args:
        model (toch.nn.Model): network model
        X_data (torch.FloatTensor): all inputs values
        y_data (torch.FloatTensor): all expected output values
        loss_func (torch.nn.modules.loss): function used to calculate the loss between output and y_data
        optimizer (torch.optim): algorithm ussed to reduce the loss
        size (int, optional): mini bach to random test. Defaults to 32.
    Returns:
        tuple: (accuracy, loss) values to mesure the model with unseen data
    """
    test_X, test_y = shufleLists(X_data, y_data)
    X, y = test_X[:size], test_y[:size]
    val_acc, val_loss = fwdPass(model, X, y, loss_func, optimizer)
    return val_acc, val_loss


def rivetModelFit(net, X_data, y_data, batch_size, epochs, lr, valPercent, model_log_file=''):
    """Fit the model to match the rivet data, separating a percent of the data for validation test
    (test the model with unseen data), and separate the data in batchs in case that might be necessary
    to avoid overflow GPU memmory.

    Args:
        net (LRModel): linear regression model
        X_data (torch.FloatTensor): all inputs values
        y_data (torch.FloatTensor): all expected output values
        batch_size (int): separate the data in smaller batches
        epochs (int): how many times train the model with the same data
        lr (float): optimazer learning rate, the amount that the weights are updated during training
        valPercent (float): percecnt of the data separated to test model validation
        model_log_file (str, optional): file path to save the log file. Defaults to ''.
    """
    val_size = int(len(X_data) * valPercent)
    optimizer = optim.Adam(net.parameters(), lr=lr)
    loss_func = nn.MSELoss()
    for ep in range(epochs):
        X_shufl, y_shufl = shufleLists(X_data, y_data)
        train_X = X_shufl[:-val_size]
        train_y = y_shufl[:-val_size]
        test_X = X_shufl[-val_size:]
        test_y = y_shufl[-val_size:]

        for i in tqdm(range(0, len(train_X), batch_size)):
            batch_X = train_X[i:i + batch_size]
            batch_y = train_y[i:i + batch_size]
            acc, loss = fwdPass(net, batch_X, batch_y, loss_func, optimizer, train=True)
            if i % 50 == 0:
                val_acc, val_loss = testPass(net, test_X, test_y, loss_func, optimizer, size=100)
                log = f'"epoch":{ep}, "acc":{round(acc, 6)}, "loss":{round(float(loss), 6)}'
                log += f', "val_acc":{round(val_acc, 6)}, "val_loss":{round(float(val_loss), 6)}'
                _logger.info(log)
                if not model_log_file or not os.path.exists(model_log_file):
                    continue
                with open(model_log_file, "a") as f:
                    f.write(f"{log}\n")


def saveModelZipData(dataDict, filePath):
    """save normalize data to be used on the ml_rivet.mll node
    Args:
        dataDict (dict): dictionary of normalize data (mean and standar deviation)
        filePath (str): path were to save the xip file
    """
    file_paths = list()
    dirpath = tempfile.mkdtemp()
    for key, value in dataDict.items():
        csvPath = os.path.join(dirpath, f'{key}.{constants.NUMPYIOFORMAT}')
        np.savetxt(csvPath, value)
        file_paths.append(csvPath)

    with ZipFile(filePath, 'w') as zip:
        for each in file_paths:
            zip.write(each)
    shutil.rmtree(dirpath)


def train(outPath,
          dataPath,
          valPercent=.1,
          epochs=1000,
          prefixFileName=''):
    """train the model with the pre generated data and save the model and normalize data ready to
    be used on the ml_rivet node

    Args:
        outPath (str): folder where to save the model and data zip
        dataPath (str): folder where to get the data files saved by getRivetsSceneData module
        valPercent (float, optional): how much data separate for validation process. Defaults to .1.
        epochs (int, optional): how may times train with the same data. Defaults to 1000.
        prefixFileName (str, optional): prefix to saved files. Defaults to ''.
    """
    if not os.path.exists(outPath):
        os.makedirs(outPath)
    # normalize the input and output data
    inputs, outputs = getDataFiles(dataPath, prefixFileName)
    inputsNorm, inputsMean, inputsStd = featNorm(inputs)
    outputsNorm, outputsMean, outputsStd = featNorm(outputs)
    inputsTensor = torch.FloatTensor(inputsNorm)
    outputsTensor = torch.FloatTensor(outputsNorm)
    # hardcoded filenames inside the zip files for now becasue are the names
    # that the node search inside the zip
    dataDict = {'in_mean': inputsMean,
                'in_std': inputsStd,
                'out_mean': outputsMean,
                'out_std': outputsStd,
                }
    zipPath = os.path.join(outPath, f'{prefixFileName}modelData.zip')
    saveModelZipData(dataDict, zipPath)
    # configurable inputs for test differents parameters (maybe a creating a confusion Matrix)
    lr = .001
    batch_size = 5000
    extraLayers = 0
    sampling = len(inputsTensor)
    logFileName = f"{prefixFileName}model_{int(time.time())}.log"  # gives a dynamic model name
    model_log_file = os.path.join(outPath, logFileName)
    # create and train model
    model = LRModel(inputsTensor.shape[1], outputsTensor.shape[1], extraLayers).to(DEVICE)
    rivetModelFit(model.float(), inputsTensor[:sampling], outputsTensor[:sampling],
                  batch_size, epochs, lr, valPercent, model_log_file)
    dummy_input = torch.randn(1, inputsTensor.shape[1]).to(DEVICE)
    # save model
    torch.onnx.export(model.float(), dummy_input, os.path.join(outPath, f'{prefixFileName}model.onnx'))


train(outPath=r'D:/projects/ml_rivet/model', dataPath=r'D:/projects/ml_rivet/data', prefixFileName='mery_')
