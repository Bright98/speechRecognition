from functions.MakeFeature import MakeFeature
from functions.TrainTest import CreateTrainAndTestData
from functions.SvmModel import CreateSVMModel
from functions.AccuracyTest import AccuracyTest
import numpy as np
import os


dataNumber1 = 0
dataNumber2 = 0
frameSize = 200
overlap = 2

dataFolder1 = "sholugh"
dataFolder2 = "khalvat"

feature_matrix = []


with os.scandir("data/" + dataFolder1) as datafolder:
    for data in datafolder:
        dataNumber1 += 1
        feature_matrix.append(
            MakeFeature(dataFolder1, dataNumber1, 1, frameSize, overlap)
        )

with os.scandir("data/" + dataFolder2) as datafolder:
    for data in datafolder:
        dataNumber2 += 1
        feature_matrix.append(
            MakeFeature(dataFolder2, dataNumber2, 2, frameSize, overlap)
        )

feature_matrix = np.array(feature_matrix)
# _____________________

# separate test and tran feature
trainFeature, trainLable, testFeature, testLable = CreateTrainAndTestData(
    feature_matrix, dataNumber1, dataNumber2
)
# _____________________

# train
CreateSVMModel(trainFeature, trainLable)
# _____________________

# test:
AccuracyTest(testFeature, testLable)
# _____________________
