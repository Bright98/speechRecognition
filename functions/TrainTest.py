import numpy as np


def CreateTrainAndTestData(feature_matrix, dataNumber1, dataNumber2):

    trainPersent = 0.7  # 70 % train
    testPersent = 0.3  # 30 % test

    trainDataNumber1 = int(dataNumber1 * trainPersent)
    testDataNumber1 = int(dataNumber1 * testPersent)
    trainDataNumber2 = int(dataNumber2 * trainPersent)
    testDataNumber2 = int(dataNumber2 * testPersent)

    trainFeature = []
    trainLable = []
    testFeature = []
    testLable = []
    row, col = feature_matrix.shape

    for i in range(trainDataNumber1):
        trainFeature.append(feature_matrix[i, 0 : col - 1])
        trainLable.append(feature_matrix[i, col - 1])

    for i in range(trainDataNumber1, dataNumber1):
        testFeature.append(feature_matrix[i, 0 : col - 1])
        testLable.append(feature_matrix[i, col - 1])

    for i in range(dataNumber1, int(dataNumber1 + trainDataNumber2)):
        trainFeature.append(feature_matrix[i, 0 : col - 1])
        trainLable.append(feature_matrix[i, col - 1])

    for i in range(int(dataNumber1 + trainDataNumber2), row):
        testFeature.append(feature_matrix[i, 0 : col - 1])
        testLable.append(feature_matrix[i, col - 1])

        # if i < trainDataNumber:
        #     trainFeature.append(feature_matrix[i, 0 : col - 1])
        #     trainLable.append(feature_matrix[i, col - 1])
        # else:
        #     testFeature.append(feature_matrix[i, 0 : col - 1])
        #     testLable.append(feature_matrix[i, col - 1])

    # for i in range(dataNumber, int(dataNumber * 2)):
    #     if i < int(dataNumber + trainDataNumber):
    #         trainFeature.append(feature_matrix[i, 0 : col - 1])
    #         trainLable.append(feature_matrix[i, col - 1])
    #     else:
    #         testFeature.append(feature_matrix[i, 0 : col - 1])
    #         testLable.append(feature_matrix[i, col - 1])

    trainFeature = np.array(trainFeature)
    trainLable = np.array(trainLable)
    testFeature = np.array(testFeature)
    testLable = np.array(testLable)

    return trainFeature, trainLable, testFeature, testLable
