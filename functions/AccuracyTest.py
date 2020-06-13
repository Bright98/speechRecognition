from sklearn.metrics import accuracy_score
from joblib import load


def AccuracyTest(testFeature, testLable):

    # load model
    svm_model = load("model/svm_model.joblib")

    # test
    new_testLable = svm_model.predict(testFeature)

    # accuracy test
    score = accuracy_score(testLable, new_testLable.round(), normalize=False)
    print("accuracy of model: ", (score / len(testLable)) * 100, "%")
    return score
