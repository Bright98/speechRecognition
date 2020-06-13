from sklearn import svm
from joblib import dump


def CreateSVMModel(trainFeature, trainLable):

    clf = svm.SVR()
    svm_model = clf.fit(trainFeature, trainLable)

    # save the model
    dump(clf, "model/svm_model.joblib")
