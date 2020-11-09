# coding=utf-8
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multiclass import OneVsOneClassifier
from fileio import readFaceRS


def predict(X_train, X_test, y_train, y_test):
    print('Start RandomForest predicting...')

    ran = RandomForestClassifier()
    ran_ovo = OneVsOneClassifier(ran)
    ran_ovo.fit(X_train, y_train.values.ravel())
    print('Accuracy score of ran_ovo: ' + '%.3f' %
          ran_ovo.score(X_test, y_test))

    ran_ovr = OneVsRestClassifier(ran)
    ran_ovr.fit(X_train, y_train.values.ravel())

    print('Accuracy score of ran_ovr: ' + '%.3f' %
          ran_ovr.score(X_test, y_test))


def run(method_readFaceRS='lbp'):
    # 读取经过上一步特征处理处理的四个输出(lbp/lbp_pca)
    X_train, X_test, y_train, y_test = readFaceRS(method_readFaceRS)

    # 取age
    y_train_age = y_train[['age']]
    y_test_age = y_test[['age']]

    predict(X_train, X_test, y_train_age, y_test_age)
