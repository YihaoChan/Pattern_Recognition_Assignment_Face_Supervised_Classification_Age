# coding=utf-8
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multiclass import OneVsOneClassifier
from fileio import readFaceRS
from sklearn.svm import SVC
import algorithms.plot as plot


def predict(X_train, X_test, y_train, y_test, method_name):
    print('Start SVM predicting...')

    svm_ovo = OneVsOneClassifier(SVC(kernel='rbf', probability=True))
    svm_ovo.fit(X_train, y_train.values.ravel())

    print('Accuracy score of svm_ovo: ' + '%.3f' %
          svm_ovo.score(X_test, y_test))

    svm_ovr = OneVsRestClassifier(SVC(kernel='rbf', probability=True))
    svm_ovr.fit(X_train, y_train.values.ravel())

    print('Accuracy score of svm_ovr: ' + '%.3f' %
          svm_ovr.score(X_test, y_test))

    plot.plot_conf_matrix(X_test, y_test, svm_ovo, method_name+'_ovo')
    plot.plot_conf_matrix(X_test, y_test, svm_ovr, method_name+'_ovr')
    plot.plot_roc(X_train, X_test, y_train, y_test,
                  svm_ovr, method_name+'_ovr')


def run(method_readFaceRS='hog'):
    # 读取经过上一步特征处理处理的四个输出(hog/hog_pca)
    X_train, X_test, y_train, y_test = readFaceRS(method_readFaceRS)

    # 取age
    y_train_age = y_train[['age']]
    y_test_age = y_test[['age']]

    method_name = method_readFaceRS + '_svm'
    predict(X_train, X_test, y_train_age, y_test_age, method_name)
