# coding=utf-8
from sklearn.ensemble import AdaBoostClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multiclass import OneVsOneClassifier
from fileio import readFaceRS
import algorithms.plot as plot


def predict(X_train, X_test, y_train, y_test, method_name):

    # =========OVO=========
    ada_ovo = OneVsOneClassifier(
        AdaBoostClassifier(
            n_estimators=200
        )
    )
    ada_ovo.fit(X_train, y_train.values.ravel())

    # =========OVR=========
    ada_ovr = OneVsRestClassifier(
        AdaBoostClassifier(
            n_estimators=200
        )
    )
    ada_ovr.fit(X_train, y_train.values.ravel())

    plot.plot_conf_matrix(X_test, y_test, ada_ovo, method_name+'_ovo')
    plot.plot_conf_matrix(X_test, y_test, ada_ovr, method_name+'_ovr')
    plot.plot_roc(X_train, X_test, y_train, y_test,
                  ada_ovr, method_name+'_ovr')


def run(method_readFaceRS='lbp_pca'):
    # 读取经过上一步特征处理处理的四个输出(lbp/lbp_pca)
    X_train, X_test, y_train, y_test = readFaceRS(method_readFaceRS)

    # 取age
    y_train_age = y_train[['age']]
    y_test_age = y_test[['age']]

    method_name = method_readFaceRS + '_adaboost'

    predict(X_train, X_test, y_train_age, y_test_age, method_name)
