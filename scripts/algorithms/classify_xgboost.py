# coding=utf-8
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multiclass import OneVsOneClassifier
from xgboost.sklearn import XGBClassifier
from fileio import readFaceRS
import algorithms.plot as plot


def predict(X_train, X_test, y_train, y_test, method_name):
    print('Start XGBoost predicting...')

    xgb_ovo = OneVsOneClassifier(XGBClassifier())
    xgb_ovo.fit(X_train, y_train.values.ravel())

    print('Accuracy score of xgb_ovo: ' + '%.3f' %
          xgb_ovo.score(X_test, y_test))

    xgb_ovr = OneVsRestClassifier(XGBClassifier())
    xgb_ovr.fit(X_train, y_train.values.ravel())

    print('Accuracy score of xgb_ovr: ' + '%.3f' %
          xgb_ovr.score(X_test, y_test))

    plot.plot_conf_matrix(X_test, y_test, xgb_ovo, method_name+'_ovo')
    plot.plot_conf_matrix(X_test, y_test, xgb_ovr, method_name+'_ovr')
    plot.plot_roc(X_train, X_test, y_train, y_test,
                  xgb_ovr, method_name+'_ovr')


def run(method_readFaceRS='resnet'):
    # 读取经过上一步特征处理处理的四个输出(ResNet/ResNet_kpca)
    X_train, X_test, y_train, y_test = readFaceRS(method_readFaceRS)

    # 取age
    y_train_age = y_train[['age']]
    y_test_age = y_test[['age']]

    method_name = method_readFaceRS + '_xgb'
    predict(X_train, X_test, y_train_age, y_test_age, method_name)
