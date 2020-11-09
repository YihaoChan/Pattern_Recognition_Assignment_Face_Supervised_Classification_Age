# coding=utf-8
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn import neighbors
from fileio import readFaceRS
import plot


def predict(X_train, X_test, y_train, y_test, k, method_name):

    print('Start knn predicting...')

    knn = neighbors.KNeighborsClassifier(
        n_neighbors=k, weights='distance', algorithm='auto',
        leaf_size=30, p=2, metric='minkowski', metric_params=None, n_jobs=-1
    )
    knn_ovo = OneVsOneClassifier(knn)
    knn_ovo.fit(X_train, y_train.values.ravel())
    print('Accuracy score of knn_ovo: ' + '%.3f' %
          knn_ovo.score(X_test, y_test))

    knn_ovr = OneVsRestClassifier(knn)
    knn_ovr.fit(X_train, y_train.values.ravel())

    print('Accuracy score of knn_ovr: ' + '%.3f' %
          knn_ovr.score(X_test, y_test))

    plot.plot_conf_matrix(X_test, y_test, knn_ovr, method_name+'_ovr')
    plot.plot_conf_matrix(X_test, y_test, knn_ovo, method_name+'_ovo')
    plot.plot_roc(X_train, X_test, y_train, y_test,
                  knn_ovr, method_name+'_ovr')


def run(method_readFaceRS='sift', k=70):

    X_train, X_test, y_train, y_test = readFaceRS(method_readFaceRS)

    # age
    y_train_age = y_train[['age']]
    y_test_age = y_test[['age']]

    method_name = method_readFaceRS + '_knn'

    predict(X_train[:, 1:], X_test[:, 1:],
            y_train_age, y_test_age, k, method_name)
