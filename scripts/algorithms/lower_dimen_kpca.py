# coding=utf-8
from sklearn.decomposition import KernelPCA
from fileio import generateUpdateFaceRS
from fileio import readFaceRS
import numpy as np


def lower_dimen(X_train, X_test):
    print('Start KernalPCA lowering dimension...')

    X_train_index = X_train[:, 0].reshape(-1, 1)
    X_test_index = X_test[:, 0].reshape(-1, 1)

    X_train = X_train[:, 1:]
    X_test = X_test[:, 1:]

    kpca = KernelPCA(n_components=99, kernel='linear')
    faceR = kpca.fit_transform(X_train)
    faceS = kpca.fit_transform(X_test)

    faceR = np.concatenate([X_train_index, faceR], axis=1)
    faceS = np.concatenate([X_test_index, faceS], axis=1)

    print('Shape of faceR: ' + str(faceR.shape))
    print('Shape of faceS: ' + str(faceS.shape))

    return faceR, faceS


def run(
    method_readFaceRS='densenet',
    method_generateUpdateFaceRS='densenet_kpca'
):
    '''
    @param：进行特征提取+降维，第一个参数为特征提取方法，第二个参数为特征提取+降维方法
    '''
    # 读取已经经过特征处理过的四个文件作为输入
    X_train, X_test, y_train, y_test = readFaceRS(method_readFaceRS)

    # 特征提取+降维
    X_pca_train, X_pca_test = lower_dimen(X_train, X_test)

    # 把特征提取+降维和标签输出
    generateUpdateFaceRS(
        X_pca_train, X_pca_test, y_train, y_test,
        method_generateUpdateFaceRS
    )
