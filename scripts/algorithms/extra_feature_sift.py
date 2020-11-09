# coding=utf-8
from fileio import generateFaceRS, readRiIndex
from sklearn.decomposition import PCA
import numpy as np
import cv2


def extra_feature(readImage):
    print('Start SIFT feature preprocessing...')

    img_array = readImage.astype('uint8')
    img_array_height = img_array.shape[0]
    sift = cv2.xfeatures2d.SIFT_create()
    pca = PCA(n_components=15)

    for i in range(img_array.shape[0]):
        img = img_array[i].reshape((128, 128))
        _, fd = sift.detectAndCompute(img, None)

        if fd is None:
            fd = np.zeros([15, 128])
        elif fd.shape[0] < 15:
            fd = np.concatenate((fd, np.zeros([15-fd.shape[0], 128])), axis=0)
        fd = pca.fit_transform(fd.T)

        print('No.' + str(i)+' shape is ' + str(fd.shape) + '\t\t' + '%.2f' %
              (i * 100 / img_array_height) + '%')
        fd = fd.reshape(1, -1)

        if (i == 0):
            face = fd
        elif (i >= 1):
            face = np.concatenate((face, fd), axis=0)

    print('Shape of the result:' + str(face.shape))

    return face


def run(method_generateFaceRS='sift'):
    # 读取原始图片与索引
    ri, index = readRiIndex()

    X = extra_feature(ri)
    X = np.concatenate([index, X], axis=1)

    # 进行SIFT特征处理后，划分数据集，并将结果写到目录里
    generateFaceRS(X, method_generateFaceRS)
