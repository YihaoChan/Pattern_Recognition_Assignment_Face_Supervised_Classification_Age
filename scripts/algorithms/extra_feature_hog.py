# coding=utf-8
from fileio import generateFaceRS, readRiIndex
from skimage.feature import hog
import numpy as np


def extra_feature(img_array):
    print('Start HOG feature preprocessing...')

    img_array_height = img_array.shape[0]

    for i in range(img_array.shape[0]):
        img = img_array[i].reshape((128, 128))

        fd = hog(
            img, orientations=6, block_norm='L1',
            pixels_per_cell=(5, 5), cells_per_block=(2, 2),
            visualize=False,
            transform_sqrt=True,
            multichannel=False
        )

        print('No.' + str(i) + ' shape is ' + str(fd.shape) + '\t\t' + '%.2f' %
              (i * 100 / img_array_height) + '%')

        fd = fd.reshape(1, fd.shape[0])

        if (i == 0):
            face = fd
        elif (i >= 1):
            face = np.concatenate((face, fd), axis=0)

    print('Shape of the result:' + str(face.shape))

    return face


def run(method_generateFaceRS='hog'):
    # 读取原始图片与索引
    ri, index = readRiIndex()

    X = extra_feature(ri)
    X = np.concatenate([index, X], axis=1)

    # 进行HOG特征处理后，划分数据集，并将结果写到目录里
    generateFaceRS(X, method_generateFaceRS)
