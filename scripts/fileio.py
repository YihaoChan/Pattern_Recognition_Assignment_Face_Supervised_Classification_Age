# coding=utf-8
from sklearn.model_selection import train_test_split
import directory as dirt
from cv2 import cv2
import pandas as pd
import numpy as np
import os
import re


def mkdir_all(datapath):
    print('Generating paths: ')
    for path in datapath:
        print(path)
        if not os.path.exists(path):
            os.mkdir(path)


def rawdata_restore():
    '''
    @description: 二进制恢复成jpg
    '''
    raw_files = os.listdir(dirt.Dirt.raw_data_path)

    count = 0

    for raw_file in raw_files:

        f = open(dirt.Dirt.raw_data_path + "//" + raw_file)

        image = np.fromfile(f, dtype=np.uint8)

        if image.size != 262144:  # 有两张256KB的图
            image = np.reshape(image, (128, 128))
            image = cv2.resize(image, (780, 380))
        else:  # 其余的图都是16KB
            image = np.reshape(image, (512, 512))
            image = cv2.resize(image, (780, 380))

        cv2.waitKey(0)

        cv2.imwrite(dirt.Dirt.restored_path + '//' + raw_file + '.jpg', image)

        count += 1

        print('%d / %d' % (count, len(raw_files)))


def generateRi():
    '''
    @description: 把所有原始图片以矩阵形式存放
    '''
    raw_files = os.listdir(dirt.Dirt.raw_data_path)

    times = 0

    for raw_file in raw_files:
        with open(dirt.Dirt.raw_data_path + raw_file) as f:
            In = np.fromfile(f, dtype=np.uint8).reshape(1, -1)

            if times == 0:
                ri = In
            else:
                if ri.shape[1] != In.shape[1]:
                    In = cv2.resize(In, (ri.shape[1], 1))
                ri = np.concatenate((In, ri), axis=0)

            times += 1

            print('ri-shape:' + str(ri.shape))

    np.savetxt(dirt.ri_path, ri, fmt='%d')

    print('ri is done:' + str(ri.shape))


def generateIndex():
    '''
    @description: 生成图片的编号索引，并存入index文件中
    '''
    raw_files = os.listdir(dirt.Dirt.raw_data_path)
    raw_files = list(map(int, raw_files))
    raw_files = np.array(raw_files, dtype='int32')
    np.savetxt(dirt.index_path, raw_files, fmt='%d')


def readRiIndex():
    '''
    @description: 读取所有图片的大矩阵中矩阵的维度和索引编号
    '''
    ri = np.loadtxt(dirt.ri_path)
    print('ri import done. ri-shape: ' + str(ri.shape))

    index = np.loadtxt(dirt.index_path, dtype='int32').reshape(-1, 1)
    print('index import done. ri-shape: ' + str(index.shape))

    return ri, index


def generateCSV():
    '''
    @description: 把原始的全部标签文本文件转换为CSV格式
    '''
    result = pd.DataFrame(
        columns=('idx', 'sex', 'age', 'race', 'face', 'prop')
    )

    with open(dirt.faceDR_path) as f:
        lines = f.readlines()

    for line in lines:
        if not re.search('_missing', line):
            index = re.match(' ([0-9]{4})', line).group(1)
            print(index)
            sex = re.search('\(_sex  (.*?)\)', line).group(1)
            age = re.search('\(_age  (.*?)\)', line).group(1)
            race = re.search('\(_race (.*?)\)', line).group(1)
            face = re.search('\(_face (.*?)\)', line).group(1)
            prop = re.search('\(_prop \'\((.*?)\)', line).group(1)
            if not prop:
                prop = 'None'
            result = result.append(pd.DataFrame({'idx': [index], 'sex': [sex], 'age': [
                                   age], 'race': [race], 'face': [face], 'prop': [prop]}), ignore_index=True)
        else:
            continue

    with open(dirt.faceDS_path) as f:
        lines = f.readlines()
    for line in lines:
        if not re.search('_missing', line):
            index = re.match(' ([0-9]{4})', line).group(1)
            print(index)
            sex = re.search('\(_sex  (.*?)\)', line).group(1)
            age = re.search('\(_age  (.*?)\)', line).group(1)
            race = re.search('\(_race (.*?)\)', line).group(1)
            face = re.search('\(_face (.*?)\)', line).group(1)
            prop = re.search('\(_prop \'\((.*?)\)', line).group(1)
            if not prop:
                prop = 'None'
            result = result.append(pd.DataFrame({'idx': [index], 'sex': [sex], 'age': [
                                   age], 'race': [race], 'face': [face], 'prop': [prop]}), ignore_index=True)
        else:
            continue

    result.to_csv(dirt.faceD_csv_path, index=False)


def generateFaceRS(X, method):
    '''
    @description: 特征处理的脚本中执行该函数，执行后进行特征处理，并划分数据集，将结果写到方法所在的目录里
    '''
    y = pd.read_csv(dirt.faceD_csv_path)

    print('X-shape: ' + str(X.shape))
    print('y-shape: ' + str(y.shape))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=7
    )

    if (method == 'hog'):
        file_path = dirt.HOGFile
    elif (method == 'resnet'):
        file_path = dirt.ResnetFile
    elif (method == 'sift'):
        file_path = dirt.SIFTFile
    elif (method == 'densenet'):
        file_path = dirt.DensenetFile
    elif (method == 'lbp'):
        file_path = dirt.LBPFile

    np.savetxt(
        file_path.faceR_path, X_train,
        fmt=['%d']+['%.6f']*(X_train.shape[1]-1)
    )
    np.savetxt(
        file_path.faceS_path, X_test,
        fmt=['%d']+['%.6f']*(X_test.shape[1]-1)
    )

    y_train.to_csv(file_path.faceDR_path, index=False)
    y_test.to_csv(file_path.faceDS_path, index=False)

    print('write file to ' + file_path._path)
    print('[write] X_train.shape, X_test.shape, y_train.shape, y_test.shape:')
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)


def readFaceRS(method):
    '''
    @description: 要用到特征处理结果的时候，执行该函数，读取特征处理后的训练、测试集结果，便于进行下一步降维/分类
    @parametre：method - 上一步生成四个文件时用的方法
    '''
    if (method == 'hog'):
        file_path = dirt.HOGFile
    elif (method == 'hog_pca'):
        file_path = dirt.HOG_PCAFile
    elif (method == 'resnet'):
        file_path = dirt.ResnetFile
    elif (method == 'resnet_kpca'):
        file_path = dirt.Resnet_KPCAFile
    elif (method == 'sift'):
        file_path = dirt.SIFTFile
    elif (method == 'sift_pca'):
        file_path = dirt.SIFT_PCAFile
    elif (method == 'lbp'):
        file_path = dirt.LBPFile
    elif (method == 'lbp_pca'):
        file_path = dirt.LBP_PCAFile
    elif (method == 'densenet'):
        file_path = dirt.DensenetFile
    elif (method == 'densenet_kpca'):
        file_path = dirt.Densenet_KPCAFile

    # 读取数据
    X_train = np.loadtxt(file_path.faceR_path)
    X_test = np.loadtxt(file_path.faceS_path)

    y_train = pd.read_csv(file_path.faceDR_path, index_col=0)
    y_test = pd.read_csv(file_path.faceDS_path, index_col=0)

    print('read file from ' + file_path._path)
    print('[read] X_train.shape, X_test.shape, y_train.shape, y_test.shape:')
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

    return X_train, X_test, y_train, y_test


def generateUpdateFaceRS(X_train, X_test, y_train, y_test, method):
    '''
    @description: 经过两种特征提取方法后的输出
    '''
    if (method == 'hog_pca'):
        file_path = dirt.HOG_PCAFile
    elif (method == 'resnet_kpca'):
        file_path = dirt.Resnet_KPCAFile
    elif (method == 'sift_pca'):
        file_path = dirt.SIFT_PCAFile
    elif (method == 'lbp_pca'):
        file_path = dirt.LBP_PCAFile
    elif (method == 'densenet_kpca'):
        file_path = dirt.Densenet_KPCAFile

    np.savetxt(
        file_path.faceR_path, X_train,
        fmt=['%d']+['%.6f']*(X_train.shape[1]-1)
    )
    np.savetxt(
        file_path.faceS_path, X_test,
        fmt=['%d']+['%.6f']*(X_test.shape[1]-1)
    )

    y_train.to_csv(file_path.faceDR_path)
    y_test.to_csv(file_path.faceDS_path)

    print('write file to ' + file_path._path)
    print('[write] X_train.shape, X_test.shape, y_train.shape, y_test.shape:')
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)


if __name__ == '__main__':
    directories_dict = dirt.Dirt.__dict__
    directories = [directories_dict[key]
                   for key in directories_dict if "__" not in key]

    mkdir_all(directories)

    if not os.path.exists(dirt.ri_path):
        generateRi()

    if not os.path.exists(dirt.index_path):
        generateIndex()

    if not os.path.exists(dirt.faceD_csv_path):
        generateCSV()

    # 生成原始图像，不必要一定运行
    genResFlag = True
    if not os.listdir(dirt.Dirt.restored_path) and genResFlag:
        rawdata_restore()
