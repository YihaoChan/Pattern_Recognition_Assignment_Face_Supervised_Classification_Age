# coding=utf-8
from tensorflow.python.keras.layers import Dense, Activation, Convolution2D, MaxPooling2D, Flatten
from tensorflow.python.keras.models import Sequential
from fileio import readFaceRS
import pandas as pd
import warnings
warnings.filterwarnings('ignore')


class CNNInputSize(object):
    def __init__(self, method):
        if method == 'densenet':
            self.height = 1001
            self.width = 1
        elif method == 'densenet_kpca':
            self.height = 100
            self.width = 1


# 建立一个用于存储和格式化读取训练数据的类
class DataSet(object):
    def __init__(self, X_train, X_test, Y_train, Y_test, CNNInputSize):

        self.X_train = X_train.reshape(
            [-1, CNNInputSize.height, CNNInputSize.width, 1])
        self.X_test = X_test.reshape(
            [-1, CNNInputSize.height, CNNInputSize.width, 1])
        self.Y_train = Y_train.values
        self.Y_test = Y_test.values
        self.img_size = 128
        self.num_classes = self.Y_train.shape[1]


# 建立一个CNN模型，一层卷积、一层池化、一层卷积、一层池化、抹平之后进行全链接、最后进行分类
def build_model(dataset):
    model = Sequential()
    model.add(
        Convolution2D(
            filters=64,
            kernel_size=(5, 5),
            padding='same',
            data_format='channels_last',
            input_shape=dataset.X_train.shape[1:]
        )
    )

    model.add(Activation('relu'))
    model.add(
        MaxPooling2D(
            pool_size=(2, 2),
            strides=(2, 2),
            padding='same'
        )
    )

    model.add(Convolution2D(filters=64, kernel_size=(5, 5), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))

    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))

    model.add(Dense(dataset.num_classes))
    model.add(Activation('softmax'))
    model.summary()
    return model


# 进行模型训练的函数，具体的optimizer、loss可以进行不同选择
def train_model(model, dataset):
    model.compile(
        optimizer='adam',  # 可选不同optimizer，例如RMSprop,Adagrad
        loss='categorical_crossentropy',  # 可以选用squared_hinge作为loss
        metrics=['accuracy'])

    # epochs为训练多少轮、batch_size为每次训练多少个样本
    model.fit(dataset.X_train, dataset.Y_train, epochs=100, batch_size=32)


def evaluate_model(model, dataset):
    print('\nTesting---------------')
    loss, accuracy = model.evaluate(dataset.X_test, dataset.Y_test)

    print('test loss;', loss)
    print('test accuracy:', accuracy)


def run(method_readFaceRS='densetnet'):

    X_train, X_test, y_train, y_test = readFaceRS(method_readFaceRS)

    # 在标签中取出age列
    y_train_age = y_train[['age']]
    y_test_age = y_test[['age']]

    # 得到one-hot编码
    y_train_age_one = pd.get_dummies(y_train_age)
    y_test_age_one = pd.get_dummies(y_test_age)

    input_size = CNNInputSize(method=method_readFaceRS)
    dataset = DataSet(X_train, X_test, y_train_age_one,
                      y_test_age_one, input_size)
    model = build_model(dataset)
    train_model(model, dataset)
    evaluate_model(model, dataset)
