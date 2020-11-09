# coding=utf-8
from fileio import generateFaceRS, readRiIndex
from torch.autograd import Variable
from torchvision import models
import directory as dirt
import torch.nn as nn
import numpy as np
import torch


def extra_feature(ri, net, use_gpu):
    print('Start ResNet feature preprocessing...')

    img_array_height = ri.shape[0]

    for i in range(img_array_height):

        eachImage = ri[i]

        # 二维变成三维
        singleChannelImage = np.reshape(eachImage, (1, 128, 128))

        # 单通道变成三通道
        multiChannelImage = np.concatenate(
            (singleChannelImage, singleChannelImage, singleChannelImage), axis=0
        )

        # numpy array转torch tensor
        multiChannelImage_tensor = torch.from_numpy(multiChannelImage)

        x = Variable(torch.unsqueeze(
            multiChannelImage_tensor, dim=0).float(), requires_grad=False
        )

        if use_gpu:
            x = x.cuda()
            net = net.cuda()

        y = net(x).cpu()
        y = y.data.numpy()

        print('No.' + str(i) + ' shape is ' + str(x.shape) + '\t\t' + '%.2f' %
              (i * 100 / img_array_height) + '%')

        if (i == 0):
            face = y
        elif (i >= 1):
            face = np.concatenate((face, y), axis=0)

    print('Shape of the result:' + str(face.shape))

    return face


def run(method_generateFaceRS='resnet'):
    ri, index = readRiIndex()

    resnet50_feature_extractor = models.resnet50(pretrained=True)
    # resnet152 = models.resnet152(pretrained = True)
    # densenet201 = models.densenet201(pretrained = True)

    resnet50_feature_extractor.fc = nn.Linear(2048, 2048)
    torch.nn.init.eye(resnet50_feature_extractor.fc.weight)

    for param in resnet50_feature_extractor.parameters():
        param.requires_grad = False

    use_gpu = torch.cuda.is_available()

    X = extra_feature(ri, resnet50_feature_extractor, use_gpu)

    X = np.concatenate([index, X], axis=1)

    # 进行ResNet特征处理后，划分数据集，并将结果写到目录里
    generateFaceRS(X, method_generateFaceRS)
