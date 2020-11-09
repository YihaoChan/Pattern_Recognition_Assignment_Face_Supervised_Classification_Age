from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import classification_report
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from itertools import cycle
import directory as dirt
from scipy import interp
import numpy as np


def plot_conf_matrix(X_test, y_test, classifier, method_name):
    class_names = ['child', 'teen', 'adult', 'senior']
    # Plot normalized confusion matrix
    disp = plot_confusion_matrix(classifier, X_test, y_test,
                                 display_labels=class_names,
                                 cmap=plt.cm.Blues,
                                 normalize='true')
    disp.ax_.set_title('Normalized confusion matrix')

    # 混淆矩阵
    confusionMatrix = disp.confusion_matrix
    print('Confusion Matrix:\n', confusionMatrix)
    plt.savefig(dirt.Dirt.pic_path+method_name+'_confM.jpg')

    # 将犯错矩阵可视化
    row_sums = np.sum(confusionMatrix, axis=1)  # 求行和
    err_matrix = confusionMatrix / row_sums  # 求错误所占百分比
    np.fill_diagonal(err_matrix, 0)
    print('Error Matrix:\n', err_matrix)

    # 存图片
    plt.matshow(err_matrix, cmap=plt.cm.gray)
    plt.savefig(dirt.Dirt.pic_path+method_name + '_errM.jpg')

    # precision、recall、f1_score，默认weighted，考虑不平衡
    y_pred = classifier.predict(X_test)
    print('Report:\n', classification_report(y_test, y_pred))


def plot_roc(X_train, X_test, y_train, y_test, classifier, method_name):
    n_classes = 4

    # onehot 编码
    class_names = ['child', 'teen', 'adult', 'senior']
    y_train = label_binarize(y_train, classes=class_names)
    y_test = label_binarize(y_test, classes=class_names)

    X = np.concatenate([X_train, X_test])
    y = np.concatenate([y_train, y_test])
    y_score = classifier.predict_proba(X_test)
    # y_score = classifier.fit(X_train, y_train).decision_function(X_test)
    print(y_score.shape)
    # Compute ROC curve and ROC area for each class
    fpr = dict()  # 伪阳性率
    tpr = dict()  # 真阳性率
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr['micro'], tpr['micro'], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc['micro'] = auc(fpr['micro'], tpr['micro'])

    # 四组ROC曲线对应的fpr（即横坐标不一样），要得到四组ROC曲线的平均曲线，进行如下的操作。
    all_fpr = np.unique(np.concatenate(
        [fpr[i] for i in range(n_classes)]))  # 拼接数组并去除重复的

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])  # 利用一维插值的方式得到tpr

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr['macro'] = all_fpr
    tpr['macro'] = mean_tpr
    roc_auc['macro'] = auc(fpr['macro'], tpr['macro'])

    print('原来四组ROC曲线的坐标数量：', fpr[0].shape[0],
          fpr[1].shape[0], fpr[2].shape[0], fpr[3].shape[0])
    print('利用插值法得到的三组ROC曲线的平均曲线的坐标数量(macro)：', all_fpr.shape[0])
    print('将所有类当做一个类得到的ROC曲线的坐标数量(micro)：', fpr['micro'].shape[0])

    # Plot all ROC curves
    plt.figure()
    lw = 2
    plt.plot(fpr['micro'], tpr['micro'],
             label='micro-average ROC curve (area = {0:0.2f})'
             ''.format(roc_auc['micro']),
             color='deeppink', linestyle=':', linewidth=4)

    plt.plot(fpr['macro'], tpr['macro'],
             label='macro-average ROC curve (area = {0:0.2f})'
             ''.format(roc_auc['macro']),
             color='navy', linestyle=':', linewidth=4)

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'green'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                 ''.format(class_names[i], roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic to multi-class')
    plt.legend(loc='lower right')
    plt.savefig(dirt.Dirt.pic_path+method_name+'_roc.jpg')
