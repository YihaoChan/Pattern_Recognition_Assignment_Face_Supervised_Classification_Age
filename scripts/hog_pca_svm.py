import algorithms.extra_feature_hog as extra_feature_hog
import algorithms.lower_dimen_pca as lower_dimen_pca
import algorithms.classify_svm as classify_svm
import time


startTime = time.time()

# 用HOG方法进行特征预处理
extra_feature_hog.run(method_generateFaceRS='hog')

# 读取HOG特征处理的结果，并用PCA方法进行特征降维
lower_dimen_pca.run(method_readFaceRS='hog',
                    method_generateUpdateFaceRS='hog_pca')

# 采用SVM进行分类，输出HOG_PCA_SVM犯错矩阵
classify_svm.run(method_readFaceRS='hog_pca')

endTime = time.time()

print('\nHOG_PCA_SVM costs %.2f seconds.' % (endTime - startTime))
