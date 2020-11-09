import algorithms.extra_feature_sift as extra_feature_sift
import algorithms.classify_svm as classify_svm
import algorithms.lower_dimen_pca as lower_dimen_pca
import time


startTime = time.time()

# 用SIFT方法进行特征预处理
extra_feature_sift.run(method_generateFaceRS='sift')

# 读取SIFT特征处理的结果，并用PCA方法进行特征降维
lower_dimen_pca.run(
    method_readFaceRS='sift', method_generateUpdateFaceRS='sift_pca',
    n_components=99
)

# 用SVM进行分类
classify_svm.run(method_readFaceRS='sift_pca')

endTime = time.time()

print('\nSIFT_PCA_SVM costs %.2f seconds.' % (endTime - startTime))
