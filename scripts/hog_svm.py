import algorithms.extra_feature_hog as extra_feature_hog
import algorithms.classify_svm as classify_svm
import time


startTime = time.time()

# 用HOG方法进行特征预处理
extra_feature_hog.run(method_generateFaceRS='hog')

# 采用SVM进行分类，输出HOG_SVM犯错矩阵
classify_svm.run(method_readFaceRS='hog')

endTime = time.time()

print('\nHOG_SVM costs %.2f seconds.' % (endTime - startTime))
