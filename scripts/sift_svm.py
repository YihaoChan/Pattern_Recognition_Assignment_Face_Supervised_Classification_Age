import algorithms.extra_feature_sift as extra_feature_sift
import algorithms.classify_svm as classify_svm
import time


startTime = time.time()

# 用SIFT方法进行特征预处理
extra_feature_sift.run(method_generateFaceRS='sift')

# 用SVM进行分类
classify_svm.run(method_readFaceRS='sift')

endTime = time.time()

print('\nSIFT_SVM costs %.2f seconds.' % (endTime - startTime))
