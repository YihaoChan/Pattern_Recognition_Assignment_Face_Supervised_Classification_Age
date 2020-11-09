import algorithms.extra_feature_sift as extra_feature_sift
import algorithms.classify_knn as classify_knn
import time


startTime = time.time()

# 用SIFT方法进行特征预处理
extra_feature_sift.run(method_generateFaceRS='sift')

# 用KNN进行分类
classify_knn.run(method_readFaceRS='sift', k=70)

endTime = time.time()

print('\nSIFT_KNN costs %.2f seconds.' % (endTime - startTime))
