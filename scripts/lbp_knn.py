import algorithms.extra_feature_lbp as extra_feature_lbp
import algorithms.classify_knn as classify_knn
import time


startTime = time.time()

# 用LBP方法进行特征预处理
extra_feature_lbp.run(method_generateFaceRS='lbp')

# 用KNN进行分类
classify_knn.run(method_readFaceRS='lbp', k=70)

endTime = time.time()

print('\nLBP_KNN costs %.2f seconds.' % (endTime - startTime))
