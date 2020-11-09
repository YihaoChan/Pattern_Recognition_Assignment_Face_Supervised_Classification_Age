import algorithms.extra_feature_lbp as extra_feature_lbp
import algorithms.classify_random as classify_random
import time


startTime = time.time()

# 用LBP方法进行特征预处理
extra_feature_lbp.run(method_generateFaceRS='lbp')

# 用RandomForest进行分类
classify_random.run(method_readFaceRS='lbp')

endTime = time.time()

print('\nLBP_Random costs %.2f seconds.' % (endTime - startTime))
