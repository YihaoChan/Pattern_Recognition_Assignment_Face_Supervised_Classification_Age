import algorithms.extra_feature_lbp as extra_feature_lbp
import algorithms.classify_cnn as classify_cnn
import time


startTime = time.time()

# 用LBP方法进行特征预处理
extra_feature_lbp.run(method_generateFaceRS='lbp')

# 用CNN进行分类
classify_cnn.run(method_readFaceRS='lbp')

endTime = time.time()

print('\nLBP_CNN costs %.2f seconds.' % (endTime - startTime))
