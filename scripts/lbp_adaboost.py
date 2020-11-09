import algorithms.extra_feature_lbp as extra_feature_lbp
import algorithms.classify_adaboost as classify_adaboost
import time


startTime = time.time()

# 用LBP方法进行特征预处理
extra_feature_lbp.run(method_generateFaceRS='lbp')

# 采用Adaboost进行分类，输出LBP_Adaboost犯错矩阵
classify_adaboost.run(method_readFaceRS='lbp')

endTime = time.time()

print('\nLBP_AdaBoost costs: %.2f seconds.' % (endTime - startTime))
