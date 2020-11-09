import algorithms.extra_feature_resnet as extra_feature_resnet
import algorithms.classify_xgboost as classifier_xgboost
import algorithms.lower_dimen_kpca as lower_dimen_kpca
import time


startTime = time.time()

# 执行ResNet50方法进行特征提取
extra_feature_resnet.run(method_generateFaceRS='resnet')

# 用kpca特征降维
lower_dimen_kpca.run(method_readFaceRS='resnet',
                     method_generateUpdateFaceRS='resnet_kpca')

# 执行XGBoost方法进行分类，输出ResNet_XGBoost犯错矩阵
classifier_xgboost.run(method_readFaceRS='resnet_kpca')

endTime = time.time()

print('\nResNet_KPCA_XGBoost costs %.2f seconds.' % (endTime - startTime))
