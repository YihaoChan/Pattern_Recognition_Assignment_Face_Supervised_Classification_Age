import algorithms.extra_feature_densenet as extra_feature_densenet
import algorithms.lower_dimen_kpca as lower_dimen_kpca
import algorithms.classify_cnn as classify_cnn
import time


startTime = time.time()

# 用DenseNet方法进行特征预处理
extra_feature_densenet.run(method_generateFaceRS='densenet')

# 读取DenseNet特征处理的结果，并用KPCA方法进行特征降维
lower_dimen_kpca.run(
    method_readFaceRS='densenet', method_generateUpdateFaceRS='densenet_kpca'
)

# 用CNN进行分类
classify_cnn.run(method_readFaceRS='densenet_kpca')

endTime = time.time()

print('\nDenseNet_KPCA_CNN costs %.2f seconds.' % (endTime - startTime))
