@echo off

rem 安装相关依赖
pip install -r requirements.txt

rem 生成必要数据文件及目录
python scripts/fileio.py

rem 执行算法组合
python scripts/hog_svm.py
python scripts/hog_pca_svm.py
python scripts/resnet_xgboost.py
python scripts/resnet_kpca_xgboost.py
python scripts/lbp_adaboost.py
python scripts/lbp_pca_adaboost.py
python scripts/densenet_cnn.py
python scripts/densenet_kpca_cnn.py
python scripts/lbp_knn.py
python scripts/lbp_pca_knn.py
python scripts/lbp_random.py
python scripts/lbp_pca_random.py
python scripts/lbp_cnn.py
python scripts/lbp_svm.py
python scripts/lbp_pca_svm.py
rem python scripts/sift_knn.py
rem python scripts/sift_pca_knn.py
rem python scripts/sift_svm.py
rem python scripts/sift_pca_svm.py
