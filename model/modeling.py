import os, time
import numpy as np
import pandas as pd
from pathlib import Path
from importlib import import_module

from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans, DBSCAN
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

from umap import UMAP
from xgboost import XGBRegressor

from setup import IndexSetup, ColumnsSetup

class EXP:

    def __init__(self):
        self.model_deep = ['ann_lin_dev', 'cnn_nn_dev', 'nn_lstm_nn_dev', 'DLinear', 'Autoformer', 'LSTNet']
        self.model_machine = ['svr_dev', 'xgb_dev']
        self.CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))
        self.ROOT_PATH = os.path.dirname(self.CURRENT_PATH)
        self.CONF_DEFAULT = {'season': "summer", 'prev_times': 192, 'next_times': 24,
                             'test_ratio': 0.2, 'train_sample_ratio': 1, 'summer_months': [4, 5, 6, 7, 8, 9],
                             'time_flag_var': 'time', 'time_var_format': '.+_d[1-7]_t_$', 'value_var_format': '^h[0-9]+'}
        self.data = pd.read_pickle(self.ROOT_PATH + '/ireland_power_1h.bin')

    def clustering(self, data, COLUMNS_SETUP, n_cluster, cluster_method="kmeans", cluster_reduce_method=False, cluster_reduce_dim=2):
        if n_cluster > 1:
            # 클러스터링용 차원축소
            time_start = time.time()
            if cluster_reduce_method:
                if cluster_reduce_method == 'tsne':
                    data = TSNE(n_components=cluster_reduce_dim).fit_transform(data)
                elif cluster_reduce_method == 'umap':
                    data = UMAP(n_components=cluster_reduce_dim).fit_transform(data)
                elif cluster_reduce_method == 'pca':
                    data = PCA(n_components=cluster_reduce_dim).fit_transform(data)
            time_end = time.time()
            reduce_time = time_end - time_start

            # 클러스터링
            if cluster_method == "kmeans":
                model_clust = KMeans(n_clusters=n_cluster, random_state=0)
                print()

            # eps 조절 -> 자동화하고 싶다
            elif cluster_method == "dbscan":
                if cluster_reduce_method == "tsne":
                    eps = 1
                elif cluster_reduce_method == "umap":
                    eps = 0.25
                else:
                    eps = 30
                model_clust = DBSCAN(eps=eps)

            time_start = time.time()
            clust_id_set = model_clust.fit_predict(data)
            # clust_id_set[clust_id_set >= 2] = 1  # 클러스터 수 반영
            time_end = time.time()
            cluster_time = time_end - time_start
            info_clust = pd.DataFrame(
                {'var_value': COLUMNS_SETUP.value_var_names, 'clust_id': clust_id_set})

        else:
            cluster_time = 0
            reduce_time = 0
            info_clust = pd.DataFrame(
                {'var_value': COLUMNS_SETUP.value_var_names, 'clust_id': 0})

        return info_clust, cluster_time, reduce_time

    def set_test(self, season):

        CONF = self.CONF_DEFAULT
        CONF['season'] = season

        INDEX_SETUP = IndexSetup(CONF, self.data)
        COLUMNS_SETUP = ColumnsSetup(CONF, self.data)

        arr_v = self.data.loc[INDEX_SETUP.train_period_index,
                              COLUMNS_SETUP.value_var_names].values.T
        arr_v = np.log(arr_v + 1)

        return CONF, INDEX_SETUP, COLUMNS_SETUP, arr_v
    
    def set_save_fold(self, model_name, n_cluster, cluster_method, season,
                     cluster_reduce_method, cluster_reduce_dim, feature_reduce_method, feature_reduce_dim):
        if n_cluster > 1:
            CLUSTER_CODE = f"{cluster_method}(nc={n_cluster})"

            if cluster_reduce_method:
                CLUSTER_CODE += f"({cluster_reduce_method},{cluster_reduce_dim})"
            else:
                CLUSTER_CODE += "(noReduce)"
        else:
            CLUSTER_CODE = "noCluster"

        if feature_reduce_method:
            FEATURE_REDUCE_CODE = f"{feature_reduce_method}({feature_reduce_dim})"
        else:
            FEATURE_REDUCE_CODE = "noFeatureReduce"

        MODEL_CODE = f"{model_name}_{CLUSTER_CODE}_{FEATURE_REDUCE_CODE}_{season}"

        SAVE_FOLD = Path(self.ROOT_PATH) / "result" / MODEL_CODE
        if not SAVE_FOLD.exists():
            SAVE_FOLD.mkdir(parents=True, exist_ok=True)
        
        return SAVE_FOLD
    
    def save_result(self, info_clust, err_info, TIME_SET, perfs, SAVE_FOLD):

        time_ = time.strftime('%Y%m%d%H%M%S', time.localtime())

        # 이진 파일로 저장
        pd.to_pickle(info_clust, SAVE_FOLD / f'info_clust_{time_}.bin')
        pd.to_pickle(err_info, SAVE_FOLD / f'error_{time_}.bin')
        pd.to_pickle(TIME_SET, SAVE_FOLD / f'time_set_{time_}.bin')
        pd.to_pickle(perfs, SAVE_FOLD / f'performance_{time_}.bin')
        # pd.to_pickle(models, SAVE_FOLD / 'models.bin')

        # 텍스트로 저장
        TEXT_FOLD = SAVE_FOLD / "view"
        TEXT_FOLD.mkdir(parents=True, exist_ok=True)

        with open(TEXT_FOLD / f'error_{time_}.txt', 'w') as f:
            f.write(str(err_info))

        with open(TEXT_FOLD / f'time_set_{time_}.txt', 'w') as f:
            f.write(str(TIME_SET))

        with open(TEXT_FOLD / f'performance_{time_}.txt', 'w') as f:
            f.write(str(perfs))

    def test_model(self, model_name, n_cluster=1, cluster_method='kmeans', season='summer',
                     cluster_reduce_method=False, cluster_reduce_dim=2,
                     feature_reduce_method=False, feature_reduce_dim=2):

        # 실행 조건 출력
        print(f"""[Test Model]
Model Name: {model_name}
Seoson: {season}
Number of Clusters: {n_cluster}
Clustering Method: {cluster_method}
Demension Reduction(clustering): {cluster_reduce_method}({cluster_reduce_dim})
Demension Reduction(feature): {feature_reduce_method}({feature_reduce_dim})""")

        # 초기 세팅
        CONF, INDEX_SETUP, COLUMNS_SETUP, arr_v = self.set_test(season)
        SAVE_FOLD = self.set_save_fold(model_name, n_cluster, cluster_method, season,
                     cluster_reduce_method, cluster_reduce_dim, feature_reduce_method, feature_reduce_dim)
        
        # 클러스터링
        info_clust, cluster_time, reduce_time = self.clustering(
            arr_v, COLUMNS_SETUP, n_cluster, cluster_method, cluster_reduce_method, cluster_reduce_dim)
        
        # 클러스터링 시간 기록
        print("\nStart clustering...")
        TIME_SET = {}
        TIME_SET['REDUCE_DIM_CLUSTER'] = reduce_time
        TIME_SET['CLUSTERING'] = cluster_time
        print(f"Dimension Reduction: {reduce_time}s")
        print(f"Clustering: {cluster_time}s")
        print("Done.")
        
        # 모델 학습에 필요한 정보 저장
        context = (model_name, n_cluster, info_clust, INDEX_SETUP, COLUMNS_SETUP, CONF,
                   feature_reduce_method, feature_reduce_dim)

        # 머신러닝 모델
        print("\nStart learning...")
        print(f"- {model_name}...")
        if model_name in self.model_machine:
            info_clust, err_info, learning_time, feature_reduce_time, perfs = self.train_machine(context)

        # 딥러닝 모델
        elif model_name in self.model_deep:
            info_clust, err_info, learning_time, perfs = self.train_deep(context)
        
        # TIME_SET['REDUCE_DIM_FEATURE'] = feature_reduce_time
        TIME_SET['LEARNING'] = learning_time
                
        # 파일에 결과 저장
        self.save_result(info_clust, err_info, TIME_SET, perfs, SAVE_FOLD)
        # print(f"Dimension Reduction: {feature_reduce_time}s")
        print(f"Learning: {learning_time}s\n")
        print("Test finished.")

    def train_machine(self, context):
        model_name, n_cluster, info_clust, INDEX_SETUP, COLUMNS_SETUP, CONF, feature_reduce_method, feature_reduce_dim = context
        
        perfs = []
        models = {}
        
        feature_reduce_time = 0
        start_time = time.time()
        for clust_id in range(n_cluster):

            # 컬럼 설정
            var_value_new = info_clust[info_clust['clust_id']
                                    == clust_id].var_value.to_list()
            var_use = COLUMNS_SETUP.base_var_names
            var_use = var_use.append(COLUMNS_SETUP.time_var_names)
            var_use = var_use.append(pd.Index(var_value_new))

            data_new = self.data.loc[:, var_use]
            index_setup = IndexSetup(CONF, data_new)
            columns_setup = ColumnsSetup(CONF, data_new)
            y_col_index = list(range(index_setup.next_times))  # 0~23: 예측 대상

            ys = []
            y_sum = data_new.loc[:, columns_setup.value_var_names].sum(axis=1)
            for next_time in range(index_setup.next_times):
                next_index = index_setup.base_index_set_train + (next_time + 1)
                y = y_sum[next_index]
                y.index = index_setup.base_index_set_train
                ys.append(y)

            Y = pd.concat(ys, axis=1)
            X = data_new.loc[index_setup.base_index_set_train,
                            columns_setup.use_var_names].values

            if model_name == "lin_dev":
                model = LinearRegression()
            elif model_name == "svr_dev":
                model = SVR(gamma='scale')
            else:
                model = XGBRegressor()

            # feature 차원축소
            if feature_reduce_method:
                start = time.time()
                if feature_reduce_method == "tsne":
                    X = TSNE(n_components=feature_reduce_dim).fit_transform(X)
                elif feature_reduce_method == "umap":
                    X = UMAP(n_components=feature_reduce_dim).fit_transform(X)
                end = time.time()
                feature_reduce_time += end - start

            ymodels = {}
            for idx in y_col_index:
                print(time.ctime(),
                    f'Cluster ID : {clust_id} / Next Time : {idx}')
                y = Y[idx].values
                model = model.fit(X, y)
                ymodels[idx] = model

            nX = data_new.loc[index_setup.base_index_set_test,
                            columns_setup.use_var_names].values

            # feature 차원축소
            if feature_reduce_method:
                start = time.time()
                if feature_reduce_method == "tsne":
                    nX = TSNE(n_components=feature_reduce_dim).fit_transform(nX)
                elif feature_reduce_method == "umap":
                    nX = UMAP(n_components=feature_reduce_dim).fit_transform(nX)
                end = time.time()
                feature_reduce_time += end - start

            rets = []
            for idx in y_col_index:
                model = ymodels[idx]
                ret = pd.Series(model.predict(
                    nX), index=index_setup.base_index_set_test+idx+1)
                rets.append(ret)

            perf = pd.concat(rets).sort_index()
            perf = pd.DataFrame({'pred_val': perf})
            perf['act_val'] = data_new.loc[perf.index,
                                        columns_setup.value_var_names].sum(axis=1)
            perf = data_new.loc[:, columns_setup.base_var_names].join(
                perf).dropna().reset_index()
            perf['clust_id'] = clust_id
            perfs.append(perf)
            models[clust_id] = ymodels

        end_time = time.time()
        learning_time = end_time - start_time
        learning_time -= feature_reduce_time

        result = pd.concat(perfs).reset_index(drop=True)

        info_test = result.groupby(
            'time')[['act_val', 'pred_val']].sum().reset_index()
        info_test['ae'] = (info_test['act_val'] - info_test['pred_val']).abs()
        info_test['se'] = (info_test['act_val'] - info_test['pred_val']).pow(2)
        info_test['ar'] = info_test['act_val'].abs()
        info_test['sr'] = info_test['act_val'].pow(2)
        info_test['ape'] = info_test['ae'] / info_test['act_val']

        test_mape = info_test['ape'].mean() * 100
        test_rmse = np.power(info_test['se'].mean(), .5)
        test_nmae = info_test['ae'].mean() / info_test['ar'].mean()
        test_nrmse = np.power(info_test['se'].mean(
        ), .5) / np.power(info_test['sr'].mean(), .5)
        err_info = pd.DataFrame({'mape': test_mape, 'rmse': test_rmse,
                                'nmae': test_nmae, 'nrmse': test_nrmse}, index=[0])
        
        return info_clust, err_info, learning_time, feature_reduce_time, perfs
    
    def train_deep(self, context):
        
        model_name, n_cluster, info_clust, INDEX_SETUP, COLUMNS_SETUP, CONF, feature_reduce_method, feature_reduce_dim = context
        
        model = import_module(model_name)
        CustomDataset = getattr(model, 'CustomDataset')
        PowerModel = getattr(model, 'PowerModel')
        training = getattr(model, 'training')
        
        perfs = []
        models = {}

        start_time = time.time()
        for clust_id in range(n_cluster):
            print(f"""Cluster : {clust_id}""")
            var_value_new = info_clust.loc[info_clust['clust_id']
                                        == clust_id, 'var_value'].to_list()

            var_use = COLUMNS_SETUP.base_var_names
            var_use = var_use.append(COLUMNS_SETUP.time_var_names)
            var_use = var_use.append(pd.Index(var_value_new))

            data_new = self.data.loc[:, var_use]
            
            if feature_reduce_method:
                if feature_reduce_method == 'tsne':
                    data_new = TSNE(n_components=feature_reduce_dim).fit_transform(data_new)
                elif feature_reduce_method == 'umap':
                    data_new = UMAP(n_components=feature_reduce_dim).fit_transform(data_new)
                elif feature_reduce_method == 'pca':
                    data_new = PCA(n_components=feature_reduce_dim).fit_transform(data_new)

            perf, model = training(CONF, data_new)
            perf['clust_id'] = clust_id
            perfs.append(perf)
            models[clust_id] = model

        end_time = time.time()
        learning_time = end_time - start_time

        result = pd.concat(perfs).reset_index(drop=True)

        info_test = result.groupby(['epoch', 'batch', 'pos'])[
            ['act_val', 'pred_val']].sum().reset_index()
        info_test['ae'] = (info_test['act_val'] - info_test['pred_val']).abs()
        info_test['se'] = (info_test['act_val'] - info_test['pred_val']).pow(2)
        info_test['ar'] = info_test['act_val'].abs()
        info_test['sr'] = info_test['act_val'].pow(2)
        info_test['ape'] = info_test['ae'] / info_test['act_val']

        err_meas = info_test.groupby(['epoch', 'batch'])
        test_mape = (err_meas['ape'].mean() * 100).rename('mape')
        test_rmse = err_meas['se'].mean().pow(1/2).rename('rmse')
        test_nmae = (err_meas['ae'].mean() /
                    err_meas['ar'].mean()).rename('nmae')
        test_nrmse = (err_meas['se'].mean().pow(1/2) /
                    err_meas['sr'].mean().pow(1/2)).rename('nrmse')
        err_test = pd.concat(
            (test_mape, test_rmse, test_nmae, test_nrmse), axis=1).reset_index()
        err_test = err_test.groupby(
            'epoch')[['mape', 'rmse', 'nmae', 'nrmse']].mean().reset_index()

        err_train = result.drop_duplicates(
            subset=['epoch', 'batch', 'clust_id']).reset_index(drop=True)
        err_train = err_train.groupby(['epoch', 'batch'])[
            'err_train'].sum().reset_index()
        err_train = err_train.groupby(
            'epoch')['err_train'].mean().reset_index()
        err_info = pd.merge(err_train, err_test)

        return info_clust, err_info, learning_time, perfs
    
    def test_all(self, model_type, n_cluster=1, cluster_method='kmeans', season='summer',
                     cluster_reduce_method=False, cluster_reduce_dim=2,
                     feature_reduce_method=False, feature_reduce_dim=2):
        
        # 실행 조건 출력
        print(f"""\n[Test All]
Model Type: {model_type}
Seoson: {season}
Number of Clusters: {n_cluster}
Clustering Method: {cluster_method}
Dimension Reduction(clustering): {cluster_reduce_method}({cluster_reduce_dim})
Dimension Reduction(feature): {feature_reduce_method}({feature_reduce_dim})""")
        
        # 모델 리스트 설정
        if model_type == "all":
            model_list = self.model_deep + self.model_machine
        elif model_type == "deep":
            model_list = self.model_deep
        elif model_type == "machine":
            model_list = self.model_machine
        
        # 초기 세팅
        CONF, INDEX_SETUP, COLUMNS_SETUP, arr_v = self.set_test(season)
        save_folds = {}
        for model_name in model_list:
            save_fold = self.set_save_fold(model_name, n_cluster, cluster_method, season,
                            cluster_reduce_method, cluster_reduce_dim, feature_reduce_method, feature_reduce_dim)
            save_folds[model_name] = save_fold
        
        # 클러스터링
        print("\nStart clustering...")
        info_clust, cluster_time, reduce_time = self.clustering(
            arr_v, COLUMNS_SETUP, n_cluster, cluster_method, cluster_reduce_method, cluster_reduce_dim)
        
        # 클러스터링 시간 기록
        TIME_SET = {}
        TIME_SET['REDUCE_DIM_CLUSTER'] = reduce_time
        TIME_SET['CLUSTERING'] = cluster_time
        print(f"Dimension Reduction: {reduce_time}s")
        print(f"Clustering: {cluster_time}s")
        print("Done.")
        
        # 모델 학습 및 평가
        print("\nStart learning...")
        for model_name in model_list:
            print(f"- {model_name}...")
            
            # 모델 학습에 필요한 정보 저장
            context = (model_name, n_cluster, info_clust, INDEX_SETUP, COLUMNS_SETUP, CONF,
                    feature_reduce_method, feature_reduce_dim)
            
            # 모델 학습 및 평가
            if model_name in self.model_deep:
                info_clust, err_info, learning_time, feature_reduce_time, perfs = self.train_deep(context)
            elif model_name in self.model_machine:
                info_clust, err_info, learning_time, feature_reduce_time, perfs = self.train_machine(context)
        
            TIME_SET['REDUCE_DIM_FEATURE'] = feature_reduce_time
            TIME_SET['LEARNING'] = learning_time
                
            # 파일에 결과 저장
            self.save_result(info_clust, err_info, TIME_SET, perfs, save_folds[model_name])
            print(f"Dimension Reduction: {feature_reduce_time}s")
            print(f"Learning: {learning_time}s\n")
        print("Test finished.")
