from modeling import Exp

exp = Exp()

for i in range(10):
    for season in ['summer', 'winter']:
        for model_name in ['ann_lin_dev', 'cnn_nn_dev', 'nn_lstm_nn_dev', 'DLinear', 'Autoformer', 'LSTNet']:
            exp.test_model(model_name=model_name, 
                            n_cluster=4, 
                            cluster_method="kmeans", 
                            season=season, 
                            cluster_reduce_method="pca",
                            cluster_reduce_dim=2)

            exp.test_model(model_name=model_name, 
                            n_cluster=1, 
                            cluster_method=None, 
                            season=season, 
                            cluster_reduce_method=None,
                            cluster_reduce_dim=None)

