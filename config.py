
BayesConfig = {
    "pbounds": {"converge_log_lr": (-6, -3), "global_log_lr": (-6, -3)},
    "max_dense_layers": 3,
    "bayes_init_points": 5,
    "bayes_num_iter": 25,
    "max_epoch": 15,
    "patience": 3
}

ModelQueueConfig = {
    "queue_size": 5,
    "log_lr_grid": [-2.5, -3.0, -3.5, -4.0]
}

GridConfig = {
    "grid_target": 85,
    "conv_reg_l2": [-5, -6],
    "dense_reg_l1": [-5, -6],
    "patience": 1,
    "max_epoch": 15
}
