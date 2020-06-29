from param_config import ParamConfig
from loss_utils import RMSELoss, LikelyLoss
from dfnn_run import mlp_run_r
import torch


# Dataset configuration
# init the parameters
param_config = ParamConfig()
param_config.config_parse('statlib_calhousing_config')
n_rule_list = torch.arange(1, 16, 1)
param_config.n_rules_list = n_rule_list

for i in torch.arange(len(param_config.dataset_list)):
    # load dataset
    dataset = param_config.get_dataset_mat(int(i))

    param_config.log.debug(f"=====starting on {dataset.name}=======")
    loss_fun = None
    if dataset.task == 'C':
        param_config.log.war(f"=====Mission: Classification=======")
        param_config.loss_fun = LikelyLoss()
    else:
        param_config.log.war(f"=====Mission: Regression=======")
        param_config.loss_fun = RMSELoss()

    train_data, test_data = dataset.get_kfold_data(param_config.n_brunches, param_config.n_agents, 0)
    mlp_run_r(train_data, test_data)

    # loss_c_train_mean = loss_c_train.mean()
    # loss_c_test_mean = loss_c_test.mean()
    # loss_d_train_mean = loss_d_train.mean()
    # loss_d_test_mean = loss_d_test.mean()
    # 
    # loss_c_train_std = loss_c_train.std()
    # loss_c_test_std = loss_c_test.std()
    # loss_d_train_std = loss_d_train.std()
    # loss_d_test_std = loss_d_test.std()
    # 
    # param_config.log.info(f"mAp of training data on centralized method: {round(float(loss_c_train_mean), 4)}"
    #                       f"/{round(float(loss_c_train_std), 4)}")
    # param_config.log.info(f"mAp of test data on centralized method: {round(float(loss_c_test_mean), 4)}"
    #                       f"/{round(float(loss_c_test_std), 4)}")
    # param_config.log.info(f"mAp of training data on distributed method"
    #                       f": {round(float(loss_d_train_mean), 4)}/{round(float(loss_d_train_std), 4)}")
    # param_config.log.info(f"mAp of test data on distributed method: {round(float(loss_d_test_mean), 4)}"
    #                       f"/{round(float(loss_d_test_std), 4)}")
