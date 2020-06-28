import torch
from utils import Logger
import os
import scipy.io as io
from param_config import ParamConfig

param_config = ParamConfig()
param_config.config_parse('eegdual_config')

n_rules_list = [3, 6, 9, 12, 15]
n_subj = 11
save_dict = dict()
acc_c_train_list = []
acc_c_test_list = []
acc_d_train_list = []
acc_d_test_list = []
for i in torch.arange(len(n_rules_list)):
    sub_fold = "eeg_dual_old"
    dir_dataset = f"./results/{sub_fold}/hdfnn_ao_{n_rules_list[int(i)]}.pt"
    save_dict = torch.load(dir_dataset)

    if i == 0:
        acc_c_train_list = save_dict["acc_c_train_list"]
        acc_c_test_list = save_dict["acc_c_test_list"]
        acc_d_train_list = save_dict["acc_d_train_list"]
        acc_d_test_list = save_dict["acc_d_test_list"]
    else:
        for j in torch.arange(n_subj):
            acc_c_train_list[int(j)] = torch.cat((acc_c_train_list[int(j)], save_dict["acc_c_train_list"][int(j)]), 0)
            acc_c_test_list[int(j)] = torch.cat((acc_c_test_list[int(j)], save_dict["acc_c_test_list"][int(j)]), 0)
            acc_d_train_list[int(j)] = torch.cat((acc_d_train_list[int(j)], save_dict["acc_d_train_list"][int(j)]), 0)
            acc_d_test_list[int(j)] = torch.cat((acc_d_test_list[int(j)], save_dict["acc_d_test_list"][int(j)]), 0)

acc_c_train_arr = []
acc_c_test_arr = []
acc_d_train_arr = []
acc_d_test_arr = []
for i in torch.arange(len(acc_c_train_list)):
    # acc_c_train_tsr = acc_c_train_list[int(i)].squeeze()
    # acc_c_test_tsr = acc_c_test_list[int(i)].squeeze()
    # acc_d_train_tsr = acc_d_train_list[int(i)].squeeze()
    # acc_d_test_tsr = acc_d_test_list[int(i)].squeeze()

    acc_c_train_mean = acc_c_train_list[int(i)].squeeze()
    acc_c_test_mean = acc_c_test_list[int(i)].squeeze()
    acc_d_train_mean = acc_d_train_list[int(i)].squeeze()
    acc_d_test_mean = acc_d_test_list[int(i)].squeeze()

    # # get mean and variance performance
    # acc_c_train_mean = acc_c_train_tsr.mean(2)
    # acc_c_test_mean = acc_c_test_tsr.mean(2)
    # acc_d_train_mean = acc_d_train_tsr.mean(2)
    # acc_d_test_mean = acc_d_test_tsr.mean(2)

    # acc_c_train_std = acc_c_train_tsr.std(2)
    # acc_c_test_std = acc_c_test_tsr.std(2)
    # acc_d_train_std = acc_d_train_tsr.std(2)
    # acc_d_test_std = acc_d_test_tsr.std(2)

    # get the best performance
    acc_c_test_best = acc_c_test_mean.max()
    best_c_mask = torch.eq(acc_c_test_mean, acc_c_test_best)
    acc_c_train_best = acc_c_train_mean[best_c_mask].max()

    acc_d_test_best = acc_d_test_mean.max()
    best_d_mask = torch.eq(acc_d_test_mean, acc_d_test_best)
    acc_d_train_best = acc_d_train_mean[best_d_mask].max()

    acc_c_train_arr.append(acc_c_train_best)
    acc_c_test_arr.append(acc_c_test_best)
    acc_d_train_arr.append(acc_d_train_best)
    acc_d_test_arr.append(acc_d_test_best)

acc_c_train = torch.tensor(acc_c_train_arr).mean()
acc_c_test = torch.tensor(acc_c_test_arr).mean()
acc_d_train = torch.tensor(acc_d_train_arr).mean()
acc_d_test = torch.tensor(acc_d_test_arr).mean()
acc_c_train_std = torch.tensor(acc_c_train_arr).std()
acc_c_test_std = torch.tensor(acc_c_test_arr).std()
acc_d_train_std = torch.tensor(acc_d_train_arr).std()
acc_d_test_std = torch.tensor(acc_d_test_arr).std()

save_dict["acc_c_train_arr"] = acc_c_train_arr
save_dict["acc_c_test_arr"] = acc_c_test_arr
save_dict["acc_d_train_arr"] = acc_d_train_arr
save_dict["acc_d_test_arr"] = acc_d_test_arr

save_dict["acc_c_train"] = acc_c_train
save_dict["acc_c_test"] = acc_c_test
save_dict["acc_d_train"] = acc_d_train
save_dict["acc_d_test"] = acc_d_test

save_dict["acc_c_train_std"] = acc_c_train_std
save_dict["acc_c_test_std"] = acc_c_test_std
save_dict["acc_d_train_std"] = acc_d_train_std
save_dict["acc_d_test_std"] = acc_d_test_std

param_config.log.info(
    f"mAp of training data on centralized method: "
    f"{round(float(acc_c_train), 4)}/{round(float(acc_c_train_std), 4)}")
param_config.log.info(
    f"mAp of test data on centralized method: "
    f"{round(float(acc_c_test), 4)}/{round(float(acc_c_test_std), 4)}")
param_config.log.info(
    f"mAp of training data on distributed method: "
    f"{round(float(acc_d_train), 4)}/{round(float(acc_d_train_std), 4)}")
param_config.log.info(
    f"mAp of test data on distributed method:"
    f" {round(float(acc_d_test), 4)}/{round(float(acc_d_test_std), 4)}")

data_save_dir = f"./results/eeg_dual/"

# if not os.path.exists(data_save_dir):
#     os.makedirs(data_save_dir)
# data_save_file = f"{data_save_dir}/h_dfnn_ao_final.pt"
# torch.save(save_dict, data_save_file)

for j in torch.arange(n_subj):
    acc_c_train_list[int(j)] = acc_c_train_list[int(j)].numpy()
    acc_c_test_list[int(j)] = acc_c_test_list[int(j)].numpy()
    acc_d_train_list[int(j)] = acc_d_train_list[int(j)].numpy()
    acc_d_test_list[int(j)] = acc_d_test_list[int(j)].numpy()

    acc_c_train_arr[int(j)] = acc_c_train_arr[int(j)].numpy()
    acc_c_test_arr[int(j)] = acc_c_test_arr[int(j)].numpy()
    acc_d_train_arr[int(j)] = acc_d_train_arr[int(j)].numpy()
    acc_d_test_arr[int(j)] = acc_d_test_arr[int(j)].numpy()

save_dict = dict()
save_dict["acc_c_train_list"] = acc_c_train_list
save_dict["acc_c_test_list"] = acc_c_test_list
save_dict["acc_d_train_list"] = acc_d_train_list
save_dict["acc_d_test_list"] = acc_d_test_list

save_dict["acc_c_train_arr"] = acc_c_train_arr
save_dict["acc_c_test_arr"] = acc_c_test_arr
save_dict["acc_d_train_arr"] = acc_d_train_arr
save_dict["acc_d_test_arr"] = acc_d_test_arr

save_dict["acc_c_train"] = acc_c_train.numpy()
save_dict["acc_c_test"] = acc_c_test.numpy()
save_dict["acc_d_train"] = acc_d_train.numpy()
save_dict["acc_d_test"] = acc_d_test.numpy()

save_dict["acc_c_train_std"] = acc_c_train_std.numpy()
save_dict["acc_c_test_std"] = acc_c_test_std.numpy()
save_dict["acc_d_train_std"] = acc_d_train_std.numpy()
save_dict["acc_d_test_std"] = acc_d_test_std.numpy()

data_save_file = f"{data_save_dir}/h_dfnn_ao_final.mat"
io.savemat(data_save_file, save_dict)
