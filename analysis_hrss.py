import torch
from utils import Logger
import os
import scipy.io as io

n_subj = 1
n_rules_list = [10]
save_dict = dict()
acc_c_train_list = []
acc_c_test_list = []
acc_d_train_list = []
acc_d_test_list = []
for i in torch.arange(len(n_rules_list)):
    dataset_name = f"hdfnn_ao_{n_rules_list[int(i)]}"
    sub_fold = "hrss"
    dir_dataset = f"./results/{sub_fold}/{dataset_name}.pt"
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

acc_c_train_tsr = acc_c_train_list[0]
acc_c_test_tsr = acc_c_test_list[0]
acc_d_train_tsr = acc_d_train_list[0]
acc_d_test_tsr = acc_d_test_list[0]

acc_c_train_mean = acc_c_train_tsr.mean(2)
acc_c_test_mean = acc_c_test_tsr.mean(2)
acc_d_train_mean = acc_d_train_tsr.mean(2)
acc_d_test_mean = acc_d_test_tsr.mean(2)

acc_c_train_std = acc_c_train_tsr.std(2)
acc_c_test_std = acc_c_test_tsr.std(2)
acc_d_train_std = acc_d_train_tsr.std(2)
acc_d_test_std = acc_d_test_tsr.std(2)

# get the best performance
acc_c_u = torch.ge(acc_c_train_mean, acc_c_test_mean)
acc_c_test_mean_best = acc_c_test_mean[acc_c_u].max()
# get index of best performance
idx_c = torch.where(acc_c_test_mean == acc_c_test_mean[acc_c_u].max())
best_c_mask = torch.eq(acc_c_test_mean, acc_c_test_mean_best)
acc_c_train_mean_best = acc_c_train_mean[best_c_mask].max()
acc_c_train_std_best = acc_c_train_std[best_c_mask].max()
acc_c_test_std_best = acc_c_test_std[best_c_mask].max()

acc_d_u = torch.ge(acc_d_train_mean, acc_d_test_mean)
acc_d_test_mean_best = acc_d_test_mean[acc_d_u].max()
idx_d = torch.where(acc_d_test_mean == acc_d_test_mean[acc_d_u].max())
best_d_mask = torch.eq(acc_d_test_mean, acc_d_test_mean_best)
acc_d_train_mean_best = acc_d_train_mean[best_d_mask].max()
acc_d_train_std_best = acc_d_train_std[best_d_mask].max()
acc_d_test_std_best = acc_d_test_std[best_d_mask].max()

save_dict["acc_c_train_tsr"] = acc_c_train_tsr
save_dict["acc_c_test_tsr"] = acc_c_test_tsr
save_dict["acc_d_train_tsr"] = acc_d_train_tsr
save_dict["acc_d_test_tsr"] = acc_d_test_tsr

save_dict["acc_c_test_mean_best"] = acc_c_test_mean_best
save_dict["acc_c_train_mean_best"] = acc_c_train_mean_best
save_dict["acc_d_test_mean_best"] = acc_d_test_mean_best
save_dict["acc_d_train_mean_best"] = acc_d_train_mean_best

save_dict["acc_c_train_std_best"] = acc_c_train_std_best
save_dict["acc_c_test_std_best"] = acc_c_test_std_best
save_dict["acc_d_train_std_best"] = acc_d_train_std_best
save_dict["acc_d_test_std_best"] = acc_d_test_std_best

Logger().info(
    f"mAp of training data on centralized method: "
    f"{round(float(acc_c_train_mean_best), 4)}/{round(float(acc_c_train_std_best), 4)}")
Logger().info(
    f"mAp of test data on centralized method: "
    f"{round(float(acc_c_test_mean_best), 4)}/{round(float(acc_c_test_std_best), 4)}")
Logger().info(
    f"mAp of training data on distributed method: "
    f"{round(float(acc_d_train_mean_best), 4)}/{round(float(acc_d_train_std_best), 4)}")
Logger().info(
    f"mAp of test data on distributed method:"
    f" {round(float(acc_d_test_mean_best), 4)}/{round(float(acc_d_test_std_best), 4)}")

data_save_dir = f"./results/hrss/"

# if not os.path.exists(data_save_dir):
#     os.makedirs(data_save_dir)
# data_save_file = f"{data_save_dir}/h_dfnn_ao_final.pt"
# torch.save(save_dict, data_save_file)

acc_c_train_tsr = acc_c_train_tsr.numpy()
acc_c_test_tsr = acc_c_test_tsr.numpy()
acc_d_train_tsr = acc_d_train_tsr.numpy()
acc_d_test_tsr = acc_d_test_tsr.numpy()

acc_c_test_mean_best = acc_c_test_mean_best.numpy()
acc_c_train_mean_best = acc_c_train_mean_best.numpy()
acc_c_train_std_best = acc_c_train_std_best.numpy()
acc_c_test_std_best = acc_c_test_std_best.numpy()

acc_d_test_mean_best = acc_d_test_mean_best.numpy()
acc_d_train_mean_best = acc_d_train_mean_best.numpy()
acc_d_train_std_best = acc_d_train_std_best.numpy()
acc_d_test_std_best = acc_d_test_std_best.numpy()

save_dict = dict()
save_dict["acc_c_train_tsr"] = acc_c_train_tsr
save_dict["acc_c_test_tsr"] = acc_c_test_tsr
save_dict["acc_d_train_tsr"] = acc_d_train_tsr
save_dict["acc_d_test_tsr"] = acc_d_test_tsr

save_dict["acc_c_test_mean_best"] = acc_c_test_mean_best
save_dict["acc_c_train_mean_best"] = acc_c_train_mean_best
save_dict["acc_d_test_mean_best"] = acc_d_test_mean_best
save_dict["acc_d_train_mean_best"] = acc_d_train_mean_best

save_dict["acc_c_train_std_best"] = acc_c_train_std_best
save_dict["acc_c_test_std_best"] = acc_c_test_std_best
save_dict["acc_d_train_std_best"] = acc_d_train_std_best
save_dict["acc_d_test_std_best"] = acc_d_test_std_best

data_save_file = f"{data_save_dir}/hdfnn_ao_final.mat"
io.savemat(data_save_file, save_dict)

# for i in torch.arange(len(n_rules_list)):
#     dataset_name = f"h_dfnn_ao"
#     sub_fold = "hrss"
#     dir_dataset = f"./results/{sub_fold}/{dataset_name}.pt"
#     save_dict = torch.load(dir_dataset)
#
#     loss_c_train_tsr = save_dict["loss_c_train_tsr"]
#     loss_c_test_tsr = save_dict["loss_c_test_tsr"]
#     loss_d_train_tsr = save_dict["loss_d_train_tsr"]
#     loss_d_test_tsr = save_dict["loss_d_test_tsr"]
#
#     loss_c_train_mean = loss_c_train_tsr.mean(2)
#     loss_c_test_mean = loss_c_test_tsr.mean(2)
#     loss_d_train_mean = loss_d_train_tsr.mean(2)
#     loss_d_test_mean = loss_d_test_tsr.mean(2)
#
#     acc_c_u = torch.ge(loss_c_train_mean, loss_c_test_mean)
#     loss_c_test_mean_cal = loss_c_test_mean[acc_c_u]
#
#     acc_c_test = loss_c_test_mean_cal.max()
#     best_c_mask = torch.eq(loss_c_test_mean, acc_c_test)
#     acc_c_train = loss_c_train_mean[best_c_mask]
#
#     acc_d_u = torch.ge(loss_d_train_mean, loss_d_test_mean)
#     loss_d_test_mean_cal = loss_d_test_mean[acc_d_u]
#     acc_d_test = loss_d_test_mean_cal.max()
#     best_d_mask = torch.eq(loss_d_test_mean, acc_d_test)
#     acc_d_train = loss_d_train_mean[best_d_mask]
#     print('lei')
