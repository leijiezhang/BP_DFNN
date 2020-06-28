import torch
import os
import scipy.io as io

data_save_dir = f"./results/loss"
data_save_file = f"{data_save_dir}/hdmm_kmm_loss.mat"

if not os.path.exists(data_save_dir):
    os.makedirs(data_save_dir)
save_dict = io.loadmat(data_save_file)
loss_mtrx = torch.tensor(save_dict["hdmm_kmm_loss"])
loss_mtrx = torch.zeros(3, 300)
loss_mtrx[0, :] = torch.tensor(errors)
loss_mtrx[1, :] = torch.tensor(errors)
loss_mtrx[2, :] = torch.tensor(errors)
save_dict = dict()
save_dict["hdmm_kmm_loss"] = loss_mtrx.numpy()
data_save_file = f"{data_save_dir}/hdmm_kmm_loss.mat"
io.savemat(data_save_file, save_dict)