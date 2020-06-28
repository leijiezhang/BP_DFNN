import torch
from utils import Logger
import os
from model import FnnNet
from neuron import NeuronC, NeuronD
import scipy.io as io

fuzy_tree_c: FnnNet = FnnNet()
fuzy_tree_d: FnnNet = FnnNet()

fuzzy_c_neuron: NeuronC = fuzy_tree_c.get_neuron_tree()[0][0]
fuzzy_c_rule = fuzzy_c_neuron.get_rules()
n_rules = fuzzy_c_rule.n_rules

mu_c = fuzzy_c_rule.center_list
sigma_c = fuzzy_c_rule.widths_list
consequent_c: torch.Tensor = fuzzy_c_rule.consequent_list
consequent_c = consequent_c.squeeze()
n_fea = mu_c.shape[1]
para_c = []
for i in torch.arange(n_rules):
    for j in torch.arange(n_fea):
        para_c.append(mu_c[i, j])
        para_c.append(sigma_c[i, j])
    for j in torch.arange(n_fea+1):
        para_c.append(consequent_c[i, j])
para_c = torch.tensor(para_c).numpy()

fuzzy_d_neuron: NeuronD = fuzy_tree_d.get_neuron_tree()[0][0]
fuzzy_d_rule = fuzzy_d_neuron.get_rules()

mu_d = fuzzy_d_rule.center_list
sigma_d = fuzzy_d_rule.widths_list
consequent_d: torch.Tensor = fuzzy_d_rule.consequent_list
consequent_d = consequent_d.squeeze()
n_fea = mu_d.shape[1]
para_d = []
for i in torch.arange(n_rules):
    for j in torch.arange(n_fea):
        para_d.append(mu_d[i, j])
        para_d.append(sigma_d[i, j])
    for j in torch.arange(n_fea+1):
        para_d.append(consequent_d[i, j])
para_d = torch.tensor(para_d).numpy()


save_dict = dict()
save_dict["sigma_d"] = sigma_d.numpy()
save_dict["consequent_d"] = consequent_d.numpy()
save_dict["mu_d"] = mu_d.numpy()
save_dict["para_d"] = para_d
save_dict["sigma_c"] = sigma_c.numpy()
save_dict["consequent_c"] = consequent_c.numpy()
save_dict["mu_c"] = mu_c.numpy()
save_dict["para_c"] = para_c

data_save_dir = f"./results/lidar/"
if not os.path.exists(data_save_dir):
    os.makedirs(data_save_dir)
data_save_file = f"{data_save_dir}/model.mat"
io.savemat(data_save_file, save_dict)
