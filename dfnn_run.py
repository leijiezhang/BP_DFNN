from param_config import ParamConfig
from dataset import Dataset
from torch.utils.data import DataLoader
from math_utils import mapminmax
from sklearn.metrics import mean_squared_error
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from typing import Dict
import torch.nn as nn
import os
import torch
from keras.utils import to_categorical
from h_utils import HNormal
from rules import RuleKmeans
from fnn_solver import FnnSolveReg
from datetime import datetime
from loss_utils import LossFunc, Map, LikelyLoss
from bp import BP_HDFNN, BP_HDFNN_C, BP_HFNN, BP_FNN, BP_FNN_S, BP_FNN_L, BP_FNN_CE, BP_FNN_M
from dataset import DatasetFNN
from model import MLP, dev_network_s, dev_network_sr, dev_network_s_r


def svc(train_fea: torch.Tensor, test_fea: torch.Tensor, train_gnd: torch.Tensor,
        test_gnd: torch.Tensor, loss_fun: LossFunc, paras: Dict):
    """
    todo: this is the method for distribute fuzzy Neuron network
    :param train_fea: training data
    :param test_fea: test data
    :param train_gnd: training label
    :param test_gnd: test label
    :param loss_fun: the loss function that used to calculate the loss of regression or accuracy of classification task
    :param paras: parameters that used for training SVC model

    :return:
    """
    """ codes for parameters 
        paras = dict()
        paras['kernel'] = 'rbf'
        paras['gamma'] = gamma
        paras['C'] = C
    """
    print("training the one-class SVM")
    train_gnd = train_gnd.squeeze()
    test_gnd = test_gnd.squeeze()
    if 'kernel' in paras:
        svc_kernel = paras['kernel']
    else:
        svc_kernel = 'rbf'
    if 'gamma' in paras:
        svc_gamma = paras['gamma']
    else:
        svc_gamma = 'scale'
    if 'C' in paras:
        svc_c = paras['C']
    else:
        svc_c = 1
    svc_train = SVC(kernel=svc_kernel, gamma=svc_gamma, C=svc_c)
    clf = make_pipeline(StandardScaler(), svc_train)
    clf.fit(train_fea.numpy(), train_gnd.numpy())
    train_gnd_hat = clf.predict(train_fea.numpy())
    test_gnd_hat = clf.predict(test_fea.numpy())

    train_acc = loss_fun.forward(train_gnd.squeeze(), torch.tensor(train_gnd_hat))
    test_acc = loss_fun.forward(test_gnd.squeeze(), torch.tensor(test_gnd_hat))

    """ following code is designed for those functions that need to output the svm results
    if test_data.task == 'C':
        param_config.log.info(f"Accuracy of training data using SVM: {train_acc:.2f}%")
        param_config.log.info(f"Accuracy of test data using SVM: {test_acc:.2f}%")
    else:
        param_config.log.info(f"loss of training data using SVM: {train_acc:.4f}")
        param_config.log.info(f"loss of test data using SVM: {test_acc:.4f}")
    """

    return 100*train_acc, 100*test_acc


def bp_hdfnn_run(param_config: ParamConfig, train_data: Dataset, test_data: Dataset):
    """
    todo: this is the method for hierarchical distribute fuzzy Neuron network using alternative optimizing
    :param param_config:
    :param train_data: training dataset
    :param test_data: test dataset
    :return:
    """
    # train_dataset = DatasetNN(x=train_data.fea_d, y=train_data.gnd_d)
    # valid_dataset = DatasetNN(x=test_data.fea_d, y=test_data.gnd_d)

    # train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
    # valid_loader = DataLoader(dataset=valid_dataset, batch_size=64, shuffle=False)

    n_batch = param_config.n_batch
    model: nn.Module = BP_HDFNN_C(n_agents=param_config.n_agents, n_brunch=param_config.n_brunches,
                                  n_fea=train_data.n_fea_d, n_rules=param_config.n_rules)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()
    valid_acc_list = []
    epochs = 350

    train_losses = []
    valid_losses = []

    for epoch in range(epochs):
        model.train()
        start_idx = epoch * n_batch
        end_idx = (epoch + 1) * n_batch
        start_idx = start_idx % (train_data.n_smpl_d_h + 1)
        end_idx = end_idx % (train_data.n_smpl_d_h + 1)

        if end_idx > start_idx:
            data = train_data.fea_d_h[:, :, start_idx:end_idx, :]
            labels = train_data.gnd_d_h[:, start_idx:end_idx]
        else:
            data_level0_1 = train_data.fea_d_h[:, :, start_idx:train_data.n_smpl_d_h, :]
            data_level0_2 = train_data.fea_d_h[:, :, 0:end_idx + 1, :]

            gnd_train_1 = train_data.gnd_d_h[:, start_idx:train_data.n_smpl_d_h]
            gnd_train_2 = train_data.gnd_d_h[:, 0:end_idx + 1]
            data = torch.cat([data_level0_1, data_level0_2], 2)
            labels = torch.cat([gnd_train_1, gnd_train_2], 1)
        optimizer.zero_grad()

        outputs = model(data)
        loss = loss_fn(outputs.t().double(), labels.t().double())
        loss.backward()
        model.para_mu.grad = model.para_mu.grad.mean(0).repeat(param_config.n_agents, 1, 1, 1)
        model.para_sigma.grad = model.para_sigma.grad.mean(0).repeat(param_config.n_agents, 1, 1, 1)
        model.para_w3.grad = model.para_w3.grad.mean(0).repeat(param_config.n_agents, 1, 1, 1)
        model.para_w5.grad = model.para_w5.grad.mean(0).repeat(param_config.n_agents, 1)
        optimizer.step()

        train_losses.append(loss.item())

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            outputs = model(test_data.fea_d_h)
            loss = loss_fn(outputs.double(), test_data.gnd_d_h.double())

            valid_losses.append(loss.item())
            predicted = torch.round(outputs)
            # _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == test_data.gnd_d_h).sum().item()
            total += test_data.gnd_d_h.shape[0] * test_data.gnd_d_h.shape[1]

        accuracy = 100 * correct / total
        valid_acc_list.append(accuracy)
        print(f"epoch : {epoch + 1}, train loss : {train_losses[-1]}, "
              f"valid loss : {valid_losses[-1]}, valid acc : {accuracy}%")
    plt.plot(torch.arange(epochs), valid_losses)
    plt.show()
    plt.figure()
    plt.plot(torch.arange(epochs), valid_acc_list)
    plt.show()

    if test_data.task == 'C':
        param_config.log.info(f"Accuracy of training data using SVM: {train_losses}")
        param_config.log.info(f"Accuracy of test data using SVM: {valid_losses}")
    else:
        param_config.log.info(f"loss of training data using SVM: {train_losses}")
        param_config.log.info(f"loss of test data using SVM: {valid_losses}")
    test_map = torch.tensor(valid_acc_list).mean()
    return test_map, train_losses


def bp_hdfnn_run1(param_config: ParamConfig, train_data: Dataset, test_data: Dataset):
    """
    todo: this is the method for hierarchical distribute fuzzy Neuron network using alternative optimizing
    :param param_config:
    :param train_data: training dataset
    :param test_data: test dataset
    :return:
    """
    # train_dataset = DatasetNN(x=train_data.fea_d, y=train_data.gnd_d)
    # valid_dataset = DatasetNN(x=test_data.fea_d, y=test_data.gnd_d)

    # train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
    # valid_loader = DataLoader(dataset=valid_dataset, batch_size=64, shuffle=False)

    n_batch = param_config.n_batch
    model_list = []
    scheduler_list = []
    optimizer_list = []
    loss_fn_list = []
    epochs = param_config.n_epoch

    # init the same parameters of models on all agents
    # parameters in level1
    para_mu_item = torch.rand(param_config.n_brunches, param_config.n_rules, train_data.n_fea_d_h)
    para_sigma_item = torch.rand(param_config.n_brunches, param_config.n_rules, train_data.n_fea_d_h)
    # parameters in level3
    para_w3_item = torch.rand(param_config.n_brunches, param_config.n_rules, train_data.n_fea_d_h + 1)
    para_level5_w_item = torch.rand(2, param_config.n_brunches)
    para_level5_b_item = torch.rand(2)
    for ii in torch.arange(param_config.n_agents):
        model_tmp: nn.Module = BP_HFNN(n_brunch=param_config.n_brunches,
                                       n_fea=train_data.n_fea_d_h, n_rules=param_config.n_rules)
        model_tmp.para_mu = torch.nn.Parameter(para_mu_item, requires_grad=True)
        model_tmp.para_sigma = torch.nn.Parameter(para_sigma_item, requires_grad=True)
        # parameters in level3
        model_tmp.para_w3 = torch.nn.Parameter(para_w3_item, requires_grad=True)
        model_tmp.level5.weight = torch.nn.Parameter(para_level5_w_item, requires_grad=True)
        model_tmp.level5.bias = torch.nn.Parameter(para_level5_b_item, requires_grad=True)
        optimizer_tmp = torch.optim.Adam(model_tmp.parameters(), lr=param_config.lr)
        # optimizer_tmp = torch.optim.SGD(model_tmp.parameters(), lr=param_config.lr)
        scheduler_tmp = torch.optim.lr_scheduler.StepLR(optimizer_tmp, step_size=10, gamma=0.5)
        model_list.append(model_tmp)
        optimizer_list.append(optimizer_tmp)
        scheduler_list.append(scheduler_tmp)
        # loss_fn_tmp = nn.MSELoss()
        loss_fn_tmp = nn.CrossEntropyLoss()
        loss_fn_list.append(loss_fn_tmp)

    valid_acc_list = []

    train_losses_tsr = torch.zeros(epochs, param_config.n_agents)
    valid_losses = []

    for epoch in range(epochs):

        start_idx = epoch * n_batch
        end_idx = (epoch + 1) * n_batch
        start_idx = start_idx % (train_data.n_smpl_d_h + 1)
        end_idx = end_idx % (train_data.n_smpl_d_h + 1)

        if end_idx > start_idx:
            data = train_data.fea_d_h[:, :, start_idx:end_idx, :]
            labels = train_data.gnd_d_h[:, start_idx:end_idx]
        else:
            data_level0_1 = train_data.fea_d_h[:, :, start_idx:train_data.n_smpl_d_h, :]
            data_level0_2 = train_data.fea_d_h[:, :, 0:end_idx + 1, :]

            gnd_train_1 = train_data.gnd_d_h[:, start_idx:train_data.n_smpl_d_h]
            gnd_train_2 = train_data.gnd_d_h[:, 0:end_idx + 1]
            data = torch.cat([data_level0_1, data_level0_2], 2)
            labels = torch.cat([gnd_train_1, gnd_train_2], 1)

        grad_para_mu = torch.zeros(param_config.n_brunches, param_config.n_rules, train_data.n_fea_d_h)
        grad_para_sigma = torch.zeros(param_config.n_brunches, param_config.n_rules, train_data.n_fea_d_h)
        grad_para_w3 = torch.zeros(param_config.n_brunches, param_config.n_rules, train_data.n_fea_d_h + 1)
        grad_level5_w = torch.zeros(2, param_config.n_brunches)
        grad_level5_b = torch.zeros(2)
        for ii in torch.arange(param_config.n_agents):
            model_list[int(ii)].train()
            optimizer_list[int(ii)].zero_grad()

            outputs = model_list[int(ii)](data[ii, :, :, :])

            model_list[int(ii)].zero_grad()
            loss_tmp = loss_fn_list[int(ii)](outputs.double(), labels[ii, :].long())
            loss_tmp.backward()
            train_losses_tsr[epoch, ii] = loss_tmp.item()

            grad_para_mu = grad_para_mu + model_list[int(ii)].para_mu.grad
            grad_para_sigma = grad_para_sigma + model_list[int(ii)].para_sigma.grad
            grad_para_w3 = grad_para_w3 + model_list[int(ii)].para_w3.grad
            grad_level5_w = grad_level5_w + model_list[int(ii)].level5.weight.grad
            grad_level5_b = grad_level5_b + model_list[int(ii)].level5.bias.grad

        grad_para_mu = grad_para_mu / param_config.n_agents
        grad_para_sigma = grad_para_sigma / param_config.n_agents
        grad_para_w3 = grad_para_w3 / param_config.n_agents
        grad_level5_w = grad_level5_w / param_config.n_agents
        grad_level5_b = grad_level5_b / param_config.n_agents
        for ii in torch.arange(param_config.n_agents):
            model_list[int(ii)].para_mu.grad = grad_para_mu
            model_list[int(ii)].para_sigma.grad = grad_para_sigma
            model_list[int(ii)].para_w3.grad = grad_para_w3
            model_list[int(ii)].level5.weight.grad = grad_level5_w
            model_list[int(ii)].level5.bias.grad = grad_level5_b
            optimizer_list[int(ii)].step()
            scheduler_list[int(ii)].step(epoch)

        model_tmp = model_list[0]
        model_tmp.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            outputs = model_tmp(test_data.fea_c_h)
            loss_fn_tmp = nn.CrossEntropyLoss()
            loss = loss_fn_tmp(outputs.double(), test_data.gnd_c_h.squeeze().long())

            valid_losses.append(loss.item())
            predict_label = outputs[:, 1] > outputs[:, 0]

            correct = torch.sum(predict_label == test_data.gnd_c_h.squeeze())
            total += test_data.gnd_c_h.shape[0] * test_data.gnd_c_h.shape[1]

            accuracy = 100 * torch.div(correct.float(), float(total))
            valid_acc_list.append(accuracy)
            print(f"epoch : {epoch + 1}, train loss : {train_losses_tsr[epoch, :]}, "
                  f"valid loss : {valid_losses[-1]}, valid acc : {accuracy}%")
    plt.figure(0)
    plt.plot(torch.arange(epochs), valid_losses, 'r--', linewidth=2, markersize=5)
    plt.title('Loss')
    plt.xlabel('Iteration')
    plt.ylabel('loss')
    color_map = ['blue', 'green', 'teal', 'm', 'purple', 'peru']
    marker_map = ['.', 'o', '^', '2', '+', 'x']
    linstyle_map = ['solid', 'dotted', 'dashed', 'dashdot', 'loosely dashdotted', 'loosely dashdotdotted']
    for jj in torch.arange(param_config.n_agents):
        plt.plot(torch.arange(epochs), train_losses_tsr[:, jj], color=color_map[int(jj)], marker='.',
                 linestyle='dashed',
                 linewidth=2, markersize=5)
    plt.show()
    plt.figure(1)
    plt.title('Accuracy on test data')
    plt.xlabel('Iteration')
    plt.ylabel('Acc')
    plt.plot(torch.arange(epochs), valid_acc_list, color='red', marker='^', linestyle='dashed',
             linewidth=2, markersize=5)
    plt.show()

    if test_data.task == 'C':
        param_config.log.info(f"Accuracy of training data using SVM: {train_losses_tsr[-1, :]}")
        param_config.log.info(f"Accuracy of test data using SVM: {valid_losses}")
    else:
        param_config.log.info(f"loss of training data using SVM: {train_losses_tsr[-1, :]}")
        param_config.log.info(f"loss of test data using SVM: {valid_losses}")
    test_map = torch.tensor(valid_acc_list).mean()
    return test_map, train_losses_tsr


def bp_fnn_run(param_config: ParamConfig, train_data: Dataset, test_data: Dataset):
    """
    todo: this is the method for hierarchical distribute fuzzy Neuron network using alternative optimizing
    :param param_config:
    :param train_data: training dataset
    :param test_data: test dataset
    :return:
    """
    n_batch = param_config.n_batch
    epochs = param_config.n_epoch
    lr = param_config.lr

    # # test the data on SVM method
    # paras = dict()
    # paras['kernel'] = 'rbf'
    # # paras['gamma'] = gamma
    # # paras['C'] = C
    # train_acc, test_acc = svc(train_data.fea, test_data.fea, train_data.gnd, test_data.gnd, Map(), paras=paras)
    # if test_data.task == 'C':
    #     param_config.log.info(f"Accuracy of training data using SVM: {train_acc:.2f}")
    #     param_config.log.info(f"Accuracy of test data using SVM: {test_acc:.2f}")
    # else:
    #     param_config.log.info(f"loss of training data using SVM: {train_acc:.4f}")
    #     param_config.log.info(f"loss of test data using SVM: {test_acc:.4f}")

    # traditional fnn
    rules = RuleKmeans()
    rules.fit(train_data.fea, param_config.n_rules)
    h_computer = HNormal()
    fnn_solver = FnnSolveReg()
    h, _ = h_computer.comute_h(train_data.fea, rules)
    # run FNN solver for given rule number
    fnn_solver.h = h
    # train_gnd_w = torch.zeros(train_data.n_smpl, 2)
    # for kk, gnd_tmp in enumerate(train_data.gnd):
    #     train_gnd_w[kk, int(gnd_tmp)] = 1
    # fnn_solver.y = train_gnd_w.double()
    fnn_solver.y = train_data.gnd
    fnn_solver.para_mu = 0.1
    # w_optimal = fnn_solver.solve()
    w_optimal = fnn_solver.solve().squeeze()
    rules.consequent_list = w_optimal

    h_test, _ = h_computer.comute_h(test_data.fea, rules)
    n_rule_test = h_test.shape[0]
    n_smpl_test = h_test.shape[1]
    n_fea_test = h_test.shape[2]
    h_cal_test = h_test.permute((1, 0, 2))  # N * n_rules * (d + 1)
    h_cal_test = h_cal_test.reshape(n_smpl_test, n_rule_test * n_fea_test)  # squess the last dimension

    # calculate Y hat
    # y_test_hat = h_cal_test.mm(rules.consequent_list.reshape(2, n_rule_test * n_fea_test).t())
    y_test_hat = h_cal_test.mm(rules.consequent_list.reshape(1, n_rule_test * n_fea_test).t())
    # y_test_hat = torch.where(y_test_hat[:, 1] > y_test_hat[:, 0], torch.tensor(1), torch.tensor(0))
    loss_func = LikelyLoss()
    acc_test_fnn = 100*loss_func.forward(test_data.gnd, y_test_hat)
    print(f"Accuracy of training data using Traditional FNN: {acc_test_fnn:.2f}")

    train_dataset = DatasetFNN(x=train_data.fea, y=train_data.gnd)
    valid_dataset = DatasetFNN(x=test_data.fea, y=test_data.gnd)

    train_loader = DataLoader(dataset=train_dataset, batch_size=n_batch, shuffle=True)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=n_batch, shuffle=False)

    # model_s: nn.Module = BP_FNN_CE(param_config.n_rules, train_data.n_fea)
    # correct = 0
    # total = 0
    # with torch.no_grad():
    #     for i, (data, labels) in enumerate(valid_loader):
    #         outputs = model_s(data)
    #
    #         # _, predicted = torch.max(outputs.data, 1)
    #         predicted = torch.where(outputs.data > 0.5, torch.tensor(1), torch.tensor(0))
    #         correct += (predicted == labels.squeeze().long()).sum().item()
    #         total += labels.size(0)
    #
    # accuracy_bp = 100 * correct / total
    # print(f"Accuracy of training data on rebuilt bp Traditional FNN before initiation: {accuracy_bp:.2f}")
    # # # set the value of sigma and mu
    # model_s.para_mu = torch.nn.Parameter(rules.center_list, requires_grad=True)
    # model_s.para_sigma = torch.nn.Parameter(rules.widths_list, requires_grad=True)
    # model_s.para_w3 = torch.nn.Parameter(rules.consequent_list, requires_grad=True)
    # model_s.eval()
    # correct = 0
    # total = 0
    # with torch.no_grad():
    #     for i, (data, labels) in enumerate(valid_loader):
    #         # h_test, _ = h_computer.comute_h(data, rules)
    #         # n_rule_test = h_test.shape[0]
    #         # n_smpl_test = h_test.shape[1]
    #         # n_fea_test = h_test.shape[2]
    #         # h_cal_test = h_test.permute((1, 0, 2))  # N * n_rules * (d + 1)
    #         # h_cal_test = h_cal_test.reshape(n_smpl_test, n_rule_test * n_fea_test)  # squess the last dimension
    #         #
    #         # # calculate Y hat
    #         # y_test_hat = h_cal_test.mm(rules.consequent_list.reshape(1, n_rule_test * n_fea_test).t())
    #         outputs = model_s(data)
    #
    #         # _, predicted = torch.max(outputs.data, 1)
    #         predicted = torch.where(outputs.data > 0.5, torch.tensor(1), torch.tensor(0))
    #         correct += (predicted == labels.squeeze().long()).sum().item()
    #         total += labels.size(0)
    #
    # accuracy_bp = 100 * correct / total
    # print(f"Accuracy of training data on rebuilt bp Traditional FNN after initiation: {accuracy_bp:.2f}")

    n_output = train_data.gnd.unique().shape[0]
    model: nn.Module = BP_FNN_L(param_config.n_rules, train_data.n_fea, n_output)

    # # set the value of sigma and mu
    # model.para_mu = torch.nn.Parameter(rules.center_list, requires_grad=True)
    # model.para_sigma = torch.nn.Parameter(rules.widths_list, requires_grad=True)
    # model.para_w3 = torch.nn.Parameter(rules.consequent_list, requires_grad=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.01)
    loss_fn = nn.CrossEntropyLoss()
    # loss_fn = nn.BCELoss()
    # loss_fn = nn.MSELoss()
    valid_acc_list = []

    train_losses = []
    valid_losses = []
    data_save_dir = f"./results/{param_config.dataset_folder}"

    if not os.path.exists(data_save_dir):
        os.makedirs(data_save_dir)
    model_save_file = f"{data_save_dir}/model{param_config.n_rules}.pkl"

    if os.path.exists(model_save_file):
        model.load_state_dict(torch.load(model_save_file))

    best_test_acc = 0
    for epoch in range(epochs):
        model.train()
        loss_tmp = 0
        for i, (data, labels) in enumerate(train_loader):
            # model.para_sigma[model.para_sigma < 0.01] = torch.tensor(0.01).double()
            optimizer.zero_grad()

            outputs = model(data)
            loss = loss_fn(outputs.double(), labels.long().squeeze(1))
            # loss = loss_fn(outputs, labels.double().squeeze(1))
            loss_tmp = loss.item()
            loss.backward()
            optimizer.step()

        train_losses.append(loss_tmp)

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for i, (data, labels) in enumerate(valid_loader):
                outputs = model(data)
                loss = loss_fn(outputs, labels.long().squeeze(1))
                # loss = loss_fn(outputs, labels.double().squeeze(1))
                loss_tmp = loss.item()

                _, predicted = torch.max(outputs.data, 1)
                # predicted = torch.where(outputs.data > 0.5, torch.tensor(1), torch.tensor(0))
                correct += (predicted == labels.squeeze().long()).sum().item()
                total += labels.size(0)

        valid_losses.append(loss_tmp)
        accuracy = 100 * correct / total
        if best_test_acc < accuracy:
            torch.save(model.state_dict(), model_save_file)
        valid_acc_list.append(accuracy)
        print(f"epoch : {epoch + 1}, train loss : {train_losses[-1]:.4f}, "
              f"valid loss : {valid_losses[-1]:.4f}, valid acc : {accuracy:.2f}%")

    # if test_data.task == 'C':
    #     param_config.log.info(f"Accuracy of training data using SVM: {train_losses:.2f}%")
    #     param_config.log.info(f"Accuracy of test data using SVM: {valid_losses:.2f}%")
    # else:
    #     param_config.log.info(f"loss of training data using SVM: {train_losses:.2f}")
    #     param_config.log.info(f"loss of test data using SVM: {valid_losses:.2f}")
    h, _ = h_computer.comute_h(train_data.fea, rules)
    # run FNN solver for given rule number
    fnn_solver.h = h
    # train_gnd_w = torch.zeros(train_data.n_smpl, 2)
    # for kk, gnd_tmp in enumerate(train_data.gnd):
    #     train_gnd_w[kk, int(gnd_tmp)] = 1
    # fnn_solver.y = train_gnd_w.double()
    fnn_solver.y = train_data.gnd
    fnn_solver.para_mu = 0.0000001
    w_optimal = fnn_solver.solve().squeeze()

    rules.update_rules(train_data.fea, model.para_mu.data)
    rules.widths_list = model.para_sigma.data
    rules.consequent_list = w_optimal

    h_test, _ = h_computer.comute_h(test_data.fea, rules)
    n_rule_test = h_test.shape[0]
    n_smpl_test = h_test.shape[1]
    n_fea_test = h_test.shape[2]
    h_cal_test = h_test.permute((1, 0, 2))  # N * n_rules * (d + 1)
    h_cal_test = h_cal_test.reshape(n_smpl_test, n_rule_test * n_fea_test)
    y_test_hat = h_cal_test.mm(rules.consequent_list.reshape(1, n_rule_test * n_fea_test).t())
    # y_test_hat = torch.where(y_test_hat[:, 1] > y_test_hat[:, 0], torch.tensor(1), torch.tensor(0))
    y_test_hat = torch.where(y_test_hat > 0.5, torch.tensor(1), torch.tensor(0))
    acc_train_num = torch.where(y_test_hat == test_data.gnd, torch.tensor(1), torch.tensor(0)).sum()
    acc_test_fnn = 100 * acc_train_num / float(test_data.n_smpl)
    print(f"Accuracy of training data using Traditional FNN: {acc_test_fnn:.2f}")

    plt.figure(0)
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('loss')
    color_map = ['blue', 'green', 'teal', 'm', 'purple', 'peru']
    marker_map = ['.', 'o', '^', '2', '+', 'x']
    linstyle_map = ['solid', 'dotted', 'dashed', 'dashdot', 'loosely dashdotted', 'loosely dashdotdotted']
    plt.plot(torch.arange(len(train_losses)), train_losses, color=color_map[int(0)], marker='.', linestyle='dashed',
             linewidth=2, markersize=5)

    plt.plot(torch.arange(len(valid_losses)), torch.tensor(valid_losses), 'r--', linewidth=2, markersize=5)
    plt.legend(['train loss', 'valid loss'])
    plt.show()
    plt.figure(1)
    plt.title('Accuracy on test data')
    plt.xlabel('Epoch')
    plt.ylabel('Acc')
    plt.plot(torch.arange(epochs), valid_acc_list, color='red', marker='^', linestyle='dashed',
             linewidth=2, markersize=5)
    plt.show()

    if test_data.task == 'C':
        param_config.log.info(f"Accuracy of training data using SVM: {train_losses[-1]}")
        param_config.log.info(f"Accuracy of test data using SVM: {valid_losses}")
    else:
        param_config.log.info(f"loss of training data using SVM: {train_losses[-1]}")
        param_config.log.info(f"loss of test data using SVM: {valid_losses}")
    test_map = torch.tensor(valid_acc_list).mean()
    return test_map, train_losses


def mlp_run(param_config: ParamConfig, train_data: Dataset, test_data: Dataset):
    """
    todo: this is the method for distribute fuzzy Neuron network
    :param param_config:
    :param train_data: training dataset
    :param test_data: test dataset
    :return:
    """

    train_dataset = DatasetFNN(x=train_data.fea, y=train_data.gnd)
    valid_dataset = DatasetFNN(x=test_data.fea, y=test_data.gnd)

    train_loader = DataLoader(dataset=train_dataset, batch_size=2000, shuffle=True)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=2000, shuffle=False)

    model = dev_network_s(train_data.fea.shape[1], train_data.gnd.unique().shape[0])
    # # compile the keras model
    # model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
    # # fit the keras model on the dataset 78.5%
    # model.fit(train_data.fea.numpy(), train_data.gnd.numpy().ravel(), epochs=1000, batch_size=200)
    # _, loss_keras = model.evaluate(test_data.fea.numpy(), test_data.gnd.numpy().ravel())
    # print(f"keras loss : {loss_keras:.2f}%")
    # compile the keras model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # fit the keras model on the dataset 78.5%
    train_y_binary = to_categorical(train_data.gnd[1:8, :].numpy())
    model.fit(train_data.fea[1:8, :].numpy(), train_y_binary, epochs=100, batch_size=50)
    test_y_binary = to_categorical(test_data.gnd.numpy())
    _, accuracy_keras = model.evaluate(test_data.fea.numpy(), test_y_binary)
    print(f"keras acc : {100*accuracy_keras:.2f}%")
    model: nn.Module = MLP()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()
    # loss_fn = nn.BCELoss()
    valid_acc_list = []
    epochs = 30000

    train_losses = []
    valid_losses = []

    for epoch in range(epochs):
        model.train()

        for i, (images, labels) in enumerate(train_loader):

            optimizer.zero_grad()

            outputs = model(images.float())
            # loss = loss_fn(outputs.double(), labels.double().squeeze(1))
            loss = loss_fn(outputs, labels.long().squeeze())
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for i, (images, labels) in enumerate(valid_loader):
                outputs = model(images.float())
                # loss = loss_fn(outputs, labels.float().squeeze(1))
                loss = loss_fn(outputs.double(), labels.long().squeeze())
                valid_losses.append(loss.item())

                # predicted = torch.where(outputs.data > 0.5, torch.tensor(1), torch.tensor(0))
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted.squeeze() == labels.squeeze().long()).sum().item()
                total += labels.size(0)

        accuracy = 100 * correct / total
        valid_acc_list.append(accuracy)
        print(f"epoch : {epoch + 1}, train loss : {train_losses[-1]}, "
              f"valid loss : {valid_losses[-1]}, valid acc : {accuracy}%")
    plt.figure(0)
    plt.plot(torch.arange(len(valid_losses)), torch.tensor(valid_losses), 'r--', linewidth=2, markersize=5)
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('loss')
    color_map = ['blue', 'green', 'teal', 'm', 'purple', 'peru']
    marker_map = ['.', 'o', '^', '2', '+', 'x']
    linstyle_map = ['solid', 'dotted', 'dashed', 'dashdot', 'loosely dashdotted', 'loosely dashdotdotted']
    plt.plot(torch.arange(len(train_losses)), train_losses, color=color_map[int(0)], marker='.', linestyle='dashed',
             linewidth=2, markersize=5)

    plt.show()
    plt.figure(1)
    plt.title('Accuracy on test data')
    plt.xlabel('Epoch')
    plt.ylabel('Acc')
    plt.plot(torch.arange(epochs), valid_acc_list, color='red', marker='^', linestyle='dashed',
             linewidth=2, markersize=5)
    plt.show()

    if test_data.task == 'C':
        param_config.log.info(f"Accuracy of training data using SVM: {train_losses}")
        param_config.log.info(f"Accuracy of test data using SVM: {valid_losses}")
    else:
        param_config.log.info(f"loss of training data using SVM: {train_losses}")
        param_config.log.info(f"loss of test data using SVM: {valid_losses}")
    test_map = torch.tensor(valid_acc_list).mean()
    return test_map, train_losses


def mlp_run_r(train_data: Dataset, test_data: Dataset):
    """
    todo: this is the method for distribute fuzzy Neuron network
    :param train_data: training dataset
    :param test_data: test dataset
    :return:
    """

    model = dev_network_s_r(train_data.fea.shape[1])
    # compile the keras model
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
    # fit the keras model on the dataset 78.5%
    model.fit(train_data.fea.numpy(), train_data.gnd.numpy(), epochs=100, batch_size=500)
    # test_y_binary = to_categorical(test_data.gnd.numpy())
    loss_train, _ = model.evaluate(train_data.fea.numpy(), train_data.gnd.numpy())
    print(f"keras test loss : {loss_train:.4f}%")
    loss_test, _ = model.evaluate(test_data.fea.numpy(), test_data.gnd.numpy())
    print(f"keras test loss : {loss_test:.4f}%")


def bp_hdfnn_kfolds(param_config: ParamConfig, dataset: Dataset):
    """
    todo: this is the method for distribute fuzzy Neuron network
    :param param_config:
    :param dataset: dataset
    :return:
    """
    loss_c_train_tsr = []
    loss_c_test_tsr = []
    loss_d_train_tsr = []
    loss_d_test_tsr = []

    for k in torch.arange(param_config.n_kfolds):
        param_config.log.info(f"start traning at {dataset.partition.current_fold + 1}-fold!")
        train_data, test_data = dataset.get_kfold_data(param_config.n_brunches, param_config.n_agents, k)

        # bp_fnn_run(param_config, train_data, test_data)
        mlp_run(param_config, train_data, test_data)

        # loss_c_train_tsr.append(train_loss_c)
        # loss_c_test_tsr.append(test_loss_c)
        #
        # loss_d_train_tsr.append(cfnn_train_loss)
        # loss_d_test_tsr.append(cfnn_test_loss)

    loss_c_train_tsr = torch.tensor(loss_c_train_tsr)
    loss_c_test_tsr = torch.tensor(loss_c_test_tsr)
    loss_d_train_tsr = torch.tensor(loss_d_train_tsr)
    loss_d_test_tsr = torch.tensor(loss_d_test_tsr)

    if dataset.task == 'C':
        param_config.log.war(f"Mean Accuracy of training data on centralized method:"
                             f" {loss_c_train_tsr.mean()}")
        param_config.log.war(f"Mean Accuracy  of test data on centralized method: "
                             f"{loss_c_test_tsr.mean()}")
        param_config.log.war(f"Mean Accuracy  of training data on distributed method:"
                             f" {loss_d_train_tsr.mean()}")
        param_config.log.war(f"Mean Accuracy  of test data on distributed method: "
                             f"{loss_d_test_tsr.mean()}")
    else:
        param_config.log.war(f"loss of training data on centralized method: "
                             f"{loss_c_train_tsr.mean()}")
        param_config.log.war(f"loss of test data on centralized method: "
                             f"{loss_c_test_tsr.mean()}")
        param_config.log.war(f"loss of training data on distributed method: "
                             f"{loss_d_train_tsr.mean()}")
        param_config.log.war(f"loss of test data on distributed method: "
                             f"{loss_d_test_tsr.mean()}")
    return loss_c_train_tsr, loss_c_test_tsr, loss_d_train_tsr, loss_d_test_tsr
