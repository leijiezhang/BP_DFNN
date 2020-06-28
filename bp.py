import torch
import torch.nn as nn


class BP_FNN_Ite(torch.nn.Module):
    def __init__(self, n_rules, n_fea):
        super(BP_FNN_Ite, self).__init__()
        self.n_rules = n_rules
        self.n_fea = n_fea

        # parameters in level1
        para_mu_item = torch.rand(self.n_rules, self.n_fea)
        self.para_mu = torch.nn.Parameter(para_mu_item, requires_grad=True)
        para_sigma_item = torch.rand(self.n_rules, self.n_fea)
        self.para_sigma = torch.nn.Parameter(para_sigma_item, requires_grad=True)
        # parameters in level3
        para_w3_item = torch.rand(2, self.n_rules, self.n_fea+1)
        self.para_w3 = torch.nn.Parameter(para_w3_item, requires_grad=True)

    def forward(self, data: torch.Tensor):
        n_batch = data.shape[0]
        result_list = torch.zeros(n_batch, 2)
        for i in torch.arange(n_batch):
            data_item = data[i, :].unsqueeze(0)
            # 1) fuzzification layer (Gaussian membership functions)
            layer_fuzzify = torch.exp(-(data_item.repeat([self.n_rules, 1]) - self.para_mu) ** 2 / (2 * self.para_sigma ** 2))
            # 2) rule layer (compute firing strength values)
            layer_rule = torch.prod(layer_fuzzify, dim=1)
            # 3) normalization layer (normalize firing strength)
            inv_frn = torch.unsqueeze(torch.reciprocal(torch.sum(layer_rule, dim=0)), 0)
            layer_normalize = torch.mul(layer_rule, inv_frn.expand_as(layer_rule))
            # 4) consequence layer (TSK-type)
            x_rep = data_item.expand([data_item.size(0), layer_fuzzify.size(1)])
            layer_conq_a = torch.add(self.para_w3[0, :, -1],
                                     torch.sum(torch.mul(self.para_w3[0, :, :-1], x_rep), dim=1))
            layer_conq_b = torch.add(self.para_w3[1, :, -1],
                                     torch.sum(torch.mul(self.para_w3[1, :, :-1], x_rep), dim=1))

            # 5) output layer for brunch level
            layer_out_tsk_a = torch.sum(torch.mul(layer_normalize, layer_conq_a), dim=0)
            layer_out_tsk_b = torch.sum(torch.mul(layer_normalize, layer_conq_b), dim=0)

            # 6) consequence layer (TSK-type) bottom layer
            result_final = torch.cat([layer_out_tsk_a.unsqueeze(0), layer_out_tsk_b.unsqueeze(0)], 0)
            result_final = torch.nn.functional.softmax(result_final, dim=0)
            result_list[i, :] = result_final

        return result_list


class BP_FNN(torch.nn.Module):
    def __init__(self, n_rules, n_fea):
        super(BP_FNN, self).__init__()
        self.n_rules = n_rules
        self.n_fea = n_fea

        # parameters in level1
        para_mu_item = torch.rand(self.n_rules, self.n_fea)
        self.para_mu = torch.nn.Parameter(para_mu_item, requires_grad=True)
        para_sigma_item = torch.rand(self.n_rules, self.n_fea)
        self.para_sigma = torch.nn.Parameter(para_sigma_item, requires_grad=True)
        # parameters in level3
        para_w3_item = torch.rand(self.n_rules, self.n_fea+1)
        self.para_w3 = torch.nn.Parameter(para_w3_item, requires_grad=True)

    def forward(self, data: torch.Tensor):
        # n_batch = data.shape[0]
        data_item = data.unsqueeze(1)
        # 1) fuzzification layer (Gaussian membership functions)
        layer_fuzzify = torch.exp(
            -(data_item.repeat([1, self.n_rules, 1]) - self.para_mu) ** 2 / (2 * self.para_sigma ** 2))
        # 2) rule layer (compute firing strength values)
        layer_rule = torch.prod(layer_fuzzify, dim=2)
        # layer_rule[layer_rule < 10 ^ (-8)] = torch.tensor(10 ^ (-8)).double()
        layer_rule = layer_rule + torch.tensor(10 ^ (-18)).double()
        # 3) normalization layer (normalize firing strength)
        inv_frn = torch.unsqueeze(torch.reciprocal(torch.sum(layer_rule, dim=1)), 1)
        layer_normalize = torch.mul(layer_rule, inv_frn.expand_as(layer_rule))
        # 4) consequence layer (TSK-type)
        x_rep = torch.unsqueeze(data, dim=1).expand([data.size(0), layer_fuzzify.size(1), layer_fuzzify.size(2)])
        layer_conq = torch.add(self.para_w3[:, 0],
                                 torch.sum(torch.mul(self.para_w3[:, 1::], x_rep), dim=2))

        # 5) output layer for brunch level
        layer_out_tsk = torch.sum(torch.mul(layer_normalize, layer_conq), dim=1)
        # 6) consequence layer (TSK-type) bottom layer
        result_list = torch.nn.functional.sigmoid(layer_out_tsk)
        # result_list = layer_out_tsk
        # if the result has Nan
        if result_list[torch.isnan(result_list)].shape[0] > 0:
            print("there is some dirty data in the final result")

        return result_list


class BP_FNN_L(torch.nn.Module):
    def __init__(self, n_rules, n_fea, n_output):
        super(BP_FNN_L, self).__init__()
        self.n_rules = n_rules
        self.n_fea = n_fea
        self.n_output = n_output

        # parameters in level1
        para_mu_item = torch.rand(self.n_rules, self.n_fea)
        self.para_mu = torch.nn.Parameter(para_mu_item, requires_grad=True)
        para_sigma_item = torch.ones(self.n_rules, self.n_fea)
        self.para_sigma = torch.nn.Parameter(para_sigma_item, requires_grad=True)
        # parameters in level3
        self.layer3 = nn.Linear(self.n_fea, self.n_rules)
        # self.layer3 = nn.Sequential(
        #     nn.Linear(self.n_fea, 2*self.n_fea),
        #     nn.ReLU(),+
        #     nn.Linear(2*self.n_fea, self.n_fea),
        #     nn.ReLU(),
        #     nn.Linear(self.n_fea, self.n_rules),
        # )
        self.layer4 = nn.Linear(self.n_rules, self.n_output)

    def forward(self, data: torch.Tensor):
        # n_batch = data.shape[0]
        data_item = data.unsqueeze(1)
        para_sigma_c = torch.log(0.01 + torch.exp(self.para_sigma))
        # 1) fuzzification layer (Gaussian membership functions)
        layer_fuzzify = torch.exp(
            -(data_item.repeat([1, self.n_rules, 1]) - self.para_mu) ** 2 / (2 * para_sigma_c ** 2))
        # 2) rule layer (compute firing strength values)
        layer_rule = torch.prod(layer_fuzzify, dim=2)
        # layer_rule[layer_rule < 10 ^ (-8)] = torch.tensor(10 ^ (-8)).double()
        layer_rule = layer_rule + torch.tensor(10 ** (-5)).double()
        # 3) normalization layer (normalize firing strength)
        inv_frn = torch.unsqueeze(torch.reciprocal(torch.sum(layer_rule, dim=1)), 1)
        layer_normalize = torch.mul(layer_rule, inv_frn.expand_as(layer_rule))
        # 4) consequence layer (TSK-type)
        output_layer3 = self.layer3(data.float())

        # 5) output layer for brunch level
        layer_out_tsk = self.layer4(torch.mul(layer_normalize.float(), output_layer3))
        # 6) consequence layer (TSK-type) bottom layer
        # result_list = torch.nn.functional.softmax(layer_out_tsk, dim=1)
        result_list = layer_out_tsk
        # if the result has Nan
        if result_list[torch.isnan(result_list)].shape[0] > 0:
            print("there is some dirty data in the final result")

        return result_list


class BP_FNN_M(torch.nn.Module):
    def __init__(self, n_rules, n_fea):
        super(BP_FNN_M, self).__init__()
        self.n_rules = n_rules
        self.n_fea = n_fea

        # parameters in level1
        para_mu_item = torch.rand(self.n_rules, self.n_fea)
        self.para_mu = torch.nn.Parameter(para_mu_item, requires_grad=True)
        para_sigma_item = torch.rand(self.n_rules, self.n_fea)
        self.para_sigma = torch.nn.Parameter(para_sigma_item, requires_grad=True)
        # parameters in level3
        self.layer_m = nn.Sequential(
            nn.Linear(18, 36),
            nn.ReLU(),
            nn.Linear(36, 18),
            nn.ReLU(),
            nn.Linear(18, 2),
            # nn.ReLU()
        )
        # self.layer3 = nn.Linear(self.n_fea, self.n_rules)
        # self.layer4 = nn.Linear(self.n_rules, 2)

    def forward(self, data: torch.Tensor):
        # n_batch = data.shape[0]
        data_item = data.unsqueeze(1)
        # 1) fuzzification layer (Gaussian membership functions)
        layer_fuzzify = torch.exp(
            -(data_item.repeat([1, self.n_rules, 1]) - self.para_mu) ** 2 / (2 * self.para_sigma ** 2))
        # 2) rule layer (compute firing strength values)
        layer_rule = torch.prod(layer_fuzzify, dim=2)
        # layer_rule[layer_rule < 10 ^ (-8)] = torch.tensor(10 ^ (-8)).double()
        layer_rule = layer_rule + torch.tensor(10 ** (-198)).double()
        # 3) normalization layer (normalize firing strength)
        inv_frn = torch.unsqueeze(torch.reciprocal(torch.sum(layer_rule, dim=1)), 1)
        layer_normalize = torch.mul(layer_rule, inv_frn.expand_as(layer_rule))
        # 4) consequence layer (TSK-type)
        output_layer3 = self.layer_m(data.float())

        # 5) output layer for brunch level
        layer_normalize_expand = layer_normalize.unsqueeze(2).repeat([1, 1, 2])
        output_layer3_expand = output_layer3.unsqueeze(1).repeat([1, self.n_rules, 1])
        layer_out_tsk = torch.mul(layer_normalize_expand.float(), output_layer3_expand)
        # 6) consequence layer (TSK-type) bottom layer
        # result_list = torch.nn.functional.softmax(layer_out_tsk, dim=1)
        result_list = layer_out_tsk.sum(1).squeeze()
        # if the result has Nan
        if result_list[torch.isnan(result_list)].shape[0] > 0:
            print("there is some dirty data in the final result")

        return result_list


class BP_FNN_S(torch.nn.Module):
    def __init__(self, n_rules, n_fea):
        super(BP_FNN_S, self).__init__()
        self.n_rules = n_rules
        self.n_fea = n_fea

        # parameters in level1
        para_mu_item = torch.rand(self.n_rules, self.n_fea)
        self.para_mu = torch.nn.Parameter(para_mu_item, requires_grad=True)
        para_sigma_item = torch.rand(self.n_rules, self.n_fea)
        self.para_sigma = torch.nn.Parameter(para_sigma_item, requires_grad=True)
        # parameters in level3
        para_w3_item = torch.rand(self.n_rules, self.n_fea+1)
        self.para_w3 = torch.nn.Parameter(para_w3_item, requires_grad=True)

    def forward(self, data: torch.Tensor):
        # n_batch = data.shape[0]
        data_item = data.unsqueeze(1)
        # 1) fuzzification layer (Gaussian membership functions)
        layer_fuzzify = torch.exp(
            -(data_item.repeat([1, self.n_rules, 1]) - self.para_mu) ** 2 / (2 * self.para_sigma ** 2))
        # 2) rule layer (compute firing strength values)
        layer_rule = torch.prod(layer_fuzzify, dim=2)
        # layer_rule_item = layer_rule
        # layer_rule_item.requires_grad = False
        # layer_rule_item[layer_rule < 10 ^ (-8)] = torch.tensor(10 ^ (-538)).double()
        # layer_rule_item[layer_rule >= 10 ^ (-8)] = torch.tensor(0).double()
        layer_rule = layer_rule + torch.tensor(10 ** (-538)).double()
        # 3) normalization layer (normalize firing strength)
        inv_frn = torch.unsqueeze(torch.reciprocal(torch.sum(layer_rule, dim=1)), 1)
        layer_normalize = torch.mul(layer_rule, inv_frn.expand_as(layer_rule))
        # 4) consequence layer (TSK-type)
        x_rep = torch.unsqueeze(data, dim=1).expand([data.size(0), layer_fuzzify.size(1), layer_fuzzify.size(2)])
        layer_conq = torch.add(self.para_w3[:, 0],
                                 torch.sum(torch.mul(self.para_w3[:, 1::], x_rep), dim=2))

        # 5) output layer for brunch level
        layer_out_tsk = torch.sum(torch.mul(layer_normalize, layer_conq), dim=1)
        # 6) consequence layer (TSK-type) bottom layer
        # result_list = torch.nn.functional.sigmoid(layer_out_tsk)
        result_list = layer_out_tsk
        # if the result has Nan
        if result_list[torch.isnan(result_list)].shape[0] > 0:
            print("there is some dirty data in the final result")

        return result_list


class BP_FNN_CE(torch.nn.Module):
    def __init__(self, n_rules, n_fea):
        super(BP_FNN_CE, self).__init__()
        self.n_rules = n_rules
        self.n_fea = n_fea

        # parameters in level1
        para_mu_item = torch.rand(self.n_rules, self.n_fea)
        self.para_mu = torch.nn.Parameter(para_mu_item, requires_grad=True)
        para_sigma_item = torch.rand(self.n_rules, self.n_fea)
        self.para_sigma = torch.nn.Parameter(para_sigma_item, requires_grad=True)
        # parameters in level3
        para_w3_item = torch.rand(2, self.n_rules, self.n_fea+1)
        self.para_w3 = torch.nn.Parameter(para_w3_item, requires_grad=True)

    def forward(self, data: torch.Tensor):
        # n_batch = data.shape[0]
        data_item = data.unsqueeze(1)
        # 1) fuzzification layer (Gaussian membership functions)
        layer_fuzzify = torch.exp(
            -(data_item.repeat([1, self.n_rules, 1]) - self.para_mu) ** 2 / (2 * self.para_sigma ** 2))
        # 2) rule layer (compute firing strength values)
        layer_rule = torch.prod(layer_fuzzify, dim=2)
        # layer_rule[layer_rule < 10 ^ (-8)] = torch.tensor(10 ^ (-8)).double()
        layer_rule = layer_rule + torch.tensor(10 ^ (-18)).double()
        # 3) normalization layer (normalize firing strength)
        inv_frn = torch.unsqueeze(torch.reciprocal(torch.sum(layer_rule, dim=1)), 1)
        layer_normalize = torch.mul(layer_rule, inv_frn.expand_as(layer_rule))
        # 4) consequence layer (TSK-type)
        x_rep = torch.unsqueeze(data, dim=1).expand([data.size(0), layer_fuzzify.size(1), layer_fuzzify.size(2)])
        layer_conq_a = torch.add(self.para_w3[0, :, -1],
                                 torch.sum(torch.mul(self.para_w3[0, :, :-1], x_rep), dim=2))
        layer_conq_b = torch.add(self.para_w3[1, :, -1],
                                 torch.sum(torch.mul(self.para_w3[1, :, :-1], x_rep), dim=2))

        # 5) output layer for brunch level
        layer_out_tsk_a = torch.sum(torch.mul(layer_normalize, layer_conq_a), dim=1)
        layer_out_tsk_b = torch.sum(torch.mul(layer_normalize, layer_conq_b), dim=1)
        # 6) consequence layer (TSK-type) bottom layer
        result_final = torch.cat([layer_out_tsk_a.unsqueeze(1), layer_out_tsk_b.unsqueeze(1)], 1)
        # result_list = torch.nn.functional.softmax(result_final, dim=1)
        result_list = result_final
        # if the result has Nan
        if result_list[torch.isnan(result_list)].shape[0] > 0:
            print("there is some dirty data in the final result")

        return result_list


class BP_HFNN(torch.nn.Module):
    def __init__(self, n_brunch, n_rules, n_fea):
        super(BP_HFNN, self).__init__()
        self.n_bruches = n_brunch
        self.n_rules = n_rules
        self.n_fea = n_fea

        # parameters in level1
        para_mu_item = torch.zeros(self.n_bruches, self.n_rules, self.n_fea)
        self.para_mu = torch.nn.Parameter(para_mu_item, requires_grad=True)
        para_sigma_item = torch.zeros(self.n_bruches, self.n_rules, self.n_fea)
        self.para_sigma = torch.nn.Parameter(para_sigma_item, requires_grad=True)
        # parameters in level3
        para_w3_item = torch.zeros(self.n_bruches, self.n_rules, self.n_fea+1)
        self.para_w3 = torch.nn.Parameter(para_w3_item, requires_grad=True)

        # # parameters in level5
        # para_w5_item = torch.zeros(self.n_bruches + 1)
        # self.para_w5 = torch.nn.Parameter(para_w5_item, requires_grad=True)

        # for classification task
        self.level5 = nn.Linear(self.n_bruches, 2)

    def forward(self, data: torch.Tensor):
        n_batch = data.shape[1]
        result_list = torch.zeros(n_batch, 2)
        for i in torch.arange(n_batch):
            data_item = data[:, i, :].unsqueeze(1)
            # 1) fuzzification layer (Gaussian membership functions)
            layer_fuzzify = torch.exp(-(data_item.repeat([1, self.n_rules, 1]) - self.para_mu) ** 2 / (2 * self.para_sigma ** 2))
            # 2) rule layer (compute firing strength values)
            layer_rule = torch.prod(layer_fuzzify, dim=2)
            # 3) normalization layer (normalize firing strength)
            inv_frn = torch.unsqueeze(torch.reciprocal(torch.sum(layer_rule, dim=1)), 1)
            layer_normalize = torch.mul(layer_rule, inv_frn.expand_as(layer_rule))
            # 4) consequence layer (TSK-type)
            x_rep = data_item.expand([data.size(0), layer_fuzzify.size(1), layer_fuzzify.size(2)])
            layer_conq_a = torch.add(self.para_w3[ :, :, -1], torch.sum(torch.mul(self.para_w3[:, :, :-1], x_rep), dim=2))

            # 5) output layer for brunch level
            layer_out_tsk_a = torch.sum(torch.mul(layer_normalize, layer_conq_a), dim=1)

            # 6) consequence layer (TSK-type) bottom layer
            result_final = self.level5(layer_out_tsk_a)
            result_final = torch.nn.functional.softmax(result_final, dim=0)
            result_list[i, :] = result_final

        return result_list


class BP_HDFNN(torch.nn.Module):
    def __init__(self, n_brunch, n_rules, n_fea, n_agents):
        super(BP_HDFNN, self).__init__()
        self.n_bruches = n_brunch
        self.n_rules = n_rules
        self.n_fea = n_fea
        self.n_agents = n_agents

        # parameters in level1
        para_mu_item = torch.rand(self.n_bruches, self.n_rules, self.n_fea)
        para_mu_tmp = para_mu_item.repeat(self.n_agents, 1, 1, 1).float()
        self.para_mu = torch.nn.Parameter(para_mu_tmp, requires_grad=True)
        para_sigma_item = torch.rand(self.n_bruches, self.n_rules, self.n_fea)
        para_sigma_tmp = para_sigma_item.repeat(self.n_agents, 1, 1, 1)
        self.para_sigma = torch.nn.Parameter(para_sigma_tmp, requires_grad=True)
        # parameters in level3
        para_w3_item = torch.rand(self.n_bruches, self.n_rules, self.n_fea+1)
        para_w3_tmp = para_w3_item.repeat(self.n_agents, 1, 1, 1)
        self.para_w3 = torch.nn.Parameter(para_w3_tmp, requires_grad=True)

        # parameters in level5
        para_w5_item = torch.rand(self.n_bruches + 1)
        para_w5_tmp = para_w5_item.repeat(self.n_agents, 1)
        self.para_w5 = torch.nn.Parameter(para_w5_tmp, requires_grad=True)

        # for classification task
        # self.bottom_layers = nn.Sequential(
        #     nn.Linear(self.n_bruches*self.n_agents, 2),
        #     nn.Sigmoid(),
        # )

    def forward(self, data: torch.Tensor):
        n_batch = data.shape[2]
        result_list = torch.zeros(self.n_agents, n_batch)
        for i in torch.arange(n_batch):
            data_item = data[:, :, i, :].unsqueeze(2)
            # 1) fuzzification layer (Gaussian membership functions)
            layer_fuzzify = torch.exp(-(data_item.repeat([1, 1, self.n_rules, 1]) - self.para_mu) ** 2 / (2 * self.para_sigma ** 2))
            # 2) rule layer (compute firing strength values)
            layer_rule = torch.prod(layer_fuzzify, dim=3)
            # 3) normalization layer (normalize firing strength)
            inv_frn = torch.unsqueeze(torch.reciprocal(torch.sum(layer_rule, dim=2)), 2)
            layer_normalize = torch.mul(layer_rule, inv_frn.expand_as(layer_rule))
            # 4) consequence layer (TSK-type)
            x_rep = data_item.expand([data.size(0), layer_fuzzify.size(1), layer_fuzzify.size(2), layer_fuzzify.size(3)])
            layer_conq_a = torch.add(self.para_w3[:, :, :, -1], torch.sum(torch.mul(self.para_w3[:, :, :, :-1], x_rep), dim=3))

            # 5) output layer for brunch level
            layer_out_tsk_a = torch.sum(torch.mul(layer_normalize, layer_conq_a), dim=2)

            # 6) consequence layer (TSK-type) bottom layer
            result_final = torch.add(self.para_w5[:, -1], torch.sum(torch.mul(self.para_w5[:, :-1], layer_out_tsk_a), dim=1))
            result_final = result_final.softmax()
            # # 6) bottom layer
            # layer_out_tsk_a = layer_out_tsk_a.view(-1)
            # result_final
            result_list[:, i] = result_final

        return result_list


class BP_HDFNN_C(torch.nn.Module):
    '''
    this is a class for classification task
    '''
    def __init__(self, n_brunch, n_rules, n_fea, n_agents):
        super(BP_HDFNN_C, self).__init__()
        self.n_bruches = n_brunch
        self.n_rules = n_rules
        self.n_fea = n_fea
        self.n_agents = n_agents

        # parameters in level1
        para_mu_item = torch.rand(self.n_bruches, self.n_rules, self.n_fea)
        para_mu_tmp = para_mu_item.repeat(self.n_agents, 1, 1, 1).float()
        self.para_mu = torch.nn.Parameter(para_mu_tmp, requires_grad=True)
        para_sigma_item = torch.rand(self.n_bruches, self.n_rules, self.n_fea)
        para_sigma_tmp = para_sigma_item.repeat(self.n_agents, 1, 1, 1)
        self.para_sigma = torch.nn.Parameter(para_sigma_tmp, requires_grad=True)
        # parameters in level3
        para_w3_item = torch.rand(self.n_bruches, self.n_rules, self.n_fea + 1)
        para_w3_tmp = para_w3_item.repeat(self.n_agents, 1, 1, 1)
        self.para_w3 = torch.nn.Parameter(para_w3_tmp, requires_grad=True)

        # for classification task
        self.bottom_layers = nn.Sequential(
            nn.Linear(self.n_bruches*self.n_agents, 2),
            nn.Sigmoid(),
        )

    def forward(self, data: torch.Tensor):
        n_batch = data.shape[2]
        result_list = torch.zeros(self.n_agents, n_batch)
        for i in torch.arange(n_batch):
            data_item = data[:, :, i, :].unsqueeze(2)
            # 1) fuzzification layer (Gaussian membership functions)
            layer_fuzzify = torch.exp(
                -(data_item.repeat([1, 1, self.n_rules, 1]) - self.para_mu) ** 2 / (2 * self.para_sigma ** 2))
            # 2) rule layer (compute firing strength values)
            layer_rule = torch.prod(layer_fuzzify, dim=3)
            # 3) normalization layer (normalize firing strength)
            inv_frn = torch.unsqueeze(torch.reciprocal(torch.sum(layer_rule, dim=2)), 2)
            layer_normalize = torch.mul(layer_rule, inv_frn.expand_as(layer_rule))
            # 4) consequence layer (TSK-type)
            x_rep = data_item.expand(
                [data.size(0), layer_fuzzify.size(1), layer_fuzzify.size(2), layer_fuzzify.size(3)])
            layer_conq_a = torch.add(self.para_w3[:, :, :, -1],
                                     torch.sum(torch.mul(self.para_w3[:, :, :, :-1], x_rep), dim=3))

            # 5) output layer for brunch level
            layer_out_tsk_a = torch.sum(torch.mul(layer_normalize, layer_conq_a), dim=2)

            # # 6) bottom layer
            layer_out_tsk_a = layer_out_tsk_a.view(-1)
            result_final = self.bottom_layers(layer_out_tsk_a)
            result_list[:, i] = result_final

        return result_list