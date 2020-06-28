import torch
from dataset import Dataset
from param_config import ParamConfig
import matplotlib.pyplot as plt


class BP_HDFNN(object):
    def __init__(self, train_data: Dataset, test_data: Dataset, param_config: ParamConfig):
        self.lr = param_config.lr
        self.n_epoch = param_config.n_epoch
        self.train_data = train_data
        self.test_data = test_data

        self.n_agents = self.train_data.n_agents
        self.n_bruches = self.train_data.n_brunch
        self.n_rules = param_config.n_rules
        self.n_batch = param_config.n_batch
        self.n_smpl_train = self.train_data.n_smpl_d
        self.n_smpl_test = self.test_data.n_smpl_d
        self.n_fea = self.train_data.n_fea_d

        # save model outputs
        self.losses = []
        self.stds = []

        self.config = param_config

        # parameters in level1
        para_mu_item = torch.rand(self.n_bruches, self.n_rules, self.n_fea)
        self.para_mu = para_mu_item.repeat(self.n_agents, 1, 1, 1)
        para_sigma_item = torch.rand(self.n_bruches, self.n_rules, self.n_fea)
        self.para_sigma = para_sigma_item.repeat(self.n_agents, 1, 1, 1)
        # parameters in level3
        para_w3_item = torch.rand(self.n_bruches, self.n_rules, self.n_fea+1)
        self.para_w3 = para_w3_item.repeat(self.n_agents, 1, 1, 1)
        # parameters in level5
        para_w5_item = torch.rand(self.n_bruches + 1)
        self.para_w5 = para_w5_item.repeat(self.n_agents, 1)

        # datas involved in the model
        self.data_level0 = torch.zeros(self.n_agents, self.n_bruches, self.n_batch, self.n_fea)
        self.o_f1 = torch.zeros(self.n_agents, self.n_bruches, self.n_rules, self.n_batch, self.n_fea)
        self.o_a1 = torch.zeros(self.n_agents, self.n_bruches, self.n_rules, self.n_batch, self.n_fea)

        self.o_f2 = torch.zeros(self.n_agents, self.n_bruches, self.n_rules, self.n_batch)
        self.o_a2 = torch.zeros(self.n_agents, self.n_bruches, self.n_rules, self.n_batch)

        ones_level3 = torch.ones(self.n_agents, self.n_bruches, self.n_batch, 1)
        self.data_level3 = torch.cat([ones_level3, self.data_level0], 3)
        self.o_f3 = torch.zeros(self.n_agents, self.n_bruches, self.n_rules, self.n_batch)
        self.o_fc3 = torch.zeros(self.n_agents, self.n_bruches, self.n_rules, self.n_batch)

        self.o_f4 = torch.zeros(self.n_agents, self.n_batch, self.n_bruches)

        ones_level5 = torch.ones(self.n_agents, self.n_batch, 1)
        self.data_level5 = torch.cat([ones_level5, self.o_f4], 2)
        self.o_f5 = torch.zeros(self.n_agents, self.n_batch)

        self.o_sigmoid = torch.zeros(self.n_agents, self.n_batch)
        self.o_ce = torch.rand(self.n_agents, self.n_batch)

        self.gnd_train = self.train_data.gnd_d
        self.gnd_test = self.test_data.gnd_d
        self.test_acc_list = []

        # gradients involved in the model
        # self.grad_ce_sigmoid = torch.ones(self.n_agents, self.n_batch)
        # self.grad_sigmoid_f5 = torch.ones(self.n_agents, self.n_batch)
        self.grad_f5_w5 = torch.zeros(self.n_agents, self.n_batch, self.n_bruches+1)
        self.grad_f5_f4 = torch.zeros(self.n_agents, self.n_bruches+1)
        self.grad_f4_f3 = torch.zeros(self.n_agents, self.n_bruches, self.n_rules, self.n_batch)
        self.grad_fc3_w3 = torch.zeros(self.n_agents, self.n_bruches, self.n_rules, self.n_batch, self.n_fea+1)
        self.grad_f3_fc3 = torch.zeros(self.n_agents, self.n_bruches, self.n_rules, self.n_batch)
        self.grad_f3_a2 = torch.zeros(self.n_agents, self.n_bruches, self.n_rules, self.n_batch)
        self.grad_a2_f2 = torch.zeros(self.n_agents, self.n_bruches, self.n_rules, self.n_batch)
        self.grad_f2_a1 = torch.zeros(self.n_agents, self.n_bruches, self.n_rules, self.n_batch, self.n_fea)
        self.grad_a1_f1 = torch.zeros(self.n_agents, self.n_bruches, self.n_rules, self.n_batch, self.n_fea)
        self.grad_f1_mu = torch.zeros(self.n_agents, self.n_bruches, self.n_rules, self.n_batch, self.n_fea)
        self.grad_f1_sigma = torch.zeros(self.n_agents, self.n_bruches, self.n_rules, self.n_batch, self.n_fea)

        self.grad_ce_f5 = torch.ones(self.n_agents, self.n_batch)
        self.grad_ce_w5 = torch.zeros(self.n_agents, self.n_batch, self.n_bruches + 1)
        self.grad_ce_f4 = torch.zeros(self.n_agents, self.n_batch, self.n_bruches + 1)
        self.grad_ce_f3 = torch.zeros(self.n_agents, self.n_bruches, self.n_rules, self.n_batch)
        self.grad_ce_w3 = torch.zeros(self.n_agents, self.n_bruches, self.n_rules, self.n_batch, self.n_fea + 1)
        self.grad_ce_fc3 = torch.zeros(self.n_agents, self.n_bruches, self.n_rules, self.n_batch)
        self.grad_ce_a2 = torch.zeros(self.n_agents, self.n_bruches, self.n_rules, self.n_batch)
        self.grad_ce_f2 = torch.zeros(self.n_agents, self.n_bruches, self.n_rules, self.n_batch)
        self.grad_ce_a1 = torch.zeros(self.n_agents, self.n_bruches, self.n_rules, self.n_batch, self.n_fea)
        self.grad_ce_f1 = torch.zeros(self.n_agents, self.n_bruches, self.n_rules, self.n_batch, self.n_fea)
        self.grad_ce_mu = torch.zeros(self.n_agents, self.n_bruches, self.n_rules, self.n_batch, self.n_fea)
        self.grad_ce_sigma = torch.zeros(self.n_agents, self.n_bruches, self.n_rules, self.n_batch, self.n_fea)

    def get_batch_data(self, batch_idx):
        start_idx = batch_idx*self.n_batch
        end_idx = (batch_idx+1)*self.n_batch
        start_idx = start_idx % (self.n_smpl_train+1)
        end_idx = end_idx % (self.n_smpl_train+1)

        if end_idx > start_idx:
            self.data_level0 = self.train_data.fea_d[:, :, start_idx:end_idx, :]
            self.gnd_train = self.train_data.gnd_d[:, start_idx:end_idx]
        else:
            data_level0_1 = self.train_data.fea_d[:, :, start_idx:self.n_smpl_train, :]
            data_level0_2 = self.train_data.fea_d[:, :, 0:end_idx+1, :]

            gnd_train_1 = self.train_data.gnd_d[:, start_idx:self.n_smpl_train]
            gnd_train_2 = self.train_data.gnd_d[:, 0:end_idx+1]
            self.data_level0 = torch.cat([data_level0_1, data_level0_2], 2)
            self.gnd_train = torch.cat([gnd_train_1, gnd_train_2], 1)
        ones_level3 = torch.ones(self.n_agents, self.n_bruches, self.n_batch, 1)
        self.data_level3 = torch.cat([ones_level3, self.data_level0], 3)
        
    def level1_f(self, x: torch.Tensor, m: torch.Tensor, sigma: torch.Tensor,):
        n_fea = x.shape[1]
        n_smpl = x.shape[0]
        m = m.repeat(n_smpl, 1)
        sigma = sigma.repeat(n_smpl, 1)
        sigma_div = torch.div(torch.ones(n_smpl, n_fea), sigma)
        f1 = torch.mul((x-m), sigma_div)
        f1 = -torch.mul(f1, f1)
        return f1

    def bp_f1_m(self, x: torch.Tensor, m: torch.Tensor, sigma: torch.Tensor):
        n_fea = x.shape[1]
        n_smpl = x.shape[0]
        m = m.repeat(n_smpl, 1)
        sigma = sigma.repeat(n_smpl, 1)
        sigma_div = torch.div(torch.ones(n_smpl, n_fea), sigma)
        d_f1_m = torch.mul((x-m), torch.mul(sigma_div, sigma_div))
        d_f1_m = torch.mul(2, d_f1_m)
        return d_f1_m

    def bp_f1_sigma(self, x: torch.Tensor, m: torch.Tensor, sigma: torch.Tensor):
        n_fea = x.shape[1]
        n_smpl = x.shape[0]
        m = m.repeat(n_smpl, 1)
        sigma = sigma.repeat(n_smpl, 1)
        sigma_div = torch.div(torch.ones(n_smpl, n_fea), sigma)
        sigma_div2 = torch.mul(sigma_div, sigma_div)
        sigma_div3 = torch.mul(sigma_div, sigma_div2)
        x_m2 = torch.mul((x-m), (x-m))
        d_f1_sigma = torch.mul(x_m2, sigma_div3)
        return d_f1_sigma

    def level1_a(self, f1: torch.Tensor):
        a1 = torch.exp(f1)
        return a1

    def bp_a1_f1(self, o_a1_item: torch.Tensor):
        # d_a1_f1 = torch.exp(f1)
        d_a1_f1 = o_a1_item
        return d_a1_f1

    def level2_f(self, a1: torch.Tensor):
        f2 = torch.prod(a1, 1)
        illegal_idx = torch.where(f2 < torch.tensor(0.1).pow(8))[0]
        f2[illegal_idx] = torch.tensor(0.1).pow(8)
        return f2

    def bp_f2_a1(self, a1: torch.Tensor):
        n_fea = a1.shape[1]
        n_smpl = a1.shape[0]
        d_f2_a1 = torch.ones(n_smpl, n_fea)
        a1_prod = torch.prod(a1, 1)
        illegal_idx = torch.where(a1_prod < torch.tensor(0.1).pow(8))[0]
        a1_prod[illegal_idx] = torch.tensor(0.1).pow(8)
        for i in torch.arange(n_fea):
            d_f2_a1[:, i] = torch.div(a1_prod, d_f2_a1[:, i])
        return d_f2_a1

    def level2_a(self, f2: torch.Tensor):
        n_rules = f2.shape[0]
        n_batch = f2.shape[1]
        a2 = torch.zeros(n_rules, n_batch)
        for i in torch.arange(n_rules):
            a2[i, :] = torch.div(f2[i, :], torch.sum(f2, 0))
        return a2

    def bp_a2_f2(self, f2: torch.Tensor):
        n_fea = f2.shape[1]
        n_smpl = f2.shape[0]
        f2_sum = torch.sum(f2, 1)
        f2_sum_squre = torch.mul(f2_sum, f2_sum)
        d_a2_f2 = torch.ones(n_smpl, n_fea)
        for i in torch.arange(n_fea):
            f2_sum_sub = f2_sum - f2[:, i]
            d_a2_f2[:, i] = torch.div(f2_sum_sub, f2_sum_squre)
        return d_a2_f2

    def fc3(self, w3: torch.Tensor, x: torch.Tensor):
        return x.mm(w3)

    def bp_fc3_w3(self, x: torch.Tensor):
        return x

    def level3_f(self, a2: torch.Tensor, w3: torch.Tensor, x: torch.Tensor):
        fc3 = x.mm(w3)
        f3 = a2*fc3
        return f3

    def bp_f3_a2(self, fc3: torch.Tensor):
        return fc3

    def bp_f3_fc3(self, a2: torch.Tensor):
        return a2

    def level4_f(self, f3: torch.Tensor):
        f4 = torch.sum(f3, 0)
        return f4

    def bp_f4_f3(self, f3: torch.Tensor):
        one_item = torch.ones(f3.shape[0], f3.shape[1])
        return one_item

    def level5_f(self, f4: torch.Tensor, w5: torch.Tensor):
        f5 = f4.mm(w5)
        return f5

    def bp_f5_f4(self, w5: torch.Tensor):
        return w5

    def bp_f5_w5(self, f4: torch.Tensor):
        return f4

    def sigmoid(self, in_data: torch.Tensor):
        n_batch = in_data.shape[1]
        n_agent = in_data.shape[0]
        one_item = torch.ones(n_agent, n_batch)
        output = torch.div(one_item, one_item+torch.exp(-in_data))
        return output

    def bp_sigmoid(self, o_sigmoid: torch.Tensor):
        n_agent = o_sigmoid.shape[0]
        n_batch = o_sigmoid.shape[1]
        one_item = torch.ones(n_agent, n_batch)
        output = torch.mul(o_sigmoid, one_item-o_sigmoid)
        return output

    def cross_entropy(self, y_hat, y):
        return -((1-y)*torch.log(1 - y_hat) + y*torch.log(y_hat))

    def bp_cross_entropy(self, y_hat: torch.Tensor, y: torch.Tensor, x: torch.Tensor):
        # n_smpl = x.shape[0]
        # output = torch.sum(torch.mul((y_hat-y), x), 1) / n_smpl
        output = torch.mul((y_hat-y), x)
        return output

    def bp_ce_sigmoid(self, y, o_sigmoid: torch.Tensor):
        return o_sigmoid - y

    def feed_forward(self, is_train=True):
        n_smpl = self.n_smpl_test
        if is_train:
            n_smpl = self.n_batch
        o_w5_item = torch.zeros(self.n_agents, n_smpl)
        for i in torch.arange(self.n_agents):
            o_f4_item = torch.zeros(self.n_bruches, n_smpl)
            for j in torch.arange(self.n_bruches):
                # get the data assigned on each brunch
                if is_train:
                    fea_item = self.data_level0[i, j, :, :]
                else:
                    fea_item = self.test_data.fea_d[i, j, :, :]
                # the output of level1's fuzzy set
                o_f2_item = torch.zeros(self.n_rules, n_smpl)
                for k in torch.arange(self.n_rules):
                    mu_item = self.para_mu[i, j, k, :]
                    sigma_item = self.para_sigma[i, j, k, :]
                    o_f1_item = self.level1_f(fea_item, mu_item, sigma_item)
                    if is_train:
                        self.o_f1[i, j, k, :, :] = o_f1_item
                    o_a1_item = self.level1_a(o_f1_item)
                    if is_train:
                        self.o_a1[i, j, k, :, :] = o_a1_item
                    o_f2_item[k, :] = self.level2_f(o_a1_item)

                if is_train:
                    self.o_f2[i, j, :, :] = o_f2_item
                o_level2_item = self.level2_a(o_f2_item)
                if is_train:
                    self.o_a2[i, j, :, :] = o_level2_item

                # data prosessing in level 3
                w3_item = self.para_w3[i, j, :, :].t()
                if is_train:
                    fea_item_expand = self.data_level3[i, j, :, :]
                else:
                    ones_level3 = torch.ones(self.n_smpl_test, 1)
                    fea_item_expand = torch.cat([ones_level3, fea_item], 1)

                o_fc3_item = fea_item_expand.mm(w3_item).t()
                if is_train:
                    self.o_fc3[i, j, :, :] = o_fc3_item
                o_f3_item = torch.mul(o_level2_item, o_fc3_item)
                if is_train:
                    self.o_f3[i, j, :, :] = o_f3_item

                # data prosessing in level4
                o_f4_item[j, :] = self.level4_f(o_f3_item)
            if is_train:
                self.o_f4[i, :, :] = o_f4_item.t()

            # data prosessing in level5
            o_f4_expand = torch.cat([torch.ones(n_smpl, 1), o_f4_item.t()], 1)
            if is_train:
                self.data_level5[i, :, :] = o_f4_expand
            w5_item = self.para_w5[i, :]
            o_w5_item[i, :] = o_f4_expand.mm(w5_item.unsqueeze(1)).squeeze()
        if is_train:
            self.o_f5 = o_w5_item
        
        o_sigmoid = self.sigmoid(o_w5_item)
        if is_train:
            self.o_sigmoid = o_sigmoid
        else:
            o_sigmoid = torch.round(o_sigmoid)
            n_correct = torch.where(o_sigmoid == self.gnd_test)[1].shape[0]
            test_acc = n_correct / self.n_smpl_test
            self.config.log.info(f"test acc: {test_acc}")
            self.test_acc_list.append(test_acc)
        
        if is_train:
            o_ce = self.cross_entropy(o_sigmoid, self.gnd_train)
            self.o_ce = o_ce

    def backward(self, is_train=True):
        self.feed_forward(is_train)
        # self.grad_ce_sigmoid = self.bp_cross_entropy(self.o_ce, self.gnd_train, self.o_sigmoid)
        # self.grad_sigmoid_f5 = self.bp_sigmoid(self.o_sigmoid)

        self.grad_ce_f5 = self.bp_ce_sigmoid(self.gnd_train, self.o_sigmoid)

        for i in torch.arange(self.n_agents):
            w5_item = self.para_w5[i, :]
            self.grad_f5_f4[i, :] = self.bp_f5_f4(w5_item)
            o_f4_expand = self.data_level5[i, :, :]
            self.grad_f5_w5[i, :, :] = self.bp_f5_w5(o_f4_expand)

            for j in torch.arange(self.n_bruches):
                o_f3_item = self.o_f3[i, j, :, :]
                self.grad_f4_f3[i, j, :, :] = self.bp_f4_f3(o_f3_item)

                o_a2_item = self.o_a2[i, j, :, :]
                o_fc3_item = self.o_fc3[i, j, :, :]
                self.grad_f3_fc3[i, j, :, :] = self.bp_f3_fc3(o_a2_item)
                self.grad_f3_a2[i, j, :, :] = self.bp_f3_a2(o_fc3_item)
                input_level3_item = self.data_level3[i, j, :, :]

                o_f2_item = self.o_f2[i, j, :, :]
                self.grad_a2_f2[i, j, :, :] = self.bp_a2_f2(o_f2_item)

                input_level1_item = self.data_level0[i, j, :, :]
                for k in torch.arange(self.n_rules):
                    o_a1_item = self.o_a1[i, j, k, :, :]
                    self.grad_f2_a1[i, j, :, :, :] = self.bp_f2_a1(o_a1_item)
                    self.grad_fc3_w3[i, j, k, :, :] = self.bp_fc3_w3(input_level3_item)
                    o_a1_item = self.o_a1[i, j, k, :, :]
                    self.grad_a1_f1[i, j, k, :, :] = self.bp_a1_f1(o_a1_item)
                    mu_item = self.para_mu[i, j, k, :]
                    sigma_item = self.para_sigma[i, j, k, :]
                    self.grad_f1_mu[i, j, k, :, :] = self.bp_f1_m(input_level1_item, mu_item, sigma_item)
                    self.grad_f1_sigma[i, j, k, :, :] = self.bp_f1_sigma(input_level1_item, mu_item, sigma_item)

        # calculate gradient
        # self.grad_ce_f5 = torch.mul(self.grad_ce_sigmoid, self.grad_sigmoid_f5)
        grad_f5_f4_expand = self.grad_f5_f4.unsqueeze(1).repeat(1, self.n_batch, 1)
        grad_ce_f5_expand = self.grad_ce_f5.unsqueeze(2).repeat([1, 1, self.n_bruches + 1])
        self.grad_ce_f4 = torch.mul(grad_ce_f5_expand, grad_f5_f4_expand)

        self.grad_ce_w5 = torch.mul(grad_ce_f5_expand, self.grad_f5_w5)

        grad_ce_f4_expand = self.grad_ce_f4[:, :, 1:].permute([0, 2, 1]).unsqueeze(2).repeat(1, 1, self.n_rules, 1)
        self.grad_ce_f3 = torch.mul(grad_ce_f4_expand, self.grad_f4_f3)
        self.grad_ce_fc3 = torch.mul(self.grad_ce_f3, self.grad_f3_fc3)

        grad_ce_f3_expand = self.grad_ce_fc3.unsqueeze(4).repeat([1, 1, 1, 1, self.n_fea + 1])
        self.grad_ce_w3 = torch.mul(grad_ce_f3_expand, self.grad_fc3_w3)

        self.grad_ce_a2 = torch.mul(self.grad_ce_f3, self.grad_f3_a2)
        self.grad_ce_f2 = torch.mul(self.grad_ce_a2, self.grad_a2_f2)
        grad_ce_f2_expand = self.grad_ce_f2.unsqueeze(4).repeat([1, 1, 1, 1, self.n_fea])
        self.grad_ce_a1 = torch.mul(grad_ce_f2_expand, self.grad_f2_a1)
        self.grad_ce_f1 = torch.mul(self.grad_ce_a1, self.grad_a1_f1)
        self.grad_ce_mu = torch.mul(self.grad_ce_f1, self.grad_f1_mu)
        self.grad_ce_sigma = torch.mul(self.grad_ce_f1, self.grad_f1_sigma)

        # update parameters
        self.para_w5 = self.para_w5 - self.lr * self.grad_ce_w5.mean(1).mean(0).repeat([self.n_agents, 1])
        self.para_w3 = self.para_w3 - self.lr * self.grad_ce_w3.mean(3).mean(0).repeat([self.n_agents, 1, 1, 1])
        self.para_mu = self.para_mu - self.lr * self.grad_ce_mu.mean(3).mean(0).repeat([self.n_agents, 1, 1, 1])
        self.para_sigma = self.para_sigma - self.lr * self.grad_ce_sigma.mean(3).mean(0).repeat([self.n_agents, 1, 1, 1])

    def tune(self):
        current_epoch = 0
        while current_epoch < self.n_epoch:
            self.config.log.info(f"runing at {current_epoch + 1}-epoch!")
            self.get_batch_data(current_epoch)
            self.backward()
            loss = self.o_ce.mean(1).mean(0)
            std = self.o_ce.mean(1).std(0)
            self.config.log.info(f"training loss: {loss} std:{std}")
            self.losses.append(loss)
            self.stds.append(std)
            if (current_epoch % 200) == 0:
                self.feed_forward(False)
            current_epoch = current_epoch+1

        plt.plot(torch.arange(self.n_epoch), self.losses)
        plt.show()
        plt.figure()
        plt.plot(torch.arange(len(self.test_acc_list)), self.test_acc_list)
        plt.show()
