# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy
import sklearn.datasets 
import sklearn.cluster
import torch


def progress_bar(batch_num, report_interval, last_loss):
    print("\r Epoch:{:3d},  Loss: {:.6f}".format(batch_num, last_loss), end='\r')


def get_iris_data():
    iris = sklearn.datasets.load_iris() # load iris data
    normalize = sklearn.preprocessing.MaxAbsScaler() # claim normalizaton method

    samples = normalize.fit_transform(iris.data) # get samples and normalize to [0,1]
    labels = iris.target # get target
    
    tensor_x = torch.Tensor(samples) # transfer data into tensor
    tensor_y = torch.Tensor(labels)
    return tensor_x, tensor_y, samples, labels


def init_antecedent_para(x, n_rules):
    clustering = sklearn.cluster.KMeans(n_clusters=n_rules, init='k-means++')
    results = clustering.fit(x)
    cy = results.labels_
    mu = results.cluster_centers_
    sig = numpy.zeros((n_rules,len(x[0])))
    for i in range(n_rules):
        idx = numpy.argwhere(cy==i)
        sig[i,:] = numpy.std(x[idx])    
    return mu, sig


def init_conseqence_para(x, labels, n_rules):
    upper=numpy.max(labels,axis=0)
    lower=numpy.min(labels,axis=0)
    dim_q = len(x[0]) + 1
    conq = (numpy.random.rand(n_rules,dim_q)*2-1)*0.01
    for i in range(n_rules):
        conq[i][0] = numpy.random.rand()*(upper-lower) + lower
    return conq


class TS_FuzzyNeural(torch.nn.Module):
    def __init__(self,sz_data, n_rules, n_in_varl, n_out_varl, init_cen, init_sig, init_conq):
        super(TS_FuzzyNeural, self).__init__()
        self.n_rules = n_rules
        self.n_in_varl = n_in_varl
        self.n_out_varl = n_out_varl
        # initialize centres of fuzzy sets
        self.cen = torch.nn.Parameter(torch.tensor(init_cen, dtype=torch.float))  
        self.cen.requires_grad = True
        # initialize widths of fuzzy sets
        self.sig = torch.nn.Parameter(torch.tensor(init_sig, dtype=torch.float)) 
        self.sig.requires_grad = True
        # initialize consequence components 
        self.conq = torch.nn.Parameter(torch.tensor(conq, dtype=torch.float))    
        self.conq.requires_grad = True

    def forward(self, x):
        #1) fuzzification layer (Gaussian membership functions)
        layer_fuzzify = torch.exp(-(torch.unsqueeze(x, dim=1) - self.cen) ** 2 / (2 * self.sig ** 2))
        
        #2) rule layer (compute firing strength values)
        layer_rule = torch.prod(layer_fuzzify, dim=2)  

        #3) normalization layer (normalize firing strength)
        inv_frn = torch.unsqueeze(torch.reciprocal(torch.sum(layer_rule,dim=1)),1)
        layer_normalize = torch.mul(layer_rule, inv_frn.expand_as(layer_rule))
        
        #4) consequence layer (TSK-type)
        tsk_a0 = torch.unsqueeze(self.conq[:,-1], dim=0).expand([x.size(0), layer_fuzzify.size(1)])
        tsk_a = torch.unsqueeze(self.conq[:,:-1], dim=0).expand([x.size(0), layer_fuzzify.size(1), layer_fuzzify.size(2)])
        x_rep = torch.unsqueeze(x, dim=1).expand([x.size(0), layer_fuzzify.size(1), layer_fuzzify.size(2)])
        layer_conq = torch.add(tsk_a0, torch.sum(torch.mul(tsk_a,x_rep),dim=2))

        #5) output layer
        layer_out = torch.sum(torch.mul(layer_normalize, layer_conq),dim=1)

        return layer_out




# Main ========================================================================
        
n_rules = 6    
n_in_varl = 4
n_out_varl = 1
max_epoch = 500
    
tensor_x, tensor_y, samples, labels = get_iris_data()
cen, sig = init_antecedent_para(samples, n_rules)
conq =  init_conseqence_para(samples, labels, n_rules)
fnn = TS_FuzzyNeural(tensor_x.size(0), n_rules, n_in_varl, n_out_varl, cen, sig, conq)

lossfunction = torch.nn.MSELoss()
optimizer = torch.optim.Rprop(fnn.parameters(), lr=1e-3)

for t in range(max_epoch):
    # Forward pass: Compute predicted y by passing x to the model
    fout = fnn(tensor_x)

    # Compute and print loss
    loss = lossfunction(fout, tensor_y)
    progress_bar(batch_num=t, report_interval=10, last_loss=loss.item())

    # Zero gradients, perform a backward pass, and update the weights.
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

