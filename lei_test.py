from fcm import FCM
import torch
import csv
import scipy.io as sio
from utils import dataset_parse


data_list = ['HRSS_anomalous_optimized']

for i in torch.arange(len(data_list)):
    dataset_parse(data_list[i])
seperator = []
sub_set1 = torch.linspace(0, 2, 1)



print('lei')