import torch
from utils import Logger
from dataset import Dataset
from partition import KFoldPartition
import yaml
import scipy.io as sio


class ParamConfig(object):
    def __init__(self, n_run=1, n_kfolds=10, n_agents=25, nrules=10):
        self.n_run = n_run  # Number of simulations
        self.n_batch = 100
        self.n_epoch = 1000
        self.n_kfolds = n_kfolds  # Number of folds
        self.n_brunches = 0

        # Network configuration
        self.n_agents = n_agents

        # self.connectivity = 0.25  # Connectivity in the networks(must be between 0 and 1)

        self.n_rules = nrules  # number of rules in stage 1
        self.n_rules_list = []

        self.dataset_list = ['CASP']
        self.dataset_folder = 'hrss'

        # set learning rate
        self.lr = 0

        self.log = None

    def config_parse(self, config_name):
        config_dir = f"./configs/{config_name}.yaml"
        config_file = open(config_dir)
        config_content = yaml.load(config_file, Loader=yaml.FullLoader)

        self.n_run = config_content['n_run']
        self.n_batch = config_content['n_batch']
        self.n_epoch = config_content['n_epoch']
        self.n_kfolds = config_content['n_kfolds']
        self.n_brunches = config_content['n_brunches']
        self.n_agents = config_content['n_agents']

        self.n_rules = config_content['n_rules']
        self.n_rules_list = config_content['n_rules_list']
        self.lr = config_content['lr']

        self.dataset_list = config_content['dataset_list']
        self.dataset_folder = config_content['dataset_folder']

        # set logger to decide whether write log into files
        if config_content['log_to_file'] == 'false':
            self.log = Logger()
        else:
            self.log = Logger(True)

    def get_dataset(self, dataset_idx=0):
        dataset_name = self.dataset_list[dataset_idx]
        dir_dataset = f"./datasets/{self.dataset_folder}/{dataset_name}.pt"

        load_data = torch.load(dir_dataset)
        dataset_name = load_data['name']
        fea: torch.Tensor = load_data['X']
        gnd: torch.Tensor = load_data['Y']

        if len(gnd.shape) == 1:
            gnd = gnd.unsqueeze(1)

        task = load_data['task']
        dataset = Dataset(fea, gnd, task, dataset_name)

        # set partition strategy
        partition_strategy = KFoldPartition(self.n_kfolds)
        partition_strategy.partition(dataset.gnd, True, 0)
        dataset.set_partition(partition_strategy)
        # dataset.normalize(-1, 1)
        return dataset

    def get_dataset_mat(self, dataset_idx=0):
        dataset_name = self.dataset_list[dataset_idx]
        dir_dataset = f"./datasets/{self.dataset_folder}/{dataset_name}"

        load_data = sio.loadmat(dir_dataset)
        dataset_name = load_data['name']
        fea: torch.Tensor = torch.tensor(load_data['X']).float()
        gnd: torch.Tensor = torch.tensor(load_data['Y']).float()

        if len(gnd.shape) == 1:
            gnd = gnd.unsqueeze(1)

        task = load_data['task']
        dataset = Dataset(fea, gnd, task, dataset_name)

        # set partition strategy
        partition_strategy = KFoldPartition(self.n_kfolds)
        partition_strategy.partition(dataset.gnd, True, 0)
        dataset.set_partition(partition_strategy)
        # dataset.normalize(-1, 1)
        return dataset
