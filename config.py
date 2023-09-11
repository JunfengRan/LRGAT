import torch
DEVICE = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
# DEVICE = torch.device('cpu')

TORCH_SEED = 129


class Config(object):
    def __init__(self):
        self.bert_cache_path = 'bert-base-chinese'
        # self.train_dataset_path = "Data/CHEF_train_lengthened.json"
        # self.test_dataset_path = "Data/CHEF_test_lengthened.json"
        self.train_dataset_path = "Data/CHEF_train.json"
        self.test_dataset_path = "Data/CHEF_test.json"
        # self.train_dataset_path = "Data/train.json"
        # self.test_dataset_path = "Data/test.json"
        
        # hyper parameter
        self.num_classes = 3
        self.epochs = 30
        self.batch_size = 4
        self.lr = 1e-5
        self.tuning_bert_rate = 1e-5
        self.gradient_accumulation_steps = 2
        self.dp = 0.1
        self.warmup_proportion = 0.1

        # gnn
        self.feat_dim = 768
        self.gnn_dims = '192'
        self.att_heads = '4'

