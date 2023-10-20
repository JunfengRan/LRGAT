import torch


class Config(object):
    def __init__(self):
        self.train_dataset_path = "data/CHEF_train_modified.json"
        self.test_dataset_path = "data/CHEF_test_modified.json"
        # self.train_dataset_path = "data/CHEF_train_lengthened.json"
        # self.test_dataset_path = "data/CHEF_test_lengthened.json"
        # self.train_dataset_path = "data/CHEF_train.json"
        # self.test_dataset_path = "data/CHEF_test.json"
        
        self.seed = 129
        self.num_labels = 3
        self.epochs = 30
        self.batch_size = 4
        self.lr = 1e-5
        self.tuning_bert_rate = 1e-5
        self.gradient_accumulation_steps = 2
        self.dp = 0.1
        self.warmup_proportion = 0.1
        self.feat_dim = 4096
        self.gnn_dims = '16,4,1'
        self.att_heads = '16,8,4'
        # self.name_list_q_gat = 'q_clause-level_gat,q_sentence-level_gat,q_text-level_gat'
        self.name_list_v_gat = 'v_clause-level_gat,v_sentence-level_gat,v_text-level_gat'
        