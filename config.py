import torch


class Config(object):
    def __init__(self):
        
        # model
        self.llama1_cache_path = "./decapoda-research/llama-7b-hf/"
        self.llama2_cache_path = './meta-llama/Llama-2-7b-hf/'
        self.llama_version = 2  # 1 or 2
        self.seed = 129
        self.dataset = 'CHEF'  # 'CHEF' or 'FEVER'
        self.debugging = False
        
        # CHEF
        if self.dataset == 'CHEF':
            self.label = 'not_gold_label'  # 'gold_label'
            # self.train_dataset_path = "data/CHEF_train_modified.json"
            # self.test_dataset_path = "data/CHEF_test_modified.json"
            self.train_dataset_path = "data/CHEF_train_correct.json"
            self.test_dataset_path = "data/CHEF_test_correct.json"
        
        # FEVER
        if self.dataset == 'FEVER':
            self.train_dataset_path = "data/FEVER_train.json"
            self.test_dataset_path = "data/FEVER_test.json"
        
        # Test
        if self.debugging:
            if self.dataset == 'CHEF':
                self.train_dataset_path = "data/c_train.json"
                self.test_dataset_path = "data/c_test.json"
            elif self.dataset == 'FEVER':
                self.train_dataset_path = "data/f_train.json"
                self.test_dataset_path = "data/f_test.json"
            else:
                raise ValueError('dataset must be CHEF or FEVER')
        
        # hyper parameters
        self.num_labels = 3
        self.feat_dim = 4096
        self.max_seq_len = 512
        self.pad_token_id = 0
        self.padding_side = 'right'  # 'left' or 'right'
        self.weight_decay = 0.01
        self.lr = 2e-4
        self.nuclear_loss_weight = 2e-5
        self.v_label_loss_weight = torch.tensor([1.0, 1.0, 1.0])
        self.label_loss_weight = torch.tensor([1.0, 1.0, 1.0])
        # self.v_label_loss_weight = torch.tensor([1.0, 0.6615, 3.7075])  # CHEF Label Count {'0': 2877, '1': 4349, '2': 776}
        # self.label_loss_weight = torch.tensor([1.0, 0.6615, 3.7075])  # CHEF Label Count {'0': 2877, '1': 4349, '2': 776}
        
        # lora parameters
        self.lora_r = 4
        self.lora_alpha = 8
        self.lora_dp = 0.05
        
        # gnn
        self.dp = 0.1
        self.gnn_dims = '16,4,1'
        self.att_heads = '16,8,4'
        self.name_list_q_gat = 'q_clause-level_gat,q_sentence-level_gat,q_text-level_gat'
        self.name_list_v_gat = 'v_clause-level_gat,v_sentence-level_gat,v_text-level_gat'
        
        # training parameters
        self.epochs = 20
        self.train_batch_size = 2
        self.eval_batch_size = 1
        self.gradient_accumulation_steps = 4
        self.warmup_proportion = 0.05