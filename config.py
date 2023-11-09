

class Config(object):
    def __init__(self):
        
        # model
        self.llama_cache_path = "./LLAMA_local/decapoda-research/llama-7b-hf/"
        self.seed = 129
        
        # dataset
        self.train_dataset_path = "data/CHEF_train_modified.json"
        self.test_dataset_path = "data/CHEF_test_modified.json"
        # self.train_dataset_path = "data/train.json"
        # self.test_dataset_path = "data/test.json"
        self.label = 'not_gold_label'
        
        # hyper parameters
        self.num_labels = 3
        self.feat_dim = 4096
        self.max_seq_len = 512
        self.pad_token_id = 0
        self.weight_decay = 0.01
        self.lr = 2e-5
        self.nuclear_loss_weight = 2e-5
        
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
        self.epochs = 10
        self.train_batch_size = 2
        self.eval_batch_size = 1
        self.gradient_accumulation_steps = 2
        self.warmup_proportion = 0.1