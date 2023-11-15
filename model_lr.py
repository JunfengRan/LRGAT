import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import LlamaForSequenceClassification
from peft import LoraConfig, get_peft_model
from accelerate import Accelerator

from config import *
from gnn_layer import *


configs = Config()
accelerator = Accelerator()
device = accelerator.device


class CustomLlamaForClassification(LlamaForSequenceClassification):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = configs.num_labels
        self.score = nn.Linear(config.hidden_size, self.num_labels, bias=False)


class CastOutputToFloat(nn.Sequential):
    def forward(self, x):
        return super().forward(x).to(torch.float32)


# Define custom loss function for nuclear norm regularization
class NuclearNormLoss(nn.Module):
    def __init__(self, weight=1.0):
        super(NuclearNormLoss, self).__init__()
        self.weight = weight

    def forward(self, input_matrix):
        # Compute the singular value decomposition
        U, S, V = torch.svd(input_matrix)
        
        # Calculate the nuclear norm (sum of singular values)
        nuclear_norm = torch.sum(S)
        
        # Apply the weight
        loss = self.weight * nuclear_norm
        
        return loss


class Network(nn.Module):
    def __init__(self, configs):
        super(Network, self).__init__()
        self.num_labels = configs.num_labels
        
        self.model = CustomLlamaForClassification.from_pretrained(
            configs.llama2_cache_path,
            load_in_8bit=True,
            device_map='auto',
        )
        
        # freeze original weights
        # list(self.model.parameters())[0].dtype

        for i, param in enumerate(self.model.parameters()):
            param.requires_grad = False  # freeze the model - train adapters later
            # print(i, 'param.requires_grad = False')
            if param.ndim == 1:
                # cast the small parameters (e.g. layernorm) to fp32 for stability
                param.data = param.data.to(torch.float32)
                # print(i, 'ndim == 1, torch.float16 to torch.float32')

        # reduce number of stored activations
        self.model.gradient_checkpointing_enable()
        self.model.enable_input_require_grads()
        
        self.model.score = nn.Linear(configs.feat_dim, self.num_labels, bias=False)

        # LoRa Adapters
        self.lora_config = LoraConfig(
            r=configs.lora_r,  # low rank
            lora_alpha=configs.lora_alpha,  # alpha scaling, scale lora weights/outputs
            # target_modules=["q_proj", "v_proj"], #if you know
            lora_dropout=configs.lora_dp,
            bias="none",
            task_type="CAUSAL_LM"  # set this for CLM or Seq2Seq
        )
        
        self.model = get_peft_model(self.model, self.lora_config)
        self.model.base_model.model.model.layers[31].self_attn.v_proj.lora_A.default = CustomGAT(configs, name="v_custom_gat")
        
        self.print_trainable_parameters(self.model)
        print(self.model)
        
        # use hook to obtain labels
        self.feat_hook_nuclear_train = torch.tensor([]).to(device)
        self.feat_hook_nuclear_eval = torch.tensor([]).to(device)
        self.feat_hook_label_train = torch.tensor([]).to(device)
        self.feat_hook_label_eval = torch.tensor([]).to(device)

        nuclear_layer = "base_model.model.model.layers.31.self_attn.v_proj.lora_A.default.gnn_layer_stack.v_sentence-level_gat"
        label_layer = "base_model.model.model.layers.31.self_attn.v_proj.lora_A.default.gnn_layer_stack.v_text-level_gat"

        for (name, module) in self.model.named_modules():
            if name == nuclear_layer:
                module.register_forward_hook(hook=self.hook_nuclear)
            elif name == label_layer:
                module.register_forward_hook(hook=self.hook_label)
        
        self.nuclear_norm_loss = NuclearNormLoss(weight=configs.nuclear_loss_weight)
        self.value_label_loss = nn.CrossEntropyLoss(weight=configs.v_label_loss_weight.to(device))
        self.label_loss = nn.CrossEntropyLoss(weight=configs.label_loss_weight.to(device))
        
        self.training_step = 0
        self.evaluation_step = 0
        
        self.train_batch_size = configs.train_batch_size
        self.eval_batch_size = configs.eval_batch_size
        self.gradient_accumulation_steps = configs.gradient_accumulation_steps

    
    def forward(self, labels, input_ids, attention_mask):
        logits = self.model(input_ids=input_ids, attention_mask=attention_mask)
        logits = F.softmax(logits.logits, dim=-1)
        
        if self.training:
            if self.evaluation_step != 0:
                self.evaluation_step = 0
                if self.feat_hook_nuclear_eval.size(0) != 0:
                    self.feat_hook_nuclear_eval = torch.tensor([]).to(device)
                if self.feat_hook_label_eval.size(0) != 0:
                    self.feat_hook_label_eval = torch.tensor([]).to(device)
            v_feat_nuclear = self.feat_hook_nuclear_train[self.training_step]
            v_feat_label = self.feat_hook_label_train[self.training_step]
        
        else:
            if self.training_step != 0:
                self.training_step = 0
                if self.feat_hook_nuclear_train.size(0) != 0:
                    self.feat_hook_nuclear_train = torch.tensor([]).to(device)
                if self.feat_hook_label_train.size(0) != 0:
                    self.feat_hook_label_train = torch.tensor([]).to(device)
            v_feat_nuclear = self.feat_hook_nuclear_eval[self.evaluation_step]
            v_feat_label = self.feat_hook_label_eval[self.evaluation_step]
            
        new_v_feat_label = v_feat_label[:, :, :self.model.num_labels]  # truncate
        new_v_feat_label = torch.sum(new_v_feat_label, dim=1) / new_v_feat_label.size(1)
        v_predictions = F.softmax(new_v_feat_label, dim=-1)
        
        return logits, v_feat_nuclear, v_predictions
    
    
    def compute_loss(self, logits, v_feat_nuclear, v_predictions, labels):
        
        # nuclear norm loss
        for i in range(v_feat_nuclear.size(0)):
            if i == 0:
                loss1 = self.nuclear_norm_loss(v_feat_nuclear[i])
            else:
                loss1 += self.nuclear_norm_loss(v_feat_nuclear[i])
        
        # value label loss
        loss2 = self.value_label_loss(v_predictions.view(-1, self.model.num_labels), labels.view(-1))
        
        # label loss
        loss3 = self.label_loss(logits.view(-1, self.model.num_labels), labels.view(-1))
        
        loss = loss3
        # loss = loss1 + loss3
        # loss = loss2 + loss3
        # loss = loss1 + loss2 + loss3
        
        return loss
    
    
    def hook_nuclear(self, module, feat_in, feat_out):
        if module.training:
            if self.feat_hook_nuclear_train.size(0) == 0:
                self.feat_hook_nuclear_train = feat_out.unsqueeze(0)
            else:
                self.feat_hook_nuclear_train = torch.cat((self.feat_hook_nuclear_train, feat_out.unsqueeze(0)), dim=0)
        else:
            if self.feat_hook_nuclear_eval.size(0) == 0:
                self.feat_hook_nuclear_eval = feat_out.unsqueeze(0)
            else:
                self.feat_hook_nuclear_eval = torch.cat((self.feat_hook_nuclear_eval, feat_out.unsqueeze(0)), dim=0)
        
        return None


    def hook_label(self, module, feat_in, feat_out):
        if module.training:
            if self.feat_hook_label_train.size(0) == 0:
                self.feat_hook_label_train = feat_out.unsqueeze(0)
            else:
                self.feat_hook_label_train = torch.cat((self.feat_hook_label_train, feat_out.unsqueeze(0)), dim=0)
        else:
            if self.feat_hook_label_eval.size(0) == 0:
                self.feat_hook_label_eval = feat_out.unsqueeze(0)
            else:
                self.feat_hook_label_eval = torch.cat((self.feat_hook_label_eval, feat_out.unsqueeze(0)), dim=0)
        
        return None
    
    
    def print_trainable_parameters(self, model):
        """
        Prints the number of trainable parameters in the model.
        """
        trainable_params = 0
        all_param = 0
        for _, param in model.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        print(
            f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
        )