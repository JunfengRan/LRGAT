import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import LlamaForSequenceClassification
from peft import LoraConfig, get_peft_model
from accelerate import Accelerator

from config import *


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
        
        self.print_trainable_parameters(self.model)
        print(self.model)

        self.label_loss = nn.CrossEntropyLoss(weight=configs.label_loss_weight.to(device))

    
    def forward(self, labels, input_ids, attention_mask):
        logits = self.model(input_ids=input_ids, attention_mask=attention_mask)
        logits = F.softmax(logits.logits, dim=-1)
        
        return logits
    
    
    def compute_loss(self, logits, labels):
        
        # label loss
        loss = self.label_loss(logits.view(-1, self.model.num_labels), labels.view(-1))
        
        return loss
    
    
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