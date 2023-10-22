#!/usr/bin/env python
# coding: utf-8

# get_ipython().system('pip install -q bitsandbytes datasets accelerate loralib')
# get_ipython().system('pip install -q git+https://github.com/huggingface/transformers.git@main ')
# get_ipython().system('pip install -q git+https://github.com/huggingface/peft.git')



# ## summary

# - decapoda-research/llama-7b-hf
# - lora fine-tune bloom: plugin/adapter
#     - freeeze original weights
#     - plugin lora adapters (peft)
# - huggingface transformers
#     - trainer.train
#         - mlm: bert
#         - clm: gpt (bloom)
#     - pipeline
#         - dataset/tasks
#         - tokenizer
#         - training (fine-tune base lora)
#         - inference



import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'  # Use GPU 0,1,2,3

import torch
import torch.nn as nn
import bitsandbytes as bnb
import transformers
from transformers import LlamaTokenizer, LlamaConfig, LlamaForSequenceClassification, DataCollator, \
    DataCollatorWithPadding, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model
from watermark import watermark
from datasets import load_dataset
from config import *
from gnn_layer import GraphAttentionLayer


configs = Config()
TORCH_SEED = configs.seed
torch.manual_seed(TORCH_SEED)
torch.cuda.manual_seed_all(TORCH_SEED)
torch.backends.cudnn.deterministic = True


print(watermark(packages='peft,torch,loralib,transformers,accelerate,datasets'))

cache_dir = "./LLAMA_local/decapoda-research/llama-7b-hf/"


# config of llama
llama_config = LlamaConfig.from_pretrained(cache_dir)
llama_config.pad_token_id = 0

# tokenizer of llama
tokenizer = LlamaTokenizer.from_pretrained(cache_dir)
# tokenizer.add_special_tokens({'pad_token': '[PAD]'})
tokenizer.pad_token_id = 0
print(tokenizer)


class CustomLlamaForClassification(LlamaForSequenceClassification):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = configs.num_labels
        self.score = nn.Linear(config.hidden_size, self.num_labels, bias=False)


model = CustomLlamaForClassification.from_pretrained(
    cache_dir,
    config=llama_config,
    load_in_8bit=True,
    device_map='auto',
)



# ### freeze original weights

# list(model.parameters())[0].dtype

for i, param in enumerate(model.parameters()):
    param.requires_grad = False  # freeze the model - train adapters later
    #     print(i, 'param.requires_grad = False')
    if param.ndim == 1:
        # cast the small parameters (e.g. layernorm) to fp32 for stability
        param.data = param.data.to(torch.float32)
#         print(i, 'ndim == 1, torch.float16 to torch.float32')

# reduce number of stored activations
model.gradient_checkpointing_enable()
model.enable_input_require_grads()



# ### LoRa Adapters

def print_trainable_parameters(model):
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


lora_config = LoraConfig(
    r=4,  # low rank
    lora_alpha=8,  # alpha scaling， scale lora weights/outputs
    # target_modules=["q_proj", "v_proj"], #if you know the 
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"  # set this for CLM or Seq2Seq
)



# ### layer settings

class CustomGAT(nn.Module):
    def __init__(self, configs, name):
        super(CustomGAT, self).__init__()
        self.name = name
        in_dim = configs.feat_dim
        self.gnn_dims = [in_dim] + [int(dim) for dim in configs.gnn_dims.strip().split(',')]

        self.gnn_layers = len(self.gnn_dims) - 1
        self.att_heads = [int(att_head) for att_head in configs.att_heads.strip().split(',')]
        
        if name == "v_custom_gat":
            self.name_list = configs.name_list_v_gat.strip().split(',')
        
        self.gnn_layer_stack = nn.ModuleDict()
        for i in range(self.gnn_layers):
            in_dim = self.gnn_dims[i] * self.att_heads[i - 1] if i != 0 else self.gnn_dims[i]
            self.gnn_layer_stack.update({
                "{}".format(self.name_list[i]): GraphAttentionLayer(self.att_heads[i], in_dim, self.gnn_dims[i + 1], configs.dp, name=self.name_list[i])
            })

    def forward(self, feat_in, adj=None):
        for i, gnn_layer in enumerate(self.gnn_layer_stack):
            feat_in = gnn_layer(feat_in, adj)
        return feat_in


model = get_peft_model(model, lora_config)

model.base_model.model.model.layers[31].self_attn.v_proj.lora_A = nn.ModuleDict({
    "v_custom_gat": CustomGAT(configs, "v_custom_gat"),
})

print_trainable_parameters(model)
print(model)



# ### pipeline

data_files = {
    "train": "data/CHEF_train_modified.json",
    "test": "data/CHEF_test_modified.json"
}

# load the dataset
dataset = load_dataset('json', data_files=data_files)


def merge_texts(example):
    combined_text = '请结合证据判断如下声明是正确的(标签为0),错误的(标签为1),还是无法判断其正误(标签为2):\n'
    combined_text += '声明:' + example['claim']
    for i in range(len(example['ranksvm'])):
        if example['ranksvm'][i] != None:
            combined_text += '\n证据{}:'.format(i+1) + example['ranksvm'][i]
    
    example['prediction'] = combined_text
    
    # if example['label'] == 0:
    #     example['prediction'] = combined_text + '结合证据判断,该声明是正确的,标签为' + str(example['label'])
    # elif example['label'] == 1:
    #     example['prediction'] = combined_text + '结合证据判断,该声明是错误的,标签为' + str(example['label'])
    # elif example['label'] == 2:
    #     example['prediction'] = combined_text + '结合证据判断,无法判断该声明的正误,标签为' + str(example['label'])
    
    return example


def reset_label(example):
    example['labels'] = example['label']
    return example


# update the dataset using the map method
dataset['train'] = dataset['train'].map(merge_texts)
dataset['test'] = dataset['test'].map(merge_texts)
dataset['train'] = dataset['train'].map(reset_label)
dataset['test'] = dataset['test'].map(reset_label)



# ### tokenize

def tokenize_function(example):
    return tokenizer(example['prediction'])


dataset = dataset.map(tokenize_function, batched=True)
all_columns = dataset.column_names['train']
columns_to_remove = [col for col in all_columns if col != 'input_ids' and col != 'attention_mask' and col != 'labels']
dataset = dataset.remove_columns(columns_to_remove)



# ### data process

# use hook to obtain labels
feat_hook_nuclear = []
feat_hook_label = []

# utilize hook
def hook_nuclear(module, feat_in, feat_out):
    feat_hook_nuclear.append(feat_out.data)  # output hook
    return None

def hook_label(module, feat_in, feat_out):
    feat_hook_label.append(feat_out.data)  # output hook
    return None


nuclear_layer = "base_model.model.model.layers.31.self_attn.v_proj.lora_A.v_custom_gat.gnn_layer_stack.v_sentence-level_gat"
label_layer = "base_model.model.model.layers.31.self_attn.v_proj.lora_A.v_custom_gat.gnn_layer_stack.v_text-level_gat"

for (name, module) in model.named_modules():
    if name == nuclear_layer:
        module.register_forward_hook(hook=hook_nuclear)
    elif name == label_layer:
        module.register_forward_hook(hook=hook_label)
        
# category matrix for confusion matrix analysis
v_category_matrix = torch.zeros(3, 3, dtype=torch.int32)
category_matrix = torch.zeros(3, 3, dtype=torch.int32)



# ### training

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


# Define custom loss function for low rank label regularization
class CustomTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.nuclear_norm_loss = NuclearNormLoss(weight=0.01)
        self.value_label_loss = nn.CrossEntropyLoss()
        self.label_loss = nn.CrossEntropyLoss()
    
    def compute_loss(self, model, inputs, return_outputs=False):
        # forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")
        labels = inputs.pop("labels")
        
        step = self.state.global_step
        v_feat_nuclear = feat_hook_nuclear[step]
        v_feat_label = feat_hook_label[step]
        
        # nuclear norm loss
        loss1 = 0
        for i in range(v_feat_nuclear.size(0)):
            loss1 += self.nuclear_norm_loss(v_feat_nuclear[i])
        
        # value label loss
        new_v_feat_label = v_feat_label[:, :, :self.model.num_labels]  # truncate
        v_predictions = torch.argmax(new_v_feat_label, dim=-1)
        loss2 = self.value_label_loss(new_v_feat_label.view(-1, self.model.num_labels), labels.view(-1))
        
        # label loss
        predictions = torch.argmax(logits, dim=-1)
        loss3 = self.label_loss(logits.view(-1, self.model.num_labels), labels.view(-1))
        
        # update category matrix
        for i in range(len(labels)):
            v_category_matrix[labels[i]][v_predictions[i]] += 1
            category_matrix[labels[i]][predictions[i]] += 1
        
        loss = loss1 + loss2 + loss3
        return (loss, outputs) if return_outputs else loss


def compute_metrics(eval_preds):
    step = self.state.global_step
    v_feat_label = feat_hook_label[step]
    new_v_feat_label = new_v_feat_label[:, :, :self.model.num_labels]  # truncate
    v_predictions = torch.argmax(new_v_feat_label, dim=-1)
    
    logits, labels = eval_preds
    predictions = torch.argmax(logits, dim=-1)
    
    p_metric = evaluate.load("precision")
    r_metric = evaluate.load('recall')
    f_metric = evaluate.load("f1")
    
    result = dict()
    
    result.update(p_metric.compute(predictions=v_predictions, references=labels, average="micro"))
    result.update(r_metric.compute(predictions=v_predictions, references=labels, average="micro"))
    result.update(f_metric.compute(predictions=v_predictions, references=labels, average="micro"))
    result.update(p_metric.compute(predictions=v_predictions, references=labels, average="macro"))
    result.update(r_metric.compute(predictions=v_predictions, references=labels, average="macro"))
    result.update(f_metric.compute(predictions=v_predictions, references=labels, average="macro"))
    
    result.update(p_metric.compute(predictions=predictions, references=labels, average="micro"))
    result.update(r_metric.compute(predictions=predictions, references=labels, average="micro"))
    result.update(f_metric.compute(predictions=predictions, references=labels, average="micro"))
    result.update(p_metric.compute(predictions=predictions, references=labels, average="macro"))
    result.update(r_metric.compute(predictions=predictions, references=labels, average="macro"))
    result.update(f_metric.compute(predictions=predictions, references=labels, average="macro"))
    
    return result


trainer = CustomTrainer(
    model=model,
    train_dataset=dataset['train'],
    eval_dataset=dataset['test'],
    args=TrainingArguments(
        num_train_epochs=5,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=2,
        warmup_ratio=0.1,
        weight_decay=0.01,
        learning_rate=5e-5,
        seed=TORCH_SEED,
        fp16=True,
        output_dir='outputs',
        logging_steps=10,
        evaluation_strategy="epoch",
    ),
    tokenizer=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer),
    compute_metrics=compute_metrics,
)

model.config.use_cache = False
trainer.train()
print("v_category_matrix:", v_category_matrix)
print("category_matrix:", category_matrix)