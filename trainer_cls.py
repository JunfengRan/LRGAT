import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
import evaluate

from watermark import watermark
from datasets import load_dataset
from transformers import LlamaTokenizer, LlamaForSequenceClassification, DataCollatorWithPadding, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model
from accelerate import Accelerator

from config import *
from gnn_layer import *


configs = Config()
accelerator = Accelerator()
device = accelerator.device
TORCH_SEED = configs.seed
torch.manual_seed(TORCH_SEED)
torch.cuda.manual_seed_all(TORCH_SEED)
torch.backends.cudnn.deterministic = True

print(watermark(packages='peft,torch,loralib,transformers,accelerate,datasets'))


# tokenizer
tokenizer = LlamaTokenizer.from_pretrained(configs.llama_cache_path)
# tokenizer.add_special_tokens({'pad_token': '[PAD]'})
tokenizer.pad_token_id = 0
print(tokenizer)


class CustomLlamaForClassification(LlamaForSequenceClassification):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = configs.num_labels
        self.score = nn.Linear(config.hidden_size, self.num_labels, bias=False)


model = CustomLlamaForClassification.from_pretrained(
    configs.llama_cache_path,
    load_in_8bit=True,
    device_map='auto',
)


# freeze original weights
# list(model.parameters())[0].dtype

for i, param in enumerate(model.parameters()):
    param.requires_grad = False  # freeze the model - train adapters later
    # print(i, 'param.requires_grad = False')
    if param.ndim == 1:
        # cast the small parameters (e.g. layernorm) to fp32 for stability
        param.data = param.data.to(torch.float32)
        # print(i, 'ndim == 1, torch.float16 to torch.float32')

# reduce number of stored activations
model.gradient_checkpointing_enable()
model.enable_input_require_grads()

model.score = nn.Linear(configs.feat_dim, configs.num_labels, bias=False).to(device)


# LoRa Adapters
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
    r=configs.lora_r,  # low rank
    lora_alpha=configs.lora_alpha,  # alpha scaling, scale lora weights/outputs
    # target_modules=["q_proj", "v_proj"], #if you know
    lora_dropout=configs.lora_dp,
    bias="none",
    task_type="CAUSAL_LM"  # set this for CLM or Seq2Seq
)


model = get_peft_model(model, lora_config)
model.base_model.model.model.layers[31].self_attn.v_proj.lora_A.default = CustomGAT(configs, name="v_custom_gat").to(device)
print_trainable_parameters(model)
print(model)


# data preprocessing
data_files = {
    "train": configs.train_dataset_path,
    "test": configs.test_dataset_path,
}

# load the dataset
dataset = load_dataset('json', data_files=data_files)


def merge_texts(example):
    combined_text = '请结合证据判断如下声明是正确的(标签为0),错误的(标签为1),还是无法判断其正误(标签为2):\n'
    combined_text += '声明:' + example['claim']
    
    if configs.label != 'gold_label':
        for i in range(len(example['ranksvm'])):
            if example['ranksvm'][i] != None and example['ranksvm'][i] != '':
                
                # set length threshold
                length_threshold = int(configs.max_seq_len / len(example['ranksvm']))
                if len(example['ranksvm'][i]) > length_threshold:
                    example['ranksvm'][i] = example['ranksvm'][i][:length_threshold]
                combined_text += '\n证据{}:'.format(i+1) + example['ranksvm'][i]
                
    else:
        for i in range(len(example['gold evidence'])):
            if example['gold evidence'][i] != None and example['gold evidence'][i] != '':
                
                # set length threshold
                length_threshold = int(configs.max_seq_len / len(example['gold evidence']))
                if len(example['gold evidence'][i]) > length_threshold:
                    example['gold evidence'][i] = example['gold evidence'][i][:length_threshold]
                combined_text += '\n证据{}:'.format(i+1) + example['gold evidence'][i]
    
    example['prediction'] = combined_text + '结合证据判断,该声明的标签为'
    
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


# tokenize
def tokenize_function(example):
    return tokenizer(example['prediction'], truncation=True, padding='max_length', max_length=configs.max_seq_len)


dataset = dataset.map(tokenize_function, batched=True)

# remove labels for huggingface trainer
all_columns = dataset.column_names['train']
columns_to_remove = [col for col in all_columns if col != 'labels' and col != 'attention_mask' and col != 'input_ids']
dataset = dataset.remove_columns(columns_to_remove)


# use hook to obtain labels
feat_hook_nuclear_train = []
feat_hook_nuclear_eval = []
feat_hook_label_train = []
feat_hook_label_eval = []

# utilize hook
def hook_nuclear(module, feat_in, feat_out):
    if module.training:
        feat_hook_nuclear_train.append(feat_out.data)
    else:
        feat_hook_nuclear_eval.append(feat_out.data)
    return None

def hook_label(module, feat_in, feat_out):
    if module.training:
        feat_hook_label_train.append(feat_out.data)
    else:
        feat_hook_label_eval.append(feat_out.data)
    return None


nuclear_layer = "base_model.model.model.layers.31.self_attn.v_proj.lora_A.default.gnn_layer_stack.v_sentence-level_gat"
label_layer = "base_model.model.model.layers.31.self_attn.v_proj.lora_A.default.gnn_layer_stack.v_text-level_gat"

for (name, module) in model.named_modules():
    if name == nuclear_layer:
        module.register_forward_hook(hook=hook_nuclear)
    elif name == label_layer:
        module.register_forward_hook(hook=hook_label)
        
# category matrix for confusion matrix analysis
v_category_matrix = torch.zeros(configs.num_labels, configs.num_labels, dtype=torch.int32)
category_matrix = torch.zeros(configs.num_labels, configs.num_labels, dtype=torch.int32)


# training

eval_step = 0

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


class CustomTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.nuclear_norm_loss = NuclearNormLoss(weight=configs.nuclear_loss_weight)
        self.value_label_loss = nn.CrossEntropyLoss()
        self.label_loss = nn.CrossEntropyLoss()
        self.gradient_accumulation_steps = configs.gradient_accumulation_steps
        self.prev_global_step = -1
        self.ga_step = 0
        self.train_batch_size = configs.train_batch_size
        self.eval_batch_size = configs.eval_batch_size
        self.is_training = False
    
    def compute_loss(self, model, inputs, return_outputs=False):
        # forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")
        labels = inputs.pop("labels")
        global feat_hook_nuclear_train, feat_hook_label_train, feat_hook_nuclear_eval, feat_hook_label_eval, eval_step
        
        if self.is_training == False and self.state.global_step != self.prev_global_step:
            self.is_training = True
            self.prev_global_step = -1
            feat_hook_nuclear_eval = []
            feat_hook_label_eval = []
        
        if self.is_training == True and self.state.global_step == self.prev_global_step and self.ga_step == 0:
            self.is_training = False
            eval_step = 0
            feat_hook_nuclear_train = []
            feat_hook_label_train = []
        
        if self.is_training:
            print('training step {}, gradient accumulation step {}'.format(self.state.global_step, self.ga_step))
            v_feat_nuclear = feat_hook_nuclear_train \
                [self.state.global_step * self.train_batch_size * self.gradient_accumulation_steps \
                    + self.ga_step * self.gradient_accumulation_steps]
            v_feat_label = feat_hook_label_train \
                [self.state.global_step * self.train_batch_size * self.gradient_accumulation_steps \
                    + self.ga_step * self.gradient_accumulation_steps]
            
            self.prev_global_step = self.state.global_step
            self.ga_step = (self.ga_step + 1) % self.gradient_accumulation_steps
        
        if not self.is_training:
            print('testing step {}'.format(eval_step))
            v_feat_nuclear = feat_hook_nuclear_eval[eval_step]
            v_feat_label = feat_hook_label_eval[eval_step]
            eval_step += 1
        
        # nuclear norm loss
        for i in range(v_feat_nuclear.size(0)):
            if i == 0:
                loss1 = self.nuclear_norm_loss(v_feat_nuclear[i])
            else:
                loss1 += self.nuclear_norm_loss(v_feat_nuclear[i])
        
        # value label loss
        new_v_feat_label = v_feat_label[:, :, :self.model.num_labels]  # truncate
        new_v_feat_label = torch.sum(new_v_feat_label, dim=1) / new_v_feat_label.size(1)
        v_predictions = F.softmax(new_v_feat_label, dim=-1)
        loss2 = self.value_label_loss(new_v_feat_label.view(-1, self.model.num_labels), labels.view(-1))
        
        # label loss
        predictions = F.softmax(logits, dim=-1)
        loss3 = self.label_loss(logits.view(-1, self.model.num_labels), labels.view(-1))
        
        # update category matrix
        for i in range(len(labels)):
            v_category_matrix[labels[i].long().cpu()][v_predictions[i].long().cpu()] += 1
            category_matrix[labels[i].long().cpu()][predictions[i].long().cpu()] += 1
        
        loss = loss1 + loss2 + loss3
        
        return (loss, outputs) if return_outputs else loss


def compute_metrics(eval_preds):
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    
    for step in range(eval_step):
        print("evluating step {} of {}".format(step, eval_step))
        v_feat_label = feat_hook_label_eval[step]
        new_v_feat_label = v_feat_label[:, :, :configs.num_labels]  # truncate
        new_v_feat_label = torch.sum(new_v_feat_label, dim=1) / new_v_feat_label.size(1)
        if step == 0:
            v_predictions = torch.argmax(new_v_feat_label, dim=-1)
        else:
            v_predictions = torch.cat((v_predictions, torch.argmax(new_v_feat_label, dim=-1)), dim=0)
    
    v_p_metric_micro = evaluate.load("precision")
    v_r_metric_micro = evaluate.load('recall')
    v_f_metric_micro = evaluate.load("f1")
    
    v_p_metric_macro = evaluate.load("precision")
    v_r_metric_macro = evaluate.load('recall')
    v_f_metric_macro = evaluate.load("f1")
    
    p_metric_micro = evaluate.load("precision")
    r_metric_micro = evaluate.load('recall')
    f_metric_micro = evaluate.load("f1")
    
    p_metric_macro = evaluate.load("precision")
    r_metric_macro = evaluate.load('recall')
    f_metric_macro = evaluate.load("f1")
    
    result = dict()
    
    result.update(v_p_metric_micro.compute(predictions=v_predictions, references=labels, average="micro"))
    result.update(v_r_metric_micro.compute(predictions=v_predictions, references=labels, average="micro"))
    result.update(v_f_metric_micro.compute(predictions=v_predictions, references=labels, average="micro"))
    
    result.update(v_p_metric_macro.compute(predictions=v_predictions, references=labels, average="macro"))
    result.update(v_r_metric_macro.compute(predictions=v_predictions, references=labels, average="macro"))
    result.update(v_f_metric_macro.compute(predictions=v_predictions, references=labels, average="macro"))
    
    result.update(p_metric_micro.compute(predictions=predictions, references=labels, average="micro"))
    result.update(r_metric_micro.compute(predictions=predictions, references=labels, average="micro"))
    result.update(f_metric_micro.compute(predictions=predictions, references=labels, average="micro"))
    
    result.update(p_metric_macro.compute(predictions=predictions, references=labels, average="macro"))
    result.update(r_metric_macro.compute(predictions=predictions, references=labels, average="macro"))
    result.update(f_metric_macro.compute(predictions=predictions, references=labels, average="macro"))
    
    return result


trainer = CustomTrainer(
    model=model,
    train_dataset=dataset['train'],
    eval_dataset=dataset['test'],
    args=TrainingArguments(
        num_train_epochs=configs.num_labels,
        per_device_train_batch_size=configs.train_batch_size,
        per_device_eval_batch_size=configs.eval_batch_size,
        gradient_accumulation_steps=configs.gradient_accumulation_steps,
        warmup_ratio=configs.warmup_proportion,
        weight_decay=configs.weight_decay,
        learning_rate=configs.lr,
        seed=TORCH_SEED,
        fp16=True,
        output_dir='outputs/trainer_cls_trial',
        evaluation_strategy="epoch",
    ),
    tokenizer=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer, padding='longest', max_length=configs.max_seq_len),
    compute_metrics=compute_metrics,
)

model.config.use_cache = False
trainer.train()
print("v_category_matrix:", v_category_matrix)
print("category_matrix:", category_matrix)