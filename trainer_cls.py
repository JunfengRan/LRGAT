import os

os.environ['CUDA_VISIBLE_DEVICES'] = '6'

import numpy as np
import torch
import torch.nn as nn
import transformers
import evaluate

from watermark import watermark
from datasets import load_dataset
from transformers import LlamaTokenizer, LlamaForSequenceClassification, DataCollatorWithPadding, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model
from accelerate import Accelerator
from torcheval.metrics import functional as FUNC

from config import *


configs = Config()
accelerator = Accelerator()
device = accelerator.device
TORCH_SEED = configs.seed
torch.manual_seed(TORCH_SEED)
torch.cuda.manual_seed_all(TORCH_SEED)
torch.backends.cudnn.deterministic = True

print(watermark(packages='peft,torch,loralib,transformers,accelerate,datasets'))


# tokenizer
access_token = "hf_token"
if configs.llama_version == 1:
    tokenizer = LlamaTokenizer.from_pretrained(configs.llama1_cache_path, use_fast=True, use_auth_token=access_token)
elif configs.llama_version == 2:
    tokenizer = LlamaTokenizer.from_pretrained(configs.llama2_cache_path, use_fast=True, use_auth_token=access_token)
else:
    raise ValueError('llama_version must be 1 or 2')
if configs.padding_side == "left":
    tokenizer.padding_side = "left"
elif configs.padding_side == "right":
    tokenizer.padding_side = "right"
else:
    raise ValueError('padding_side must be left or right')
tokenizer.pad_token_id = tokenizer.unk_token_id
print(tokenizer)


# model
class CustomLlamaForClassification(LlamaForSequenceClassification):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = configs.num_labels
        self.score = nn.Linear(config.hidden_size, self.num_labels, bias=False)


if configs.llama_version == 1:
    llama_cache_path = configs.llama1_cache_path
elif configs.llama_version == 2:
    llama_cache_path = configs.llama2_cache_path
else:
    raise ValueError('llama_version must be 1 or 2')

model = CustomLlamaForClassification.from_pretrained(
    llama_cache_path,
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
print_trainable_parameters(model)
print(model)


# data preprocessing
data_files = {
    "train": configs.train_dataset_path,
    "test": configs.test_dataset_path,
}

# load the dataset
dataset = load_dataset('json', data_files=data_files)


def chef_merge_texts(example):
    combined_text = '请结合证据判断:下述声明是被证据支持的(标签为0),下述声明是被证据驳斥的(标签为1),或是证据提供的信息不足以进行判断(标签为2):\n'
    combined_text += '声明:' + example['claim']
    evidence_flag = 0
    
    if configs.label != 'gold_label':
        for i in range(len(example['ranksvm'])):
            if example['ranksvm'][i] != None and example['ranksvm'][i] != '' and example['ranksvm'][i] != ' ':
                
                evidence_flag = 1
                
                # set length threshold
                length_threshold = int(configs.max_seq_len / len(example['ranksvm']))
                if len(example['ranksvm'][i]) > length_threshold:
                    example['ranksvm'][i] = example['ranksvm'][i][:length_threshold]
                combined_text += '\n证据{}:'.format(i+1) + example['ranksvm'][i]
                
    else:
        for i in range(len(example['gold evidence'])):
            if example['gold evidence'][i] != None and example['gold evidence'][i] != '' and example['gold evidence'][i] != ' ':
                
                evidence_flag = 1
                
                # set length threshold
                length_threshold = int(configs.max_seq_len / len(example['gold evidence']))
                if len(example['gold evidence'][i]) > length_threshold:
                    example['gold evidence'][i] = example['gold evidence'][i][:length_threshold]
                combined_text += '\n证据{}:'.format(i+1) + example['gold evidence'][i]
                
    if evidence_flag == 0:
        combined_text += '\n证据: 未提供证据.'
    
    example['prediction'] = combined_text + '结合证据判断,标签为'
    
    # if example['label'] == 0:
    #     example['prediction'] = combined_text + '结合证据判断,该声明是被证据支持的,标签为' + str(example['label'])
    # elif example['label'] == 1:
    #     example['prediction'] = combined_text + '结合证据判断,该声明是被证据驳斥的,标签为' + str(example['label'])
    # elif example['label'] == 2:
    #     example['prediction'] = combined_text + '结合证据判断,证据提供的信息不足以进行判断,标签为' + str(example['label'])
    
    return example


def fever_merge_texts(example):
    combined_text = 'Please use the evidence to determine whether the following claim is supported by it (label 0),\
        refuted by it (label 1), or the evidence does not provide sufficient information to make a judgment (label 2):\n'
    combined_text += 'Claim: ' + example['claim']
    annotation_id = None
    evidence_flag = 0
    
    for i in range(len(example['evidence'])):
        if example['evidence']['{}'.format(i)] != None and example['evidence']['{}'.format(i)] != '' and example['evidence']['{}'.format(i)] != ' ':       
            evidence_flag = 1
            combined_text += '\nEvidence{}: '.format(i+1) + example['evidence']['{}'.format(i)]
    
    if evidence_flag == 0:
        combined_text += '\nEvidence: No evidence is provided.'
    
    example['prediction'] = combined_text + 'Judging from the evidence, the label is'
    
    # if example['label'] == 0:
    #     example['prediction'] = combined_text + 'Judging from the evidence, the claim is supported by the evidence, \
    #         so the label is' + str(example['label'])
    # elif example['label'] == 1:
    #     example['prediction'] = combined_text + 'Judging from the evidence, the claim is refuted by the evidence, \
    #         so the label is' + str(example['label'])
    # elif example['label'] == 2:
    #     example['prediction'] = combined_text + 'Judging from the evidence, the evidence does not provide sufficient information \
    #         to make a judgment, so the label is' + str(example['label'])
    
    return example


def reset_label(example):
    example['labels'] = example['label']
    
    return example

# update the dataset using the map method
if configs.dataset == 'CHEF':
    dataset['train'] = dataset['train'].map(chef_merge_texts)
    dataset['test'] = dataset['test'].map(chef_merge_texts)
elif configs.dataset == 'FEVER':
    dataset['train'] = dataset['train'].map(fever_merge_texts)
    dataset['test'] = dataset['test'].map(fever_merge_texts)

dataset['train'] = dataset['train'].map(reset_label)
dataset['test'] = dataset['test'].map(reset_label)


# tokenize
def tokenize_function(example):
    return tokenizer(example['prediction'], truncation=True, padding='max_length', max_length=configs.max_seq_len, return_tensors='pt')


dataset = dataset.map(tokenize_function, batched=True)

# remove labels for huggingface trainer
all_columns = dataset.column_names['train']
columns_to_remove = [col for col in all_columns if col != 'labels' and col != 'attention_mask' and col != 'input_ids']
dataset = dataset.remove_columns(columns_to_remove)


# training
class CustomTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


def compute_metrics(eval_preds):
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    
    
    # torcheval
    if 0 not in predictions:
        predictions = np.append(predictions, [0])
        labels = np.append(labels, [0])
    if 1 not in predictions:
        predictions = np.append(predictions, [1])
        labels = np.append(labels, [1])
    if 2 not in predictions:
        predictions = np.append(predictions, [2])
        labels = np.append(labels, [2])
    
    # transform from numpy to tensor
    predictions = torch.from_numpy(predictions)
    labels = torch.from_numpy(labels)
    
    p_micro_result = FUNC.multiclass_precision(predictions, labels, average="micro", num_classes=configs.num_labels)
    r_micro_result = FUNC.multiclass_recall(predictions, labels, average="micro", num_classes=configs.num_labels)
    f_micro_result = FUNC.multiclass_f1_score(predictions, labels, average="micro", num_classes=configs.num_labels)
    
    p_macro_result = FUNC.multiclass_precision(predictions, labels, average="macro", num_classes=configs.num_labels)
    r_macro_result = FUNC.multiclass_recall(predictions, labels, average="macro", num_classes=configs.num_labels)
    f_micro_result = FUNC.multiclass_f1_score(predictions, labels, average="macro", num_classes=configs.num_labels)
    
    
    # # evaluate
    # p_metric = evaluate.load("precision")
    # r_metric = evaluate.load('recall')
    # f_metric = evaluate.load("f1")
    
    # p_micro_result = p_metric.compute(predictions=predictions, references=labels, average="micro")
    # r_micro_result = r_metric.compute(predictions=predictions, references=labels, average="micro")
    # f_micro_result = f_metric.compute(predictions=predictions, references=labels, average="micro")
    
    # p_macro_result = p_metric.compute(predictions=predictions, references=labels, average="macro")
    # r_macro_result = r_metric.compute(predictions=predictions, references=labels, average="macro")
    # f_macro_result = f_metric.compute(predictions=predictions, references=labels, average="macro")
    
    
    # category matrix for confusion matrix analysis
    category_matrix = torch.zeros(configs.num_labels, configs.num_labels, dtype=torch.int32)
    
    for i in range(len(predictions)):
        category_matrix[labels[i]][predictions[i]] += 1
    
    result = dict()
    result['micro_precision'] = p_micro_result
    result['micro_recall'] = r_micro_result
    result['micro_f1'] = f_micro_result
    result['macro_precision'] = p_macro_result
    result['macro_recall'] = r_macro_result
    result['macro_f1'] = f_micro_result
    result['category_matrix'] = category_matrix
    
    return result


trainer = CustomTrainer(
    model=model,
    train_dataset=dataset['train'],
    eval_dataset=dataset['test'],
    args=TrainingArguments(
        num_train_epochs=configs.epochs,
        per_device_train_batch_size=configs.train_batch_size,
        per_device_eval_batch_size=configs.eval_batch_size,
        gradient_accumulation_steps=configs.gradient_accumulation_steps,
        warmup_ratio=configs.warmup_proportion,
        weight_decay=configs.weight_decay,
        learning_rate=configs.lr,
        seed=TORCH_SEED,
        data_seed=TORCH_SEED,
        fp16=True,
        output_dir='outputs/trainer_cls',
        evaluation_strategy="epoch",
    ),
    tokenizer=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer, padding='longest', max_length=configs.max_seq_len),
    compute_metrics=compute_metrics,
)

model.config.use_cache = False
trainer.train()