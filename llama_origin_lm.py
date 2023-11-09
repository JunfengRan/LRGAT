import os

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import numpy as np
import torch
import torch.nn as nn
import transformers
import evaluate

from watermark import watermark
from datasets import load_dataset
from transformers import LlamaTokenizer, LlamaForCausalLM, DataCollatorForLanguageModeling, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model

from config import *


configs = Config()
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


# model
model = LlamaForCausalLM.from_pretrained(
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


class CastOutputToFloat(nn.Sequential):
    def forward(self, x):
        return super().forward(x).to(torch.float32)


model.lm_head = CastOutputToFloat(model.lm_head)


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
    
    # example['prediction'] = combined_text + '结合证据判断,该声明的标签为'
    
    if example['label'] == 0:
        example['prediction'] = combined_text + '结合证据判断,该声明是正确的,标签为' + str(example['label'])
    elif example['label'] == 1:
        example['prediction'] = combined_text + '结合证据判断,该声明是错误的,标签为' + str(example['label'])
    elif example['label'] == 2:
        example['prediction'] = combined_text + '结合证据判断,无法判断该声明的正误,标签为' + str(example['label'])
    
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
columns_to_remove = [col for col in all_columns if col != 'attention_mask' and col != 'input_ids']
dataset = dataset.remove_columns(columns_to_remove)


# training
class CustomTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


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
        output_dir='outputs/llama_origin_lm',
        evaluation_strategy="epoch",
    ),
    tokenizer=tokenizer,
    data_collator=DataCollatorForLanguageModeling(tokenizer,mlm=False),
)

model.config.use_cache = False
trainer.train()