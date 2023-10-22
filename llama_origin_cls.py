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


model = get_peft_model(model, lora_config)
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



# ### training

def compute_metrics(eval_preds):
    metrics = evaluate.load("f1")
    logits, labels = eval_preds
    predictions = torch.argmax(logits, dim=-1)
    return metrics.compute(predictions=predictions, references=labels, average="micro")


trainer = Trainer(
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



# # ### inference
#
# batch = tokenizer("“Training models with PEFT and LoRa is cool” ->: ", return_tensors='pt')
#
# with torch.cuda.amp.autocast():
#     output_tokens = model.generate(**batch, max_new_tokens=50)
#
# print('\n\n', tokenizer.decode(output_tokens[0], skip_special_tokens=True))
#
# batch = tokenizer(
#     "“An important paradigm of natural language processing consists of large-scale pre-training on general domain data and adaptation to particular tasks or domains.” ->: ",
#     return_tensors='pt')
#
# with torch.cuda.amp.autocast():
#     output_tokens = model.generate(**batch, max_new_tokens=50)
#
# print('\n\n', tokenizer.decode(output_tokens[0], skip_special_tokens=True))
#
# trainer.data_collator