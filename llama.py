#!/usr/bin/env python
# coding: utf-8

# get_ipython().system('pip install -q bitsandbytes datasets accelerate loralib')
# get_ipython().system('pip install -q git+https://github.com/huggingface/transformers.git@main ')
# get_ipython().system('pip install -q git+https://github.com/huggingface/peft.git')


# ## summary

# - decapoda-research/llama-7b-hf
# - lora fine-tune bloom: 可插拔式的（plugin/adapter）
#     - freeeze original weights
#     - plugin lora adapters (peft)
# - huggingface transformers 库
#     - trainer.train 的参数及过程；
#     - mlm 与 clm 的差异：（都是 unsupervised learning，都可以自动地构建 input/labels）
#         - mlm：bert
#         - clm：gpt（bloom）
#     - pipeline
#         - dataset/tasks
#         - tokenizer
#         - training (fine-tune base lora)
#         - inference

# ## base model & lora adapters



import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'  # Use GPU 0,1,2,3


import torch
import torch.nn as nn
import bitsandbytes as bnb
from transformers import LlamaTokenizer, LlamaConfig, LlamaForCausalLM, DataCollator, DataCollatorWithPadding
from peft import LoraConfig, get_peft_model
from watermark import watermark
import transformers
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling


TORCH_SEED = 129
torch.manual_seed(TORCH_SEED)
torch.cuda.manual_seed_all(TORCH_SEED)
torch.backends.cudnn.deterministic = True


print(watermark(packages='peft,torch,loralib,transformers,accelerate,datasets'))

cache_dir = "./LLAMA_local/decapoda-research/llama-7b-hf/"

model = LlamaForCausalLM.from_pretrained(
    cache_dir,
    load_in_8bit=True,
    device_map='auto',
)

tokenizer = LlamaTokenizer.from_pretrained(cache_dir)
tokenizer.add_special_tokens({'pad_token': '[PAD]'})

# model.config
LlamaConfig.from_pretrained(cache_dir)

print(tokenizer)



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


class CastOutputToFloat(nn.Sequential):
    def forward(self, x):
        return super().forward(x).to(torch.float32)


model.lm_head = CastOutputToFloat(model.lm_head)



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


config = LoraConfig(
    r=16,  # low rank
    lora_alpha=32,  # alpha scaling， scale lora weights/outputs
    # target_modules=["q_proj", "v_proj"], #if you know the 
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"  # set this for CLM or Seq2Seq
)


model = get_peft_model(model, config)
print_trainable_parameters(model)

print(model)



# ## pipeline

data_files = {
    "train": "data/CHEF_train_modified.json",
    "test": "data/CHEF_test_modified.json"
}

# 加载数据集
dataset = load_dataset('json', data_files=data_files)


def merge_texts(example):
    # 获取claim
    combined_text = 'claim:' + example['claim']
    # 获取evidence里的text并拼接
    for sentence in example['ranksvm']:
        combined_text += '\nevidence:' + sentence
    # 更新input_text字段
    example['prediction'] = combined_text + ' ->: ' + 'label:' + str(example['label'])
    # 更新target_label字段

    return example


# 使用map方法更新数据集
dataset['train'] = dataset['train'].map(merge_texts)
dataset['test'] = dataset['test'].map(merge_texts)



# ### tokenize

# 获取所有列的名称
all_columns = dataset.column_names['train']
# 移除'prediction'，留下其他列的名称
columns_to_remove = [col for col in all_columns if col != 'prediction']
# 使用remove_columns方法移除其他列
dataset = dataset.remove_columns(columns_to_remove)


def tokenize_function(example):
    return tokenizer(example['prediction'], truncation=True, padding='max_length', max_length=2048)


dataset = dataset.map(tokenize_function, batched=True)
dataset = dataset.remove_columns('prediction')



# ### training

trainer = Trainer(
    model=model,
    train_dataset=dataset['train'],
    args=TrainingArguments(
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        warmup_steps=100,
        max_steps=200,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=1,
        output_dir='outputs'
    ),
    data_collator=DataCollatorForLanguageModeling(tokenizer,mlm=False)
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