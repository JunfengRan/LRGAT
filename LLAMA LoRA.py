#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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

# In[1]:


import torch
import torch.nn as nn
import bitsandbytes as bnb 
from transformers import LlamaTokenizer, LlamaConfig, LlamaForCausalLM
from peft import LoraConfig, get_peft_model 


# In[3]:


# get_ipython().run_line_magic('load_ext', 'watermark')


# In[4]:


# get_ipython().run_line_magic('watermark', '--iversions')


# In[9]:


from watermark import watermark
print(watermark(packages='peft,torch,loralib,transformers,accelerate,datasets'))


# In[ ]:


model = LlamaForCausalLM.from_pretrained(
    "decapoda-research/llama-7b-hf", 
    load_in_8bit=True, 
    device_map='auto',
)

tokenizer = LlamaTokenizer.from_pretrained("decapoda-research/llama-7b-hf")
# tokenizer.pad_token = tokenizer.

# In[ ]:


# model.config
LlamaConfig.from_pretrained("decapoda-research/llama-7b-hf")


# In[ ]:


# print(model)


# In[ ]:


# model.transformer.word_embeddings
# model.get_input_embeddings()


# In[ ]:


print(tokenizer)


# ### freeze original weights

# In[ ]:


# list(model.parameters())[0].dtype


# In[ ]:


for i, param in enumerate(model.parameters()):
    param.requires_grad = False  # freeze the model - train adapters later
#     print(i, 'param.requires_grad = False')
    if param.ndim == 1:
        # cast the small parameters (e.g. layernorm) to fp32 for stability
        param.data = param.data.to(torch.float32)
#         print(i, 'ndim == 1, torch.float16 to torch.float32')


# In[ ]:


# reduce number of stored activations
model.gradient_checkpointing_enable()  
model.enable_input_require_grads()


# In[ ]:


class CastOutputToFloat(nn.Sequential):
    def forward(self, x): 
        return super().forward(x).to(torch.float32)
model.lm_head = CastOutputToFloat(model.lm_head)


# ### LoRa Adapters

# In[ ]:


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


# In[ ]:


from peft import LoraConfig, get_peft_model 
config = LoraConfig(
    r=16, #low rank
    lora_alpha=32, #alpha scaling， scale lora weights/outputs
    # target_modules=["q_proj", "v_proj"], #if you know the 
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM" # set this for CLM or Seq2Seq
)


# In[ ]:


model = get_peft_model(model, config)
print_trainable_parameters(model)


# In[ ]:


print(model)


# ## pipeline

# ### data

# In[ ]:


import transformers
from datasets import load_dataset
dataset = load_dataset("Abirate/english_quotes")


# In[ ]:


print(dataset)


# In[ ]:


dataset['train']


# In[ ]:


dataset['train'].to_pandas()


# In[ ]:


dataset['train']['quote'][:4]


# In[ ]:


dataset['train']['author'][:4]


# In[ ]:


dataset['train'][:4]


# In[ ]:


str(dataset['train']['tags'][0])


# In[ ]:


def merge(row):
    row['prediction'] = row['quote'] + ' ->: ' + str(row['tags'])
    return row
dataset['train'] = dataset['train'].map(merge)


# In[ ]:


dataset['train']['prediction'][:5]


# In[ ]:


dataset['train'][4]


# In[ ]:


tokenizer(dataset['train']['prediction'][:4])


# ### tokenize

# In[ ]:


dataset = dataset.map(lambda samples: tokenizer(samples['prediction']), batched=True)


# In[ ]:


# 'input_ids', 'attention_mask'
print(dataset)


# ### training

# In[ ]:


from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling


# In[ ]:


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
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
)
model.config.use_cache = False  
trainer.train()


# ### inference

# In[ ]:


batch = tokenizer("“Training models with PEFT and LoRa is cool” ->: ", return_tensors='pt')

with torch.cuda.amp.autocast():
    output_tokens = model.generate(**batch, max_new_tokens=50)

print('\n\n', tokenizer.decode(output_tokens[0], skip_special_tokens=True))


# In[ ]:



batch = tokenizer("“An important paradigm of natural language processing consists of large-scale pre-training on general domain data and adaptation to particular tasks or domains.” ->: ", return_tensors='pt')

with torch.cuda.amp.autocast():
   output_tokens = model.generate(**batch, max_new_tokens=50)

print('\n\n', tokenizer.decode(output_tokens[0], skip_special_tokens=True))


# In[ ]:


trainer.data_collator

