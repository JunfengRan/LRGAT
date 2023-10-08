import os

os.environ['CUDA_VISIBLE_DEVICES'] = '4,5'  # Use GPU 4,5


import torch
import torch.nn as nn
import transformers
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import DataCollator, DataCollatorWithPadding
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
import numpy as np
from torcheval.metrics import functional as FUNC


TORCH_SEED = 129
torch.manual_seed(TORCH_SEED)
torch.cuda.manual_seed_all(TORCH_SEED)
torch.backends.cudnn.deterministic = True


model = AutoModelForSequenceClassification.from_pretrained("bert-base-chinese", num_labels=3)

tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")

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
    example['prediction'] = combined_text
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
    return tokenizer(example['prediction'], padding='max_length', truncation=True, max_length=512)


dataset = dataset.map(tokenize_function, batched=True)
dataset = dataset.remove_columns('prediction')



# ### training

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return FUNC.multiclass_f1_score(predictions, labels, average="micro", num_classes=3)


trainer = Trainer(
    model=model,
    train_dataset=dataset['train'],
    eval_dataset=dataset['test'],
    args=TrainingArguments(
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        warmup_steps=100,
        max_steps=200,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=1,
        output_dir="test_trainer",
        evaluation_strategy="epoch"
    ),
    compute_metrics=compute_metrics,
    data_collator=DataCollatorForLanguageModeling(tokenizer,mlm=False)
)

model.config.use_cache = False
trainer.train()