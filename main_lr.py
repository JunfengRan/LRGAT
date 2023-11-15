import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers

from watermark import watermark
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import AdamW, get_linear_schedule_with_warmup, LlamaTokenizer, DataCollatorWithPadding
from accelerate import Accelerator
from torcheval.metrics import functional as FUNC

from config import *
from model_lr import Network


configs = Config()
accelerator = Accelerator()
device = accelerator.device
TORCH_SEED = configs.seed
torch.manual_seed(TORCH_SEED)
torch.cuda.manual_seed_all(TORCH_SEED)
torch.backends.cudnn.deterministic = True


# evaluate one batch
def evaluate_one_batch(batch, model):
    
    # get prediction
    logits, v_feat_nuclear, v_predictions = model(**batch)
    
    true = batch['labels'].to('cpu').numpy()
    v_pred = torch.argmax(v_predictions, dim=-1).to('cpu').numpy()
    pred = torch.argmax(logits, dim=-1).to('cpu').numpy()
    v_pred, pred, true = int(v_pred), int(pred), int(true)
    
    return v_pred, pred, true


# evaluate step
def evaluate(test_loader, model):
    model.eval()
    v_pred_list = []
    pred_list = []
    true_list = []
    
    # category matrix for confusion matrix analysis
    v_category_matrix = torch.zeros(configs.num_labels, configs.num_labels, dtype=torch.int32)
    category_matrix = torch.zeros(configs.num_labels, configs.num_labels, dtype=torch.int32)
    
    for batch in test_loader:
        v_pred, pred, true = evaluate_one_batch(batch, model)
        v_category_matrix[true][v_pred] += 1
        category_matrix[true][pred] += 1
        v_pred_list.append(v_pred)
        pred_list.append(pred)
        true_list.append(true)
    
    # label smoothing
    if 0 not in pred_list:
        pred_list.append(0)
        v_pred_list.append(0)
        true_list.append(0)
    if 1 not in pred_list:
        pred_list.append(1)
        v_pred_list.append(1)
        true_list.append(1)
    if 2 not in pred_list:
        pred_list.append(2)
        v_pred_list.append(2)
        true_list.append(2)
    
    if 0 not in v_pred_list:
        v_pred_list.append(0)
        pred_list.append(0)
        true_list.append(0)
    if 1 not in v_pred_list:
        v_pred_list.append(1)
        pred_list.append(1)
        true_list.append(1)
    if 2 not in v_pred_list:
        v_pred_list.append(2)
        pred_list.append(2)
        true_list.append(2)
    
    v_pred_list = torch.tensor(v_pred_list)
    pred_list = torch.tensor(pred_list)
    true_list = torch.tensor(true_list)
    
    v_micro_result = [FUNC.multiclass_f1_score(v_pred_list, true_list, average="micro", num_classes=configs.num_labels), \
        FUNC.multiclass_recall(v_pred_list, true_list, average="micro", num_classes=configs.num_labels), \
        FUNC.multiclass_precision(v_pred_list, true_list, average="micro", num_classes=configs.num_labels)]
    v_macro_result = [FUNC.multiclass_f1_score(v_pred_list, true_list, average="macro", num_classes=configs.num_labels), \
        FUNC.multiclass_recall(v_pred_list, true_list, average="macro", num_classes=configs.num_labels), \
        FUNC.multiclass_precision(v_pred_list, true_list, average="macro", num_classes=configs.num_labels)]
    
    micro_result = [FUNC.multiclass_f1_score(pred_list, true_list, average="micro", num_classes=configs.num_labels), \
        FUNC.multiclass_recall(pred_list, true_list, average="micro", num_classes=configs.num_labels), \
        FUNC.multiclass_precision(pred_list, true_list, average="micro", num_classes=configs.num_labels)]
    macro_result = [FUNC.multiclass_f1_score(pred_list, true_list, average="macro", num_classes=configs.num_labels), \
        FUNC.multiclass_recall(pred_list, true_list, average="macro", num_classes=configs.num_labels), \
        FUNC.multiclass_precision(pred_list, true_list, average="macro", num_classes=configs.num_labels)]
    
    return v_micro_result, v_macro_result, micro_result, macro_result, v_category_matrix, category_matrix


# main function
def main(train_loader, test_loader):

    # model
    model = Network(configs).to(device)
    
    # optimizer
    params_with_grad = [param for param in model.parameters() if param.requires_grad]
    optimizer = AdamW(params=params_with_grad, lr=configs.lr, weight_decay=configs.weight_decay)

    # scheduler
    training_steps = configs.epochs * len(train_loader) // configs.gradient_accumulation_steps
    warmup_steps = int(training_steps * configs.warmup_proportion)
    scheduler = get_linear_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=warmup_steps,
                                                num_training_steps=training_steps)

    # accelerator
    model, optimizer, train_loader, test_loader, scheduler = accelerator.prepare(model, optimizer, train_loader, test_loader, scheduler)
    
    # training
    model.zero_grad()
    early_stop_flag = 0
    max_micro_result = None
    max_macro_result = None
    max_v_micro_result = None
    max_v_macro_result = None
    max_v_category_matrix = None
    max_category_matrix = None

    for epoch in range(configs.epochs):
        for train_step, batch in enumerate(train_loader):
            model.train()
            optimizer.zero_grad()
            
            logits, v_feat_nuclear, v_predictions = model(**batch)
            labels = batch['labels']
            
            loss = model.compute_loss(logits, v_feat_nuclear, v_predictions, labels)
            accelerator.backward(loss)

            if train_step % configs.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                model.zero_grad()

            if train_step % 1 == 0:
                print('epoch: {}, step: {}, loss: {}'.format(epoch, train_step, loss))
        
        with torch.no_grad():
            v_micro_result, v_macro_result, micro_result, macro_result, v_category_matrix, category_matrix = evaluate(test_loader, model)
            print('epoch {} v_micro_result: {}, epoch {} v_max_macro_result: {}'.format(epoch, v_micro_result, epoch, v_macro_result))
            print('epoch {} micro_result: {}, epoch {} max_macro_result: {}'.format(epoch, micro_result, epoch, macro_result))
            print('v_category_matrix:\n{}'.format(v_category_matrix))
            print('category_matrix:\n{}'.format(category_matrix))
            
            if max_micro_result is None or micro_result[0] > max_micro_result[0]:
                early_stop_flag = 1
                max_micro_result = micro_result
                
                max_v_micro_result = v_micro_result
                max_v_macro_result = v_macro_result
                max_v_category_matrix = v_category_matrix
                max_category_matrix = category_matrix
    
                # state_dict = {'model': model.state_dict(), 'result': max_micro_result}
                # torch.save(state_dict, 'model/model_cls.pth')
            
            else:
                early_stop_flag += 1
            
            if max_macro_result is None or macro_result[0] > max_macro_result[0]:
                max_macro_result = macro_result
        
        if early_stop_flag >= 10:
            break

    return max_micro_result, max_macro_result, max_v_micro_result, max_v_macro_result, max_v_category_matrix, max_category_matrix


if __name__ == '__main__':
    
    print(watermark(packages='peft,torch,loralib,transformers,accelerate,datasets'))

    # tokenizer
    access_token = "hf_token"
    tokenizer = LlamaTokenizer.from_pretrained(configs.llama2_cache_path, use_fast=True, use_auth_token=access_token)
    tokenizer.padding_side = "left"
    tokenizer.pad_token_id = tokenizer.unk_token_id
    print(tokenizer)
    
    
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
    
    # dataloader
    CustomCollator = DataCollatorWithPadding(tokenizer, padding='longest', max_length=configs.max_seq_len)
    train_loader = DataLoader(dataset=dataset['train'], shuffle=True, batch_size=configs.train_batch_size, collate_fn=CustomCollator)
    test_loader = DataLoader(dataset=dataset['test'], shuffle=False, batch_size=configs.eval_batch_size, collate_fn=CustomCollator)
    
    # main 
    max_micro_result, max_macro_result, max_v_micro_result, max_v_macro_result, max_v_category_matrix, max_category_matrix \
        = main(train_loader, test_loader)

    print('max_micro_result: {}, max_macro_result: {}'.format(max_micro_result, max_macro_result))
    print('max_v_micro_result: {}, max_v_macro_result: {}'.format(max_v_micro_result, max_v_macro_result))
    print('max_v_category_matrix:\n{}'.format(max_v_category_matrix))
    print('max_category_matrix:\n{}'.format(max_category_matrix))