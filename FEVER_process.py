import os
import json
import random


evidence_bucket = []

def convert_jsonl_to_json(file_path, new_file_path):
    with open(file_path, 'r') as f:
        data = [json.loads(line) for line in f]
    
    for item in data:
        if item['verifiable'] == 'VERIFIABLE':
            item['verifiable'] = 0
        elif item['verifiable'] == 'NOT VERIFIABLE':
            item['verifiable'] = 1
        
        if item['label'] == 'SUPPORTS':
            item['label'] = 0
        elif item['label'] == 'REFUTES':
            item['label'] = 1
        elif item['label'] == 'NOT ENOUGH INFO':
            item['label'] = 2
        
        evidence_dict = {}
        annotation_id = None
        index = 0
        for i in range(5):
            evidence_dict[i] = ''
        for i in range(len(item['evidence'])):
            if index == 5:
                break
            if item['evidence'][i] != None and item['evidence'][i] != '' and item['evidence'][i] != ' ':
                new_evidence = item['evidence'][i][0]
                if new_evidence[1] != None:
                    if annotation_id == None:
                        annotation_id = new_evidence[1]
                    if new_evidence[1] == annotation_id:
                        if len(new_evidence) >= 5:
                            evidence_dict[index] += new_evidence[4]
                            evidence_bucket.append(new_evidence[4])
                            index += 1
                else:
                    if random.random() > 0.95:
                        evidence_dict[index] += evidence_bucket[random.randint(0, len(evidence_bucket) - 1)]
                        index += 1
        item['evidence'] = evidence_dict
    
    with open(new_file_path[:-1], 'w') as f:
        json.dump(data, f)

data_dir = 'data/raw'
new_data_dir = 'data'
for file_name in os.listdir(data_dir):
    if file_name.endswith('.jsonl'):
        file_path = os.path.join(data_dir, file_name)
        new_file_path = os.path.join(new_data_dir, file_name)
        convert_jsonl_to_json(file_path, new_file_path)