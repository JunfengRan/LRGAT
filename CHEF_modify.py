import json
import random


source_file_path = 'data/raw/CHEF_train.json'
target_file_path = 'data/CHEF_train_modified.json'


def convert_claimId(obj):
    """
    recursively transform the claimId in obj from string to int
    """
    if isinstance(obj, list):
        for item in obj:
            convert_claimId(item)
    elif isinstance(obj, dict):
        if 'claimId' in obj and isinstance(obj['claimId'], str):
            try:
                obj['claimId'] = int(obj['claimId'])
            except ValueError:
                pass  # If claimId cannot be converted to int, ignore it
        for key, value in obj.items():
            convert_claimId(value)


with open(source_file_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

# Count the number of instances for each label
label_count = {'0': 0, '1': 0, '2': 0}
for d in data:
    label_count['{}'.format(d['label'])] += 1

# Calculate the minimum number of instances for each label
min_count = min(label_count.values())

# Remove some label-0 and label-1 instances to have the same number of instances for each label
new_data = []
new_label_count = {'0': 0, '1': 0, '2': 0}
for d in data:
    if new_label_count['{}'.format(d['label'])] < min_count:
        new_data.append(d)
        new_label_count['{}'.format(d['label'])] += 1

convert_claimId(new_data)

with open(target_file_path, 'w') as file:
    json.dump(new_data, file, ensure_ascii=False, indent=4)
