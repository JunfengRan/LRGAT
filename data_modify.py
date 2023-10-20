import json

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

with open('data/CHEF_test.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

convert_claimId(data)

with open(target_file_path, 'w') as file:
    json.dump(renewed_data, file, ensure_ascii=False, indent=4)