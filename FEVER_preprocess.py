import os
import json
from typing import List

# Path to the file
# file_path = 'data/raw/paper_dev.jsonl'
file_path = 'data/raw/train.jsonl'
# file_path = 'data/raw/paper_test.jsonl'
# Function to process the data from a file
def process_data_from_file(file_path) -> List:
    processed_data = []

    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            # Parse the JSON line
            item = json.loads(line)
            processed_data.append(item)

    return processed_data

# Assume the file_path variable contains the correct path to the file.
# Now we will process the data from the file.
processed_data_from_file = process_data_from_file(file_path)
# processed_data_from_file.extend(process_data_from_file(file_path1))
# processed_data_from_file.extend(process_data_from_file(file_path2))

# Output the processed data
print(processed_data_from_file[2])
import os
import glob
import json
# This is a mock path for demonstration purposes.
# In a real-world scenario, you would replace this with the actual path to your folder.
folder_path = 'data/raw/useful-wiki-pages.jsonl'
all_data_from_jsonl_files = []
with open(folder_path, 'r', encoding='utf-8') as file:
    for line in file:
        # Parse the JSON line and append to the list
        all_data_from_jsonl_files.append(json.loads(line))

# Assume we have the correct folder path.
# Now we will read all the jsonl files in the folder.
def extract_sentence(text, sentence_id):
    # Split the text into sentences based on the period followed by a space or the end of the text
    sentences = text.split('. ')
    # Add the last sentence if it does not end with a period (in case the text does not end with a period)
    if not text.endswith('.'):
        last_sentence = text.rsplit('. ', 1)[-1]
        if last_sentence:  # make sure it's not an empty string
            sentences.append(last_sentence)
    # Try to return the sentence at the specified index, if it exists
    # try:
        return sentences[sentence_id]
    # except IndexError:
    #     return "The sentence ID provided is out of range."
print(len(all_data_from_jsonl_files))
for item in processed_data_from_file:
    if item['label'] != "NOT ENOUGH INFO":
        for evidence in item['evidence']:
            for tiny_evidence in evidence:
                assert type(tiny_evidence) == list # tiny_evidence:[ , , , ]
                Wikipedia_URL,sentence_ID = tiny_evidence[-2],tiny_evidence[-1]
                # source_sentence = ''
                for page in all_data_from_jsonl_files:
                    if Wikipedia_URL == page['id']:
                        text = page['lines']
                        lines = text.strip().split('\n')
                        source_sentence  = lines[sentence_ID]
                        to_remove = f'{sentence_ID}\t'
                        source_sentence = source_sentence.lstrip(to_remove)
                        # print('source_sentence',source_sentence)
                        tiny_evidence.append(source_sentence)
                        continue
            print(item['claim'])
            print(evidence)


file_path = "data/raw/train_modified.jsonl"
with open(file_path, 'w') as file:
    for dictionary in processed_data_from_file:
        # transform every dictionary into a JSON string and write it to a new line in the file
        json_record = json.dumps(dictionary)
        file.write(json_record + '\n')