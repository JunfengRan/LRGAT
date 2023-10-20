import json


def segment_sentences(text):
    # use Jieba to segment sentences
    sentences = []
    temp_sentence = ""
    for char in text:
        if char in ['。', '！', '？']:
            temp_sentence += char
            sentences.append(temp_sentence)
            temp_sentence = ""
        else:
            temp_sentence += char

    # strip empty sentences
    sentences = [s.strip() for s in sentences if s.strip()]

    return sentences


def find_sentence_containing(subclause, text):
    sentences = segment_sentences(text)

    for sentence in sentences:
        if subclause in sentence:
            return sentence

    return None


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


# test
# text = "今天天气真好！我们一起去公园吧。"
# sentences = segment_sentences(text)
# for sentence in sentences:
#     print(sentence)

source_file_path = 'data/CHEF_test.json'
target_file_path = 'data/CHEF_test_lengthened.json'

with open(source_file_path, 'r') as f:
    data = json.load(f)
convert_claimId(data)
renewed_data = []
for item in data:
    # text = item['evidence']['0']['text']
    evidence = item['evidence']
    text = ''
    for k, v in evidence.items():
        text += v['text'] + '。'
    tfidf = item['tfidf']  # list
    cossim = item['cossim']
    ranksvm = item['ranksvm']
    renewed_tfidf = []
    renewed_cossim = []
    renewed_ranksvm = []
    for subclause in tfidf:
        result = find_sentence_containing(subclause, text)
        renewed_tfidf.append(result)
    for subclause in cossim:
        result = find_sentence_containing(subclause, text)
        renewed_cossim.append(result)
    for subclause in ranksvm:
        result = find_sentence_containing(subclause, text)
        renewed_ranksvm.append(result)
    item['tfidf'] = renewed_tfidf
    item['cossim'] = renewed_cossim
    item['ranksvm'] = renewed_ranksvm
    renewed_data.append(item)

with open(target_file_path, 'w') as file:
    json.dump(renewed_data, file, ensure_ascii=False, indent=4)
    
    