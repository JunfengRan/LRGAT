import pickle, json, decimal, math
import torch


def to_np(x):
    return x.data.cpu().numpy()


def logistic(x):
    return 1 / (1 + math.exp(-x))

# Categories = 0, 1, 2
def cal_metric(pred, true):
    tp, fp, fn = 0, 0, 0
    if pred == true:
        tp += 1
    else:
        for i in range(3):
            if pred == i and true != i:
                fp += 1
            if pred != i and true == i:
                fn += 1
    return [tp, fp, fn]


def eval_func(all_result):
    precision = all_result[0] / (all_result[0] + all_result[1] + 1e-6)
    recall = all_result[0] / (all_result[0] + all_result[2] + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    return [f1, precision, recall]


def float_n(value, n='0.0000'):
    value = decimal.Decimal(str(value)).quantize(decimal.Decimal(n))
    return float(value)


def write_b(b, b_path):
    with open(b_path, 'wb') as fw:
        pickle.dump(b, fw)


def read_b(b_path):
    with open(b_path, 'rb') as fr:
        b = pickle.load(fr)
    return b


def read_json(json_file):
    with open(json_file, 'r', encoding='utf-8') as fr:
        js = json.load(fr)
    return js
