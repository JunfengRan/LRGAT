# from sklearn.metrics import multiclass_f1_score
from sklearn.metrics import f1_score
import json
import re
import numpy as np
file_path = 'data/CHEF_test_result.json'

from sklearn.metrics import f1_score
import numpy as np


def calculate_multiclass_f1(y_true, y_pred, unknown_label=None):
    """
    计算三分类F1值

    参数：
    y_true: 真实的类别标签列表
    y_pred: 预测的类别标签列表
    unknown_label: 缺失输出的样本所对应的特殊类别标签（默认为None，表示不考虑缺失输出）

    返回：
    f1_macro: 宏平均F1值
    f1_micro: 微平均F1值
    f1_weighted: 加权平均F1值
    """
    # 处理缺失输出的样本
    if unknown_label is not None:
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        unknown_mask = (y_true == unknown_label) | (y_pred == unknown_label)
        y_true_known = y_true[~unknown_mask]
        y_pred_known = y_pred[~unknown_mask]
    else:
        y_true_known = y_true
        y_pred_known = y_pred

    # 计算宏平均F1值
    f1_macro = f1_score(y_true_known, y_pred_known, average='macro')

    # 计算微平均F1值
    f1_micro = f1_score(y_true_known, y_pred_known, average='micro')

    # 计算加权平均F1值
    f1_weighted = f1_score(y_true_known, y_pred_known, average='weighted')

    return f1_macro, f1_micro, f1_weighted


# 假设有一些样本没有输出结果，可以将这些样本标记为"未知"类别（例如，用-1表示）
unknown_label = -1
y_true_with_unknown = [0, 1, 2, 1, -1, 0, 0, 1]
y_pred_with_unknown = [0, 2, 1, 1, -1, 0, 0, 1]


with open(file_path,'r') as f:
    data = json.load(f)
true_labels = []
predicted_labels = []
label_pattern = r'label value is (\d)'
# 从response里面提取标签
for item in data:
    true_labels.append(int(item['label']))
    label_values = re.findall(label_pattern, item['response'])
    print(label_values)
    if not label_values:
        predicted_label = -1
    else:
        predicted_label = int(label_values[0])
    predicted_labels.append(predicted_label)
    # print(item['response'])


f1_macro, f1_micro, f1_weighted = calculate_multiclass_f1(true_labels, predicted_labels)

print("宏平均F1值:", f1_macro)
print("微平均F1值:", f1_micro)
print("加权平均F1值:", f1_weighted)
