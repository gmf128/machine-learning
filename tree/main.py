import time

import pandas as pd
import math
import csv
import DecisionTree
from sklearn.metrics import f1_score

decisionTree = None

condition_dict = {}
sideeffect_dict = {}
def pre_processing(data, contains_labels=True):
    # 清洗掉不需要的列
    data = data.drop(columns=['recordId', 'drugName', 'reviewComment', 'date'])
    if not contains_labels:
        data = data.drop(columns=['rating'])
    # 将字符串转成数字，便于后续处理
    global condition_dict
    global sideeffect_dict
    if len(condition_dict) == 0:
        condition_dict = {k: i for i, k in enumerate(data['condition'].unique())}
    data['condition'] = data['condition'].map(condition_dict)
    if len(sideeffect_dict) == 0:
        sideeffect_dict = {'No Side Effects': 0, 'Mild Side Effects': 1, 'Moderate Side Effects': 2, 'Severe Side Effects': 3, 'Extremely Severe Side Effects': 4}
    data['sideEffects'] = data['sideEffects'].map(sideeffect_dict)
    if contains_labels:
        data['rating'] = data['rating'].astype(int)
    # 分开attributes和label
    attributes = data[['condition', 'usefulCount', 'sideEffects']].values.tolist()
    labels = None
    if contains_labels:
        labels = data['rating'].values.tolist()

    return attributes, labels

def train():
    training_data = pd.read_csv("./dataset/training.csv")
    rows, labels = pre_processing(training_data)
    sets = []
    for i in range(0, len(rows)):
        row = rows[i]
        rows[i].append(labels[i])
        sets.append(row)
    rows = sets
    DecisionTree.update_default(rows)
    global decisionTree
    decisionTree = DecisionTree.buildDecisionTree(rows)


def validation():
    validation_data = pd.read_csv("./dataset/validation.csv")
    data, labels = pre_processing(validation_data)
    predictions = []
    for row in data:
        predictions.append(DecisionTree.classify(row, decisionTree))
    print(predictions)
    micro_score = f1_score(labels, predictions, average='micro')
    macro_score = f1_score(labels, predictions, average='macro')
    print("Micro-F1 score:", micro_score)
    print("Macro-F1 score:", macro_score)

def test():
    test_data = pd.read_csv("./dataset/testing.csv")
    data, labels = pre_processing(test_data, False)
    predictions = []
    for row in data:
        predictions.append(DecisionTree.classify(row, decisionTree))
    test_data = test_data.drop(columns=['rating'])
    test_data.insert(7, 'rating', predictions)
    test_data.to_csv('./testing_result.csv')

if __name__ == "__main__":
    begin = time.time()
    train()
    end = time.time()
    print("train_time: {}".format(end - begin))
    begin = time.time()
    validation()
    end = time.time()
    print("validation_time: {}".format(end - begin))
    test()
    DecisionTree.classify([4, 101, 0], decisionTree)