import re
import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
from itertools import combinations as C
import math
from data import data
import pandas as pd


class LDA:
    def __str__(self):
        return f'weight: {self._w}, bias: {self._b}'
        
    def __init__(self, x_train, y_train, c1 = 1, c2 = 1) -> None:
        S = np.zeros((x_train.shape[1],x_train.shape[1]))
        self._n_features = x_train.shape[1]
        self._n_classes = len(np.unique(y_train))
        self._class_type = []

        SP = np.zeros((self._n_features, self._n_features))
        SN = np.zeros((self._n_features, self._n_features))
        
        m1 = None
        m2 = None
        for c in np.unique(y_train):
            self._class_type.append(c)
            x_c = np.array(x_train)[y_train == c]
            mean_c = np.mean(x_c, axis=0)
            for i in x_c:
                diff = i-mean_c
                m = np.zeros((2, len(diff)))
                m[0] = diff
                m = m.T.dot(m)
                if c:
                    SP += m
                else:
                    SN += m
            if c:
                SP /= x_c.shape[0]-1
                S += (SP*x_c.shape[0])
                m1 = mean_c
            else:
                SN /= x_c.shape[0]-1
                S += SN*x_c.shape[0]
                m2 = mean_c
        S /= x_train.shape[0]
        self._w = np.array([round(i,2) for i in(np.dot((m1-m2), np.linalg.inv(S)))])
        self._b = round(-0.5*self._w.dot(m1+m2) - math.log(c1/c2), 2)
    
    def validation(self, x_test, y_ans):
        score = 0
        for c in np.unique(y_ans):
            x_c = np.array(x_test)[y_ans == c]
            for i in x_c:
                # pos = self.transform(i)
                pos = np.dot(i, self._w.T)+self._b
                if c == 1 and pos > 0:
                    score += 1
                elif c == 0 and pos <= 0:
                    score += 1
        return round(score*100/x_test.shape[0], 2)

    def get_tp_tn_fp_fn(self, x_test, y_ans, cc):
        b = round(self._b-math.log(cc), 2)
        tp = tn = fp = fn = 0
        for c in np.unique(y_ans):
            x_c = np.array(x_test)[y_ans == c]
            for i in x_c:
                pos = np.dot(i, self._w.T)+b
                if c == 1 and pos > 0:
                    tp += 1
                elif c == 1 and pos <= 0:
                    fn += 1
                elif c == 0 and pos > 0:
                    fp += 1
                elif c == 0 and pos <= 0:
                    tn += 1
        return tp,tn,fp,fn

    def transform(self, x):
        pos = np.dot(x, self._w.T)+self._b
        if pos > 0:
            return 1
        else:
            return 0

datas_label1 = []
datas_label2 = []
datas_label3 = []

f = open('iris.txt', 'r')
for line in f.readlines():
    s = re.split(r'\s+', line)
    data_input = data(s[4])
    data_input[:] = [float(i) for i in s[:4]]
    if len(data_input) == 1:
        datas_label1.append(data_input)
    elif len(data_input) == 2:
        datas_label2.append(data_input)
    elif len(data_input) == 3:
        datas_label3.append(data_input)

train_datas = []
test_datas = []
for i in range(25):
    train_datas.append(datas_label2[i].set_label(1))
    train_datas.append(datas_label3[i].set_label(0))
for i in range(25,50):
    test_datas.append(datas_label2[i].set_label(1))
    test_datas.append(datas_label3[i].set_label(0))


step2 = LDA(np.array([i[:2] for i in train_datas]), np.array([len(i) for i in train_datas]))
step2_validation = step2.validation(np.array([i[:2] for i in test_datas]), np.array([len(i) for i in test_datas]))
print(f'step2: {step2}')
print(f'step2_validation: {step2_validation}')
step4 = LDA(np.array([i[:2] for i in test_datas]), np.array([len(i) for i in test_datas]))
step4_validation = step4.validation(np.array([i[:2] for i in train_datas]), np.array([len(i) for i in train_datas]))
print(f'step4: {step4}')
print(f'step4_validation: {step4_validation}')
print(f'step2n4`s validation {(step2_validation+step4_validation)/2}')


print("==================(3)===================")
train_datas = []
test_datas = []
for i in range(25):
    train_datas.append(datas_label2[i].set_label(0))
    train_datas.append(datas_label3[i].set_label(1))
for i in range(25,50):
    test_datas.append(datas_label2[i].set_label(0))
    test_datas.append(datas_label3[i].set_label(1))
step1 = LDA(np.array([i[:] for i in train_datas]), np.array([len(i) for i in train_datas]))
print("step1:" ,step1)
tpr = []
fpr = []
for i in np.arange(0.01,10,0.01):
    tp,tn,fp,fn = step1.get_tp_tn_fp_fn(np.array([i[:] for i in test_datas]), np.array([len(i) for i in test_datas]), i)
    tpr.append(tp/(tp+fn))
    fpr.append(fp/(fp+tn))
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([-0.05, 1.0])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.savefig("step1.png")
plt.clf()
step3 = LDA(np.array([i[2:] for i in train_datas]), np.array([len(i) for i in train_datas]))
step2 = LDA(np.array([i[:2] for i in train_datas]), np.array([len(i) for i in train_datas]))
print("step2:",step2)
tpr = []
fpr = []
for i in np.arange(0.01,10,0.01):
    tp,tn,fp,fn = step2.get_tp_tn_fp_fn(np.array([i[:2] for i in test_datas]), np.array([len(i) for i in test_datas]), i)
    tpr.append(tp/(tp+fn))
    fpr.append(fp/(fp+tn))
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([-0.05, 1.0])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.savefig("step2.png")
plt.clf()
step3 = LDA(np.array([i[2:] for i in train_datas]), np.array([len(i) for i in train_datas]))
print("step3:",step3)
tpr = []
fpr = []
for i in np.arange(0.01,10,0.01):
    tp,tn,fp,fn = step3.get_tp_tn_fp_fn(np.array([i[2:] for i in test_datas]), np.array([len(i) for i in test_datas]), i)
    tpr.append(tp/(tp+fn))
    fpr.append(fp/(fp+tn))
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([-0.05, 1.0])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.savefig("step3.png")
plt.clf()

print("==================(4)===================")

train_datas = []
for i in range(25):
    train_datas.append(datas_label1[i].set_label(1))
    train_datas.append(datas_label2[i].set_label(0))
LDA1 = LDA(np.array([i[2:] for i in train_datas]), np.array([len(i) for i in train_datas]))

train_datas = []
for i in range(25):
    train_datas.append(datas_label1[i].set_label(1))
    train_datas.append(datas_label3[i].set_label(0))
LDA2 = LDA(np.array([i[2:] for i in train_datas]), np.array([len(i) for i in train_datas]))

train_datas = []
for i in range(25):
    train_datas.append(datas_label2[i].set_label(1))
    train_datas.append(datas_label3[i].set_label(0))
LDA3 = LDA(np.array([i[2:] for i in train_datas]), np.array([len(i) for i in train_datas]))

test_datas = []
for i in range(25,50):
    test_datas.append(datas_label1[i].set_label(1))
    test_datas.append(datas_label2[i].set_label(2))
    test_datas.append(datas_label3[i].set_label(3))
step3_score = 0
for i in test_datas:
    vote = []
    if LDA1.transform(i[2:]) == 1:
        vote.append(1)
    else:
        vote.append(2)
    if LDA2.transform(i[2:]) == 1:
        vote.append(1)
    else:
        vote.append(3)
    if LDA3.transform(i[2:]) == 1:
        vote.append(2)
    else:
        vote.append(3)
    predict = max(vote,key=vote.count)
    # print(f'{len(i)}: {predict}')
    if len(i) == predict:
        step3_score += 1
step3_score = step3_score*100/len(test_datas)
print("step3_score:",step3_score)
# ====================================================================================================
train_datas = []
for i in range(25, 50):
    train_datas.append(datas_label1[i].set_label(1))
    train_datas.append(datas_label2[i].set_label(0))
LDA1 = LDA(np.array([i[2:] for i in train_datas]), np.array([len(i) for i in train_datas]))

train_datas = []
for i in range(25, 50):
    train_datas.append(datas_label1[i].set_label(1))
    train_datas.append(datas_label3[i].set_label(0))
LDA2 = LDA(np.array([i[2:] for i in train_datas]), np.array([len(i) for i in train_datas]))

train_datas = []
for i in range(25, 50):
    train_datas.append(datas_label2[i].set_label(1))
    train_datas.append(datas_label3[i].set_label(0))
LDA3 = LDA(np.array([i[2:] for i in train_datas]), np.array([len(i) for i in train_datas]))

test_datas = []
for i in range(25):
    test_datas.append(datas_label1[i].set_label(1))
    test_datas.append(datas_label2[i].set_label(2))
    test_datas.append(datas_label3[i].set_label(3))
step4_score = 0
for i in test_datas:
    vote = []
    if LDA1.transform(i[2:]) == 1:
        vote.append(1)
    else:
        vote.append(2)
    if LDA2.transform(i[2:]) == 1:
        vote.append(1)
    else:
        vote.append(3)
    if LDA3.transform(i[2:]) == 1:
        vote.append(2)
    else:
        vote.append(3)
    predict = max(vote,key=vote.count)
    # print(f'{len(i)}: {predict}')
    if len(i) == predict:
        step4_score += 1
step4_score = step4_score*100/len(test_datas)
print("step4_score:",step4_score)
print("step34",(step3_score+step4_score)/2)