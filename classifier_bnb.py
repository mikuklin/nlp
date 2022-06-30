import math
from collections import Counter

def pretrain(texts):
    """
    """
def preproc(text):
    newtext = ''
    for c in text:
        if (not c.isalpha() and not c.isdigit() and not c.isspace()):
            newtext = newtext + ' ' + c.lower() + ' '
        else:
            newtext = newtext + c.lower()
    return newtext

def tokenize(text):
    text = text.split()
    text = list(filter(lambda x: x != '', text))
    return text

def NBOW(words, n):
    new_list  = [''] * (len(words) - n + 1)
    for i in range(len(words) - n + 1):
        for j in range(n):
            new_list[i] = new_list[i] + words[i + j]
    return Counter(new_list)

def train(texts, labels, pretrain_params=None):
    all_texts = []
    pos_texts = []
    neg_texts = []
    for i in range(len(texts)):
        text = preproc(texts[i])
        text = tokenize(text)
        text = set(text)
        text = list(text)
        all_texts = all_texts + text
        if (labels[i] == 'pos'):
            pos_texts = pos_texts + text
        else:
            neg_texts = neg_texts + text
    BOAW = NBOW(all_texts, 1)
    BOPW = NBOW(pos_texts, 1)
    BONW = NBOW(neg_texts, 1)
    votes = Counter(labels)
    P_pos = votes['pos']/sum(votes.values())
    P_neg = votes['neg']/sum(votes.values())
    train_param = [BOAW, BOPW, BONW, P_pos, P_neg, votes['pos'], votes['neg']]
    return train_param

def classify(texts, params):
    ans = ['pos'] * len(texts)
    Val_empty_pos = math.log(params[3])
    Val_empty_neg = math.log(params[4])
    for word in list(params[0]):
        Val_empty_pos = Val_empty_pos + math.log(1  - (params[1][word] + 1)/(params[5] + 2))
        Val_empty_neg = Val_empty_neg + math.log(1  - (params[2][word] + 1)/(params[6] + 2))
    for i in range(len(texts)):
        text = preproc(list(texts)[i])
        words = tokenize(text)
        words = set(words)
        words = list(words)
        BOW = NBOW(words, 1)
        Val_pos = Val_empty_pos
        Val_neg = Val_empty_neg
        for word in list(BOW):
            Val_pos = Val_pos - math.log(1 - (params[1][word] + 1)/(params[5] + 2)) + math.log((params[1][word] + 1)/(params[5] + 2))
            Val_neg = Val_neg - math.log(1 - (params[2][word] + 1)/(params[6] + 2)) + math.log((params[2][word] + 1)/(params[6] + 2))
        if (Val_pos >= Val_neg):
            ans[i] = 'pos'
        else:
            ans[i] = 'neg'
    return ans
            
