import math
from collections import Counter
def pretrain(texts):
    """
    """
def preproc(text):
    text = text.lower()
    newtext = ''
    for c in text:
        if (not c.isalpha() and not c.isdigit() and not c.isspace()):
            newtext = newtext + ' ' + c + ' '
        else:
            newtext = newtext + c
    return newtext

def tokenize(text):
    text = text.split(" ")
    text = list(filter(lambda x: x != '', text))
    return text

def NBOW(words, n):
    new_list  = [''] * (len(words) - n + 1)
    for i in range(len(words) - n + 1):
        for j in range(n):
            new_list[i] = new_list[i] + words[i + j]
    return Counter(new_list)
def train(texts, labels, pretrain_params=None):
    all_texts = ''
    pos_texts = ''
    neg_texts = ''
    for i in range(len(texts)):
        text = preproc(texts[i]) 
        all_texts = all_texts + text + ' '
        if (labels[i] == 'pos'):
            pos_texts = pos_texts + text + ' '
        else:
            neg_texts = neg_texts + text + ' '
    all_words = tokenize(all_texts)
    pos_words = tokenize(pos_texts)
    neg_words = tokenize(neg_texts)
    BOAW = NBOW(all_words, 1)
    BOPW = NBOW(pos_words, 1)
    BONW = NBOW(neg_words, 1)
    votes = Counter(labels)
    P_pos = votes['pos']/sum(votes.values())
    P_neg = votes['neg']/sum(votes.values())
    train_param = [BOAW, BOPW, BONW, P_pos, P_neg]
    return train_param

def classify(texts, params):
    ans = ['pos'] * len(texts)
    L_voc = len(list(params[0]))
    S_pos = sum(params[1].values())
    S_neg = sum(params[2].values())
    for i in range(len(texts)):
        text = preproc(texts[i])
        words = tokenize(text)
        BOW = NBOW(words, 1)
        Val_pos = math.log(params[3])
        Val_neg = math.log(params[4])
        for word in list(BOW):
            Val_pos = Val_pos + BOW[word] * math.log((params[1][word] + 1) /(S_pos + L_voc))
            Val_neg = Val_neg + BOW[word] * math.log((params[2][word] + 1) /(S_neg + L_voc))
        if (Val_pos >= Val_neg):
            ans[i] = 'pos'
        else:
            ans[i] = 'neg'
    return ans
            
