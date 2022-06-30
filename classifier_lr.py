from collections import Counter
import numpy as np
from scipy.sparse import csr_matrix

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


def sigmoid(z):
    nmax = 1.0 - np.finfo(float).epsneg
    nmin = np.finfo(float).tiny
    bad = 1/(1 + np.exp(-z))
    return np.where(bad == 1.0, nmax, np.where(bad == 0.0, nmin, bad))


def init(n):
    return np.zeros(n)


def loopcount(X, y, w, a):
    ans = 0
    accur = 0
    N = len(y)
    M = len(w)
    for i in range(N):
        z = 0
        for j in range(M):
            z += X[i, j] * w[j]
        h = sigmoid(z)
        if (abs(y[i] - h) < 0.5):
            accur += 1
        ans += y[i] * np.log(h) + (1 - y[i])*np.log(1 - h)
    ans /= -N
    for i in range(len(w)):
        ans += a * w[i] ** 2
    return ans, accur / N


def matrixcount(X, y, w, a):
    ans = a * (np.sum(np.power(w, 2)) - w[0] ** 2)
    Z = X.dot(w)
    h = sigmoid(Z)
    delta = y - h
    accur = (abs(delta) < 0.5).sum()/len(y)
    ans = ans - np.sum(y * np.log(h) + (1 - y) * np.log(1 - h)) / len(y)
    return ans, accur

def gradcount(X, y, w, a):
    Z = X.dot(w)
    h = sigmoid(Z)
    w[0] = 0
    grad = X.transpose().dot((h - y))/len(y) + 2 * a * w
    return grad


def gradescent(X, y ,w, a, epoch, lrate):
    accur_hist = np.zeros(epoch)
    L_hist = np.zeros(epoch)
    for i in range(epoch):
        grad = gradcount(X, y, w, a)
        w = w - lrate * grad
    return accur_hist, L_hist, w


def batched_gradescent(X, y, w, a, epoch, lrate):
    accur_hist = np.zeros(epoch)
    L_hist = np.zeros(epoch)
    batch = 100
    for i in range(epoch):
        if (i * batch % len(y) < (i + 1) * batch % len(y)):
            X_batched = X[i * batch % len(y): (i + 1) * batch % len(y),:]
            y_batched = y[i * batch % len(y): (i + 1) * batch % len(y)]
        else:
            X_batched = X[i * batch % len(y): len(y),:]
            y_batched = y[i * batch % len(y): len(y)]
        grad = gradcount(X_batched, y_batched, w, a)
        w = w - lrate * grad
        if (i %1000 == 0):
            lrate /= 1.2
    return accur_hist, L_hist, w
    
    
def create(texts, labels, voc, size):
    voc_len = len(voc)
    ctr = Counter(voc)
    data = np.empty(size + len(labels), dtype = np.uint32)
    indices = np.empty(size + len(labels), dtype = np.uint32)
    indptr = np.zeros(len(texts) + 1, dtype = np.uint32)
    indexes = {voc[i]:i + 1 for i in range(voc_len)}
    ptr = 0
    pos = 0
    for i in range(len(texts)):
        text = tokenize(preproc(texts[i]))
        BOW = NBOW(text, 1)+ NBOW(text, 2)
        bowlist = list(BOW)
        length = len(bowlist)
        data[pos] = 1
        indices[pos] = 0
        for j in range(length):
            if (ctr[bowlist[j]]):
                data[pos + j + 1] = BOW[bowlist[j]]
                indices[pos + j + 1] = indexes[bowlist[j]]
            else:
                ptr -= 1
                pos -= 1
        ptr += length + 1
        pos += length + 1
        indptr[i + 1] = ptr
    X = csr_matrix((data, indices, indptr), shape=(len(texts), voc_len + 1))
    return X


def NBOW(words, n):
    new_list  = [''] * (len(words) - n + 1)
    for i in range(len(words) - n + 1):
        for j in range(n):
            new_list[i] = new_list[i] + ' ' + words[i + j]
    return Counter(new_list)


def train(texts, labels, pretrain_params=None):
    voc = []
    size = 0
    for i in range(len(texts)):
        text = tokenize(preproc(texts[i]))
        text = NBOW(text, 2)+ NBOW(text, 1)
        size += len(list(text))
        voc.extend(list(text))
    voc = list(set(voc))
    X = create(texts, labels, voc, size)
    y = np.array(np.array(labels) == 'pos')
    w = init(X.shape[1])
    hist, L, w = batched_gradescent(X, y, w, 1e-7, 24000, 5e-2)
    return w, voc


def classify(texts, params):
    w = params[0]
    voc = params[1]
    ctr = Counter(voc)
    ans = ['pos'] * len(texts)
    size = 0
    for i in range(len(texts)):
        text = tokenize(preproc(texts[i]))
        text = NBOW(text, 2)+ NBOW(text, 1)
        for word in list(text):
            size += (ctr[word] != 0)
    X, indexes = create(texts, ans, voc, size)
    probs = X.dot(w)
    for i in range(len(ans)):
        if (probs[i] < 0.5):
            ans[i] = 'neg'
    return ans

