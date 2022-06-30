from collections import Counter
import numpy as np

def download():
    Data = open("./glove.6B.50d.txt", "r")
    return list(Data)

def get_word(index, Data):
    return Data[index].split()[0]

def get_words(Data):
    arr = [''] * len(Data)
    for i in range(len(Data)):
        arr[i] = get_word(i, Data)
    dic = {arr[i]:i for i in range(len(arr))}
    return dic

def preproc(text):
    text = text.lower()
    newtext = ''
    for c in text:
        if (not c.isalpha() and not c.isspace()):
            newtext = newtext + ' '
        else:
            newtext = newtext + c
    return newtext


def tokenize(text):
    text = text.split(" ")
    text = list(filter(lambda x: x != '', text))
    return text

def embedding(word, Data, words):
    index = words.get(word)
    if (index != None):
        array = Data[index].split()
        array = np.array([float(x) for x in array[1:]])
        return array
    else:
        return np.array([0.0045])

def GloVe_text(text, Data, words):
    array = np.zeros(50)
    count = 0
    BOW = Counter(text)
    for word in BOW:
        if  (word != ''):
            emb = embedding(word, Data, words)
            if (emb[0] != 0.0045):
                array += BOW[word] * emb
                count += BOW[word]
    return array / count

def GloVe_texts(texts, Data, words):
    matrix = np.zeros((len(texts), 50))
    for i in range(len(texts)):
        matrix[i] = GloVe_text(texts[i], Data, words)
    return matrix

def relu(z):
    return np.where(z < 0, 0, z)

def sigmoid(z):
    nmax = 1.0 - np.finfo(float).epsneg
    nmin = np.finfo(float).tiny
    bad = 1/(1 + np.exp(-z))
    return np.where(bad == 1.0, nmax, np.where(bad == 0.0, nmin, bad))

def tanh(z):
    return np.tanh(z)

def init_params(layer_sizes, activation):
    weights = {}
    if (activation == "relu"):
        for i in range(len(layer_sizes) - 1):
            weights[('w',i+1)] = np.random.randn(layer_sizes[i] + 1, layer_sizes[i + 1]) * (
            np.sqrt(2 / (layer_sizes[i] + 1)))    
    else:
        for i in range(len(layer_sizes) - 1):
            weights[('w',i+1)] = np.random.randn(layer_sizes[i] + 1, layer_sizes[i + 1]) * (
            np.sqrt(1 / (layer_sizes[i] + 1)))
    return weights

def fully_connected(a_prev, W, activation):
    A_prev = np.c_[np.ones(a_prev.shape[0]), a_prev]
    Z = A_prev.dot(W)
    if (activation == "relu"):
        A = relu(Z)
    if (activation == "tanh"):
        A = tanh(Z)
    if (activation == "sigmoid"):
        A = sigmoid(Z)
    if (activation == "linear"):
        A = Z
    return A, Z

def ffnn(X, params, activation):
    A_arr = [None] * (len(params) + 1)
    Z_arr = [None] * (len(params) + 1)
    for i in range(len(params) + 1):
        if (i == 0):
            A_arr[i], Z_arr[i] = np.array(X), np.array(X)
        else:
            A_arr[i], Z_arr[i] = fully_connected(A_arr[i - 1], params['w', i], activation)
    return A_arr, Z_arr, Z_arr[-1]

def softmax_crossentropy(ZL, Y):
    sub = ZL.max(axis = 1)
    Z_new = ZL - np.vstack(sub)
    sft = np.exp(Z_new) / np.vstack(np.sum(np.exp(Z_new), axis=1))
    if (Y[0][0] == None):
        return sft
    CE = -1/Y.shape[0] * np.sum(np.log(sft) * Y)
    dZL = 1/Y.shape[0] * (sft - Y)
    return sft, CE, dZL


def ffnn_backward(dZL, caches, activation):
    A_arr = caches[0]
    Z_arr = caches[1]
    W_dic = caches[2]
    grads = {}
    grads['z', len(Z_arr) -1] = dZL
    if (activation == "tanh"):
        for i in range(len(Z_arr) - 2, 0, -1):
            A_prev = np.array(A_arr[i])
            grads['z', i] = (1 - A_prev*A_prev) * (grads['z', i+1] @ W_dic['w', i+1][1:].transpose())
        for i in range(1, len(W_dic) + 1, 1):
            A_prev = np.c_[np.ones(A_arr[i-1].shape[0]), A_arr[i-1]].transpose()
            grads['w', i] = A_prev @ grads['z', i]
    return grads
    
def sgd_step(params, grads, learning_rate):
    W_dic = params
    for i in range(len(W_dic)):
        W_dic['w', i+1] -= learning_rate * grads['w', i+1]
    return W_dic

def check(res, ans):
    temp = res * ans
    print(np.sum(np.sum(temp, axis = 1) > 0.5) / res.shape[0])
    
def train_ffnn(Xtrain, Ytrain, layer_sizes, learning_rate, num_epochs, batch):
    W_dic = init_params(layer_sizes, 'tanh')
    for epoch in range(num_epochs):
        if (epoch * batch % len(Ytrain) < (epoch + 1) * batch % len(Ytrain)):
            X_batched = Xtrain[epoch * batch % len(Ytrain): (epoch + 1) * batch % len(Ytrain),:]
            y_batched = Ytrain[epoch * batch % len(Ytrain): (epoch + 1) * batch % len(Ytrain)]
        else:
            X_batched = Xtrain[epoch * batch % len(Ytrain): len(Ytrain),:]
            y_batched = Ytrain[epoch * batch % len(Ytrain): len(Ytrain)]
        A_ar, Z_ar, ZL = ffnn(X_batched, W_dic, 'tanh')
        sft, CE, dZL = softmax_crossentropy(ZL, y_batched)
        grads = ffnn_backward(dZL, (A_ar, Z_ar, W_dic), 'tanh')
        W_dic = sgd_step(W_dic, grads, learning_rate)
    return W_dic, sft

def pretrain(texts):
    """
    """
def train(texts, labels, pretrain_params=None):
    np.random.seed(10)
    text = [None] * len(texts)
    labels = labels
    Data = download()
    all_words = get_words(Data)
    for i in range(len(texts)):
        text[i] = tokenize(preproc(texts[i]))
        text[i] = list(map(lambda s: s.strip(), text[i]))
    labels = list(map(lambda s: s.strip(), labels))
    Y = np.zeros((len(labels), 2))
    for i in range(len(labels)):
        if (labels[i] == 'pos'):
            Y[i, 1] = 1
        else:
            Y[i, 0] = 1 
    matrix = GloVe_texts(text, Data, all_words)
    W_dic, sft = train_ffnn(matrix, Y, [50, 200, 100, 2], 3e-2, 2000, 500)
    return W_dic

def classify(texts, params):
    Data = download()
    text = [None] * len(texts)
    all_words = get_words(Data)
    for i in range(len(texts)):
        text[i] = tokenize(preproc(texts[i]))
        text[i] = list(map(lambda s: s.strip(), text[i]))
    matrix = GloVe_texts(text, Data, all_words)
    W_dic = params
    A_ar, Z_ar, ZL = ffnn(matrix, W_dic, 'tanh')
    sft = softmax_crossentropy(ZL, [[None]])
    ans = sft[:,1]
    return np.where(ans >= 0.5, 'pos', 'neg')
