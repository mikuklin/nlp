import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing
from scipy.special import expit

def get_batch(arr, batch_size, i):
    l = len(arr)
    if (i * batch_size % l < (i + 1) * batch_size % l):
        arr_batched=arr[i * batch_size % l:(i + 1) * batch_size % l]
    else:
        arr_batched=np.concatenate((arr[i * batch_size % l:l], arr[:(i + 1) * batch_size % l]))
    return arr_batched
    
def batch_generator(words_idxs, docs_idxs, probs, nb=5, batch_size=100):
    pos1 = 0
    pos2 = 0
    l = len(words_idxs)
    words_rand = np.random.choice(len(probs), nb * len(probs), p=probs)
    while True:
        words_idxs_batched = get_batch(words_idxs, batch_size, pos1)
        docs_idxs_batched = get_batch(docs_idxs, batch_size, pos1)
        labels = np.ones(batch_size)
        yield words_idxs_batched, docs_idxs_batched, labels
        labels = np.zeros(batch_size)
        for j in range(nb):
            words_idxs_batched = get_batch(words_rand, batch_size, pos2)
            yield words_idxs_batched, docs_idxs_batched, labels
            pos2 += 1
        pos1 += 1

class Doc2Vec:
    def __init__(self, dic_len, doc_len, emb_len=500, low=-1e-3, high=1e-3):
        self.word_emb = np.random.uniform(low, high, size=(dic_len, emb_len))
        self.doc_emb = np.random.uniform(low, high, size=(doc_len, emb_len))
    def train(self, words_idxs, docs_idxs, labels, lr=0.1):
        g_words = self.word_emb[words_idxs]
        g_docs = self.doc_emb[docs_idxs]
        vec = expit(np.array([g_words[i] @ g_docs[i] for i in range(len(words_idxs))])) - labels
        for i in range(len(words_idxs)):
            self.doc_emb[docs_idxs[i]] -= g_words[i] * vec[i] * lr
            self.word_emb[words_idxs[i]] -= g_docs[i] * vec[i] * lr

def train_log_reg(X, y_train, y_dev):
    scaler = preprocessing.StandardScaler()
    X = scaler.fit_transform(X)
    grid={"C":np.logspace(-5,3,7)}
    logreg=LogisticRegression(random_state=42, max_iter=2000)
    logreg_cv=GridSearchCV(logreg,grid,cv=10)
    logreg_cv.fit(X[:15000], y_train)
    print("tuned hpyerparameters :",logreg_cv.best_params_)
    print(logreg_cv.score(X[0:15000], y_train))
    print(logreg_cv.score(X[15000:25000], y_dev))

def get_emb(embs, all_texsts, some_texts):
    idxs = list([all_texsts.index(some_texts[i]) for i in range(len(some_texts))])
    return embs[idxs]

def pretrain(texts):
    texts = [item for sublist in texts for item in sublist]
    np.random.seed(42)
    vectorizer = CountVectorizer(min_df = 5, max_df=9e-1, ngram_range=(1,3))
    X = vectorizer.fit_transform(texts)
    shape = X.get_shape()
    probs = np.power(np.sum(X, axis = 0), 3/4)
    probs /= np.sum(probs)
    probs = np.squeeze(np.asarray(probs))
    words_idxs = X.nonzero()[1]
    docs_idxs = X.nonzero()[0]
    doc_len = shape[0]
    dic_len = shape[1]
    print(dic_len)
    print(doc_len)
    print(len(words_idxs))
    perm = np.random.permutation(len(words_idxs))
    words_idxs = words_idxs[perm]
    docs_idxs = docs_idxs[perm]
    embeddings = Doc2Vec(dic_len, doc_len)
    nb = 5
    generator = batch_generator(words_idxs, docs_idxs, probs, nb, 1000)
    l_r = 0.1
    iters = 400000
    for i in range(iters):
        l_r *= 0.95
        wi, di, lab = next(generator)
        embeddings.train(wi, di, lab)
    return embeddings.doc_emb, texts
   
def train(texts, labels, pretrain_params=None):
    np.random.seed(42)
    #texts = [item for sublist in texts for item in sublist]
    emb = pretrain_params[0]
    all_texts = pretrain_params[1]
    X = get_emb(emb, all_texts, texts)
    scaler = preprocessing.StandardScaler()
    X = scaler.fit_transform(X)
    grid={"C":np.logspace(-5,3,7)}
    logreg=LogisticRegression(random_state=42, max_iter=2000)
    logreg_cv=GridSearchCV(logreg,grid,cv=10)
    logreg_cv.fit(X, labels)
    return logreg_cv, emb, all_texts, scaler

def classify(texts, params):
    np.random.seed(42)
    #texts = [item for sublist in texts for item in sublist]
    logreg = params[0]
    emb = params[1]
    all_texts = params[2]
    scaler = params[3]
    X = get_emb(emb, all_texts, texts)
    X = scaler.transform(X)
    preds = logreg.predict(X)
    return preds
