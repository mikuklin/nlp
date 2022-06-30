import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import torchtext
class LSTMCell(nn.Module):
    def __init__(self, W, U, b_w, b_u):
        super(LSTMCell, self).__init__()
        self.W = W.T
        self.U = U.T
        self.b_w = b_w
        self.b_u = b_u
    def forward(self, c_h, x):
        c, h = c_h
        chunks1 = x @ self.W + self.b_w
        chunks2 = h @ self.U + self.b_u
        chunks = chunks1 + chunks2
        i, f, o, c_ = chunks.chunk(4, 1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        c_ = torch.tanh(c_)
        c = f * c + i * c_
        h = o * torch.tanh(c)
        return (c, h)

class LSTMLayer(nn.Module):
    def __init__(self, rnn_size):
        super(LSTMLayer, self).__init__()
        self.rnn_size = rnn_size
        self.W = nn.Parameter(torch.Tensor(4 * rnn_size, rnn_size), requires_grad = True)
        self.U = nn.Parameter(torch.Tensor(4 * rnn_size, rnn_size), requires_grad = True)
        self.b_w = nn.Parameter(torch.Tensor(4 * rnn_size), requires_grad = True)
        self.b_u = nn.Parameter(torch.Tensor(4 * rnn_size), requires_grad = True)
    def forward(self, input, c_h):
        input = torch.unbind(input)
        output = []
        for i, x in enumerate(input): #x[batch_size, rnn_size]
            cell = LSTMCell(self.W, self.U, self.b_w, self.b_u)
            c_h = cell(c_h, x)
            c, h = c_h
            output.append(h)
        return torch.stack(output), (c, h)

class LSTM(nn.Module):
    def __init__(self, rnn_size, layer_num, dropout):
        super(LSTM, self).__init__()
        self.layer_num = layer_num
        self.layers = nn.ModuleList([LSTMLayer(rnn_size) for i in range(layer_num)])
        self.drop = nn.Dropout(p = dropout)
    def forward(self, input, c_h):
        for i, layer in enumerate(self.layers):
            input, c_h[i] = layer(input, c_h[i])
            input = self.drop(input)
        return input, c_h

class PTBLM(nn.Module):
    def __init__(self, voc_size, rnn_size, layer_num, dropout, weight_range = 0.1):
        super(PTBLM, self).__init__()
        self.emb = nn.Embedding(voc_size, rnn_size)
        self.drop = nn.Dropout(p = dropout)
        self.lstm = LSTM(rnn_size, layer_num, dropout)
        self.linear = nn.Linear(rnn_size, voc_size)
        self.weight_range = weight_range
        self.init_params()
    def init_params(self):
        for param in self.lstm.parameters():
            nn.init.uniform_(param, -self.weight_range, self.weight_range)
    def forward(self, x, c_h):
        x = self.emb(x)
        x = self.drop(x)
        x, c_h = self.lstm(x, c_h) 
        x = self.linear(x)
        x = x.reshape(-1, x.shape[2])
        return x, c_h

def generate(tokens, batch_size, num_of_steps, batch_num):
    batches = []
    for i in range(batch_num):
        batch = []
        for j in range(batch_size):
            batch.append(tokens[i * num_of_steps + j * num_of_steps * batch_num:
                                i * num_of_steps + j * num_of_steps * batch_num + num_of_steps])
        batches.append(batch)
    return np.array(batches)

def batch_generator(tokens, batch_size, num_of_steps):
    batch_num = (len(tokens) - 1) // (batch_size * num_of_steps)
    X_batches = generate(tokens, batch_size, num_of_steps, batch_num)
    y_batches = generate(tokens[1:], batch_size, num_of_steps, batch_num)
    for i in range(X_batches.shape[0]):
        yield X_batches[i].T, y_batches[i].T
        i += 1
def cross_entropy(y, output):
    output = torch.exp(output)
    output = output/output.sum(axis = 1)[:,None]
    probs = output[range(len(y)), y]
    ce = torch.mean(-torch.log(probs))
    return ce
    
def perplexy(generator, model, c_h):
    with torch.no_grad():
        loss = 0
        n = 0
        for (x, y) in generator:
            x = torch.tensor(x, dtype=torch.int64)
            y = torch.tensor(y, dtype=torch.int64)
            y = y.reshape(-1)
            output, c_h = model(x, c_h)
            loss += cross_entropy(y, output)
            n += 1
        perp = np.exp(loss/n)
    return perp
def make_dic(f):
    tokens = []
    for item1 in f:
        for item2 in item1.split():
            tokens.append(item2)
        tokens.append('<eos>')
    s = sorted(set(tokens))
    ind2word = {x:s[x] for x in range(len(s))}
    word2ind = {s[x]:x for x in range(len(s))}
    return ind2word, word2ind

def file_to_token(f, word2ind):
    tokens = []
    for item1 in f:
        for item2 in item1.split():
            tokens.append(item2)
        tokens.append('<eos>')
    for i in range(len(tokens)):
        tokens[i] = word2ind[tokens[i]]
    return tokens


def train(token_list, word_to_id, id_to_word):

    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    l = len(token_list)
    vocab_size = len(word_to_id)
    epoches = 20
    dropout = 0.2
    batch_size = 64
    num_of_steps = 35
    layer_num = 2
    rnn_size = 256
    learning_rate = 0.01
    epoch_decay = 6
    lr_decay = 0.9

    model = PTBLM(vocab_size, rnn_size, layer_num, dropout)
    for epoch in range(epoches):
        gen = batch_generator(token_list, batch_size, num_of_steps)
        c_h = [(torch.zeros((batch_size, rnn_size)),
        torch.zeros((batch_size, rnn_size)))
        for i in range(layer_num)]
        if epoch >= epoch_decay:
            learning_rate *= lr_decay
        optimizer = optim.Adam(model.parameters(), learning_rate)

        model.train()
        for (x, y) in gen:
            x = torch.tensor(x, dtype=torch.int64)
            y = torch.tensor(y, dtype=torch.int64)
            y = y.reshape(-1)
            c_h = [(c_h[i][0].detach(), c_h[i][1].detach()) for i in range(layer_num)]
            optimizer.zero_grad()
            output, c_h = model(x, c_h)
            loss = cross_entropy(y, output)
            loss.backward()
            optimizer.step()
    
        model.eval()
        gen = batch_generator(token_list, batch_size, num_of_steps)
        c_h = [(torch.zeros((batch_size, rnn_size)),
        torch.zeros((batch_size, rnn_size))) for i in range(layer_num)]
        perp1 = perplexy(gen, model, [(torch.zeros((batch_size, rnn_size)),
        torch.zeros((batch_size, rnn_size))) for i in range(layer_num)])
        print("Perplexy on epoch", epoch, ": ", perp1.item())
    
    return model, rnn_size, layer_num

def next_proba_gen(token_gen, params, hidden_state=None):
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    model = params[0]
    rnn_size = params[1]
    layer_num = params[2]
    model.eval()
    with torch.no_grad():
        for batch in token_gen:
            batch_size = batch.shape[0] 
            batch = [[x] for x in batch]
            batch = np.array(batch).T
            batch = torch.tensor(batch, dtype=torch.int64)
            batch_size = batch.shape[0] 
            if (hidden_state == None):
                hidden_state = [(torch.zeros((batch_size, rnn_size)),
                torch.zeros((batch_size, rnn_size))) for i in range(layer_num)]
            output, hidden_state = model(batch, hidden_state)
            output = torch.exp(output)
            probs = output/output.sum(axis = 1)[:,None]
            probs = np.array(probs)
            yield probs, hidden_state

def model_generator(model, ind2word, c_h, temp):
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    #prev = np.random.randint(0, len(ind2word))
    prev = 43
    prev = torch.tensor(prev).flatten()
    l = 0
    model.eval()
    with torch.no_grad():
        while (ind2word[prev.item()] != '<eos>' or l == 0):
            print(ind2word[prev.item()], end = ' ')
            output, c_h = model(prev, c_h)
            output /= temp
            output = torch.exp(output)
            probs = output/output.sum(axis = 1)[:,None]
            prev = random.choices(np.arange(len(ind2word)), probs.flatten())
            prev = torch.tensor(prev).flatten()
            l += 1

