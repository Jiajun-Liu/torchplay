
##########################################################################
# reference: 
# https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html
##########################################################################

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)
def get_correct_nums(preds,labels):
    return preds.argmax(dim=1).eq(labels).sum().item()


training_data = [
    ("The dog ate the apple".split(), ["DET", "NN", "V", "DET", "NN"]),
    ("Everybody read that book".split(), ["NN", "V", "DET", "NN"])
]

word_to_ix = {}
for sent, tags in training_data:
    for word in sent:
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)
tag_to_ix = {"DET": 0, "NN": 1, "V": 2}
EMBEDDING_DIM = 6
HIDDEN_DIM = 6

class myLSTM(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
        super(myLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        
        # https://www.cnblogs.com/picassooo/p/12535916.html
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores


model = myLSTM(EMBEDDING_DIM, HIDDEN_DIM, len(word_to_ix), len(tag_to_ix))
loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

for epoch in range(300):  # again, normally you would NOT do 300 epochs, it is toy data
    for sentence, tags in training_data:
        model.zero_grad()
        sentence_in = prepare_sequence(sentence, word_to_ix)
        targets = prepare_sequence(tags, tag_to_ix)
        tag_scores = model(sentence_in)
        loss = loss_function(tag_scores, targets)
        loss.backward()
        optimizer.step()

test_data = ("Everybody ate the apple".split(),["NN","V","DET","NN"])
test_in = prepare_sequence(test_data[0],word_to_ix)
test_targets = prepare_sequence(test_data[1],tag_to_ix)
test_tag_scores = model(test_in)
num_corrects = get_correct_nums(test_tag_scores,test_targets)
print("accurancy: ",num_corrects/4)

# final results: acc 1