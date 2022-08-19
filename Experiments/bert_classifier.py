# Libraries
import pandas as pd
import torch
import argparse

# Models
import torch.nn as nn
from transformers import BertTokenizer, BertModel
import torch.optim as optim
# data
from bert_preprocess import load_dataset

# Evaluation
# from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# initialize tokenizer and device
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
device = 'cpu'

# Model parameter
MAX_SEQ_LEN = 128
PAD_INDEX = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
UNK_INDEX = tokenizer.convert_tokens_to_ids(tokenizer.unk_token)

# Model
class BERT(nn.Module):

    def __init__(self, num_classes=2):
        super(BERT, self).__init__()

        self.num_classes = num_classes

        options_name = "bert-base-uncased"
        self.encoder = BertModel.from_pretrained(options_name)
        self.linear = nn.Linear(768, self.num_classes)

    def forward(self, inp):
        encoder_out = self.encoder(**inp)
        CLS_vector = encoder_out.last_hidden_state[0][0]
        logits = self.linear(CLS_vector)

        return logits

"""
Training
"""
hyper_params = {
    'num_epochs': 1,
    'lr': 2e-5
}

def train(model, train_loader, criterion, optimizer, fp):

    model = model.train()
    print("Running training loop...")
    for epoch in range(hyper_params['num_epochs']):
        for i, (inps, lab) in enumerate(train_loader):
            out = model(inps)

            loss = criterion(out, lab)
            print(epoch, i, ": ", loss)
            # update weights
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # bug with using pandas df with torch Dataset; actual length is 1 less
            if i == len(train_loader) - 1:
                break
    print("Saving model...")
    torch.save(model.state_dict(), fp)

def test(model, test_loader):

    num_correct = 0

    for i, (inps, lab) in enumerate(test_loader):
        print(i)
        out = model(inps)
        pred = torch.argmax(out).item()
        label = lab.item()
        if pred == label:
            num_correct += 1
        # bug with using pandas df with torch Dataset; actual length is 1 less
        if i == len(test_loader) - 1:
            break
    accuracy = num_correct / i
    return accuracy

# TODO: add main function for below
if __name__ == "__main__":
    # parse command line arguments (whether train or test)
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--dataset', type=str, help="dataset used for training/testing")
    parser.add_argument('--num_classes', type=int, help="number of classes", default=2)
    parser.add_argument('--model_fp', type=str, help="Model file path (fp) to save or load model. \
    If training, fp used to save. If testing, model state dict loaded from fp.")
    args = parser.parse_args()
    # throw error if not at least one argument entered
    if not (args.test or args.train):
        parser.error('--train or --test required')
    
    # initialize model
    model = BERT(num_classes=args.num_classes)
    
    print("Preprocessing Data...")
    train_loader, test_loader = load_dataset(args.dataset, tokenizer)

    if args.train:
        # train
        optimizer = optim.Adam(model.parameters(), lr=hyper_params['lr'])
        criterion = nn.CrossEntropyLoss()
        train(model, train_loader, criterion, optimizer, args.model_fp)
    if args.test:
        # load saved model and evaluate
        model.load_state_dict(torch.load(args.model_fp, map_location=torch.device('cpu')))
        acc = test(model, test_loader)
        print(acc)