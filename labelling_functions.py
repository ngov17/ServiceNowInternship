from bert_classifier import BERT
import torch
from transformers import BertTokenizer
import numpy as np
from dataset_utils import split_data, datasets
import pickle
import argparse

"""
All labelling functions take in X, a corpus of text, as input and returns 
the corresponding label for each document in the corpus as a np array
"""

def bert_lf(X):
    # load model and tokenizer
    model = BERT(num_classes=2)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model.load_state_dict(torch.load('../SavedModels/toxic_model.pt', map_location=torch.device('cpu')))
    res = []
    for i, x in enumerate(X):
        print(i)
        inp = tokenizer(x, max_length=512, return_tensors="pt")
        out = model(inp)
        res.append(torch.argmax(out))
    res = np.array(res)
    # Accuracy calc:
    acc = np.sum(np.where(res == test_data['label'], 1, 0)) / len(res)
    print(acc)
    res = np.expand_dims(res, axis=1) # Shape: (num_data_points, 1)
    return res

def logistic_tfidf_lf(X):
    model = pickle.load(open('../SavedModels/logistic_reg_tfidf.sav', 'rb'))
    vectorizer = pickle.load(open('../SavedModels/logistic_reg_tfidf_vec.sav', 'rb'))
    X = vectorizer.transform(X.to_list())
    res = model.predict(X)
    res = np.expand_dims(res, axis=1)
    return res

def svm_tfidf_lf(X):
    model = pickle.load(open('../SavedModels/svm_tfidf.sav', 'rb'))
    vectorizer = pickle.load(open('../SavedModels/svm_tfidf_vec.sav', 'rb'))
    X = vectorizer.transform(X.to_list())
    res = model.predict(X)
    res = np.expand_dims(res, axis=1)
    return res


def save_label_matrix(lfs, data, fp):
    arrs = []
    for lf in lfs:
        print("Applying label fuction: ", lf)
        arr = lf(data['text'])
        print(arr.shape)
        arrs.append(arr)
    label_matrix = np.hstack(arrs)
    print(label_matrix.shape)
    with open(fp, 'wb') as f:
        np.save(f, label_matrix)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, help="dataset used for training/testing")
    parser.add_argument('--fp', type=str, help="file path to save label matrix")
    args = parser.parse_args()

    lfs = [bert_lf, logistic_tfidf_lf, svm_tfidf_lf]
    dataset = datasets[args.dataset]()
    train_data, test_data = split_data(dataset, 0.1)
    save_label_matrix(lfs, test_data, args.fp)

# dataset = datasets['toxic']()
# train_data, test_data = split_data(dataset, 0.1)
# with open('preds_lab_model_toxic.npy', 'rb') as f:
#     preds = np.load(f)

# acc = np.sum(np.where(preds == test_data['label'], 1, 0)) / len(preds)
# print(acc)

