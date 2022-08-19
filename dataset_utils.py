import pandas as pd
import numpy as np
import glob
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
import json

"""
Generates imbalanced data favouring imbalance_class according to imbalance ratio
(binary case)
"""
def generate_imbalanced_data(df, no_data_points=1000, imbalance_ratio=0.8, imbalance_class=1):
    other_class = 1 - imbalance_class
    counter = {1: 0, 0: 0}
    no_imbalance_class = int(no_data_points * imbalance_ratio)
    no_other_class = no_data_points - no_imbalance_class
    print(no_imbalance_class, no_other_class)
    data = defaultdict(list)
    for i in range(len(df)):
        lab = df['label'].iloc[i]
        if lab == imbalance_class:
            if counter[lab] < no_imbalance_class:
                data['text'].append(df['text'].iloc[i])
                data['label'].append(lab)
                counter[lab] += 1
        else:
            if counter[lab] < no_other_class:
                data['text'].append(df['text'].iloc[i])
                data['label'].append(lab)
                counter[lab] += 1
        if counter[imbalance_class] == no_imbalance_class and counter[other_class] == no_other_class:
            break
        
    return pd.DataFrame.from_dict(data)

"""
: df - daaframe to split data on
: param train_test_ratio - percentage of data used for training. Remaining used for test. No val set
"""
def split_data(df, train_ratio):
    _l = len(df)
    # idx_train = int(train_test_split * _l)

    # train_data = dataset[0:idx_train]
    # train_data = train_data.reset_index(drop=True)
    # test_data = dataset[idx_train:]
    # test_data = test_data.reset_index(drop=True)
    test_size = _l - int(train_ratio * _l)
    df_train, df_test = train_test_split(
        df, test_size=test_size, random_state=123, stratify=df.label
    )
    df_train = df_train.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)
    return df_train, df_test

def save_data_stats(df_train, fn, df_test=None, train_test_split=0.1):
    if df_test is None:
        total = len(df_train)
        no_train = int(train_test_split * len(df_train))
        no_test = len(df_train) - no_train
        label_dist = dict(df_train['label'].value_counts())
    else:
        pass
    stats = {
        "Total Number of Data points: " : total,
        "Numer of training examples: " : no_train,
        "Numer of test examples: " : no_test,
        "Label distribution: " : label_dist
    }
    with open(fn,"w") as f:
        for k, v in stats.items():
            f.write(k)
            f.write(" : ")
            f.write(str(v))
            f.write("\n")
"""
Vectorize data
:param method - 'tfidf' or 'count'
"""
def vectorize_data(df, method='tf-idf'):

    if method == 'count':
        # count vectorizer with unigram and bigram 
        vectorizer = CountVectorizer(ngram_range=(1, 5))
    else:
        #tf-idf vectorizer
        vectorizer=TfidfVectorizer(use_idf=True, max_df=0.95)
    vectorizer.fit(df['text'].to_list())
    return vectorizer

"""
Data Retrieval functions
"""
def get_fraud_data():
    data = pd.read_csv('../data/creditcard.csv')
    features = data.iloc[:, :-1]
    features = features.iloc[:, 1:]
    labels = data.iloc[:, -1]
    
    return features, labels

def get_clinc_data():
    with open("../data/data_full.json") as f:
        data = json.load(f)
    
    df = defaultdict(list) # dict dataframe (converted later)
    labels = {} #label to id
    i = 0 # keeps track of id for label
    for d in data['train']:
        text, label = d[0], d[1]
        df['text'].append(text)
        if label not in labels:
            df['label'].append(i)
            labels[label] = i
            i += 1
        else:
            df['label'].append(labels[label])
    
    for d in data['train']:
        text, label = d[0], d[1]
        df['text'].append(text)
        if label not in labels:
            df['label'].append(i)
            labels[label] = i
            i += 1
        else:
            df['label'].append(labels[label])
    
    df = pd.DataFrame.from_dict(df)
    return df

def get_news_data():
    news_data = pd.read_csv('../data/news.csv')
    # 1 - REAL, 0 - FAKE
    news_data['label'] = news_data['label'].map(lambda x: 1 if x == 'REAL' else 0)
    return news_data

def get_news_data_imbalanced():
    data = get_news_data()
    return generate_imbalanced_data(data, no_data_points=3000)

def get_youtube_spam_data():
    filenames = sorted(glob.glob("./../data/Youtube*.csv"))
    dfs = []
    for i, filename in enumerate(filenames, start=1):
        df = pd.read_csv(filename)
        # Lowercase column names
        df.columns = map(str.lower, df.columns)
        # Remove comment_id field
        df = df.drop("comment_id", axis=1)
        # Add field indicating source video
        df["video"] = [i] * len(df)
        # Rename fields
        df = df.rename(columns={"class": "label", "content": "text"})
        # Shuffle order
        df = df.sample(frac=1, random_state=123).reset_index(drop=True)
        dfs.append(df)

    df_train = pd.concat(dfs[:4])
    # 1 - HAM, 0 - SPAM
    df_train['label'] = df_train["label"].map(lambda x: 1 if x == 0 else 0)
    return df_train

def get_youtube_data_imbalanced():
    data = get_youtube_spam_data()
    return generate_imbalanced_data(data, no_data_points=900)

def get_toxic_data():
    data = pd.read_csv('../data/train.csv')
    toxic_data = defaultdict(list)
    for i in range(5000):
        d = data.iloc[i]
        toxic_data['text'].append(d['comment_text'])
        toxic_data['label'].append(d['toxic'])
    return pd.DataFrame.from_dict(toxic_data)

"""
Map to retrieve relevant data from dataset
"""
datasets = {
    'news': get_news_data,
    'news_imbalanced': get_news_data_imbalanced,
    'youtube_spam': get_youtube_spam_data,
    'youtube_spam_imbalanced': get_youtube_data_imbalanced,
    'toxic': get_toxic_data,
    'clinc': get_clinc_data,
    'fraud': get_fraud_data
}

data = get_toxic_data()
save_data_stats(data, 'toxic_comment_stats.txt')