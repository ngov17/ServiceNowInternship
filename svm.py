from sklearn import svm
from dataset_utils import vectorize_data, split_data, datasets
import pickle
from sklearn.model_selection import train_test_split
import argparse

def train_nontext_data(features, labels):
    model = svm.SVC()
    model.fit(X=features, y=labels)

    print("Saving Model...")
    pickle.dump(model, open('../SavedModels/svm_fraud.sav', 'wb'))

    return model

def train(df, train_data):
    vectorizer = vectorize_data(df)
    X_train = vectorizer.transform(train_data['text'].to_list())
    model = svm.SVC()
    model.fit(X=X_train, y=train_data['label'])

    # Save model and vectorizer
    pickle.dump(model, open('../SavedModels/svm_tfidf.sav', 'wb'))
    pickle.dump(vectorizer, open('../SavedModels/svm_tfidf_vec.sav', 'wb'))

    return model, vectorizer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, help="dataset used for training/testing")
    args = parser.parse_args()
    # Get data
    df = datasets[args.dataset]()

    # split data into train/test sets
    train_data, test_data = split_data(df, train_ratio=0.1)

    print(len(train_data), len(test_data))

    model, vectorizer = train(df, train_data)
    X_test = vectorizer.transform(test_data['text'].to_list())

    # test model
    acc = model.score(X_test, test_data['label'])
    print(acc)

    # features, labels = datasets['fraud']()
    # f_train, f_test, l_train, l_test = train_test_split(
    #     features, labels, test_size=0.9, random_state=123, stratify=labels
    # )
    # print(len(f_train), len(f_test), len(l_train), len(l_test))
    
    # model = train_nontext_data(f_train, l_train)
    # acc = model.score(f_test, l_test)
    # print(acc)