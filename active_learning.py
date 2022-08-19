from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from dataset_utils import vectorize_data, split_data, datasets, save_data_stats
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Active Learning Hyper Parameters
k = 15
num_iter = 10

df = datasets['youtube_spam']()
    
# split data into train/test sets and vectorize
val_data, test_data = split_data(df, train_ratio=0.1)
print(len(val_data), len(test_data))
vectorizer = vectorize_data(df)
X_test = vectorizer.transform(test_data['text'].to_list())

# sample k datapoints to start
train_set = val_data.sample(n=k)
val_data = val_data.drop(train_set.index)

# initialize model
# ensemble model - Logistic, Random Forest, SVM, soft voting
models = [('lr',LogisticRegression()),('svm',svm.SVC(probability=True)), ('rf', RandomForestClassifier(n_estimators=50, random_state=1))]
model = VotingClassifier(estimators=models, voting='soft')
# Logistic Regression Model
# model = LogisticRegression(C=1e3, solver="liblinear")

# Selection functions
def margin_sampling(probas_val):
    rev = np.sort(probas_val, axis=1)[:, ::-1]
    values = rev[:, 0] - rev[:, 1]
    selection = np.argsort(values)[:k]
    return selection

# For plotting graph
accuracies = []
sizes = []
acc = 0
while acc < 0.9131652661064426 and len(val_data) >= k:
    # Append to sizes
    sizes.append(len(train_set))

    print( "Size of train set: ", len(train_set), "Size of val set: ", len(val_data))
    X_train = vectorizer.transform(train_set['text'].to_list())
    # train model
    model.fit(X=X_train, y=train_set['label'])
    # score model
    acc = model.score(X_test, test_data['label'])
    print("Accuracy: ", acc)
    # Append to accuracies
    accuracies.append(acc)

    # get probabilities on validation set
    vec_val_data = vectorizer.transform(val_data['text'].to_list())
    val_probs = model.predict_proba(vec_val_data)
    indices = margin_sampling(val_probs)
    
    # choose k points to retrain based on selection function
    new_set = val_data.iloc[indices]
    val_data = val_data.drop(new_set.index)
    # concatenate to original
    train_set = pd.concat([train_set, new_set])

# plot graph
plt.plot(sizes, accuracies)
plt.xlabel("Size of train set")
plt.ylabel("Accuracy")
plt.show()