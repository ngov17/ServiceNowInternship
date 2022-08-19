
- dataset_utils.py contains helper functions to retrieve dataset_utils and contains the names of each dataset:
        datasets = {
        'news': get_news_data,
        'news_imbalanced': get_news_data_imbalanced,
        'youtube_spam': get_youtube_spam_data,
        'youtube_spam_imbalanced': get_youtube_data_imbalanced,
        'toxic': get_toxic_data,
        'clinc': get_clinc_data,
        'fraud': get_fraud_data
    }
- to run svm or logistic regression:
    python logistic_regression.py/svm --dataset [dataset_name]
    For ex:
    python logistic_regression.py --dataset news

- to train bert classifier:
    python bert_classifier --train --dataset [dataset_name] --num_classes [number of classes] --model_fp [model filepath to save model]
- to test bert classifier:
    python bert_classifier --test --dataset [dataset_name] --num_classes [number of classes] --model_fp [model filepath to load saved model]

- to save a label matrix, make sure the filepaths to the model are correct in the code and run:
    python labelling_functions.py --dataset [dataset] --fp [file path to save label matrix]

- Once labelling matrix has been created, type the file name in snorkel_functions.py and run:
    python snorkel_functions.py
  to save snorkel predictions based on either MajorityVoteModel or LabelModel (can uncomment appropriate one in code)
  
- to run active learning:
    python active_learning.py
  This will generate a graph showing the active learning perfomance against the size of the train set