# Supress warnings
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np

# from snorkel.labeling.model import MajorityLabelVoter
# with open('label_matrix_toxic.npy', 'rb') as f:
#     L_train = np.load(f)
# majority_model = MajorityLabelVoter(cardinality=2) # chooses label based on majority vote from LFs
# preds_train = majority_model.predict(L=L_train)
# print(len(preds_train))
# with open('preds_maj_toxic.npy', 'wb') as f:
#     np.save(f, preds_train)
# exit(0)

from snorkel.labeling.model import LabelModel

with open('label_matrix_toxic.npy', 'rb') as f:
    L_train = np.load(f)
label_model = LabelModel(cardinality=2, verbose=True)
label_model.fit(L_train=L_train, n_epochs=500, log_freq=100, seed=123)

preds_train = label_model.predict(L_train)

with open('preds_lab_model_toxic.npy', 'wb') as f:
    np.save(f, preds_train)

exit(0)
