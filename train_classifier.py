import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np


data_dict = pickle.load(open('./data.pickle', 'rb'))

raw_data = data_dict['data']
raw_labels = data_dict['labels']

filtered_samples = [(d, l) for d, l in zip(raw_data, raw_labels) if len(d) == 42]

if not filtered_samples:
    raise ValueError('No samples with 42 features were found. Check data preprocessing.')

data, labels = map(np.asarray, zip(*filtered_samples))

x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

model = RandomForestClassifier()

model.fit(x_train, y_train)

y_predict = model.predict(x_test)

score = accuracy_score(y_predict, y_test)

print('{}% of samples were classified correctly !'.format(score * 100))

f = open('model.p', 'wb')
pickle.dump({'model': model}, f)
f.close()
