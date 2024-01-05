import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import *
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from tqdm import tqdm
import numpy as np
import sys
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder

# Set random seeds for consistency
random_seed = 42  # You can use any integer as the seed

# Set random seed for NumPy
np.random.seed(random_seed)

fp = open("file_name.csv", 'r')
no_lines = sum([1 for l in tqdm(fp)])
fp = open("file_name.csv", 'r')
lines = [fp.readline() for i in tqdm(range(1, no_lines + 1))]
print(no_lines, len(lines))

X = []
Y = []
for l in tqdm(lines):
    try:
        data_point = [float(i) for i in l.split(',')[:-1]]
        # Randomly select 400 features
        selected_features = np.random.choice(data_point, size=700, replace=False)
        X.append(selected_features)
        Y.append(l.split(',')[-1].strip())
    except:
        continue

X = np.array(X)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=random_seed)
le = LabelEncoder()
Y_train = le.fit_transform(Y_train)
Y_test = le.transform(Y_test)
clf_1 = LogisticRegression(multi_class='multinomial', max_iter=10000, solver='lbfgs', C=1.0)
clf_2 = RandomForestClassifier(n_estimators=100, max_depth=None, min_samples_split=2, min_samples_leaf=1,
                                max_features='auto')
clf_3 = XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    min_child_weight=1,
    subsample=0.8,
    colsample_bytree=0.8,
    gamma=0,
    reg_lambda=1,
    objective='multi:softmax',
    eval_metric='mlogloss',
    num_class=len(set(Y))
)

models = [("Logistic Regression", clf_1), ("RandomForestClassifier", clf_2), ("XG", clf_3)]
f = open('classification_results_random700.txt', 'w')
for name, model in models:
    model.fit(X_train, Y_train)
    y_pred = model.predict(X_test)
    f.write("\n\nModel Name: {}\n".format(name))
    f.write(classification_report(Y_test, y_pred))
    f.write("Accuracy: {} ".format(accuracy_score(Y_test, y_pred)))
    print("Model Name : {} Model Accuracy: {}".format(name, accuracy_score(Y_test, y_pred)))

