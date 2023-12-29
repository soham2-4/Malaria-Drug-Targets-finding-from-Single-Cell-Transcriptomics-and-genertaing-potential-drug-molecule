import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import *
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from tqdm import tqdm
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Set random seeds for consistency
random_seed = 42
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
        X.append(np.array([float(i) for i in l.split(',')[:-1]]))
        Y.append(l.split(',')[-1].strip())
    except:
        continue

X = np.array(X)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=random_seed)
le = LabelEncoder()
Y_train = le.fit_transform(Y_train)
Y_test = le.transform(Y_test)

# Iterate over different values of k
for k_value in range(200, 500, 50):  # Adjust the range and step as needed
    # Feature selection using Mutual Information
    k_best = SelectKBest(score_func=mutual_info_classif, k=k_value)
    X_train_mi = k_best.fit_transform(X_train, Y_train)
    X_test_mi = k_best.transform(X_test)

    # Get the selected feature indices
    selected_feature_indices = k_best.get_support()

    # Get the original feature names
    original_feature_names = [str(i) for i in range(X.shape[1])]

    # Print the selected feature names
    selected_feature_names = [original_feature_names[i] for i, selected in enumerate(selected_feature_indices) if selected]
    f = open('classification_results_k{}.txt'.format(k_value), 'w')
    print('Selected Feature Names for k={}:'.format(k_value))
    print(selected_feature_names)
    f.write(', '.join(selected_feature_names))

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

    for name, model in models:
        model.fit(X_train_mi, Y_train)
        y_pred = model.predict(X_test_mi)
        f.write("\n\nModel Name: {}\n".format(name))
        f.write(classification_report(Y_test, y_pred))
        f.write("Accuracy: {} ".format(accuracy_score(Y_test, y_pred)))
        print("Model Name : {} Model Accuracy: {}".format(name, accuracy_score(Y_test, y_pred)))
    f.close()

