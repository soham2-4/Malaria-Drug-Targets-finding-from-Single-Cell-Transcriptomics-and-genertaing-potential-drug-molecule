import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import *
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier  # Import XGBoost
from tqdm import tqdm
import numpy as np
import sys


# Set random seeds for consistency
random_seed = 42  # You can use any integer as the seed

# Set random seed for NumPy
np.random.seed(random_seed)

# Redirect output to a text file
with open('classification_results.txt', 'w') as f:
    original_stdout = sys.stdout
    sys.stdout = f

    df = pd.read_csv('file_name.csv')

    #df.drop(columns={'Unnamed: 0'}, inplace=True)

    x = df.drop(['stage1'], axis=1)
    y = df['stage1']

    X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=random_seed)

    print('SVM')
    np.random.seed(random_seed) 
    model = SVC(
        C=1.0,
        kernel='rbf',
        gamma='scale',
        degree=3,
        probability=True
    )
    model.fit(X_train, Y_train)
    np.random.seed(random_seed) 
    y_pred = model.predict(X_test)
    print(classification_report(Y_test, y_pred))
    print("Accuracy:", accuracy_score(Y_test, y_pred))
 
    print('\nLR')
    np.random.seed(random_seed) 
    clf = LogisticRegression(
        multi_class='multinomial',  # Use multinomial for multi-class problems
        max_iter=10000,
        solver='lbfgs',  # Choose an appropriate solver
        C=1.0,  # Regularization parameter (adjust as needed)
    )
    clf.fit(X_train, Y_train)
    np.random.seed(random_seed) 
    y_pred_log = clf.predict(X_test)
    print(classification_report(Y_test, y_pred_log))
    print("Accuracy:", accuracy_score(Y_test, y_pred_log))

    print('\nRF')
    np.random.seed(random_seed) 
    RF = RandomForestClassifier(
        n_estimators=100,  # Number of trees
        max_depth=None,  # Max depth of each tree
        min_samples_split=2,  # Minimum samples required to split an internal node
        min_samples_leaf=1,  # Minimum samples required to be at a leaf node
        max_features='auto',  # Number of features to consider for the best split
    )
    RF.fit(X_train, Y_train)
    np.random.seed(random_seed) 
    y_pred_RF = RF.predict(X_test)
    print(classification_report(Y_test, y_pred_RF))
    print("Accuracy:", accuracy_score(Y_test, y_pred_RF))

    print('\nXGBoost')
    np.random.seed(random_seed) 
    xgb_model = XGBClassifier(
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
        num_class=len(df['stage1'].unique())
    )
    xgb_model.fit(X_train, Y_train)
    np.random.seed(random_seed) 
    y_pred_xgb = xgb_model.predict(X_test)
    print(classification_report(Y_test, y_pred_xgb))
    print("Accuracy:", accuracy_score(Y_test, y_pred_xgb))

    # Restore the original stdout
    sys.stdout = original_stdout

print('Classification results saved to classification_results.txt')

