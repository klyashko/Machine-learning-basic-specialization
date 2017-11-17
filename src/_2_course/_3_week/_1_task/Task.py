import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction import DictVectorizer as DV
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


def write_answer_5(auc):
    with open("preprocessing_lr_answer5.txt", "w") as fout:
        fout.write(str(auc))


data = pd.read_csv('data/data.csv')
X = data.drop('Grant.Status', 1)
y = data['Grant.Status']

numeric_cols = ['RFCD.Percentage.1', 'RFCD.Percentage.2', 'RFCD.Percentage.3',
                'RFCD.Percentage.4', 'RFCD.Percentage.5',
                'SEO.Percentage.1', 'SEO.Percentage.2', 'SEO.Percentage.3',
                'SEO.Percentage.4', 'SEO.Percentage.5',
                'Year.of.Birth.1', 'Number.of.Successful.Grant.1', 'Number.of.Unsuccessful.Grant.1']
categorical_cols = list(set(X.columns.values.tolist()) - set(numeric_cols))

X_real_zeros = X[numeric_cols].fillna(0)

X_cat = X[categorical_cols].fillna('NA').astype(str)

encoder = DV(sparse=False)
X_cat_oh = encoder.fit_transform(X_cat.T.to_dict().values())

(X_train_real_zeros, X_test_real_zeros, y_train, y_test) = train_test_split(X_real_zeros, y, test_size=0.3, random_state=0, stratify=y)
(X_train_cat_oh, X_test_cat_oh) = train_test_split(X_cat_oh, test_size=0.3, random_state=0, stratify=y)

scaler = StandardScaler()

X_train_real_zeros = scaler.fit_transform(X_train_real_zeros)
X_test_real_zeros = scaler.transform(X_test_real_zeros)

# place your code here
transform = PolynomialFeatures(2)
# X_train = np.hstack((X_train_real_zeros, X_train_cat_oh))
# X_test = np.hstack((X_test_real_zeros, X_test_cat_oh))

X_train = X_train_real_zeros
X_test = X_test_real_zeros

X_train = transform.fit_transform(X_train)
X_test = transform.transform(X_test)

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train = np.hstack((X_train, X_train_cat_oh))
X_test = np.hstack((X_test, X_test_cat_oh))

param_grid = {'C': [0.01, 0.05, 0.1, 0.5, 1, 5, 10]}
cv = 3

grid_cv = GridSearchCV(LogisticRegression(fit_intercept=False, class_weight='balanced'), param_grid=param_grid, cv=cv)
grid_cv.fit(X_train, y_train)

roc_score = roc_auc_score(y_test, grid_cv.predict_proba(X_test)[:, 1])
print(roc_score)

write_answer_5(roc_score)
