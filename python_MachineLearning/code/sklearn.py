import sklearn
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold

# %%
print(sklearn.__version__)
# %%
iris = load_iris()
iris_data = iris.data
iris_label = iris.target
print('iris target값 :', iris_label)
print('iris target명 :', iris.target_names)

# %%
iris_df = pd.DataFrame(data=iris_data, columns=iris.feature_names)
iris_df['label'] = iris.target
print(iris_df.head())

X_train, X_test, Y_train, Y_test = train_test_split(iris_data, iris_label, test_size=0.2, random_state=11)
# %%
df_clf = DecisionTreeClassifier(random_state=11)
df_clf.fit(X_train, Y_train)

pred = df_clf.predict(X_test)

print('예측 정확도: {0:.4f}'.format(accuracy_score(Y_test, pred)))
# %%
iris = load_iris()
print(type(iris))
kyes = iris.keys()
print(kyes)

# %%
print('feature names type:', type(iris.feature_names))
print('feature names shape:', len(iris.feature_names))
print(iris.feature_names)

print('\ntarget_names type:', type(iris.target_names))
print('target_names shape:', iris.target_names.shape)
print(iris.target_names)

print('\ndata type:', type(iris.data))
print('data shape:', iris.data.shape)
print(iris.data)

print('\ntarget type:', type(iris.target))
print('target shape:', iris.target.shape)
print(iris.target)
# %%
iris = load_iris()
dt_clf = DecisionTreeClassifier()
train_data = iris.data
train_label = iris.target
dt_clf.fit(train_data, train_label)

pred = dt_clf.predict(train_data)
print(accuracy_score(train_label, pred))
# %%
dt_clf = DecisionTreeClassifier()
iris_data = load_iris()

x_train, x_test, y_train, y_test = train_test_split(iris_data.data, iris_data.target, test_size=0.3, random_state=121)
dt_clf.fit(x_train, y_train)
pred = dt_clf.predict(x_test)
print('%0.4f' % (accuracy_score(y_test, pred)))

# %%
iris = load_iris()
features = iris.data
label = iris.target
dt_clf = DecisionTreeClassifier(random_state=156)

kfold = KFold(n_splits=5)
cv_accuracy = []
print('data shape:', features.shape[0])

n_iter = 0
for train_index, test_index in kfold.split(features):
    n_iter += 1
    x_train, x_test = features[train_index], features[test_index]
    y_train, y_test = label[train_index], label[test_index]

    dt_clf.fit(x_train, y_train)
    pred = dt_clf.predict(x_test)

    accuracy = np.round(accuracy_score(y_test, pred), 4)
    train_size = x_train.shape[0]
    test_size = x_test.shape[0]
    print('\n #%d 교차 검증 정확도 : %.1f, 학습 데이터 크기 : %d, 검증 데이터 크기 : %d' % (n_iter, accuracy, train_size, test_size))
    print('\n #%d 검증 세트 인덱스 :' % n_iter, test_index)
    cv_accuracy.append(accuracy)

print('\n ## 평균 검증 정확도:', np.mean(cv_accuracy))

# %%
iris = load_iris()
iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
iris_df['label'] = iris.target
# print(iris_df['label'].value_counts())
kfold = KFold(n_splits=3)
n_iter = 0

for train_index, test_index in kfold.split(iris_df):
    n_iter += 1
    label_train = iris_df['label'].iloc[train_index]
    label_test = iris_df['label'].iloc[test_index]
    print('## 교차 검증 : {0}'.format(n_iter))
    print('학습 레이블 데이터 분포 : \n', label_train.value_counts())
    print('검증 레이블 데이터 분포 : \n', label_test.value_counts())

# %%
skf = StratifiedKFold(n_splits=3)
n_iter = 0

for train_index, test_index in skf.split(iris_df, iris_df['label']):
    n_iter += 1
    label_train = iris_df['label'].iloc[train_index]
    label_test = iris_df['label'].iloc[test_index]
    print('## 교차 인증 : {0}'.format(n_iter))
    print('학습 레이블 데이터 분포 : \n', label_train.value_counts())
    print('검증 레이블 데이터 분포 : \n', label_test.value_counts())
# %%
iris = load_iris()
features = iris.data
label = iris.target
dt_clf = DecisionTreeClassifier(random_state=156)
cv_accuracy = []
skf = StratifiedKFold(n_splits=3)
n_iter = 0

for train_index, test_index in skf.split(features, label):
    n_iter += 1
    x_train, x_test = features[train_index], features[test_index]
    y_train, y_test = label[train_index], label[test_index]

    dt_clf.fit(x_train, y_train)
    pred = dt_clf.predict(x_test)

    accuracy = np.round(accuracy_score(y_test, pred), 4)
    train_size = x_train.shape[0]
    test_size = x_test.shape[0]
    print('\n #%d 교차 검증 정확도 : %.1f, 학습 데이터 크기 : %d, 검증 데이터 크기 : %d' % (n_iter, accuracy, train_size, test_size))
    print('\n #%d 검증 세트 인덱스 :' % n_iter, test_index)
    cv_accuracy.append(accuracy)

print('## 교차 검증별 정확도 :', np.round(cv_accuracy, 4))
print('## 평균 검증 정확도 : ', np.mean(cv_accuracy))
