# %%
from sklearn.base import BaseEstimator
import numpy as np
import pandas as pd
from python_MachineLearning.code.Titanic import transform_feature
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# 정확도
# %%
class MyDummyClassifier(BaseEstimator):
    def fit(self, x, y=None):
        pass

    def predict(self, x):
        pred = np.zeros((x.shape[0], 1))
        for i in range(x.shape[0]):
            if x['Sex'].iloc[i] == 1:
                pred[i] = 0
            else:
                pred[i] = 1

        return pred


titanic_df = pd.read_csv('./python_MachineLearning/dataset/titanic/titanic_train.csv')
y_titanic_df = titanic_df['Survived']
x_titanic_df = titanic_df.drop('Survived', axis=1)
x_titanic_df = transform_feature(x_titanic_df)

x_train, x_test, y_train, y_test = train_test_split(x_titanic_df, y_titanic_df, test_size=0.2, random_state=0)

myclf = MyDummyClassifier()

myclf.fit(x_train, y_train)

mypredictions = myclf.predict(x_test)

print('Dummy Classifier의 정확도 : {0:.4f}'.format(accuracy_score(y_test, mypredictions)))

# %%
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd


class MyFakeClassifier(BaseEstimator):
    def fit(self, x, y):
        pass

    def predict(self, x):
        return np.zeros((len(x), 1), dtype=bool)


digits = load_digits()

y = (digits.target == 7).astype(int)
x_train, x_test, y_train, y_test = train_test_split(digits.data, y, random_state=11)

print('레이블 테스트 세트 크기 :', y_test.shape)
print('테스트 세트 레이블 0과 1의 분포도')
print(pd.Series(y_test).value_counts())

fakeclf = MyFakeClassifier()
fakeclf.fit(x_train, y_train)
fakepred = fakeclf.predict(x_test)

print('모든 예측을 0으로 하여도 정확도는 : {0:.3f}'.format(accuracy_score(y_test, fakepred)))

# 오차 행렬
# %%

from sklearn.metrics import confusion_matrix

print(confusion_matrix(y_test, fakepred))

# 정밀도와 재현율
# %%

from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix


def get_clf_eval(y_test, pred):
    confusion = confusion_matrix(y_test, pred)
    accuracy = accuracy_score(y_test, pred)
    precision = precision_score(y_test, pred)
    recall = recall_score(y_test, pred)

    print('오차 행렬')
    print(confusion)
    print('정확도 : {0:.4f}, 정밀도 : {1:.4f}, 재현율 : {2:.4f}'.format(accuracy, precision, recall))


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

titanic_df = pd.read_csv('./python_MachineLearning/dataset/titanic/titanic_train.csv')
y_titanic_df = titanic_df['Survived']
x_titanic_df = titanic_df.drop('Survived', axis=1)
x_titanic_df = transform_feature(x_titanic_df)

x_train, x_test, y_train, y_test = train_test_split(x_titanic_df, y_titanic_df, test_size=0.20, random_state=11)

lr_clf = LogisticRegression()
lr_clf.fit(x_train, y_train)
pred = lr_clf.predict(x_test)
get_clf_eval(y_test, pred)

# %%
