# %%
from sklearn.base import BaseEstimator
import numpy as np
import pandas as pd
from python_MachineLearning.code.Titanic import transform_feature
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


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