import sklearn
# %%
print(sklearn.__version__)
# %%
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
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

from sklearn.metrics import accuracy_score
print('예측 정확도: {0:.4f}'.format(accuracy_score(Y_test, pred)))
