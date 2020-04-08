import pandas as pd
# %%
titanic_df = pd.read_csv('./python_MachineLearning/dataset/titanic/titanic_train.csv')
print(titanic_df.head())
print(type(titanic_df))
