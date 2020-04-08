import pandas as pd
titanic_df = pd.read_csv('./python_MachineLearning/dataset/titanic/titanic_train.csv')
# %%

print(titanic_df.head())
print(type(titanic_df))

print('Data Shape : ' + str(titanic_df.shape))
# %%
print(titanic_df.info())
# %%
print(titanic_df.describe())
# %%
value_count = titanic_df['Pclass'].value_counts()
print(value_count)
print(type(value_count))
# %%
value = titanic_df.values
print(value)
value_tolist = titanic_df.values.tolist()
print(value_tolist)
value_to_dict = titanic_df.to_dict('list')
print(value_to_dict)

# %%
titanic_df['Age_0'] = 0
print(titanic_df.head())
titanic_df['Age_by_10'] = titanic_df['Age'] * 10
titanic_df['Family_No'] = titanic_df['SibSp'] + titanic_df['Parch'] + 1

# %%
titanic_df.drop('Age_0', axis=1, inplace=True)

# %%
titanic_df.drop(['Age_by_10', 'Family_No'], axis=1, inplace=True)

# %%
indexes = titanic_df.index
print('Index 객체 array 값:\n' + str(indexes.values))

# %%
series_fair = titanic_df['Fare']
print('max value :', series_fair.max())
print('sum value :', series_fair.sum())
print('sum() fair series :', sum(series_fair))
print('Fair Series + 3:\n', (series_fair + 3).head())

# %%
print(titanic_df[titanic_df['Pclass'] == 3].head())

# %%
# titanic_sorted = titanic_df.sort_values(by=['Name'])
titanic_sorted = titanic_df.sort_values(by=['Pclass', 'Name'], ascending=False )

print(titanic_sorted)

# %%
titanic_df['Age'].fillna(titanic_df['Age'].mean(), inplace=True)
titanic_df['Embarked'].fillna('S', inplace=True)
print(titanic_df.isna().sum())

