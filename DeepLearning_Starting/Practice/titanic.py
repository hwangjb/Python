import pandas as pd

train = pd.read_csv('D:/ProgramDir/GIT/Python/DeepLearning_Starting/dataset/train.csv')
test = pd.read_csv('D:/ProgramDir/GIT/Python/DeepLearning_Starting/dataset/test.csv')
# %%
print(train.head())

# %%
print(test.head())

# %%
print(train.shape)
print(train.info())
print(test.shape)
print(test.info())

# %%
print(train.isnull().sum())
print(test.isnull().sum())

# %%
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()


# %%
def bar_chart(feature):
    survived = train[train['Survived'] == 1][feature].value_counts()
    dead = train[train['Survived'] == 0][feature].value_counts()
    df = pd.DataFrame([survived, dead])
    df.index = ['Survived', 'Dead']
    df.plot(kind='bar', stacked=True)
    plt.show()


# %%
bar_chart('Sex')

# %%
bar_chart('Pclass')

# %%
bar_chart('SibSp')

# %%
bar_chart('Parch')

# %%
bar_chart('Embarked')

# %%
train_test_data = [train, test]
for dataset in train_test_data:
    dataset['Title'] = dataset['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
# %%
print(train['Title'].value_counts())

# %%
title_mapping = {"Mr": 0, "Miss": 1, "Mrs": 2, "Master": 3, "Dr": 3, "Rev": 3, "Col": 3, "Major": 3, "Mlle": 3, "Countess": 3, "Ms": 3, "Lady": 3, "Jonkheer": 3, "Don": 3, "Dona": 3, "Mme": 3, "Capt": 3, "Sir": 3}
for dataset in train_test_data:
    dataset['Title'] = dataset['Title'].map(title_mapping)
# %%
print(train.head())

# %%
bar_chart('Title')
# %%
train.drop('Name', axis=1, inplace=True)
test.drop('Name', axis=1, inplace=True)
# %%
print(train.head())
