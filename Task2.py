#importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


#Loading Data
data = pd.read_csv("train.csv")
print(data.head())
print("Shape of dataset :", data.shape)
print(data.describe())
print(data.info())



#Cleaning data
print("Null values in dataset :\n", data.isnull().sum())

#Flling null values of Age
data['Age'].fillna(data['Age'].mean(), inplace=True)
print(data.info())

data.drop(columns={'Cabin'}, inplace=True)

print(data['Embarked'].mode()[0])

data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)

print("Null values after replacing null values('Age') :\n", data.isnull().sum())

print("Shape of dataset after removing null values :", data.shape)
print(data.describe())
print(data.info())

print("Duplicate values in dataset :", data.duplicated().sum())

#Analysing Data
print(data['Survived'].value_counts())

# sns.countplot(x = 'Survived', data=data)
# plt.show()

print(data['Sex'].value_counts())

# sns.countplot(x='Sex', data=data)
# plt.show()

# sns.countplot(x='Sex', hue='Survived', data=data)
# plt.show()

# sns.countplot(x='Pclass', data=data)
# plt.show()

# sns.countplot(x='Pclass', hue='Survived', data=data)
# plt.show()

print(data['Embarked'].value_counts())

#Encoding
data.replace({'Sex':{'male': 0, 'female': 1}, 'Embarked': {'S': 0, 'C': 1, 'Q': 2}}, inplace=True)
print(data.describe)


X = data.drop(columns=["PassengerId", "Name", "Ticket", "Survived"], axis=1)
y = data['Survived']

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state=2)


model = LogisticRegression()
model.fit(X_train, y_train)

X_train_prediction = model.predict(X_train)
print("Accuracy score of train data :", accuracy_score(y_train, X_train_prediction))

X_test_prediction = model.predict(X_test)
print("Accuracy of test data :", accuracy_score(y_test, X_test_prediction))


print(data.head(10))
print(data.tail(10))
print(data['Parch'].value_counts())
print(data['SibSp'].value_counts())
print(data['Fare'].value_counts())
print(data['Age'].value_counts())

new_data = pd.DataFrame({"Pclass": [3], "Sex": [0], "Age": [45], "SibSp": [1], "Parch": [0], "Fare": [200], "Embarked": [1]})
prediction = model.predict(new_data)
prediction[0]

if prediction[0] == 1:
    print("SURVIVED")
else:
    print("NOT SURVIVED")

