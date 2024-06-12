#importing libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

#Loading dataset
data = pd.read_csv("spam.csv", encoding='latin-1')

print(data.head())
print("Shape of dataset :", data.shape)
print(data.describe())
print("Information about dataset :\n", data.info)
print("Columns of dataset :", data.columns)

#Cleaning Dataset
print("Null Values in dataset :\n", data.isnull().sum())
data.drop(columns={'Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'}, inplace=True)
print(data.head())
print("Shape of dataset after removing null values :", data.shape)
print("Columns after removing useless columns :", data.columns)
print(data.isnull().sum())
print("Duplicate Values in dataset :", data.duplicated().sum())
data.drop_duplicates(inplace=True)
print("Duplicate Values in dataset after removing duplicates :", data.duplicated().sum())


#Renaming remaining columns
data.rename(columns={'v1': 'Category', 'v2': 'SMS'}, inplace=True)
print(data.head())
print("Columns after changing name:", data.columns)

#Labeling
data.loc[data['Category'] == 'spam','Category']=0
data.loc[data["Category"] == 'ham','Category']=1

X = data['SMS']
y = data['Category']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=10)

feature_extraction = TfidfVectorizer(min_df = 1, stop_words='english', lowercase=True)

X_trian_features = feature_extraction.fit_transform(X_train)
X_test_features = feature_extraction.transform(X_test)

y_train = y_train.astype('int')
y_test = y_test.astype('int')


#Creating and training model
model = LogisticRegression()
model.fit(X_trian_features, y_train)

#Predictions
prediction_on_test_data = model.predict(X_test_features)
print("Accuracy score :", accuracy_score(y_test, prediction_on_test_data))


input = ["WINNER!! As a valued network customer you have been selected to receivea �900 prize reward! To claim call 09061701461. Claim code KL341. Valid 12 hours only."]
input_sms = feature_extraction.transform(input)

prediction = model.predict(input_sms)
print(prediction)

if prediction == 0:
    print("Spam")
else:
    print("Ham")








