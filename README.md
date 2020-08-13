# Machine Learning models using Ensemble modelling

import pandas as pd
import numpy as np

import seaborn as sns
#grid search is used to find the optimal paramters for each model
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
#Importing KNN, Logistic and Rf classifier from sklearn library
from sklearn.neighbors import KNeighborsClassifier
#Voting Classifier is used to make the prediction by majority vote
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

# Reading the csv dataset
df = pd.read_csv('C:/Users/AWU/Desktop/CSV Files/diabetes.csv')

# Returning first few observations
df.head()

#dataset Size
df.shape

#List of Column Names
df.columns.tolist()

#Data-types of each column
df.dtypes

# Returning no. of missing values in each column
df.isnull().sum()

df.info()

df.describe()

# Replacing 0 by NaN

df[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']] = df[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']].replace(0,np.NaN)

# Re-checking no. of missing values in each column 

df.isnull().sum()

# From the above we found that there are 5 variables having misssing values (Glucose,BloodPressure,SkinThickness,Insulin,BMI)
# Filling NaN/Missing values 

df['Glucose'].fillna(df['Glucose'].median(), inplace = True)
df['BloodPressure'].fillna(df['BloodPressure'].median(), inplace = True)
df['SkinThickness'].fillna(df['SkinThickness'].median(), inplace = True)
df['Insulin'].fillna(df['Insulin'].median(), inplace = True)
df['BMI'].fillna(df['BMI'].mean(), inplace = True)

# Checking missing values correctly replaced

df.isnull().sum()

# Finding corelation among the dataset
df.corr()

# Getting unique values 

df['Pregnancies'].unique()

# Finding counts of unique values and sorting it in ascending order

df['Pregnancies'].value_counts().sort_values()

#Plotting a graph based on the Outcome Column
sns.countplot(x='Outcome', data=df)
plt.show()

#Count of Diabetes and Non-Diabetes patients 
diabetes_count = len(df.loc[df['Outcome'] == 1])
no_diabetes_count=len(df.loc[df['Outcome']==0])
(diabetes_count, no_diabetes_count)

#Plotting graphs for all the parameters except the Outcome ie.,target parameter

cols=['Pregnancies','Glucose','BloodPressure','SkinThickness',
      'Insulin','BMI','DiabetesPedigreeFunction','Age']
num=df[cols]
for i in num.columns:
    plt.hist(num[i])
    plt.title(i)
    plt.show()

#here my target column is outcome hence i am splitting the dataset into two where in one the input(X) and the other target(Y)
X = df.drop(columns = ['Outcome'])
y = df['Outcome']

#Training the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y)

#Performing KNN algorithm for the diabetes dataset
knn = KNeighborsClassifier()

#create a dictionary of all values we want to test for n_neighbors
params_knn = {'n_neighbors': np.arange(1, 25)}

#useing gridsearch to test all values for n_neighbors
knn_gs = GridSearchCV(knn, params_knn, cv=5)

#fit model to training data
knn_gs.fit(X_train, y_train)

#save best model
knn_best = knn_gs.best_estimator_

#check best n_neigbors value
print(knn_gs.best_params_)

#Performing Random Forest Algorithm for the diabetes dataset 
rf = RandomForestClassifier()

#create a dictionary of all values we want to test for n_estimators
params_rf = {'n_estimators': [50, 100, 200]}

#useing gridsearch to test all values for n_estimators
rf_gs = GridSearchCV(rf, params_rf, cv=5)

#fit model to training data
rf_gs.fit(X_train, y_train)

#save best model
rf_best = rf_gs.best_estimator_

#check best n_estimators value
print(rf_gs.best_params_)

#Performing logistic regression Algorithm to the diabetes dataset
log_reg = LogisticRegression()

#fit the model to the training data
log_reg.fit(X_train, y_train)

#Testing the accuracy for all the three models for the diabetes dataset
print('knn: {}'.format(knn_best.score(X_test, y_test)))
print('rf: {}'.format(rf_best.score(X_test, y_test)))
print('log_reg: {}'.format(log_reg.score(X_test, y_test)))

#Performing Ensembly Modelling for the diabetes dataset

#create a dictionary of our models
estimators=[('knn', knn_best), ('rf', rf_best), ('log_reg', log_reg)]

#create our voting classifier, inputting our models
ensemble = VotingClassifier(estimators, voting='hard')

#fit model to training data
ensemble.fit(X_train, y_train)

#test our model on the test data
ensemble.score(X_test, y_test)

#Question 1
#Display accuracies for different values of n in KNN Classifier 

a_index=list(range(1,11))
a=pd.Series()
x=[0,1,2,3,4,5,6,7,8,9,10]
for i in list(range(1,11)):
    model=KNeighborsClassifier(n_neighbors=i) 
    model.fit(X_train,y_train)
    prediction=model.predict(X_test)
    a=a.append(pd.Series(metrics.accuracy_score(prediction,y_test)))
plt.plot(a_index, a)
plt.xticks(x)
plt.show()
print('Accuracies for different values of n are:',a.values)

# Question 2
# Display number of rows with respect to the columns

print("Total number of rows : {0}".format(len(df)))
print("Number of rows with 0 Pregnancies: {0}".format(len(df.loc[df['Pregnancies'] == 0])))
print("Number of rows with 0 Glucose: {0}".format(len(df.loc[df['Glucose'] == 0])))
print("Number of rows with 0 BloodPressure: {0}".format(len(df.loc[df['BloodPressure'] == 0])))
print("Number of rows with 0 SkinThickness: {0}".format(len(df.loc[df['SkinThickness'] == 0])))
print("Number of rows with 0 Insulin: {0}".format(len(df.loc[df['Insulin'] == 0])))
print("Number of rows with 0 BMI: {0}".format(len(df.loc[df['BMI'] == 0])))
print("Number of rows with 0 DiabetesPedigreeFunction: {0}".format(len(df.loc[df['DiabetesPedigreeFunction'] == 0])))
print("Number of rows with 0 Ages: {0}".format(len(df.loc[df['Age'] == 0])))
