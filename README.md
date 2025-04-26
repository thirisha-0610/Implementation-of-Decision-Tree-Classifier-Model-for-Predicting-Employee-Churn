# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Load Data – Import the employee dataset with relevant features and churn labels.

2.Preprocess Data – Handle missing values, encode categorical features, and split into train/test sets.

3.Initialize Model – Create a DecisionTreeClassifier with desired parameters.

4.Train Model – Fit the model on the training data.

5.Evaluate Model – Predict on test data and check accuracy, precision, recall, etc.

6.Visualize & Interpret – Visualize the tree and identify key features influencing churn.

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: THIRISHA A
RegisterNumber: 212223040228
*/
```
```
import pandas as pd
data = pd.read_csv("C:\\Users\\admin\\OneDrive\\Desktop\\Folders\\ML\\DATASET-20250226\\Employee.csv")
data.head()
```
![437670168-567758eb-0f85-441b-ade9-678ddc444918](https://github.com/user-attachments/assets/b884019b-11ff-4105-ae39-13ac41d982d0)
```
data.info()
data.isnull().sum()
data['left'].value_counts()
```

![437670226-d5001b6d-4eb6-413f-afa8-f6e9bc630313](https://github.com/user-attachments/assets/048c0517-6de2-47d1-b585-da0812e8fc0b)
```
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data['salary'] = le.fit_transform(data['salary'])
data.head()
```
![437670270-9bc66997-20e2-4dc7-b3b5-5e3171c16592](https://github.com/user-attachments/assets/59c53867-5de8-4214-b999-ab1300b8c2e2)
```
x=data[['satisfaction_level','last_evaluation','number_project','average_montly_hours','time_spend_company','Work_accident','promotion_last_5years','salary']]
x.head()
```
![437670288-eb7b31b2-d2ea-45a4-b3aa-ebfda6acb53a](https://github.com/user-attachments/assets/c3ad5f96-f66c-4623-b01d-dc0f3272cb7c)
```
y=data['left']

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state =100)
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion='entropy')
dt.fit(x_train,y_train)
y_predict=dt.predict(x_test)

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_predict)
accuracy
```
![437670323-fe3d2d63-08d6-4dd3-9936-e3f2dd580a56](https://github.com/user-attachments/assets/87e543c5-b4f5-4777-a2d7-e25aa47d1932)
```
dt.predict([[0.5,0.8,9,260,6,0,1,2]])
```
![437670427-e0b69b05-af3f-4043-b94f-5fbcb82b7e2b](https://github.com/user-attachments/assets/5e944bc4-db9a-411d-a88d-4f1549d477bc)
```
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree # Import the plot_tree function

plt.figure(figsize=(8,6))
plot_tree(dt,feature_names=x.columns,class_names=['salary','left'],filled=True)
plt.show()
```

![437670402-f76a887e-0cfa-484b-b7a9-4fe5bc89ec36](https://github.com/user-attachments/assets/b4006042-6038-4c1a-9a12-95a562eeaa97)


## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
