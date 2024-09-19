# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the standard libraries.
2. Upload the dataset and check for any null values using .isnull() function.
3. Import LabelEncoder and encode the dataset.
4. Import DecisionTreeRegressor from sklearn and apply the model on the dataset.
5. Predict the values of arrays.
6. Import metrics from sklearn and calculate the MSE and R2 of the model on the dataset.
7. Predict the values of array.
8. Apply to new unknown values. 

## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: MOULIDHAR
RegisterNumber: 212223240042
import pandas as pd
data=pd.read_csv("Salary.csv")
data.head()

data.info()

data.isnull().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()

data["Position"]=le.fit_transform(data["Position"])
data.head()

x=data[["Position","Level"]]
x.head()

y=data[["Salary"]]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.2,random_state=2)

from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)

from sklearn import metrics
mse=metrics.mean_squared_error(y_test, y_pred)
mse

r2=metrics.r2_score(y_test,y_pred)
r2

dt.predict([[5,6]])
*/
```

## Output:
## Initial dataset:
![exp 7 ml 1](https://github.com/Rama-Lekshmi/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/118541549/3013f142-8d1c-42cc-9f11-751ce674ae84)
## Data Info:
![exp 7 ml 2](https://github.com/Rama-Lekshmi/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/118541549/f42b2f9a-d161-4205-90ca-368733c1c156)
## Optimization of null values:
![exp 7 ml 3](https://github.com/Rama-Lekshmi/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/118541549/d0eefc95-353f-4c35-b696-de3a978d8124)
## Converting string literals to numerical values using label encoder:
![exp 7 ml 4](https://github.com/Rama-Lekshmi/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/118541549/4d7e562c-9a2a-48ca-ae0e-5b5f1a429e91)
## Assigning x and y values:
![exp 7 ml 5](https://github.com/Rama-Lekshmi/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/118541549/c56e3af8-065e-45de-93a5-ce9910f6638d)
## Mean Squared Error:
![exp 7 ml 6](https://github.com/Rama-Lekshmi/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/118541549/29e082dc-8dc0-4836-ba51-334497170c7a)
## R2 (variance):
![exp 7 ml 7](https://github.com/Rama-Lekshmi/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/118541549/0fe196e4-baea-44e6-8a90-240185d0cbe7)
## Prediction:
![exp 7 ml 8](https://github.com/Rama-Lekshmi/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/118541549/e69d2f5c-67f7-462a-9ddc-20a51c383016)

## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.

