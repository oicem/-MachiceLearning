''' 
    This is the tool to select which one module is preform best 
    You just need a data file that have depended variable in last 
    and data must not contains any String variables
'''

# importing requires modules 

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score as ac
from sklearn.metrics import accuracy_score as acs
# taking input from user

path = input("Enter Csv Data path : ")
Which = input("Enter Reg for Regression or cla for classification : ")
results = []
reg_results = ["Linear Regression ", " Polynominal Regression " , " SVR " , " Decision Tree ", " Random Forest "]
cla_results = ["Logistic Regression ", "K-Nearest Neighbors ", " SVC ", "Kernal SVM " , " Naive Bayes " , " Decision Tress " , "Random Forest "]
# importing data 

data = pd.read_csv(path)
X = data.iloc[:,:-1].values
y = data.iloc[:, -1].values

# splitting data into Train and test

from sklearn.model_selection import train_test_split

X_train , X_test, y_train, y_test = train_test_split(X,y, test_size=0.25, random_state = 0)

# feature scaling the X_test and X_train

from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
X_train_Scaler=ss.fit_transform(X_train)
X_test_Scaler =ss.fit_transform(X_test) 

# first for regression

if(Which.lower()=="reg" or Which.lower()=="regression"):
    
    # multiRegression 
    
    from sklearn.linear_model import LinearRegression
    
    reg = LinearRegression()
    reg.fit(X_train, y_train)
    results.append(ac(y_test,reg.predict(X_test)))
    
    # polyregression
    
    from sklearn.preprocessing import PolynomialFeatures
    
    pf = PolynomialFeatures(degree=4)
    X_poly = pf.fit_transform(X_train)
    reg_poly = LinearRegression()
    reg_poly.fit(X_poly, y_train)
    results.append(ac(y_test, reg_poly.predict(pf.fit_transform(X_test))))
    
    # SVM
    
    from sklearn.svm import SVR 
    
    reg = SVR(degree=4)
    reg.fit(X_train_Scaler, y_train)
    results.append(ac(y_test, reg.predict(X_test_Scaler)))
    
    # Desision Tree
    
    from sklearn.tree import DecisionTreeRegressor
    
    reg = DecisionTreeRegressor(random_state=0)
    reg.fit(X_train, y_train)
    results.append(ac(y_test, reg.predict(X_test)))
    
    # random forest
    
    from sklearn.ensemble import RandomForestRegressor
    
    reg = RandomForestRegressor(n_estimators=200, random_state=0)
    reg.fit(X_train, y_train)
    results.append(ac(y_test, reg.predict(X_test)))

else :
    # this is for classification
    
    # Logisitic Classification
    
    from sklearn.linear_model import LogisticRegression
    
    cla = LogisticRegression()
    cla.fit(X_train_Scaler, y_train)
    results.append(acs(y_test, cla.predict(X_test_Scaler)))
    
    # K_Nearnest
    
    from sklearn.neighbors import KNeighborsClassifier
    
    cla = KNeighborsClassifier(n_neighbors=10)
    cla.fit(X_train_Scaler, y_train)
    results.append(acs(y_test, cla.predict(X_test_Scaler)))
    
    # SVM
    
    from sklearn.svm import SVC
    
    cla = SVC(kernel = 'linear',random_state=0)
    cla.fit(X_train_Scaler, y_train)
    results.append(acs(y_test, cla.predict(X_test_Scaler)))
    
    # Kernal 
    
    cla = SVC(random_state=0)
    cla.fit(X_train_Scaler, y_train)
    results.append(acs(y_test, cla.predict(X_test_Scaler)))
    
    # Naive Bayes
    
    from sklearn.naive_bayes import GaussianNB
    
    cla = GaussianNB()
    cla.fit(X_train_Scaler, y_train)
    results.append(acs(y_test, cla.predict(X_test_Scaler)))
    
    # Tree
    
    from sklearn.tree import DecisionTreeClassifier
    
    cla = DecisionTreeClassifier(random_state=(0))
    cla.fit(X_train_Scaler, y_train)
    results.append(acs(y_test, cla.predict(X_test_Scaler)))
    
    # random
    
    from sklearn.ensemble import RandomForestClassifier
    
    cla = RandomForestClassifier(random_state=(0), n_estimators=200)
    cla.fit(X_train_Scaler, y_train)
    results.append(acs(y_test, cla.predict(X_test_Scaler)))
    
 
    
i = 0
if(Which.lower()=="reg" or Which.lower()=="regression"):
    for value in results:
      print(reg_results[i] , value*100)
      i = i+1 
else :
   for value in results:
      print(cla_results[i] , value*100)
      i = i+1   
