import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

dataset =pd.read_csv("./Salary Data.csv")
dataset=dataset.dropna()
print(dataset)
dataset=dataset[['Years of Experience', 'Salary']]
print(dataset)
X= dataset.iloc[:,0:1].values
print(X)
Y=dataset.iloc[:,1:]
print(Y)
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=0)
model = LinearRegression()
model.fit(X_train,Y_train)
dataset[dataset['Years of Experience'] == 5]
model.predict([[5]])
plt.scatter(X_train,Y_train,color='black')
plt.plot(X_train,model.predict(X_train),color="blue")
plt.show()