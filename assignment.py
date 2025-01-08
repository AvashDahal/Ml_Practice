import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

dataset =pd.read_csv("./Salary Data.csv")
dataset=dataset.dropna()
print(dataset)
dataset=dataset[['Age', 'Salary']]
print(dataset)
X= dataset["Age"].values.reshape(-1,1)
print(X)
Y=dataset["Salary"].values
print(Y)
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=0)
model = LinearRegression()
model.fit(X_train,Y_train)
dataset[dataset['Age'] == 20]
model.predict([[20]])
plt.scatter(X_train,Y_train,color='green')
plt.plot(X_train,model.predict(X_train),color="blue")
plt.xlabel('Age')
plt.ylabel('Salary')
plt.title('Salary vs Age Linear Regression')
plt.legend()
plt.show()