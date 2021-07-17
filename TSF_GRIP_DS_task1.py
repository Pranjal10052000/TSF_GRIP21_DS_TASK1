print("SOLUTION TO TSF_GRIP21_DS_TASK1 BY PRANJAL KALEKAR")

#importing required libraries
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


#reading data
data = pd.read_csv("data.csv")
print("data")
print(data.head())

#extracting data
X = np.array(data.Hours).reshape(-1,1)
y = np.array(data.Scores)


#training data
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.5, random_state=1)
lm = LinearRegression(fit_intercept = True)
lm.fit(X_train,y_train)

print("model is trained succesfully")
#visualizing trainde model
plt.figure()
plt.scatter(X_train,y_train)
plt.plot(X_train,lm.coef_*X_train+lm.intercept_)
plt.title("Trained model")
plt.xlabel("X_train")
plt.ylabel("y_train")


#testingdata
print("score of linear_regression model on test data ", lm.score(X_test,y_test))

#prediction for the given point in the task
def pred_score(hours):
    score = lm.predict(np.array(hours).reshape(-1,1))
    return print("if you study ", hours," hours, You can get ",score[0], " % score")
 
print("Lets predict score while studing for 9.5 hours")
pred_score(9.5)