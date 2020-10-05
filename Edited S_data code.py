import pandas as pd# Importing all libraries required in this notebook
import numpy as np  
import matplotlib.pyplot as plt
import seaborn as sns  
%matplotlib inline


s_data = pd.read_csv("http://bit.ly/w-data")# Reading data from remote link
print("Data imported successfully")
s_data.describe()
s_data.head(10)


x=s_data['Hours']# Plotting the distribution of scores
y=s_data['Scores']
plt.scatter(x,y, label='Data points', color='Red',marker='o')
plt.xlabel('Hours studied')
plt.ylabel('Score Obtained')
plt.title('Hours vs Scores')
plt.grid()



relation=s_data.corr()#Relation
sns.heatmap(relation,annot=True, cmap='mako')
plt.title('There is 98% correation between Hours and Scores')
plt.show()
print('There is 98% correation between Hours and Scores')

X = s_data.iloc[:, :-1].values#Splitting data  
y = s_data.iloc[:, 1].values  
from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                            test_size=0.2, random_state=0) 


from sklearn.linear_model import LinearRegression# Importing Linear Regression Model  
regressor = LinearRegression()  
regressor.fit(X_train, y_train) 
print("Training complete.")


line = regressor.coef_*X+regressor.intercept_# Plotting the regression line
plt.scatter(X, y, color='green' )# Plotting for the test data
plt.plot(X, line, color='blue')
plt.show()


print(X_test) # Testing data - In Hours
y_pred = regressor.predict(X_test) # Predicting the scores


df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})# Comparing Actual vs Predicted  
df 


graph=df.head()#Graph Comparsion
graph.plot(kind='bar',figsize=(20,5))
plt.grid(which='both',color='blue',linestyle='-',linewidth=0.5)
plt.show()

hours = 9.25#Predicting Score of a student studied 9.25 hours
print("No of Hours = {}".format(hours))
print("Predicted Score = ",regressor.predict(np.array(hours).reshape(1,-1))[0])


from sklearn import metrics#Evaluating the performane of the model  
print('Mean Absolute Error:',metrics.mean_absolute_error(y_test, y_pred)) 
