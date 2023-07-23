import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import sklearn 

#importing datasets
df1=pd.read_csv("D:\IBM_MentalHelth_dataset\mental-and-substance-use-as-share-of-disease.csv")
df2=pd.read_csv("D:\IBM_MentalHelth_dataset\prevalence-by-mental-and-substance-use-disorder.csv")
print(df1.head())
print(df2.head())
#merging the two datasets
df=pd.merge(df1,df2)
'''print(df)
#data cleaning'''
df.drop('Code', axis=1,inplace=True)
'''print(df.size)
print(df.shape)'''
df=df.set_axis(['Country','Year','schizophrenia','bipolar_disorder','Eating_disorder','Anxiety','Drug_usage','depression','alcohol','mental fitness'],axis='columns')
'''for col in df.columns:
  
    print(col)'''
  
#data visualization

dt=df.drop(['Country'],axis=1)

for col in dt.columns:
   print(col)
    
data_corr=dt.corr()
sb.heatmap(data_corr,annot=True,cmap='coolwarm',linewidths=1)
plt.show()
sb.pairplot(dt, x_vars=["schizophrenia","bipolar_disorder","Eating_disorder","Anxiety"], y_vars=["mental fitness"], hue="Year" ,diag_kind="kde")
plt.show()

from sklearn.preprocessing import LabelEncoder
l=LabelEncoder() #labelencoder normalisez values
for i in dt.columns:
  if dt[i].dtype == 'object':
    dt[i]=l.fit_transform(df[i])


#data splitting
x=dt.drop(['mental fitness'],axis=1)
y=dt['mental fitness']
from sklearn.model_selection import train_test_split
xtrain,xtest, ytrain, ytest=train_test_split(x,y,test_size=.20,random_state=2)
#random_state ensures that your data splitting is deterministic. it does so by setting the seed to the random generator
print(xtrain.shape)
print(ytrain.shape)
print(xtest.shape)
print(ytest.shape)

#Model Training
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Assuming you have the data split into X_train, X_test, y_train, y_test
# X_train and X_test are your feature data
# y_train and y_test are your target (label) data

# Initialize the Linear Regression model
model = LinearRegression()

# Train the model using the training data
model.fit(xtrain, ytrain)

# Make predictions using the testing data
ypred = model.predict(xtest)

# Evaluate the model's performance
mse = mean_squared_error(ytest, ypred)
r2 = r2_score(ytest, ypred)
#r2 is referred as coefficient of determination, tells about goodness of fit of the model. we have used the function directly


print("Mean Squared Error:", mse)
print("R-squared:", r2)
#implementing random forest

from sklearn.ensemble import RandomForestRegressor


# Initialize the Random Forest Regressor model
rf_model = RandomForestRegressor(n_estimators=100, random_state=2)

# Train the model using the training data
rf_model.fit(xtrain, ytrain)

# Make predictions using the testing data
ypred = rf_model.predict(xtest)

# Evaluate the model's performance
mse_RF = mean_squared_error(ytest, ypred)
r2_RF = r2_score(ytest, ypred)

print("Mean Squared Error (RandomForest):", mse_RF)
print("R-squared(RandomForest):", r2_RF)

if mse < mse_RF:
    print("\nLinear Regression model has lower Mean Squared Error, indicating better performance on the testing data.")
else:
    print("\nRandom Forest model has lower Mean Squared Error, indicating better performance on the testing data.")

if r2> r2_RF:
    print("Linear Regression model has higher R-squared, indicating better goodness of fit on the testing data.")
else:
    print("Random Forest model has higher R-squared, indicating better goodness of fit on the testing data.")


import pickle









